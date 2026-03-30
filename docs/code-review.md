# Code Review — vkusvillbot

Дата: 2026-03-20

---

## Находки

| # | Severity | Файл | Строки | Тип | Описание |
|---|----------|------|--------|-----|----------|
| 1 | CRITICAL | db.py | 1084 | Bug | Regex `\\d` вместо `\d` в `_parse_weight()` — ищет литеральные символы `\d`, вес никогда не парсится |
| 2 | CRITICAL | db.py | 833 | Bug | Regex `[\\w\\d]+` в `_tokenize_query()` — аналогичная проблема, FTS-токенизация сломана |
| 3 | CRITICAL | db.py | 56–61 | Concurrency | `sqlite3.connect()` без `check_same_thread=False` в async-коде — race condition при параллельных handler-ах |
| 4 | CRITICAL | main.py | 332, 359 | Race Condition | `pending_topics` (dict) используется без синхронизации из async handler-ов — потеря routing при параллельных сообщениях |
| 5 | HIGH | llm_client.py, embeddings_client.py | 57, 58 | Resources | Новый `httpx.AsyncClient` создаётся на каждый запрос — нет переиспользования TCP-соединений |
| 6 | HIGH | mcp_client.py | 19–26 | Resources | Ручной вызов `__aenter__`/`__aexit__` вместо `async with` — при ошибке в connect() ресурсы не освободятся |
| 7 | HIGH | main.py | 144–150 | Error Handling | `except Exception: return` без логирования в `safe_progress()` — ошибки прогресс-коллбэка тихо проглатываются |
| 8 | HIGH | main.py | 553–554 | Data Integrity | `save_message()` вызывается дважды без общей транзакции — при ошибке между вызовами user-сообщение сохранится, assistant — нет |
| 9 | HIGH | sgr_agent.py | 322–337 | Error Handling | Нет backoff/rate-limiting при повторяющихся ошибках tool — при падении MCP быстро исчерпываются шаги SGR и лимиты OpenRouter |
| 10 | HIGH | main.py | 573–608 | Telegram API | Fallback с Markdown на plain-text при ошибке парсинга, но `to_telegram_markdown()` может вернуть невалидный Markdown |
| 11 | HIGH | vector_index.py | 64 | Logic | `strict=False` в `zip(ids, distances)` — при расхождении длин результаты молча теряются |
| 12 | HIGH | db.py | 208–217 | Performance | SELECT после INSERT вместо `RETURNING` или `lastrowid` — лишний запрос к БД |
| 13 | MEDIUM | sgr_agent.py | 52–55 | Bug | Жадный regex `\{.*\}` (re.S) для извлечения JSON — при нескольких `{...}` в ответе LLM захватит от первого `{` до последнего `}` |
| 14 | MEDIUM | main.py | 552 | Reliability | Нет timeout для `agent.run()` — при зависании LLM/MCP handler будет ждать бесконечно |
| 15 | MEDIUM | sgr_agent.py | 183–196 | Logic | При повторных ошибках парсинга JSON цикл тратит шаги SGR на просьбы «повтори в JSON» без ограничения попыток |
| 16 | MEDIUM | sgr_agent.py | 168–173 | Memory | Список messages растёт без ограничения (до 24+ сообщений за сессию) — риск исчерпания контекстного окна LLM |
| 17 | MEDIUM | product_retriever.py | 247–248 | Error Handling | `emb[0]` без проверки длины — при пустом ответе embeddings будет IndexError |
| 18 | MEDIUM | sgr_agent.py | 322–337 | Error Handling | `except Exception` перехватывает слишком широко — может скрыть системные ошибки (хотя не BaseException) |
| 19 | MEDIUM | manual_llm.py | 20–21 | Error Handling | `except Exception: pass` при форматировании JSON — молчаливое игнорирование ошибок |
| 20 | MEDIUM | main.py | 507–514 | Resources | Отмена `typing_task` может оставить HTTP-запрос `send_chat_action` в подвешенном состоянии |
| 21 | MEDIUM | main.py | 106–110 | Logic | Логика разбиения текста с backslash-escaping хрупкая — может неправильно обработать текст с `\` |
| 22 | LOW | sgr_agent.py | 154–162 | Validation | Нет валидации `cart_items` (xml_id, q) перед отправкой в MCP |
| 23 | LOW | db.py | 340 | Design | При отсутствии FTS-индекса поиск откатывается на LIKE без уведомления — пользователь не знает о деградации |
| 24 | LOW | main.py | 528–543 | Security | Весь диалог пользователя логируется в файл, включая потенциально приватную информацию |
| 25 | LOW | formatting.py | 8–9 | Code Quality | `except Exception` при импорте `telegramify_markdown` с `# pragma: no cover` — может скрыть ошибки импорта |

---

## Детали критических находок

### 1. Двойное экранирование regex (CRITICAL)

**db.py:833** — `_tokenize_query()`:
```python
tokens = re.findall(r"[\\w\\d]+", query, flags=re.UNICODE)
```

**db.py:1084** — `_parse_weight()`:
```python
match = re.search(r"(\\d+[\\.,]?\\d*)\\s*([^\\d\\s]+)", text)
```

В raw-строках `r"..."` обратный слеш не экранируется Python-ом, поэтому `\\d` — это два символа: `\` и `d`. Regex ищет литеральную последовательность `\d` вместо класса цифр.

**Последствия:**
- `_tokenize_query()` не извлечёт ни одного токена → FTS-поиск вернёт пустые результаты
- `_parse_weight()` не распарсит вес → расчёт цены за кг/л не работает

**Исправление:** Убрать лишний `\`: `r"[\w\d]+"` и `r"(\d+[.,]?\d*)\s*([^\d\s]+)"`.

### 2. SQLite без check_same_thread (CRITICAL)

**db.py:56–61:**
```python
self.conn = sqlite3.connect(self.db_path)
```

Бот работает через aiogram (async), handler-ы могут выполняться в разных потоках/корутинах одновременно. SQLite по умолчанию запрещает доступ из другого потока (`check_same_thread=True`).

**Последствия:** `ProgrammingError` при конкурентном доступе, потенциальная corruption БД.

**Исправление:** Добавить `check_same_thread=False` или мигрировать на `aiosqlite`.

### 3. Race condition в pending_topics (CRITICAL)

**main.py:332, 359:**
```python
pending_topics: dict[int, tuple[dict[str, int], float]] = {}
# В on_forum_topic_created:
pending_topics[int(message.chat.id)] = (routing, time.monotonic())
# В on_text:
pending_routing = consume_pending_routing(int(message.chat.id))
```

Обычный dict без синхронизации. При быстрых последовательных сообщениях в один чат routing может потеряться или перезаписаться.

**Исправление:** Использовать `asyncio.Lock` вокруг доступа к `pending_topics`.

---

## Положительные стороны

- **Строгий JSON-контракт с LLM** — агент принимает только `tool_call`/`final`, невалидные ответы обрабатываются
- **Логирование диалогов** — полная история через `dialog_logger` для отладки
- **Защита от утечки токена** — httpx логгер выставлен на WARNING
- **Graceful fallback** — при ошибке инструмента агент продолжает работу с TOOL_ERROR
- **Разбиение длинных сообщений** — `_split_text()` корректно обрабатывает лимиты Telegram
- **Семантический + полнотекстовый поиск** — комбинация FAISS и FTS5 с ранжированием
- **Модульная архитектура** — чёткое разделение: config, db, llm_client, sgr_agent, retriever
- **CI/CD** — автодеплой через GitHub Actions на self-hosted runner

---

## Общая оценка: 5 / 10

**Обоснование:**

Архитектура проекта продуманная — модули хорошо разделены, контракт с LLM строгий, есть CI/CD. Однако два CRITICAL-бага в регулярных выражениях буквально ломают ключевую функциональность (FTS-поиск и парсинг веса). SQLite без учёта async-контекста и race condition в `pending_topics` создают реальные риски data corruption на production.

На уровне ресурсов: httpx-клиент создаётся заново на каждый LLM/embeddings запрос, что при нагрузке приведёт к исчерпанию соединений. Error handling местами слишком широкий (`except Exception: pass`), что скрывает реальные проблемы.

**Приоритеты исправления:**
1. **Немедленно:** regex в db.py (#1, #2) — ломают функциональность
2. **Срочно:** SQLite threading (#3), pending_topics sync (#4)
3. **На этой неделе:** httpx client reuse (#5), transaction safety (#8), timeout (#14)
4. **Планово:** error handling улучшения, messages limit, валидация
