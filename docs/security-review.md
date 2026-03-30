# Security Review — vkusvillbot

Дата: 2026-03-20

---

## Сводка

| Severity | Кол-во | Ключевые риски |
|----------|:------:|----------------|
| CRITICAL | 1 | Реальные секреты в .env на диске (требуется ротация) |
| HIGH | 8 | Prompt injection, нет rate limiting, утечка ошибок пользователю, нет валидации /city /diet |
| MEDIUM | 7 | PII в логах, API key как str (не SecretStr), docker volume mount, MCP URL в конфиге |
| LOW | 5 | f-string в PRAGMA (хардкод), логи без ротации, SQLite без шифрования |
| INFO | 4 | .gitignore ок, httpx логирование заглушено, FTS5 санитизация корректна |

---

## Находки

### CRITICAL

| # | Файл:строка | Описание | Рекомендация |
|---|-------------|----------|--------------|
| 1 | `.env`:1-3 | Реальные production-секреты в plaintext: Telegram bot token, OpenRouter API key, proxy IP. Файл исключён из git через .gitignore и не попадал в историю, но был прочитан в рамках аудита | **Немедленно ротировать** оба секрета (BotFather `/revoke`, OpenRouter regenerate). Использовать менеджер секретов или `chmod 600` |

### HIGH

| # | Файл:строка | Описание | Рекомендация |
|---|-------------|----------|--------------|
| 2 | `sgr_agent.py:173` | **LLM Prompt Injection.** Пользовательский текст из Telegram вставляется в messages как `role: "user"` без санитизации. Можно внедрить `"Ignore all previous instructions..."`, имитировать `TOOL_RESULT`, подменить JSON-формат | Санитизировать user_text: экранировать паттерны `TOOL_RESULT`, `TOOL_ERROR`, `{"action":`. Ограничить длину до 2000 символов |
| 3 | `sgr_agent.py:316-320` | **TOOL_RESULT spoofing.** Результат инструмента вставляется как `"TOOL_RESULT {tool}: {json}"` с `role: "user"`. Пользователь может сымитировать этот формат в своём сообщении | Использовать `role: "tool"` или уникальный маркер/nonce |
| 4 | `main.py:410-439` | **Injection через /diet и /city.** Текст записывается в БД и подставляется в system prompt LLM (`prompts.py:13-14`) без валидации длины и содержимого. Вектор persistent prompt injection | Ограничить `diet_notes` ≤ 500 символов, `city` ≤ 100 символов. Фильтровать injection-паттерны |
| 5 | `main.py:447-646` | **Нет rate limiting.** Любой Telegram-пользователь может спамить, запуская SGR-циклы (12 шагов LLM + MCP). Нет ограничения concurrency | Per-user throttle через aiogram middleware. `asyncio.Semaphore` на 10 одновременных запросов |
| 6 | `sgr_agent.py:177` | **Нет общего timeout на SGR-цикл.** До 12 итераций × 60с = 720с (12 минут) на один запрос | `asyncio.wait_for(agent.run(...), timeout=120)` |
| 7 | `main.py:616,630` | **Утечка ошибок пользователю.** `f"Ошибка MCP: {exc}"` / `f"Ошибка: {exc}"` — технические детали (URL, пути, трейсбеки) отправляются в Telegram | Generic сообщение пользователю, детали только в лог |
| 8 | `llm_client.py:61` | **Тело HTTP-ответа в исключении.** `raise LLMError(f"OpenRouter error: {resp.status_code} {resp.text}")` — может содержать rate limit info, API details. Может всплыть к пользователю через main.py:630 | В исключение только status_code, тело в лог |
| 9 | `main.py:526` | **Нет валидации длины пользовательского ввода.** `text = message.text or ""` — Telegram допускает до 4096 символов, нет фильтрации null bytes и control characters | Обрезать до 2000 символов, удалять `\x00`-`\x1f` (кроме `\n`, `\t`) |

### MEDIUM

| # | Файл:строка | Описание | Рекомендация |
|---|-------------|----------|--------------|
| 10 | `main.py:528-543` | **PII в логах.** `dialog_logger.info()` пишет полный текст пользователей и ответов с tg_id в `logs/dialog.log`. Персональные данные в plaintext | Ротация логов (`RotatingFileHandler`), retention policy, маскирование tg_id |
| 11 | `sgr_agent.py:181` | **Raw LLM response в логах.** `dialog_logger.info("LLM_RAW ...")` — полный ответ LLM может содержать PII из контекста диалога | Ограничить длину логируемого ответа |
| 12 | `llm_client.py:24` | **API key как str.** `self.api_key = api_key` — обычная строка в памяти, доступна через stack trace. В config.py используется `repr=False`, но не `SecretStr` | `pydantic.SecretStr` для api_key и token. `.get_secret_value()` только при отправке |
| 13 | `sgr_agent.py:156-157` | **cart_items без ограничений.** Pydantic валидирует типы (`xml_id: int`, `q: float`), но `q` может быть 999999999 или отрицательным. Нет лимита количества | `q: float = Field(gt=0, le=999)`, max 50 элементов |
| 14 | `config.yaml:8` | **MCP URL закоммичен.** `https://mcp001.vkusvill.ru/mcp` раскрывает внутреннюю инфраструктуру | Вынести в env, в YAML — placeholder |
| 15 | `docker-compose.yml:11` | **Volume mount всего проекта.** `./:/app` маунтит включая `.env`, `.git/`, `logs/` | Для production — маунтить только необходимое |
| 16 | `Dockerfile:13` | **Dev-зависимости в production.** `pip install -e .[dev]` ставит pytest, ruff | Multi-stage build или убрать `[dev]` |

### LOW

| # | Файл:строка | Описание | Рекомендация |
|---|-------------|----------|--------------|
| 17 | `db.py:123,127` | **f-string в SQL DDL.** `f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"` — значения из хардкода, не из user input. Безопасно, но хрупко | Whitelist-валидация имён |
| 18 | `db.py:259-261` | **role без whitelist.** `save_message(role)` принимает произвольную строку | `assert role in {"user", "assistant"}` |
| 19 | `logging.py:21` | **Лог без ротации.** `FileHandler` — файл растёт без ограничений | `RotatingFileHandler` (50MB, 5 backup) |
| 20 | `db.py:58` | **SQLite без шифрования.** Хранит tg_id, историю, настройки питания (PII) | Рассмотреть sqlcipher или ограничить доступ к файлу |
| 21 | `sgr_agent.py:215-286` | **Tool args без клэмпинга.** `page`, `limit`, `product_id` из LLM-ответа не ограничены по диапазону | `page = max(1, min(page, 100))`, `limit = max(1, min(limit, 50))` |

### INFO

| # | Файл:строка | Описание |
|---|-------------|----------|
| 22 | `.gitignore` | Корректно исключает `.env`, `*.db`, `*.sqlite*`, `*.faiss`, `logs/`. `.env` никогда не попадал в git |
| 23 | `config.py:18,45-46` | `repr=False` на `token`, `api_key`, `proxy_url` — предотвращает случайный вывод |
| 24 | `logging.py:10-13` | httpx логгер принудительно на WARNING — защита от утечки Telegram-токена в URL |
| 25 | `db.py:845-852` | FTS5 санитизация корректна: `re.sub(r"[^0-9a-zа-яё_]+", "", token)` удаляет спецсимволы FTS5 |

---

## Анализ по OWASP Top 10

| Категория | Статус | Комментарий |
|-----------|--------|-------------|
| **A01: Broken Access Control** | N/A | Telegram бот, нет web-интерфейса, авторизация через Telegram |
| **A02: Cryptographic Failures** | LOW | API key как str, не SecretStr. Нет шифрования БД с PII |
| **A03: Injection** | **HIGH** | SQL — защищён (параметризация). LLM Prompt Injection — **не защищён** |
| **A04: Insecure Design** | **HIGH** | Нет rate limiting, нет timeout, нет валидации длины ввода |
| **A05: Security Misconfiguration** | MEDIUM | MCP URL в репозитории, dev-deps в production, volume mount |
| **A06: Vulnerable Components** | LOW | Зависимости актуальны. `fastmcp==2.12.4` и `mcp==1.17.0` зафиксированы |
| **A07: Auth Failures** | N/A | Нет собственного auth — Telegram handles |
| **A08: Data Integrity Failures** | LOW | Нет проверки целостности конфига, YAML без подписи |
| **A09: Logging Failures** | MEDIUM | PII в логах, нет ротации, нет маскирования |
| **A10: SSRF** | N/A | Нет user-controlled URL requests |

---

## Зависимости

| Зависимость | Версия | CVE | Статус |
|-------------|--------|-----|--------|
| aiogram | >=3.4.1 | — | Актуальна |
| httpx | >=0.28.1 | — | Актуальна |
| pydantic | >=2.7.0 | — | Актуальна |
| faiss-cpu | >=1.9.0 | — | Актуальна |
| fastmcp | ==2.12.4 | — | Зафиксирована, проверять обновления |
| mcp | ==1.17.0 | — | Зафиксирована, проверять обновления |
| PyYAML | >=6.0.1 | — | `yaml.safe_load` используется корректно |
| numpy | >=1.26.0 | — | Актуальна |

**Рекомендация:** Добавить `pip-audit` или `safety` в CI.

---

## Топ-5 приоритетных действий

1. **Ротировать секреты** — Telegram token и OpenRouter API key были прочитаны в рамках аудита
2. **Rate limiting + timeout** — per-user throttle, `asyncio.wait_for(agent.run(), timeout=120)`, `Semaphore(10)`
3. **Санитизация user input перед LLM** — фильтрация injection-паттернов, ограничение длины, валидация /city /diet
4. **Не отправлять str(exc) пользователю** — generic сообщение, детали в лог
5. **PII в логах** — `RotatingFileHandler`, маскирование tg_id, retention policy
