# Architecture Review — vkusvillbot

Дата: 2026-03-20

---

## Сводка оценок

| Критерий | Оценка | Вердикт |
|----------|:------:|---------|
| Separation of Concerns | 5/10 | `db.py` — god object, `main.py` смешивает composition root с transport |
| Coupling | 6/10 | Нет циклов, DAG неглубокий, но `main.py` знает про всё |
| Cohesion | 5/10 | Инфра-модули идеальны, но три ключевых файла — каша из обязанностей |
| Extensibility | 4/10 | Tool dispatch на if/elif, нет абстракций для VectorIndex и Database |
| Testability | 5/10 | 7 тестов, частичный DI, side effects в конструкторах |
| Error Boundaries | 7/10 | Кастомные ошибки, graceful degradation, но нет retry и сырые сообщения |
| Configuration | 7/10 | Pydantic + YAML + env, repr=False для секретов, но ленивая валидация |
| **Средняя** | **5.6/10** | |

---

## 1. Separation of Concerns — 5/10

### Что хорошо

Инфраструктурные клиенты чётко изолированы — каждый отвечает за один внешний API:
- `llm_client.py` — HTTP к OpenRouter LLM
- `embeddings_client.py` — HTTP к OpenRouter Embeddings
- `mcp_client.py` — MCP-протокол к ВкусВилл
- `vector_index.py` — FAISS-обёртка
- `formatting.py` — Markdown-конверсия
- `config.py` — только загрузка конфигурации

### Что плохо

**`db.py` — God Object (1250 строк).** Совмещает минимум 6 обязанностей:

| Обязанность | Строки | Примеры методов |
|-------------|--------|-----------------|
| DDL/миграции | 66–118 | `_ensure_schema()`, `_ensure_column()` |
| CRUD пользователей | 201–231 | `get_or_create_user()` |
| CRUD сообщений/сессий | 233–307 | `save_message()`, `save_session()` |
| FTS5 поиск | 129–193, 661–683 | `ensure_fts()`, `_tokenize_query()` |
| Поиск товаров | 328–384 | `search_products()` |
| Nutrition-логика | 434–584 | `nutrition_query()`, `_extract_nutrition_metrics()` |
| Embeddings/upsert | 586–789 | `upsert_products_from_mcp()` |
| Парсинг свойств | 1042–1065 | `_parse_weight()`, `_normalize_category_tokens()` |

**`main.py` (657 строк) — God Function.** `main()` и `on_text()` совмещают:
- Composition root (сборка зависимостей, строки 272–330)
- Telegram transport (обработка сообщений, topic routing, строки 332–460)
- Presentation (форматирование, разбиение, pseudo-streaming, строки 85–270)
- Progress management (два механизма: DraftProgress и MessageProgress)
- Error handling (fallback routing, markdown→plain)

**`sgr_agent.py` — смешанная ответственность.** Агент одновременно:
- Оркестрирует LLM-цикл (reasoning → tool_call → final)
- Диспатчит инструменты через if/elif на 90 строк (строки 213–304)
- Маппит данные из MCP (`_mcp_item_to_compact()`, `_merge_items()`, `_normalize_weight()`)
- Управляет корзиной

### Дублирование между db.py и product_retriever.py

5 идентичных функций:
- `_normalize_category_tokens()` — db.py:903 / product_retriever.py:20
- `_match_categories()` — db.py:930 / product_retriever.py:29
- `_extract_nutrition_metrics()` — db.py:1017 / product_retriever.py:46
- `_parse_filter_expr()` — db.py:943 / product_retriever.py:141
- `_compare()` — db.py:967 / product_retriever.py:166

---

## 2. Coupling — 6/10

### Матрица зависимостей

```
Efferent coupling (от кого зависит модуль):

main.py          → 12 модулей (composition root)
sgr_agent.py     → 5 (db, mcp_client, models, product_retriever, prompts)
product_retriever → 3 (db, embeddings_client, vector_index)
prompts.py       → 1 (models)
остальные 11     → 0

Afferent coupling (кто зависит от модуля):

db               ← 3 (product_retriever, sgr_agent, main)
models           ← 3 (prompts, sgr_agent, main)
mcp_client       ← 2 (sgr_agent, main)
embeddings_client ← 2 (product_retriever, main)
vector_index     ← 2 (product_retriever, main)
manual_llm       ← 0 (никто)
```

### Циклические зависимости

**Нет.** Граф — строгий DAG. Глубина: 3 уровня (`main → sgr_agent → product_retriever → db`).

### Проблемы

**Дублирование `SgrConfig`.** В `config.py:49` — Pydantic BaseModel, в `sgr_agent.py:111` — dataclass. Одинаковые поля, два разных типа. В `main.py:320-327` происходит ручной маппинг:

```python
sgr_config = SgrConfig(  # sgr_agent.SgrConfig (dataclass)
    max_steps=settings.sgr.max_steps,  # config.SgrConfig (Pydantic)
    max_items_per_search=settings.sgr.max_items_per_search,
    ...
)
```

**`main.py` знает про всё.** 12 из 15 модулей — для composition root это нормально, но main.py также содержит бизнес-логику (topic routing, progress), что делает его hub-модулем.

---

## 3. Cohesion — 5/10

| Модуль | Обязанностей | Cohesion |
|--------|:------------:|----------|
| config.py | 1 | Высокая |
| models.py | 1 | Высокая |
| logging.py | 1 | Высокая |
| prompts.py | 1 | Высокая |
| llm_client.py | 1 | Высокая |
| embeddings_client.py | 1 | Высокая |
| mcp_client.py | 1 | Высокая |
| vector_index.py | 1 | Высокая |
| formatting.py | 1 | Высокая |
| manual_llm.py | 1 | Высокая |
| telegram_draft.py | 2 | Средняя |
| product_retriever.py | 3 | Средняя |
| **sgr_agent.py** | **4** | **Низкая** |
| **main.py** | **5+** | **Низкая** |
| **db.py** | **6+** | **Низкая** |

10 из 15 модулей имеют идеальную cohesion (1 обязанность). Проблема сконцентрирована в трёх самых больших файлах.

---

## 4. Extensibility — 4/10

### Добавить новый инструмент в SGR-агент

Нужно:
1. Дописать `elif tool == "new_tool":` в `sgr_agent.py:run()` (строки 213–304) — 90-строчный if/elif
2. Обновить `_summarize_tool_result()` (строка 382)
3. Обновить промпт в `prompts.py`

**Нет реестра инструментов, нет базового класса Tool, нет маппинга `name → handler`.** Всё на if/elif.

### Заменить FAISS на другой поисковый движок

`FaissVectorIndex` — конкретный класс без абстракции. Используется напрямую в `product_retriever.py` и `main.py`. Нет Protocol/ABC `VectorIndex`. Замена требует изменений в 3+ файлах.

### Заменить LLM-провайдера

**Есть Protocol `LLMClient`** в `sgr_agent.py:21` — можно подменить реализацию (что и делает `manual_llm.py`). Это единственная нормальная абстракция.

**Есть Protocol `EmbeddingsClient`** в `embeddings_client.py:12` — `ProductRetriever` принимает через DI.

### Заменить базу данных

Невозможно без переписывания `db.py` целиком. Класс `Database` жёстко привязан к SQLite. Нет Repository/DAO.

---

## 5. Testability — 5/10

### Текущее покрытие

| Тестовый файл | Тестов | Что тестирует |
|---------------|:------:|---------------|
| test_config.py | 1 | Settings.load() с env overrides |
| test_sgr_parser.py | 3 | Парсинг JSON-ответа LLM |
| test_message_threads.py | 1 | Thread-scoped история |
| test_vector_search.py | 2 | nutrition_query + semantic_search |
| **Итого** | **7** | |

**Не покрыто:** `SgrAgent.run()` (основной цикл!), все Telegram-хэндлеры, `mcp_client`, `formatting`, `telegram_draft`.

### DI и side effects

- `ProductRetriever.__init__` — принимает `db`, `embeddings`, `index` через DI. Хорошо.
- `SgrAgent.__init__` — принимает `mcp`, `llm`, `db`, `retriever`, `config`, `profile`. Хорошо.
- `Database.__init__` — **side effect**: открывает SQLite, создаёт таблицы, мигрирует схему (`_ensure_schema()`). Тесты обязаны создавать реальный файл БД.
- `main()` — **не принимает параметров**, всё собирается внутри. Нетестируема.

---

## 6. Error Boundaries — 7/10

### Уровни обработки ошибок

| Уровень | Реализация | Качество |
|---------|-----------|----------|
| Tool level | `sgr_agent.py:322` — общий `except Exception`, ошибка → `TOOL_ERROR` в контекст LLM | Работает, но нет отдельной обработки per-tool |
| Agent level | `sgr_agent.py:339-366` — grace-попытка при исчерпании шагов, fallback "Не успел завершить запрос" | Хорошо |
| Transport level | `main.py` — `_safe_send` с fallback routing, Markdown→plain, Draft→MessageProgress | Отлично, 5+ fallback-механизмов |
| Infra level | `db.py:191` — FTS5 fallback на LIKE; `vector_index.py:9` — faiss import fallback | Частично |

### Кастомные исключения

Определены и используются: `MCPError`, `LLMError`, `EmbeddingsError`, `VectorIndexError`, `TelegramAPIError`, `TelegramBadRequest`.

### Graceful degradation

- FAISS недоступен → TOOL_ERROR → LLM переключается на другие инструменты
- FTS5 недоступен → fallback на LIKE
- MCP недоступен → ошибка передаётся LLM как TOOL_ERROR
- telegramify_markdown недоступен → ручной escape
- Draft API недоступен → fallback на edit-message
- Topics не поддерживаются → работа без draft-стриминга

### Проблемы

- Нет retry-логики ни на одном уровне
- Сырые технические сообщения для пользователя: `"Ошибка: {exc}"`
- `sgr_agent.py:322` — ошибки инструментов логируются без traceback
- `mcp_client.py:31` — `json.loads` оборачивается в MCPError без `from exc`

---

## 7. Configuration — 7/10

### Архитектура

Pydantic `BaseModel` с иерархией секций: `Settings → AppConfig, TelegramConfig, MCPConfig, DBConfig, VectorConfig, LLMConfig, SgrConfig`.

Три источника (по приоритету): env vars > YAML (`config.yaml`) > defaults в Pydantic-моделях.

### Сильные стороны

- `repr=False` для секретов (`telegram.token`, `llm.api_key`, `telegram.proxy_url`, `llm.proxy_url`)
- httpx логгер принудительно на WARNING — защита от утечки токена в логи
- YAML молча пропускается если не существует — бот может работать только на env vars
- Все компоненты принимают зависимости через конструктор — легко переопределить для тестов

### Проблемы

| Проблема | Пример |
|----------|--------|
| Ленивая валидация API key | `OPENROUTER_API_KEY` проверяется при первом запросе, а не при старте |
| Нет fail-fast для зависимостей | MCP connect в `main.py:293` без try/except — бот крашится |
| Нет проверки диапазонов | `temperature` может быть -100, `max_steps` может быть 0 |
| Хардкод MCP URL | `https://mcp001.vkusvill.ru/mcp` как дефолт |
| Дублирование SgrConfig | Pydantic в config.py vs dataclass в sgr_agent.py |

---

## Топ-5 архитектурных рекомендаций

### 1. Разбить db.py на модули (Приоритет: высокий)

**Проблема:** God Object на 1250 строк с 6+ обязанностями. Дублирование 5 функций с product_retriever.py.

**Решение:**
```
db/
  schema.py          — DDL, миграции, _ensure_schema()
  repository.py      — CRUD users, messages, sessions
  product_queries.py — search_products(), nutrition_query(), FTS
  embeddings_store.py — upsert_embeddings(), load_embeddings()
  utils.py           — _parse_weight(), _normalize_category_tokens(), _compare()
                       (общий для db и product_retriever)
```

**Эффект:** Устранение дублирования (~250 строк), каждый модуль < 300 строк, тестируемость отдельных слоёв.

### 2. Вынести tool dispatch из sgr_agent.py в реестр (Приоритет: высокий)

**Проблема:** 90-строчный if/elif для диспатча инструментов. Добавление нового tool требует правки run() и _summarize_tool_result().

**Решение:**
```python
# tools.py
TOOL_HANDLERS: dict[str, Callable] = {
    "vkusvill_products_search": handle_search,
    "vkusvill_products_details": handle_details,
    "vkusvill_cart_create": handle_cart,
    ...
}
```

**Эффект:** SgrAgent.run() становится чистым оркестратором. Добавление tool — одна функция + одна строка в словарь.

### 3. Отделить Telegram transport от main.py (Приоритет: средний)

**Проблема:** main.py — и composition root, и transport, и presentation, и progress management.

**Решение:** Вынести в `telegram_bot.py` (хэндлеры, topic routing, progress) и `presentation.py` (split_text, pseudo_stream, draft). В main.py оставить только сборку зависимостей.

**Эффект:** main.py < 50 строк, каждый модуль тестируем отдельно.

### 4. Добавить абстракции VectorIndex и Database (Приоритет: средний)

**Проблема:** Замена FAISS или SQLite требует правки 3+ файлов. Нет интерфейсов.

**Решение:** `Protocol VectorIndex` (search, load) + `Protocol DatabasePort` (get_user, save_message, search_products). ProductRetriever и SgrAgent зависят от Protocol, не от конкретных классов.

**Эффект:** Замена storage/search — один файл, без правки потребителей.

### 5. Убрать дублирование SgrConfig и добавить fail-fast валидацию (Приоритет: низкий)

**Проблема:** Два SgrConfig (Pydantic и dataclass) с ручным маппингом. API key не проверяется при старте.

**Решение:** Использовать один SgrConfig из config.py. Добавить `Settings.validate_critical()` при старте — проверить API key, MCP connect, наличие FAISS.

**Эффект:** Бот падает при старте с понятным сообщением, а не при первом запросе пользователя.
