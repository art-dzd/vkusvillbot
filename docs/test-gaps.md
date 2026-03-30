# Test Gaps: vkusvillbot

Дата: 2026-03-20

---

## Общая статистика

| Метрика | Значение |
|---------|----------|
| Тестовых файлов | 4 |
| Тестовых функций | 7 |
| Строк тестов | ~322 |
| Исходных .py файлов | 16 |
| Покрыто полностью | 1 (config.py) |
| Покрыто частично | 4 (db.py, sgr_agent.py, product_retriever.py, vector_index.py) |
| Без тестов | 9 модулей |
| conftest.py | отсутствует |

**Конфигурация pytest:** `pyproject.toml` → `addopts = "-q"`, `pythonpath = ["src"]`, pytest ≥ 8.0.0

---

## Покрытые модули

### tests/test_config.py (1 тест)

| Тест | Что проверяет |
|------|---------------|
| `test_settings_load_env_overrides` | Settings.load() — YAML + env override. Проверяет приоритет env над YAML для token, api_key, proxy_url, db_path |

**Fixtures:** `tmp_path`, `monkeypatch`
**Качество:** Хорошее — проверяет основной сценарий. Не хватает: невалидный YAML, отсутствующий файл, пустые значения.

### tests/test_sgr_parser.py (3 теста)

| Тест | Что проверяет |
|------|---------------|
| `test_parse_tool_call` | parse_llm_output() → ToolCall: парсинг JSON с action=tool_call |
| `test_parse_final` | parse_llm_output() → FinalAnswer: парсинг JSON с action=final, cart_items |
| `test_parse_json_with_noise` | Извлечение JSON из текста с "шумом" вокруг — реальный сценарий LLM |

**Качество:** Хорошее. Не хватает: невалидный JSON, пустой ответ, неизвестный action, отсутствие обязательных полей, nested JSON.

### tests/test_message_threads.py (1 тест)

| Тест | Что проверяет |
|------|---------------|
| `test_history_scoped_by_thread_id` | Database: миграция thread_id, изоляция истории по thread_id (100, 200, None) |

**Fixtures:** `tmp_path`
**Качество:** Отличное — тестирует миграцию + бизнес-логику изоляции тредов.

### tests/test_vector_search.py (2 теста)

| Тест | Что проверяет |
|------|---------------|
| `test_nutrition_query_supports_multi_sort` | Database.nutrition_query(): мультисортировка protein DESC + price ASC |
| `test_product_retriever_semantic_search` | ProductRetriever: FAISS search → правильный товар, price_per_l |

**Fixtures:** `tmp_path`, `FakeEmbeddings` (custom mock)
**Качество:** Отличное — интеграционный тест с реальным FAISS индексом и SQLite.

---

## Непокрытые модули

### 1. main.py — Telegram dispatcher
**Риск: CRITICAL**

| Функция | Строк | Описание |
|---------|:-----:|----------|
| `main()` | ~80 | Entry point: dispatcher, middleware, aiogram bot |
| `_pseudo_stream_plain()` | ~60 | Поэтапное редактирование сообщения (стриминг) |
| `_safe_send()` | ~30 | Отправка с fallback при routing error |
| `_topic_ctx()` | ~20 | Определение thread/topic контекста |
| `_split_text()` | ~15 | Разбивка текста на куски < 4096 |
| `_reply_root_message_id()` | ~10 | Root message в цепочке replies |
| `MessageProgress` | ~25 | Класс progress для streaming |
| `_typing_loop()` | ~10 | Async typing indicator |

**Почему критично:** Основной entry point приложения. Сложная логика topics/threads, fallback routing, текстовый splitting, exception handling (MCPError, TelegramBadRequest). Без тестов невозможно гарантировать корректность обработки сообщений.

**Что тестировать первым:** `_split_text()` (чистая функция), `_topic_ctx()`, `_safe_send()` с мокнутым bot.

---

### 2. sgr_agent.py — Agent loop (частично покрыт)
**Риск: CRITICAL**

| Непокрытая функция | Строк | Описание |
|--------------------|:-----:|----------|
| `SgrAgent.run()` | ~120 | Основной SGR-цикл: reasoning → tool_call → final |
| `_summarize_tool_result()` | ~60 | Компактификация результатов tool |
| `_format_tool_args()` | ~15 | Форматирование аргументов для лога |
| `compact_search()` | ~25 | Сжатие поисковых результатов |
| `compact_details()` | ~20 | Сжатие деталей товаров |
| `_merge_items()` | ~30 | Merge MCP + local результатов |
| `_mcp_item_to_compact()` | ~15 | Конвертация MCP item |

**Покрыто:** parse_llm_output(), _extract_json() (test_sgr_parser.py)
**Что тестировать первым:** `SgrAgent.run()` с мокнутым LLM-клиентом — основной бизнес-цикл. `compact_search()`, `compact_details()` — чистые функции.

---

### 3. llm_client.py — LLM API client
**Риск: CRITICAL**

| Функция | Описание |
|---------|----------|
| `OpenRouterClient.__init__()` | Инициализация httpx client с proxy |
| `OpenRouterClient.chat()` | POST к OpenRouter, парсинг response |
| `_encode_header_value()` | Кодирование HTTP заголовков |

**Почему критично:** Ядро агента. Без LLM бот мёртв. Нет тестов на HTTP errors (400/401/500), malformed responses, timeout, missing fields в ответе.

---

### 4. mcp_client.py — MCP protocol client
**Риск: HIGH**

| Функция | Описание |
|---------|----------|
| `VkusvillMCP.connect()` | Подключение к MCP серверу |
| `VkusvillMCP.close()` | Закрытие соединения |
| `VkusvillMCP.search()` | Поиск товаров через MCP |
| `VkusvillMCP.details()` | Детали товара |
| `VkusvillMCP.cart()` | Создание корзины |

**Почему важно:** Поиск и корзина — основные фичи бота. Нет тестов на connection failures, timeout, malformed MCP response, reconnect.

---

### 5. embeddings_client.py — Embeddings API
**Риск: HIGH**

| Функция | Описание |
|---------|----------|
| `OpenRouterEmbeddingsClient.embed()` | POST к OpenRouter для embeddings |
| `_encode_header_value()` | Кодирование заголовков |

**Почему важно:** Семантический поиск зависит от эмбеддингов. Нет тестов на API errors, пустой input, batching.

---

### 6. telegram_draft.py — Telegram Draft API
**Риск: HIGH**

| Функция | Описание |
|---------|----------|
| `TelegramAPI.call()` | Generic HTTP к Telegram |
| `TelegramAPI.send_message_draft()` | Streaming через Draft API |
| `DraftProgress.set/add/flush()` | Progress management |

**Почему важно:** Streaming progress в topics. Новый API, может быть нестабилен.

---

### 7. formatting.py — Markdown escaping
**Риск: MEDIUM**

| Функция | Описание |
|---------|----------|
| `escape_markdown_v2()` | Экранирование спецсимволов для MarkdownV2 |
| `to_telegram_markdown()` | HTML → Telegram Markdown |

**Почему тестировать:** Легко написать, много edge-cases (backslash, кириллица, вложенное форматирование).

---

### 8. prompts.py — System prompt builder
**Риск: MEDIUM**

| Функция | Описание |
|---------|----------|
| `build_system_prompt()` | Сборка system prompt с tools, user profile |

**Почему тестировать:** Изменение промпта может сломать весь агент. Чистая функция, легко тестировать.

---

### 9. product_retriever.py (частично покрыт)
**Риск: MEDIUM**

| Непокрытая функция | Описание |
|--------------------|----------|
| `_normalize_category_tokens()` | Нормализация категорий |
| `_match_categories()` | Матчинг по категориям |
| `_parse_filter_expr()` | Парсинг фильтров (<=, >=, =) |
| `_compare()` | Сравнение значений |
| `_sort_items()` | Сортировка результатов |
| `_apply_filters()` | Применение фильтров |
| `_normalize_weight()` | Нормализация веса (г→кг, мл→л) |
| `_price_metrics()` | Вычисление цена/литр, цена/кг |

**Покрыто:** semantic_search(), nutrition_query().

---

### 10. logging.py, manual_llm.py
**Риск: LOW**

Утилиты инициализации и debug-инструменты. Не критичны.

---

## Качество существующих тестов

### Сильные стороны
- Используют pytest fixtures (`tmp_path`, `monkeypatch`) — нет файлового мусора
- Реалистичные данные (настоящая SQLite схема, FAISS индекс, YAML конфиги)
- `FakeEmbeddings` — правильный мок, следует Protocol
- `test_parse_json_with_noise` — реальный сценарий LLM с шумом
- `test_history_scoped_by_thread_id` — тестирует миграцию + бизнес-логику

### Слабые стороны
- **Нет conftest.py** — каждый файл создаёт свои fixtures с нуля
- **Нет параметризации** — `@pytest.mark.parametrize` не используется
- **Нет негативных сценариев** — только happy path (нет тестов на ошибки)
- **Нет async тестов** — основной цикл асинхронный, но asyncio.run() только в одном тесте
- **Нет тестов на граничные условия:** пустые строки, None, огромные входы, Unicode edge cases
- **Скорость:** тесты быстрые (нет сети, всё в tmp_path), но мало

---

## Приоритетный план тестирования

### P0 — Критично, писать немедленно

| # | Что тестировать | Тип | Effort | Impact |
|:-:|----------------|-----|:------:|:------:|
| 1 | **SgrAgent.run()** — основной SGR-цикл с мокнутым LLM | Unit + Integration | L | CRITICAL |
| 2 | **OpenRouterClient.chat()** — HTTP errors, timeout, malformed response | Unit | M | CRITICAL |
| 3 | **_split_text()** — граничные случаи (пустая строка, ровно 4096, > 4096) | Unit | S | HIGH |

### P1 — Важно, в ближайшем спринте

| # | Что тестировать | Тип | Effort | Impact |
|:-:|----------------|-----|:------:|:------:|
| 4 | **VkusvillMCP** — search/details/cart с мокнутым MCP | Unit | M | HIGH |
| 5 | **parse_llm_output()** — негативные сценарии (невалидный JSON, unknown action, пустой input) | Unit | S | HIGH |
| 6 | **compact_search(), compact_details()** — чистые функции, легко тестировать | Unit | S | MEDIUM |
| 7 | **formatting.py** — escape_markdown_v2 с edge-cases (backslash, кириллица, вложенный MD) | Unit | S | MEDIUM |

### P2 — Желательно

| # | Что тестировать | Тип | Effort | Impact |
|:-:|----------------|-----|:------:|:------:|
| 8 | **OpenRouterEmbeddingsClient.embed()** — HTTP errors, пустой batch | Unit | M | MEDIUM |
| 9 | **build_system_prompt()** — разные профили, tools, пустой toolkit | Unit | S | MEDIUM |
| 10 | **product_retriever helpers** — _parse_filter_expr, _price_metrics, _normalize_weight | Unit | M | MEDIUM |
| 11 | **TelegramAPI + DraftProgress** — streaming с мокнутым httpx | Unit | M | MEDIUM |
| 12 | **conftest.py** — общие fixtures (db, config, fake_embeddings, fake_llm) | Infra | M | HIGH |

### P3 — Опционально

| # | Что тестировать | Тип | Effort | Impact |
|:-:|----------------|-----|:------:|:------:|
| 13 | **main.py handlers** — Telegram обработчики с мокнутым bot | Integration | L | MEDIUM |
| 14 | **Database** — остальные методы (get_or_create_user, etc.) | Unit | M | LOW |
| 15 | **E2E SGR flow** — полный цикл user→bot→LLM→MCP→response | Integration | XL | HIGH |

---

## Целевые показатели

| Метрика | Текущее | Цель (P0+P1) | Цель (все) |
|---------|:-------:|:------------:|:----------:|
| Тестовых функций | 7 | ~30 | ~60 |
| Покрытых модулей | 5/16 | 10/16 | 14/16 |
| Негативные сценарии | 0 | ~10 | ~20 |
| conftest.py | нет | есть | есть |
| Время выполнения | <1с | <3с | <5с |
