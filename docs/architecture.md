# Архитектура vkusvillbot

## Назначение
Персональный Telegram-бот для подбора товаров ВкусВилл. Ведёт диалог через SGR-цикл (Schema-Guided Reasoning), вызывает MCP-инструменты и локальный semantic search, возвращает ответ в Markdown. Однопользовательский, ориентирован на Москву.

## Стек технологий
- **Python 3.11**, aiogram 3.x, pydantic 2.x
- **SQLite** + FTS5 + FAISS (`faiss-cpu`) — данные и поиск
- **OpenRouter** (httpx) — LLM chat-completions + embeddings
- **fastmcp** — MCP-клиент к `https://mcp001.vkusvill.ru/mcp`
- **telegramify-markdown** — Markdown → MarkdownV2
- **Docker Compose**, GitHub Actions (self-hosted macmini)

## Модули и их роли

| Модуль | Класс / Функция | Роль |
|--------|-----------------|------|
| `main.py` | `main()`, обработчики | Точка входа, aiogram dispatcher, routing по thread/topic |
| `sgr_agent.py` | `SgrAgent`, `ToolCall`, `FinalAnswer` | SGR-цикл: LLM → tool_call/final → ответ |
| `llm_client.py` | `OpenRouterClient` | Chat-completions через OpenRouter |
| `embeddings_client.py` | `OpenRouterEmbeddingsClient` | Embeddings через OpenRouter |
| `mcp_client.py` | `VkusvillMCP` | MCP-обёртка: search, details, cart |
| `db.py` | `Database` | SQLite: users, messages, sessions, products, embeddings, FTS |
| `product_retriever.py` | `ProductRetriever` | FAISS-семантика + FTS5 boost + фильтры по КБЖУ/цене |
| `vector_index.py` | `FaissVectorIndex` | Загрузка/поиск/пересборка FAISS-индекса |
| `config.py` | `Settings`, `SgrConfig`, `LLMConfig`... | Конфигурация из `config.yaml` + env vars |
| `prompts.py` | `build_system_prompt()` | System prompt с описанием инструментов |
| `models.py` | `UserProfile` | Dataclass профиля пользователя |
| `formatting.py` | `to_telegram_markdown()` | Markdown → MarkdownV2 для Telegram |
| `telegram_draft.py` | `TelegramAPI`, `DraftProgress` | Drafts API для прогрессивного стриминга ответа |
| `logging.py` | `setup_logging()`, `setup_dialog_logger()` | Логирование, подавление httpx INFO |
| `manual_llm.py` | `ManualLLM` | Интерактивный LLM для отладки |

## Зависимости между модулями

```
main.py
├── config.py ─────────────── конфигурация (используется всеми)
├── logging.py ─────────────── инициализация логов
├── db.py ──────────────────── единственный Database на процесс
├── mcp_client.py ──────────── VkusvillMCP.connect()
├── llm_client.py ──────────── OpenRouterClient
├── embeddings_client.py ───── OpenRouterEmbeddingsClient
├── vector_index.py ────────── FaissVectorIndex.load()
├── product_retriever.py ───── ProductRetriever
│   ├── db.py
│   ├── embeddings_client.py
│   └── vector_index.py
├── sgr_agent.py ───────────── SgrAgent (создаётся на каждое сообщение)
│   ├── db.py
│   ├── mcp_client.py
│   ├── product_retriever.py
│   ├── prompts.py
│   │   └── models.py
│   └── models.py
├── telegram_draft.py ──────── TelegramAPI, DraftProgress
└── formatting.py ──────────── to_telegram_markdown()
```

Циклических зависимостей нет.

## Точки входа

### Telegram-бот (`python -m vkusvillbot.main`)
Команды:
- `/start` — приветствие, статус бота
- `/help` — справка по командам
- `/diet <текст>` — задать особенности питания
- `/city <город>` — задать город

Обработчики:
- **`on_text`** — основной: запускает `SgrAgent.run()`, стримит прогресс, сохраняет диалог
- **`on_forum_topic_created`** — routing для forum topics

### CLI-скрипты
- `scripts/build_vector_index.py` — сборка FAISS-индекса из эмбеддингов
- `scripts/manual_sgr.py` — интерактивная отладка SGR-цикла

API-сервера нет — чистый polling-бот.

## Data flow

```
Telegram сообщение
       ↓
  on_text(message)
       ↓
  ┌─ определить thread_id (topic routing)
  ├─ загрузить UserProfile из БД
  ├─ загрузить историю (thread-scoped)
  └─ инициализировать прогресс (DraftProgress или MessageProgress)
       ↓
  SgrAgent.run(text, history, progress)
       ↓
  ┌─ build_system_prompt(profile) + history + user_text
  └─ LOOP (макс 12 шагов):
       │
       ├─ LLM.chat(messages) ──→ OpenRouter ──→ JSON
       │
       ├─ parse → ToolCall?
       │   ├─ MCP: search/details/cart ──→ mcp001.vkusvill.ru
       │   │   └─ результат → upsert в SQLite
       │   ├─ local_products_search ──→ FTS5 в SQLite
       │   ├─ local_semantic_search ──→ embed(query) → FAISS → SQLite
       │   ├─ local_product_details ──→ SQLite
       │   └─ local_nutrition_query ──→ SQLite (фильтр по КБЖУ)
       │   └─ результат → TOOL_RESULT в messages → continue LOOP
       │
       └─ parse → FinalAnswer?
           ├─ если cart_items → mcp.cart() → ссылка на корзину
           └─ ответ → to_telegram_markdown() → Telegram
                                                   ↓
                                          save_message() в SQLite
```

## IO-операции

| Компонент | Тип IO | Куда |
|-----------|--------|------|
| `Database` | SQLite | `../vkusvill.db` |
| `OpenRouterClient` | HTTP POST | `openrouter.ai/api/v1/chat/completions` |
| `OpenRouterEmbeddingsClient` | HTTP POST | `openrouter.ai/api/v1/embeddings` |
| `VkusvillMCP` | HTTP (MCP) | `mcp001.vkusvill.ru/mcp` |
| `FaissVectorIndex` | файл | `data/products.faiss` |
| `TelegramAPI` | HTTP POST | Telegram Bot API |
| `dialog_logger` | файл | `logs/dialog.log` |

## Схема БД

```
users (id, tg_id, city, diet_notes, created_at)
sessions (id, user_id, last_intent, last_context, updated_at)
messages (id, user_id, thread_id, role, content, created_at)
  └─ idx: (user_id, thread_id, id)
products (id, xml_id, name, description_short, description_full,
          composition, nutrition, storage_conditions, price_current,
          rating_avg, rating_count, unit, weight_value, weight_unit,
          url, category_json, updated_at)
products_fts (name, description_short, description_full, composition)
  └─ триггеры синхронизации с products
product_embeddings (product_id, embedding_model, content_hash, embedding, updated_at)
```

## Конфигурация

Приоритет: **env vars > config.yaml > defaults в коде**.

Обязательные env vars: `TELEGRAM_BOT_TOKEN`, `OPENROUTER_API_KEY`.
Опциональные: `OPENROUTER_PROXY_URL`, `OPENROUTER_EMBEDDING_MODEL`, `TELEGRAM_PROXY_URL`, `DB_PATH`, `VECTOR_INDEX_PATH`.

SGR-параметры (`SgrConfig`): `max_steps=12`, `max_items_per_search=10`, `temperature=0.4`, `history_messages=10`, `local_fresh_hours=24`, `use_mcp_refresh=true`.

## Запуск

| Способ | Команда | Назначение |
|--------|---------|------------|
| Локально | `python -m vkusvillbot.main` | Разработка |
| Docker (dev) | `docker compose up --build` | Dev с watchfiles auto-reload |
| Docker (prod) | `docker compose up -d --build` | Production (Dockerfile без watchfiles) |
| Автодеплой | `git push origin main` → GitHub Actions → `deploy_macmini.sh` | CI/CD на macmini |

## Границы и ограничения
- Контракт с LLM строгий: только JSON формата `tool_call`/`final`.
- Продовая актуализация данных — через MCP, локальная БД — для скорости.
- Семантический поиск зависит от наличия FAISS-индекса (`data/products.faiss`).
- Без `OPENROUTER_API_KEY` не работают ни chat, ни embeddings.
- Прогресс-стриминг через `sendMessageDraft` — Telegram 9.0+ API, с fallback на `edit_text`.
