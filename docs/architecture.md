# Архитектура vkusvillbot

## Назначение
`vkusvillbot` — персональный Telegram-бот для подбора товаров ВкусВилл.
Бот ведёт диалог, вызывает MCP-инструменты, использует локальную SQLite-базу и возвращает ответ в Markdown для Telegram.

## Основные компоненты
- **Telegram runtime (`src/vkusvillbot/main.py`)**: aiogram-диспетчер, обработка сообщений, topic/thread-маршрутизация, прогресс и безопасная отправка длинных ответов.
- **SGR-агент (`src/vkusvillbot/sgr_agent.py`)**: цикл reasoning с шагами `tool_call`/`final`, строгий JSON-контракт LLM, сбор финального ответа и создание корзины.
- **LLM-клиенты**:
  - `OpenRouterClient` (`src/vkusvillbot/llm_client.py`) — chat-completions;
  - `OpenRouterEmbeddingsClient` (`src/vkusvillbot/embeddings_client.py`) — embeddings.
- **Интеграция с MCP (`src/vkusvillbot/mcp_client.py`)**: `vkusvill_products_search`, `vkusvill_product_details`, `vkusvill_cart_link_create`.
- **Данные (`src/vkusvillbot/db.py`)**: SQLite-слой для пользователей, истории сообщений, сессий, каталога товаров и эмбеддингов.
- **Поиск (`src/vkusvillbot/product_retriever.py`)**: FAISS-семантика + FTS5 boost + фильтры по нутриентам/цене.
- **Векторный индекс (`src/vkusvillbot/vector_index.py`)**: загрузка/поиск/пересборка FAISS-индекса.

## Поток данных (боевой сценарий)
1. Пользователь отправляет текст в Telegram.
2. `main.py` определяет контекст треда (`message_thread_id`, `direct_messages_topic_id` или fallback по reply-chain).
3. Загружается профиль пользователя и история сообщений из SQLite.
4. `SgrAgent` формирует prompt (`prompts.py`), отправляет запрос в OpenRouter и получает JSON.
5. При `tool_call` вызывается локальный или MCP-инструмент, результат возвращается в LLM как `TOOL_RESULT`.
6. При `final` бот форматирует ответ и отправляет пользователю; при наличии `cart_items` создаётся корзина.
7. Диалог и ответ сохраняются в SQLite с привязкой к thread/topic.

## Локальные данные и схема
`Database` автосоздаёт/мигрирует базовые таблицы:
- `users` — профиль пользователя (`tg_id`, `city`, `diet_notes`).
- `sessions` — последний интент/контекст.
- `messages` — история диалога (включая `thread_id`).
- `product_embeddings` — эмбеддинги по модели.

Дополнительно используется таблица `products` (если база каталога уже подготовлена), FTS-таблица `products_fts` и триггеры синхронизации.

## Конфигурация
Конфиг собирается из `config.yaml` + переменных окружения (`Settings.load()`):
- Telegram: токен и UI-флаги;
- LLM/Embeddings: модель, API key, proxy;
- MCP URL;
- SQLite path;
- FAISS index path и параметры retrieval;
- SGR-параметры (`max_steps`, `temperature`, и т.д.).

## Границы и ограничения
- Контракт с LLM строгий: только JSON формата `tool_call`/`final`.
- Продовая актуализация данных идёт через MCP, локальная БД — для скорости и аналитики.
- Семантический поиск зависит от наличия FAISS-индекса (`data/products.faiss`).
- Без `OPENROUTER_API_KEY` невозможно выполнять ни chat, ни embeddings.
