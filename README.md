# ВкусВилл ИИ-бот

Персональный бот для поиска товаров ВкусВилл, сравнения и формирования корзины. Ориентирован быстрые сценарии: найти продукты, сравнить цены/вес/КБЖУ, собрать корзину и получить ссылку.

## Возможности

- Поиск товаров и сравнение по цене, весу, рейтингу.
- Детальные карточки (состав, пищевая ценность, свойства) через MCP.
- Формирование корзины и выдача ссылки.
- Учёт города и особенностей питания.
- Многошаговая логика (SGR) для уточнений и подбора.

## Как работает

1. LLM возвращает строгий JSON с действием (`tool_call` или `final`).
2. Код выполняет MCP-инструменты и возвращает результаты обратно в LLM.
3. Финальный ответ отдаётся пользователю (HTML-разметка).

## MCP-инструменты

- `vkusvill_products_search(q, page)` — поиск товаров.
- `vkusvill_product_details(id)` — подробности товара.
- `vkusvill_cart_link_create(products)` — создать ссылку на корзину.

## Быстрый старт (локально)

```bash
cd vkusvillbot
python -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
python -m vkusvillbot.main
```

## Запуск в Docker (dev)

```bash
docker compose up --build
```

Контейнер запускается в dev-режиме с авто-перезапуском при изменении файлов.

## Конфигурация

- `.env` — секреты:
  - `TELEGRAM_BOT_TOKEN`
  - `OPENROUTER_API_KEY`
  - `OPENROUTER_PROXY_URL`
- `config.yaml` — настройки:
  - URL MCP-сервера
  - параметры SGR
  - логирование

Файл-пример: `.env.example`.

## Структура проекта

- `src/vkusvillbot/main.py` — входная точка Telegram-бота.
- `src/vkusvillbot/sgr_agent.py` — SGR-цикл и вызовы MCP.
- `src/vkusvillbot/llm_client.py` — клиент OpenRouter (с прокси).
- `src/vkusvillbot/prompts.py` — системный промпт LLM.
- `scripts/manual_sgr.py` — ручная отладка SGR.
- `tests/` — тесты.
