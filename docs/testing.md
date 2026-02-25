# Тестирование

## Quality gate
Базовый набор проверок перед интеграцией изменений:
```bash
./.venv/bin/python -m ruff check .
./.venv/bin/python -m mypy src
./.venv/bin/python -m pytest -q
```

## Что покрыто автотестами
Текущие тесты в `tests/` закрывают ключевые стабильные контракты:
- `tests/test_config.py` — приоритет env-переменных над `config.yaml`.
- `tests/test_sgr_parser.py` — парсинг LLM-JSON (`tool_call`/`final`) и обработка "шумного" текста.
- `tests/test_vector_search.py` — фильтрация/сортировка nutrition и semantic retrieval через FAISS.
- `tests/test_message_threads.py` — корректная миграция `thread_id` и изоляция истории по тредам.

## Локальный pre-commit
`.githooks/pre-commit` запускает:
- `ruff check .` (обязательный);
- `mypy src` (best effort);
- `pytest` (best effort);
- `eslint .` при наличии `package.json`.

## Ручной smoke после изменений
1. Запустить бота локально.
2. Отправить запрос "найди молоко" и убедиться, что приходит список товаров.
3. Проверить сценарий корзины (через `cart_items` или прямой вызов `vkusvill_cart_link_create`).
4. Для тредов Telegram проверить, что ответы остаются в том же контексте.

## Границы тестовой стратегии
- Интеграционные вызовы MCP/OpenRouter в CI не мокируются в этом проекте.
- Нет отдельного e2e-контура для Telegram API.
- Качество retrieval зависит от актуальности `data/products.faiss`; это операционный контроль, не unit-level.
