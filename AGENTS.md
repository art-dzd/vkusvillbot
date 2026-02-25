# Индекс проекта vkusvillbot

Индексный файл проекта `vkusvillbot`.
Подробности и регламенты вынесены в `docs/`.

## Кратко о проекте
- Telegram-бот для поиска и сравнения товаров ВкусВилл.
- Работает через SGR-цикл: LLM -> инструменты -> финальный ответ.
- Комбинирует MCP ВкусВилл и локальную SQLite + FAISS.
- Поддерживает сбор корзины с выдачей ссылки.

## Стек
- Python 3.11, aiogram, pydantic.
- SQLite, FTS5, FAISS (`faiss-cpu`).
- OpenRouter (chat + embeddings), httpx.
- Docker Compose, GitHub Actions (self-hosted macmini).

## Принципы работы
1. Контракт с LLM строгий: только JSON `tool_call`/`final`.
2. Локальный retrieval используется по умолчанию, MCP — по задаче.
3. История диалогов хранится с привязкой к thread/topic.
4. Конфиг только через `config.yaml` + `.env`/env overrides.
5. Ошибки должны быть видимы в логах, без "тихих" падений.

## Каноническая документация
- `docs/architecture.md` — компоненты, слои и потоки данных.
- `docs/deploy.md` — запуск, автодеплой, rollback и post-check.
- `docs/testing.md` — quality gate и текущая тестовая стратегия.
- `docs/commands.md` — команды разработки и эксплуатации.
- `docs/design/sgr-loop.md` — внутренний контракт SGR-цикла.
- `docs/design/local-retrieval.md` — дизайн локального semantic search.

## Ограничения
- Без `OPENROUTER_API_KEY` не работают LLM и embeddings.
- Без `data/products.faiss` semantic search недоступен.
- Не обходить JSON-контракт промптов в `prompts.py`.
- Не ломать thread-scoped хранение истории в SQLite.
- Деплой считается завершённым только после health/log checks.

## Базовые команды
```bash
./.venv/bin/python -m ruff check .
./.venv/bin/python -m mypy src
./.venv/bin/python -m pytest -q
```

## Минимальный деплойный контур
- `git push origin main`.
- GitHub Actions: `.github/workflows/deploy-macmini.yml`.
- На macmini запускается `scripts/deploy_macmini.sh`.
- Проверки после релиза: `docker compose ps` и `docker compose logs --tail=200 bot`.

## Где смотреть дальше
- Операционные команды: `docs/commands.md`.
- Архитектура и границы модулей: `docs/architecture.md`.
- Дизайн-контракты агента и retrieval: `docs/design/*`.
