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

### Архитектура и дизайн
- `docs/architecture.md` — компоненты, слои и потоки данных.
- `docs/design/sgr-loop.md` — внутренний контракт SGR-цикла.
- `docs/design/local-retrieval.md` — дизайн локального semantic search.

### Операции
- `docs/deploy.md` — запуск, автодеплой, rollback и post-check.
- `docs/testing.md` — quality gate и текущая тестовая стратегия.
- `docs/commands.md` — команды разработки и эксплуатации.

### Аудиты и ревью (2026-03-20)
- `docs/quality-score.md` — сводная оценка проекта: 5.9/10.
- `docs/code-review.md` — 25 находок (4 CRITICAL, 8 HIGH). Оценка: 5/10.
- `docs/architecture-review.md` — 7 критериев, средняя: 5.6/10.
- `docs/security-review.md` — 25 находок (1 CRITICAL, 8 HIGH).
- `docs/simplifier-report.md` — 15 находок, ~250 строк потенциальной экономии.
- `docs/test-gaps.md` — 7 тестов на 16 модулей, приоритетный план покрытия.
- `docs/dependency-audit.md` — 13 зависимостей, 1 неиспользуемая, CVE-статус.
- `docs/ci-review.md` — CI/CD pipeline: 3.2/10 (нет CI-проверок, только CD).

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
