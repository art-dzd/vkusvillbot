# Деплой и эксплуатация

## Варианты запуска
- **Локально (venv)** — для разработки и отладки.
- **Docker Compose** — основной dev/runtime контур.
- **Автодеплой на macmini** — через GitHub Actions self-hosted runner.

## Локальный запуск
1. Создать окружение и установить зависимости:
```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
```
2. Подготовить конфиг: `cp .env.example .env` и заполнить секреты.
3. Запустить бота:
```bash
python -m vkusvillbot.main
```

## Docker Compose
```bash
docker compose up --build
```
Текущий `docker-compose.yml` запускает бота через `watchfiles`, поэтому изменения Python-кода подхватываются автоматически.

## Автодеплой на macmini
Workflow: `.github/workflows/deploy-macmini.yml`.
Триггер: push в `main` и ручной `workflow_dispatch`.
Runner labels: `self-hosted`, `macmini`, `vkusvillbot`.

Сценарий деплоя (`scripts/deploy_macmini.sh`):
1. `git fetch --all --prune`
2. `git reset --hard origin/main`
3. `git clean -ffdx` (с исключениями `.env`, `data`, `logs`)
4. `docker compose up -d --build`
5. `docker compose ps`

## Пост-деплой проверки
- `docker compose ps` — контейнер в состоянии `Up`.
- `docker compose logs --tail=200 bot` — нет циклических падений.
- В Telegram проходит smoke-сценарий: `/start` и один запрос на поиск товара.

## Rollback
1. На macmini перейти в `/Users/macmini/dev-server/vkusvillbot`.
2. Переключиться на стабильный коммит/тег.
3. Выполнить `docker compose up -d --build`.
4. Повторить post-check.

## Риски
- Скрипт деплоя использует `reset --hard` и удаляет неотслеживаемые файлы (кроме исключений).
- При потере `.env` бот не стартует из-за отсутствия ключей.
- Без свежего FAISS-индекса качество локального semantic search заметно падает.
