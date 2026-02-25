# Команды проекта

## Подготовка окружения
```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
cp .env.example .env
```

## Запуск
```bash
python -m vkusvillbot.main
```

## Docker
```bash
docker compose up --build
docker compose up -d --build
docker compose ps
docker compose logs --tail=200 bot
```

## Проверки качества
```bash
./.venv/bin/python -m ruff check .
./.venv/bin/python -m mypy src
./.venv/bin/python -m pytest -q
```

## Работа с retrieval/индексом
Сборка или обновление FAISS-индекса:
```bash
python scripts/build_vector_index.py
python scripts/build_vector_index.py --force --batch-size 64
```

Ручная отладка SGR-цикла:
```bash
python scripts/manual_sgr.py
```

## Эксплуатационные команды на macmini
```bash
/Users/macmini/dev-server/vkusvillbot/scripts/deploy_macmini.sh
docker compose ps
docker compose logs --tail=200 bot
```
