# CI/CD Review: vkusvillbot

Дата: 2026-03-20

---

## Общая картина

```
git push main ──► GitHub Actions (deploy-macmini.yml)
                       │
                       ▼  self-hosted runner [macmini]
                  deploy_macmini.sh
                       │
                  ┌────┴────┐
                  │ git fetch│
                  │ git reset│
                  │ git clean│
                  └────┬────┘
                       │
                  docker compose up -d --build
                       │
                  docker compose ps  (ручная проверка)
```

**Вердикт: CI отсутствует. Есть только CD.** Push в main сразу деплоит без каких-либо автоматических проверок.

---

## Что есть

### GitHub Actions Workflows

| Workflow | Файл | Триггер | Что делает |
|----------|-------|---------|-----------|
| Deploy to macmini | `deploy-macmini.yml` | push main, manual | Запускает deploy_macmini.sh |

**Один workflow. Одна задача. Ноль проверок.**

### deploy_macmini.sh

| Шаг | Команда | Оценка |
|-----|---------|--------|
| 1. Проверка репо | `[ -d .git ]` | ✅ OK |
| 2. Fetch | `git fetch --all --prune` | ✅ OK |
| 3. Reset | `git reset --hard origin/main` | ⚠️ Уничтожает локальные изменения |
| 4. Clean | `git clean -ffdx -e .env -e data -e logs` | ✅ Сохраняет .env, data, logs |
| 5. .env check | Warning если нет | ✅ OK, но не блокирует |
| 6. Docker check | `command -v docker` | ✅ OK |
| 7. Build & run | `docker compose up -d --build` | ✅ OK |
| 8. Status | `docker compose ps` | ⚠️ Не проверяет здоровье |

**Хорошо:** concurrency group предотвращает параллельные деплои.
**Плохо:** после `docker compose ps` нет проверки, что контейнер реально healthy.

### Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir --upgrade pip
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .[dev]
CMD ["python", "-m", "vkusvillbot.main"]
```

| Критерий | Оценка | Комментарий |
|----------|:------:|-------------|
| Multi-stage build | ❌ | Один stage, dev-зависимости в production образе |
| Non-root user | ❌ | Запускается от root |
| .dockerignore | ❌ | Не найден — копируется всё, включая .git, tests, docs |
| Layer caching | ⚠️ | pyproject.toml копируется перед src — хорошо, но `-e .` ломает кэш |
| Image size | ⚠️ | `pip install -e .[dev]` тащит pytest+ruff в production |
| Healthcheck | ❌ | Нет HEALTHCHECK в Dockerfile |
| PYTHONDONTWRITEBYTECODE | ✅ | Правильно для контейнера |
| PYTHONUNBUFFERED | ✅ | Правильно для логов |

### docker-compose.yml

```yaml
services:
  bot:
    build: .
    env_file: .env
    volumes:
      - ./:/app                          # весь репо монтируется
      - ./data/vkusvill.db:/app/data/vkusvill.db
    command: ["watchfiles", ..., "python -m vkusvillbot.main"]
```

| Критерий | Оценка | Комментарий |
|----------|:------:|-------------|
| Healthcheck | ❌ | Нет healthcheck (бот не имеет HTTP endpoint) |
| Restart policy | ❌ | Нет `restart:` — при падении не рестартует |
| Prod/dev разделение | ❌ | watchfiles в command — это dev-режим |
| Volumes | ⚠️ | `./:/app` монтирует весь репо, включая .env и .git |
| Resource limits | ❌ | Нет mem_limit / cpus |

**Проблема:** docker-compose.yml настроен для разработки (watchfiles), но deploy_macmini.sh запускает именно его. В production должен быть отдельный `docker-compose.prod.yml` или override.

### Инструменты качества

| Инструмент | Настроен? | В CI? | Конфигурация |
|------------|:---------:|:-----:|-------------|
| ruff | ✅ pyproject.toml | ❌ | E, F, I; line-length=100 |
| mypy | ❌ | ❌ | Упоминается в docs/testing.md, но нет конфига |
| pytest | ✅ pyproject.toml | ❌ | addopts="-q", pythonpath=["src"] |
| coverage | ❌ | ❌ | Не настроен |
| pre-commit | ❌ | ❌ | Нет .pre-commit-config.yaml |
| pip-audit | ❌ | ❌ | Нет проверки CVE |
| Makefile | ❌ | — | Нет |

---

## Чего не хватает в CI

### CRITICAL — без этого деплоить опасно

| # | Проверка | Почему критично | Effort |
|:-:|----------|-----------------|:------:|
| 1 | **Lint (ruff check)** | Синтаксические ошибки и unused imports попадут в prod | S |
| 2 | **Tests (pytest)** | Регрессии попадут в prod без обнаружения | S |
| 3 | **Разделение dev/prod** | watchfiles + dev-зависимости + root user в production | M |

### HIGH — сильно улучшит надёжность

| # | Проверка | Почему важно | Effort |
|:-:|----------|-------------|:------:|
| 4 | **Type check (mypy)** | Ловит TypeError до runtime | M |
| 5 | **Health check после деплоя** | Деплой может быть "успешным", но бот не работает | S |
| 6 | **Non-root user в Docker** | CIS benchmark, defense in depth | S |
| 7 | **Multi-stage Dockerfile** | Убрать dev tools из prod, уменьшить surface | M |

### MEDIUM — best practices

| # | Проверка | Описание | Effort |
|:-:|----------|----------|:------:|
| 8 | **Coverage gate** | Минимум 60% coverage, fail если ниже | S |
| 9 | **pip-audit** | Автоматическая проверка CVE в зависимостях | S |
| 10 | **.dockerignore** | Исключить .git, tests, docs, *.md, .env | S |
| 11 | **docker-compose.prod.yml** | Отдельный compose без watchfiles, с restart policy | S |
| 12 | **Rollback** | Автоматический откат при неудачном деплое | M |
| 13 | **Pre-commit hooks** | Локальные проверки до push | M |

---

## Рекомендуемый CI pipeline

```yaml
# .github/workflows/ci.yml
name: CI
on:
  pull_request:
  push:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install ruff
      - run: ruff check .
      - run: ruff format --check .

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e .[dev] mypy
      - run: mypy src/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e .[dev] pytest-cov
      - run: pytest --cov=vkusvillbot --cov-fail-under=50

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install pip-audit
      - run: pip-audit .

  deploy:
    needs: [lint, typecheck, test]
    if: github.ref == 'refs/heads/main'
    runs-on: [self-hosted, macmini, vkusvillbot]
    steps:
      - run: /Users/macmini/dev-server/vkusvillbot/scripts/deploy_macmini.sh
```

### Рекомендуемый Dockerfile (prod)

```dockerfile
# Build stage
FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

# Runtime stage
FROM python:3.11-slim
RUN groupadd -r appuser && useradd -r -g appuser appuser
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src ./src
USER appuser
CMD ["python", "-m", "vkusvillbot.main"]
```

### Рекомендуемый post-deploy health check

```bash
# В deploy_macmini.sh после docker compose up:
echo "[deploy] health check"
for i in $(seq 1 30); do
  if docker compose exec bot python -c "import vkusvillbot; print('ok')" 2>/dev/null; then
    echo "[deploy] ✅ bot healthy"
    exit 0
  fi
  sleep 2
done
echo "[deploy] ❌ bot failed to start" >&2
docker compose logs --tail=50 bot
exit 1
```

---

## Оценка

| Критерий | Оценка | Комментарий |
|----------|:------:|-------------|
| CI проверки | 1/10 | Нет вообще — ни lint, ни test, ни typecheck |
| CD pipeline | 6/10 | Работает, concurrency, но нет health check и rollback |
| Dockerfile | 4/10 | Функционален, но root, dev deps, нет multi-stage |
| Docker Compose | 3/10 | Dev-mode в prod, нет restart, нет healthcheck |
| Безопасность | 2/10 | Root user, нет pip-audit, нет .dockerignore |
| **Среднее** | **3.2/10** | |
