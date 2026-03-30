# Quality Score: vkusvillbot

Дата: 2026-03-20

---

## Оценки по доменам

| Домен | Оценка | Обоснование |
|-------|:------:|-------------|
| **Архитектура** | 5.5/10 | SoC 5, Coupling 6, Cohesion 5, Extensibility 4, Testability 5, Error Boundaries 7, Config 7. Инфра-модули (11 из 15) идеальны, но три ключевых файла — god objects: `db.py` (1250 строк, 6+ обязанностей), `main.py` (657 строк, composition root + transport + presentation), `sgr_agent.py` (tool dispatch на 90-строчном if/elif). Дублирование 5 функций между db.py и product_retriever.py. Нет абстракций VectorIndex/Database. Единственные Protocol — LLMClient и EmbeddingsClient |
| **Качество кода** | 5/10 | 25 находок code review: 4 CRITICAL (regex `\\d` вместо `\d` в db.py:833,1084 — FTS5/парсинг молча ломается; SQLite без `check_same_thread`; race condition в pending_topics), 8 HIGH, 9 MEDIUM, 4 LOW. 15 находок simplifier (~250 строк экономии): 90-строчный if/elif dispatch, 60-строчный _summarize_tool_result, дублирование SgrConfig. Стиль кода консистентный, type hints частичные |
| **Безопасность** | 4/10 | 1 CRITICAL (реальные секреты в .env — требуется ротация), 8 HIGH (prompt injection без санитизации, TOOL_RESULT spoofing, нет rate limiting, нет timeout на SGR-цикл до 12 мин, утечка str(exc) пользователю, нет валидации длины ввода и /city /diet). PII в логах без ротации и маскирования. API key как str, не SecretStr. MCP URL в config.yaml. OWASP: A03 Injection HIGH, A04 Insecure Design HIGH |
| **Тестирование** | 5/10 | 7 тестов в 4 файлах. Покрыто: Settings.load(), парсинг JSON LLM-ответа, thread-scoped история, nutrition/semantic search. **Не покрыто:** SgrAgent.run() (основной бизнес-цикл!), все Telegram-хэндлеры, mcp_client, formatting, telegram_draft. Database.__init__ с side effects мешает unit-тестам. main() не принимает параметров — нетестируема |
| **Документация** | 7/10 | CLAUDE.md — структурированный индекс. docs/architecture.md, deploy.md, testing.md, commands.md, design/sgr-loop.md, design/local-retrieval.md — каноническая документация покрывает основные аспекты. Inline-комментарии минимальны, но код в целом читаем. Нет API-документации (не критично для бота) |
| **CI/CD** | 7/10 | GitHub Actions с self-hosted macmini runner. Автодеплой на `git push origin main`. Pipeline: `.github/workflows/deploy-macmini.yml` → `scripts/deploy_macmini.sh`. Post-deploy checks: `docker compose ps` + `docker compose logs`. Нет staging-среды, нет canary/blue-green, нет автоматического rollback. Линтер (ruff) и mypy в базовых командах, но не обязательны в CI gate |
| **Зависимости** | 8/10 | Все зависимости актуальны. fastmcp==2.12.4 и mcp==1.17.0 зафиксированы — хорошо для стабильности, но требуют ручного обновления. yaml.safe_load() везде. Нет pip-audit/safety в CI. Нет lock-файла (requirements.txt с ranges) |

---

## Общий Quality Score: 5.9/10

```
Архитектура      ████████░░░░░░░░░░░░  5.5
Качество кода    ██████████░░░░░░░░░░  5.0
Безопасность     ████████░░░░░░░░░░░░  4.0
Тестирование     ██████████░░░░░░░░░░  5.0
Документация     ██████████████░░░░░░  7.0
CI/CD            ██████████████░░░░░░  7.0
Зависимости      ████████████████░░░░  8.0
─────────────────────────────────────
Среднее                               5.9
```

**Вердикт:** Рабочий MVP с хорошей инфраструктурной базой, но с серьёзными архитектурными и security-долгами. Инфра-модули (11 из 15) написаны качественно. Проблемы сконцентрированы в трёх файлах (db.py, main.py, sgr_agent.py) и в отсутствии security-практик. Бот работает, но масштабирование и поддержка будут болезненными без рефакторинга.

---

## Топ-5 приоритетных действий

### 1. Ротировать секреты и закрыть security-дыры (Срочность: немедленно)

- Ротировать Telegram token и OpenRouter API key
- Добавить rate limiting (per-user throttle через aiogram middleware)
- Добавить `asyncio.wait_for(agent.run(), timeout=120)` на SGR-цикл
- Заменить `str(exc)` на generic сообщения для пользователя
- Санитизировать user input: ограничение длины, фильтрация injection-паттернов

**Источники:** security-review #1-9

### 2. Исправить CRITICAL-баги в db.py (Срочность: высокая)

- Заменить `\\d` на `\d` в regex (db.py:833, 1084) — FTS5 и парсинг весов сломаны
- Добавить `check_same_thread=False` или убедиться в однопоточности
- Исправить race condition в pending_topics (main.py:332)

**Источники:** code-review #1-4

### 3. Разбить db.py на модули (Срочность: средняя)

God Object на 1250 строк → `db/schema.py`, `db/repository.py`, `db/product_queries.py`, `db/embeddings_store.py`, `db/utils.py`. Устранит дублирование 5 функций с product_retriever.py (~250 строк экономии). Каждый модуль < 300 строк.

**Источники:** architecture-review #1, simplifier-report #1-3

### 4. Вынести tool dispatch в реестр и покрыть тестами (Срочность: средняя)

- Заменить 90-строчный if/elif на `TOOL_HANDLERS: dict[str, Callable]`
- Написать тесты на SgrAgent.run() — основной бизнес-цикл не покрыт
- Добавить тесты на Telegram-хэндлеры и mcp_client

**Источники:** architecture-review #2, code-review #12, simplifier-report #4-5

### 5. Добавить PII-защиту и ротацию логов (Срочность: средняя)

- `RotatingFileHandler` вместо `FileHandler` (50MB, 5 backup)
- Маскирование tg_id в логах
- `SecretStr` для api_key и token
- Retention policy для dialog.log

**Источники:** security-review #10-12, #19

---

## Сильные стороны

- **Инфраструктурные модули** — 11 из 15 модулей имеют идеальную cohesion (1 обязанность). llm_client, embeddings_client, mcp_client, vector_index, formatting, config — чётко изолированы
- **Graceful degradation** — 6 уровней fallback: FAISS → TOOL_ERROR, FTS5 → LIKE, MCP → TOOL_ERROR, telegramify → ручной escape, Draft API → edit-message, Topics → без стриминга
- **Кастомные исключения** — MCPError, LLMError, EmbeddingsError, VectorIndexError, TelegramAPIError с осмысленной иерархией
- **Конфигурация** — Pydantic + YAML + env с `repr=False` для секретов, httpx логгер на WARNING
- **DI в ключевых компонентах** — ProductRetriever и SgrAgent принимают зависимости через конструктор
- **.gitignore** — корректно исключает .env, *.db, *.faiss, logs/
- **Документация** — структурированный CLAUDE.md + каноническая docs/ с архитектурой, деплоем, тестированием
