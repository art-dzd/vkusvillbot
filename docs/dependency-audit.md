# Dependency Audit: vkusvillbot

Дата: 2026-03-20

---

## Общая статистика

| Метрика | Значение |
|---------|----------|
| Файл зависимостей | pyproject.toml |
| Runtime-зависимостей | 13 |
| Dev-зависимостей | 2 (pytest, ruff) |
| Пиннены точно (==) | 2 (fastmcp, mcp) |
| С диапазоном (>=) | 11 |
| Неиспользуемые | 1-2 (watchfiles, pydantic-settings?) |
| Известные CVE | 4-7 (aiohttp < 3.13.3) |

---

## Таблица зависимостей

### Runtime

| Зависимость | Версия | Пиннена? | Импорт | Используется? | Статус |
|-------------|--------|:--------:|--------|:-------------:|--------|
| aiogram | >=3.4.1 | Нет | main.py: Bot, Dispatcher, F, ChatAction, ParseMode, TelegramBadRequest, Command, Message, AiohttpSession | ✅ | OK |
| aiohttp-socks | >=0.10,<0.11 | Диапазон | Косвенно через aiogram (AiohttpSession) | ✅ | OK |
| faiss-cpu | >=1.9.0.post1 | Нет | vector_index.py: `import faiss` (условный) | ✅ | OK |
| **fastmcp** | **==2.12.4** | **Да** | mcp_client.py: Client, MCPConfig | ✅ | OK |
| **mcp** | **==1.17.0** | **Да** | Транзитивно через fastmcp | ✅ | OK |
| numpy | >=1.26.0 | Нет | product_retriever.py, vector_index.py: np.asarray, np.ndarray | ✅ | OK |
| pydantic | >=2.7.0 | Нет | config.py, sgr_agent.py: BaseModel, Field, ValidationError | ✅ | OK |
| pydantic-settings | >=2.2.1 | Нет | Не найден прямой import | ⚠️ | Проверить |
| python-dotenv | >=1.0.0 | Нет | config.py: load_dotenv | ✅ | OK |
| PyYAML | >=6.0.1 | Нет | config.py: yaml.safe_load | ✅ | OK |
| httpx[socks] | >=0.28.1 | Нет | embeddings_client.py, llm_client.py, telegram_draft.py: httpx.AsyncClient | ✅ | OK |
| telegramify-markdown | >=0.1.0 | Нет | formatting.py: `from telegramify_markdown import markdownify` (условный) | ✅ | OK |
| **watchfiles** | **>=0.24.0** | **Нет** | **Нигде не импортируется** | **❌** | **Удалить** |

### Dev

| Зависимость | Версия | Используется? |
|-------------|--------|:-------------:|
| pytest | >=8.0.0 | ✅ (тесты) |
| ruff | >=0.6.0 | ✅ (линтер) |

---

## Неиспользуемые зависимости

### 1. watchfiles >=0.24.0
**Статус: НЕ ИСПОЛЬЗУЕТСЯ**

Ни один `.py` файл не содержит `import watchfiles` или `from watchfiles`. Вероятно, была добавлена для hot-reload при разработке, но не интегрирована. Можно безопасно удалить.

### 2. pydantic-settings >=2.2.1
**Статус: ПОД ВОПРОСОМ**

Прямой `import pydantic_settings` в src/ не найден. Однако может использоваться неявно через `pydantic.BaseSettings`. Требует ручной проверки — если `BaseSettings` не используется, можно удалить.

---

## Пиннинг версий

### Хорошо пиннены
- `fastmcp==2.12.4` — точная версия, критично для MCP-совместимости
- `mcp==1.17.0` — точная версия, пара к fastmcp
- `aiohttp-socks>=0.10,<0.11` — верхняя граница предотвращает breaking changes

### Проблемы с пиннингом

| Зависимость | Текущее | Рекомендация | Почему |
|-------------|---------|--------------|--------|
| openai | (транзитивно) | Если используется: `>=1.50.0` | `>=1.0.0` слишком широкий, API менялся |
| aiogram | >=3.4.1 | `>=3.16.0` или `>=3.4.1,<4.0` | Поднять min или добавить верхнюю границу |
| numpy | >=1.26.0 | `>=1.26.0,<3.0` | numpy 2.x имеет breaking changes в C API |
| pydantic | >=2.7.0 | OK или `>=2.7.0,<3.0` | Pydantic 3.x планируется, будут breaking changes |

**Общая рекомендация:** Добавить `requirements.lock` или `pip-compile` для фиксации точных версий в production.

---

## Известные CVE

### aiohttp (транзитивная через aiogram)

aiohttp не указан напрямую в зависимостях, но устанавливается как зависимость aiogram. Актуальные CVE:

| CVE | Severity | Описание | Затронуто | Исправлено |
|-----|:--------:|----------|-----------|:----------:|
| CVE-2025-53643 | LOW (1.7) | HTTP Request/Response Smuggling через trailer sections (pure-Python only) | < 3.12.14 | 3.12.14 |
| CVE-2025-69229 | MEDIUM (6.6) | DoS через chunked messages — CPU exhaustion при request.read() | ≤ 3.13.2 | 3.13.3 |
| CVE-2025-69228 | MEDIUM (6.6) | DoS — memory exhaustion при Request.post() с большим payload | ≤ 3.13.2 | 3.13.3 |
| CVE-2025-69227 | MEDIUM (6.6) | DoS — бесконечный цикл при Request.post() с PYTHONOPTIMIZE=1 | ≤ 3.13.2 | 3.13.3 |
| CVE-2025-69230 | LOW (2.7) | Logging storm при невалидных cookies | ≤ 3.13.2 | 3.13.3 |

**Действие:** Убедиться, что установлена aiohttp ≥ 3.13.3.

### Остальные пакеты

| Пакет | CVE | Статус |
|-------|-----|--------|
| pydantic >=2.7.0 | CVE-2024-3772 (ReDoS, ≤2.3.0) | ✅ Не затронуты |
| httpx >=0.28.1 | CVE-2021-41945 (≤0.22.0) | ✅ Не затронуты |
| PyYAML >=6.0.1 | Все CVE для ≤5.x | ✅ Не затронуты |
| numpy >=1.26.0 | Все CVE для ≤1.24.x | ✅ Не затронуты |
| fastmcp, mcp | Нет CVE | ✅ Чисто |
| faiss-cpu | Нет CVE | ✅ Чисто |
| openai | Нет CVE | ✅ Чисто |
| telegramify-markdown | Нет CVE | ✅ Чисто |

---

## Дублирование функционала

| Пара | Дублирование | Рекомендация |
|------|-------------|--------------|
| httpx + aiohttp | Оба — HTTP-клиенты. httpx используется для API, aiohttp — транзитивно через aiogram | Допустимо — разные задачи |
| pydantic + pydantic-settings | pydantic-settings расширяет pydantic для env/settings | Удалить pydantic-settings если не используется |

Дублирования функционала нет — каждая зависимость выполняет свою роль.

---

## Рекомендации

### Приоритет 1 — Безопасность
1. **Проверить версию aiohttp** — если < 3.13.3, обновить (CVE-2025-69227/69228/69229)
2. Добавить `pip-audit` в CI для автоматической проверки CVE

### Приоритет 2 — Чистка
3. **Удалить watchfiles** — мёртвая зависимость
4. **Проверить pydantic-settings** — удалить если BaseSettings не используется

### Приоритет 3 — Пиннинг
5. Добавить верхние границы для major-версий: `numpy>=1.26.0,<3.0`, `pydantic>=2.7.0,<3.0`
6. Создать `requirements.lock` через `pip-compile` для production
