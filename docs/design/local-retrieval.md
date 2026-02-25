# Local retrieval

## Цель
Ускорить ответы и снизить нагрузку на MCP за счёт локального поиска по SQLite + FAISS.

## Схема
1. Текст запроса пользователя -> embeddings (`OpenRouterEmbeddingsClient`).
2. Поиск ближайших товаров в `FaissVectorIndex`.
3. Добор метаданных из SQLite (`products`) в исходном порядке id-хитов.
4. Опциональный lexical boost через FTS5 (`products_fts`).
5. Применение фильтров (`protein/fat/carbs/kcal/price/rating/...`).
6. Сортировка по `SortSpec` и возврат страницы (`limit/offset`).

## Поддерживаемые метрики
- Nutrition: `protein`, `fat`, `carbs`, `kcal` (парсинг из текстового поля).
- Price normalization: `price_per_kg`, `price_per_l`, `price_per_100`.
- Доп.сигналы ранжирования: `similarity`, `lex_match`.

## Источник данных
- Базовый каталог лежит в SQLite (`products`).
- Эмбеддинги хранятся в `product_embeddings` с `content_hash`.
- Индекс строится `scripts/build_vector_index.py` в `data/products.faiss`.

## Ограничения и эксплуатация
- Если `faiss-cpu` не установлен, semantic search недоступен.
- Несвежий индекс ведёт к устаревшей выдаче.
- При отсутствии `products` retrieval-механизм деградирует до MCP-вызовов.
