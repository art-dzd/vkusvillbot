import asyncio
import json

import numpy as np

from vkusvillbot.db import Database
from vkusvillbot.product_retriever import ProductRetriever, SortSpec
from vkusvillbot.vector_index import FaissVectorIndex


class FakeEmbeddings:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.mapping.get(text, [0.0, 0.0, 0.0]) for text in texts]


def _create_products_schema(db: Database) -> None:
    db.conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS products (
          id INTEGER PRIMARY KEY,
          xml_id INTEGER,
          name TEXT,
          description_short TEXT,
          description_full TEXT,
          composition TEXT,
          brand TEXT,
          nutrition TEXT,
          price_current REAL,
          rating_avg REAL,
          rating_count INTEGER,
          unit TEXT,
          weight_value REAL,
          weight_unit TEXT,
          url TEXT,
          category_json TEXT,
          updated_at TEXT
        );
        """
    )
    db.conn.commit()


def test_nutrition_query_supports_multi_sort(tmp_path) -> None:
    db = Database(str(tmp_path / "test.db"))
    try:
        _create_products_schema(db)
        rows = [
            (
                1,
                1001,
                "Творог 5%",
                "Молочный продукт",
                None,
                None,
                None,
                "Белки 16 г, Жиры 5 г, Углеводы 3 г, 120 ккал",
                120.0,
                4.8,
                120,
                "шт",
                200.0,
                "г",
                "https://example/1",
                json.dumps(["молочные"], ensure_ascii=False),
                "2025-01-01T00:00:00+00:00",
            ),
            (
                2,
                1002,
                "Йогурт",
                "Молочный продукт",
                None,
                None,
                None,
                "Белки 4 г, Жиры 2 г, Углеводы 10 г, 80 ккал",
                60.0,
                4.5,
                20,
                "шт",
                150.0,
                "г",
                "https://example/2",
                json.dumps(["молочные"], ensure_ascii=False),
                "2025-01-01T00:00:00+00:00",
            ),
            (
                3,
                1003,
                "Протеиновый батончик",
                "Снэк",
                None,
                None,
                None,
                "Белки 20 г, Жиры 8 г, Углеводы 15 г, 220 ккал",
                140.0,
                4.2,
                10,
                "шт",
                60.0,
                "г",
                "https://example/3",
                json.dumps(["снэки"], ensure_ascii=False),
                "2025-01-01T00:00:00+00:00",
            ),
        ]
        db.conn.executemany(
            """
            INSERT INTO products (
              id, xml_id, name, description_short, description_full, composition, brand,
              nutrition, price_current, rating_avg, rating_count, unit,
              weight_value, weight_unit, url, category_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        db.conn.commit()

        items = db.nutrition_query(
            limit=10,
            include_missing=True,
            sort=[{"field": "protein", "dir": "desc"}, {"field": "price", "dir": "asc"}],
        )
        assert [item["id"] for item in items][:3] == [3, 1, 2]
    finally:
        db.close()


def test_product_retriever_semantic_search(tmp_path) -> None:
    db = Database(str(tmp_path / "test.db"))
    try:
        _create_products_schema(db)
        db.conn.executemany(
            """
            INSERT INTO products (
              id, xml_id, name, description_short, nutrition,
              price_current, rating_avg, rating_count,
              unit, weight_value, weight_unit,
              url, category_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    10,
                    2010,
                    "Молоко безлактозное",
                    "Молоко",
                    "Белки 3.0 г, Жиры 3.2 г, Углеводы 4.7 г, 60 ккал",
                    150.0,
                    4.7,
                    50,
                    "шт",
                    970.0,
                    "мл",
                    "https://example/10",
                    json.dumps(["молочные"], ensure_ascii=False),
                    "2025-01-01T00:00:00+00:00",
                ),
                (
                    11,
                    2011,
                    "Хлеб цельнозерновой",
                    "Хлеб",
                    "Белки 8 г, Жиры 2 г, Углеводы 45 г, 240 ккал",
                    90.0,
                    4.3,
                    10,
                    "шт",
                    400.0,
                    "г",
                    "https://example/11",
                    json.dumps(["хлеб"], ensure_ascii=False),
                    "2025-01-01T00:00:00+00:00",
                ),
            ],
        )
        db.conn.commit()

        index_path = tmp_path / "products.faiss"
        index = FaissVectorIndex(index_path)
        index.build_and_save(
            [10, 11],
            np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32"),
        )

        retriever = ProductRetriever(
            db=db,
            embeddings=FakeEmbeddings({"молоко": [1.0, 0.0, 0.0]}),
            index=index,
            candidate_pool=10,
            fts_boost=False,
        )

        items = asyncio.run(
            retriever.semantic_search(
                "молоко",
                limit=2,
                sort=[SortSpec(field="similarity", direction="desc")],
            )
        )
        assert items
        assert items[0]["id"] == 10
        assert items[0]["price_per_l"] is not None
    finally:
        db.close()
