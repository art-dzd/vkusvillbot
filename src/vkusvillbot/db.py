from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

UPSERT_PRODUCT_SQL = """
    INSERT INTO products (
        id, xml_id, name, slug, description_short, description, brand,
        price_current, price_old, discount_percent, currency,
        rating_avg, rating_count, unit, weight_value, weight_unit,
        url, images_json, category_json, updated_at
    ) VALUES (
        :id, :xml_id, :name, :slug, :description_short, :description, :brand,
        :price_current, :price_old, :discount_percent, :currency,
        :rating_avg, :rating_count, :unit, :weight_value, :weight_unit,
        :url, :images_json, :category_json, :updated_at
    )
    ON CONFLICT(id) DO UPDATE SET
        xml_id=COALESCE(excluded.xml_id, products.xml_id),
        name=COALESCE(excluded.name, products.name),
        slug=COALESCE(excluded.slug, products.slug),
        description_short=COALESCE(excluded.description_short, products.description_short),
        description=COALESCE(excluded.description, products.description),
        brand=COALESCE(excluded.brand, products.brand),
        price_current=COALESCE(excluded.price_current, products.price_current),
        price_old=COALESCE(excluded.price_old, products.price_old),
        discount_percent=COALESCE(excluded.discount_percent, products.discount_percent),
        currency=COALESCE(excluded.currency, products.currency),
        rating_avg=COALESCE(excluded.rating_avg, products.rating_avg),
        rating_count=COALESCE(excluded.rating_count, products.rating_count),
        unit=COALESCE(excluded.unit, products.unit),
        weight_value=COALESCE(excluded.weight_value, products.weight_value),
        weight_unit=COALESCE(excluded.weight_unit, products.weight_unit),
        url=COALESCE(excluded.url, products.url),
        images_json=COALESCE(excluded.images_json, products.images_json),
        category_json=COALESCE(excluded.category_json, products.category_json),
        updated_at=excluded.updated_at
    """


@dataclass
class User:
    id: int
    tg_id: int
    city: str | None
    diet_notes: str | None


class Database:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()

    def close(self) -> None:
        self.conn.close()

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              tg_id INTEGER UNIQUE,
              city TEXT,
              diet_notes TEXT,
              created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS sessions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              last_intent TEXT,
              last_context TEXT,
              updated_at TEXT DEFAULT (datetime('now')),
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS messages (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              role TEXT NOT NULL,
              content TEXT NOT NULL,
              created_at TEXT DEFAULT (datetime('now')),
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);
            CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
            """
        )
        self.conn.commit()

    def _table_exists(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        return row is not None

    def get_or_create_user(self, tg_id: int) -> User:
        row = self.conn.execute(
            "SELECT id, tg_id, city, diet_notes FROM users WHERE tg_id = ?",
            (tg_id,),
        ).fetchone()
        if row:
            return User(*row)
        self.conn.execute(
            "INSERT INTO users (tg_id, city, diet_notes) VALUES (?, ?, ?)",
            (tg_id, "Moscow", None),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id, tg_id, city, diet_notes FROM users WHERE tg_id = ?",
            (tg_id,),
        ).fetchone()
        return User(*row)

    def update_user_city(self, tg_id: int, city: str) -> None:
        self.conn.execute(
            "UPDATE users SET city = ? WHERE tg_id = ?",
            (city, tg_id),
        )
        self.conn.commit()

    def update_user_diet_notes(self, tg_id: int, diet_notes: str) -> None:
        self.conn.execute(
            "UPDATE users SET diet_notes = ? WHERE tg_id = ?",
            (diet_notes, tg_id),
        )
        self.conn.commit()

    def save_session(self, user_id: int, last_intent: str, last_context: dict) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        context_json = json.dumps(last_context, ensure_ascii=False)
        row = self.conn.execute(
            "SELECT id FROM sessions WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row:
            self.conn.execute(
                """
                UPDATE sessions
                SET last_intent = ?, last_context = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (last_intent, context_json, ts, user_id),
            )
        else:
            self.conn.execute(
                """
                INSERT INTO sessions (user_id, last_intent, last_context, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, last_intent, context_json, ts),
            )
        self.conn.commit()

    def save_message(self, user_id: int, role: str, content: str) -> None:
        if not content:
            return
        self.conn.execute(
            "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        self.conn.commit()

    def get_recent_messages(self, user_id: int, limit: int = 10) -> list[dict[str, str]]:
        rows = self.conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        history = [{"role": row["role"], "content": row["content"]} for row in rows]
        history.reverse()
        return history

    def has_products(self) -> bool:
        return self._table_exists("products")

    def ensure_product_columns(self) -> None:
        if not self.has_products():
            return
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(products)").fetchall()}

        def add_column(name: str, col_type: str) -> None:
            if name not in cols:
                self.conn.execute(f"ALTER TABLE products ADD COLUMN {name} {col_type}")

        add_column("description_short", "TEXT")
        add_column("description_full", "TEXT")
        add_column("composition", "TEXT")
        add_column("nutrition", "TEXT")
        add_column("storage_conditions", "TEXT")
        self.conn.commit()

    def search_products(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, object]]:
        if not self.has_products():
            return []
        tokens = _tokenize_query(query)
        if not tokens:
            tokens = [query.strip()]
        clauses = []
        params: list[str] = []
        for token in tokens:
            clauses.append("(name LIKE ? OR description_short LIKE ? OR description_full LIKE ?)")
            like = f"%{token}%"
            params.extend([like, like, like])
        where_sql = " AND ".join(clauses) if clauses else "1=1"
        params.extend([limit, offset])
        rows = self.conn.execute(
            f"""
            SELECT id, xml_id, name, price_current, rating_avg, rating_count, unit,
                   weight_value, weight_unit, url, category_json, description_short, updated_at
            FROM products
            WHERE {where_sql}
            ORDER BY (rating_avg IS NULL), rating_avg DESC, rating_count DESC,
                     price_current ASC, updated_at DESC
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        return [_product_row_to_item(row) for row in rows]

    def get_product_details(self, product_id: int) -> dict[str, object] | None:
        if not self.has_products():
            return None
        row = self.conn.execute(
            """
            SELECT id, xml_id, name, description_full, description_short,
                   price_current, price_old, discount_percent, rating_avg, rating_count,
                   unit, weight_value, weight_unit, url, category_json, updated_at,
                   composition, nutrition, storage_conditions
            FROM products
            WHERE id = ?
            """,
            (product_id,),
        ).fetchone()
        if not row:
            return None
        props = self.conn.execute(
            "SELECT name, value FROM product_properties WHERE product_id = ?",
            (product_id,),
        ).fetchall() if self._table_exists("product_properties") else []
        return _product_details_from_row(row, props)

    def get_top_protein(self, limit: int = 5) -> list[dict[str, object]]:
        if not self.has_products():
            return []
        rows = self.conn.execute(
            """
            SELECT id, xml_id, name, nutrition, price_current, rating_avg,
                   unit, weight_value, weight_unit, url, category_json, updated_at
            FROM products
            WHERE nutrition IS NOT NULL AND trim(nutrition) <> ''
            """,
        ).fetchall()
        scored: list[dict[str, object]] = []
        for row in rows:
            protein = _extract_protein_per_100g(row["nutrition"])
            if protein is None:
                continue
            item = _product_row_to_item(row)
            item["protein_per_100g"] = protein
            scored.append(item)
        scored.sort(key=lambda item: item["protein_per_100g"], reverse=True)
        return scored[: max(1, limit)]

    def nutrition_query(
        self,
        query: str | None = None,
        limit: int = 10,
        page: int = 1,
        min_protein: float | None = None,
        max_protein: float | None = None,
        min_fat: float | None = None,
        max_fat: float | None = None,
        min_carbs: float | None = None,
        max_carbs: float | None = None,
        min_kcal: float | None = None,
        max_kcal: float | None = None,
        sort_by: str | None = None,
        order: str = "desc",
        include_missing: bool = False,
    ) -> list[dict[str, object]]:
        if not self.has_products():
            return []

        tokens = _tokenize_query(query or "")
        clauses = []
        params: list[str] = []
        for token in tokens:
            clauses.append("(name LIKE ? OR description_short LIKE ? OR description_full LIKE ?)")
            like = f"%{token}%"
            params.extend([like, like, like])
        where_sql = " AND ".join(clauses) if clauses else "1=1"

        rows = self.conn.execute(
            f"""
            SELECT id, xml_id, name, nutrition, price_current, rating_avg,
                   unit, weight_value, weight_unit, url, category_json, updated_at
            FROM products
            WHERE {where_sql}
            """,
            params,
        ).fetchall()

        def within(value: float | None, min_v: float | None, max_v: float | None) -> bool:
            if value is None:
                return include_missing
            if min_v is not None and value < min_v:
                return False
            if max_v is not None and value > max_v:
                return False
            return True

        items: list[dict[str, object]] = []
        for row in rows:
            metrics = _extract_nutrition_metrics(row["nutrition"])
            protein = metrics.get("protein")
            fat = metrics.get("fat")
            carbs = metrics.get("carbs")
            kcal = metrics.get("kcal")

            if not within(protein, min_protein, max_protein):
                continue
            if not within(fat, min_fat, max_fat):
                continue
            if not within(carbs, min_carbs, max_carbs):
                continue
            if not within(kcal, min_kcal, max_kcal):
                continue

            item = _product_row_to_item(row)
            item.update(
                {
                    "protein_per_100g": protein,
                    "fat_per_100g": fat,
                    "carbs_per_100g": carbs,
                    "kcal_per_100g": kcal,
                }
            )
            items.append(item)

        sort_key = (sort_by or "protein").lower()
        reverse = order.lower() != "asc"

        def sort_value(item: dict[str, object]) -> float:
            value_map = {
                "protein": item.get("protein_per_100g"),
                "fat": item.get("fat_per_100g"),
                "carbs": item.get("carbs_per_100g"),
                "kcal": item.get("kcal_per_100g"),
                "price": item.get("price"),
                "rating": item.get("rating"),
            }
            value = value_map.get(sort_key)
            if value is None:
                return float("-inf") if reverse else float("inf")
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("-inf") if reverse else float("inf")

        items.sort(key=sort_value, reverse=reverse)
        if limit < 1:
            limit = 10
        offset = max(0, (page - 1) * limit)
        return items[offset : offset + limit]

    def upsert_products_from_mcp(self, items: list[dict[str, object]]) -> None:
        if not self.has_products():
            return
        self.ensure_product_columns()
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for item in items:
            mapped = _map_mcp_item(item, ts)
            if mapped is None:
                continue
            self._upsert_product(mapped)
            self._insert_price_history(mapped)
        self.conn.commit()

    def update_product_details_from_mcp(self, details: dict[str, object]) -> None:
        if not self.has_products() or not details:
            return
        self.ensure_product_columns()
        product_id = details.get("id")
        if product_id is None:
            return
        try:
            product_id = int(product_id)
        except (TypeError, ValueError):
            return
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        description_full = _normalize_field(details.get("description"))
        composition, nutrition, storage, props_items = _extract_properties(
            details.get("properties") or []
        )
        updates: dict[str, object] = {}
        if description_full:
            updates["description_full"] = description_full
        if composition:
            updates["composition"] = composition
        if nutrition:
            updates["nutrition"] = nutrition
        if storage:
            updates["storage_conditions"] = storage

        if updates:
            set_clause = ", ".join(f"{col} = ?" for col in updates)
            values = list(updates.values())
            values.extend([ts, product_id])
            self.conn.execute(
                f"UPDATE products SET {set_clause}, updated_at = ? WHERE id = ?",
                values,
            )

        if props_items and self._table_exists("product_properties"):
            self.conn.execute(
                "DELETE FROM product_properties WHERE product_id = ?",
                (product_id,),
            )
            for name, value in props_items:
                self.conn.execute(
                    """
                    INSERT INTO product_properties (product_id, name, value, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (product_id, name, value, ts),
                )
        self.conn.commit()

    def is_stale(self, updated_at: str | None, max_age_hours: int) -> bool:
        if not updated_at:
            return True
        try:
            dt = datetime.fromisoformat(updated_at)
        except ValueError:
            return True
        age = datetime.now(timezone.utc) - dt
        return age.total_seconds() > max_age_hours * 3600

    def _upsert_product(self, item: dict[str, object]) -> None:
        self.conn.execute(UPSERT_PRODUCT_SQL, item)

    def _insert_price_history(self, item: dict[str, object]) -> None:
        product_id = int(item["id"])
        if not self._table_exists("price_history"):
            return
        if not _should_insert_price_history(
            self.conn,
            product_id,
            item.get("price_current"),
            item.get("price_old"),
            item.get("discount_percent"),
        ):
            return
        self.conn.execute(
            """
            INSERT INTO price_history (product_id, price_current, price_old, discount_percent, ts)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                product_id,
                item.get("price_current"),
                item.get("price_old"),
                item.get("discount_percent"),
                item.get("updated_at"),
            ),
        )


def _tokenize_query(query: str) -> list[str]:
    query = (query or "").lower()
    tokens = re.findall(r"[\\w\\d]+", query, flags=re.UNICODE)
    return [t for t in tokens if len(t) >= 2]


def _normalize_field(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\u00a0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text or None


def _extract_protein_per_100g(text: str | None) -> float | None:
    if not text:
        return None
    normalized = _normalize_field(text)
    if not normalized:
        return None
    match = re.search(
        r"белк\w*[^0-9]*([0-9]+[.,]?[0-9]*)\s*г",
        normalized.lower(),
    )
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


def _extract_nutrition_metrics(text: str | None) -> dict[str, float | None]:
    if not text:
        return {"protein": None, "fat": None, "carbs": None, "kcal": None}
    normalized = _normalize_field(text)
    if not normalized:
        return {"protein": None, "fat": None, "carbs": None, "kcal": None}
    lower = normalized.lower()

    def pick(pattern: str) -> float | None:
        match = re.search(pattern, lower)
        if not match:
            return None
        try:
            return float(match.group(1).replace(",", "."))
        except ValueError:
            return None

    protein = pick(r"белк\w*[^0-9]*([0-9]+[.,]?[0-9]*)")
    fat = pick(r"жир\w*[^0-9]*([0-9]+[.,]?[0-9]*)")
    carbs = pick(r"углевод\w*[^0-9]*([0-9]+[.,]?[0-9]*)")
    kcal = pick(r"([0-9]+[.,]?[0-9]*)\s*к?кал")

    return {"protein": protein, "fat": fat, "carbs": carbs, "kcal": kcal}


def _extract_properties(
    properties: list[dict[str, object]],
) -> tuple[str | None, str | None, str | None, list[tuple[str, str | None]]]:
    composition = None
    nutrition = None
    storage = None
    items: list[tuple[str, str | None]] = []
    for prop in properties:
        name = _normalize_field(prop.get("name"))
        value = _normalize_field(prop.get("value"))
        if not name:
            continue
        items.append((name, value))
        lname = name.lower()
        if "состав" in lname and composition is None:
            composition = value
        if (
            "пищевая ценность" in lname or "энергетическая ценность" in lname
        ) and nutrition is None:
            nutrition = value
        if "услов" in lname and "хран" in lname and storage is None:
            storage = value
    items.sort(key=lambda pair: pair[0])
    return composition, nutrition, storage, items


def _parse_weight(value: object | None) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, dict):
        raw_val = value.get("value") or value.get("weight")
        raw_unit = value.get("unit")
        try:
            return (
                float(raw_val) if raw_val is not None else None,
                str(raw_unit) if raw_unit else None,
            )
        except (TypeError, ValueError):
            return None, str(raw_unit) if raw_unit else None
    text = _normalize_field(value)
    if not text:
        return None, None
    match = re.search(r"(\\d+[\\.,]?\\d*)\\s*([^\\d\\s]+)", text)
    if not match:
        return None, None
    number = float(match.group(1).replace(",", "."))
    unit = match.group(2)
    return number, unit


def _map_mcp_item(item: dict[str, object], ts: str) -> dict[str, object] | None:
    product_id = item.get("id")
    if product_id is None:
        return None
    try:
        product_id = int(product_id)
    except (TypeError, ValueError):
        return None
    xml_id = item.get("xml_id") or item.get("xmlId") or product_id
    try:
        xml_id = int(xml_id) if xml_id is not None else None
    except (TypeError, ValueError):
        xml_id = None
    price_current = item.get("price") or item.get("price_current")
    price_old = item.get("price_old") or item.get("old_price")
    discount_percent = item.get("discount_percent") or item.get("discount")
    weight_value, weight_unit = _parse_weight(item.get("weight"))
    category = item.get("category")
    images = item.get("images")
    return {
        "id": product_id,
        "xml_id": xml_id,
        "name": _normalize_field(item.get("name")),
        "slug": _normalize_field(item.get("slug")),
        "description_short": _normalize_field(
            item.get("description") or item.get("description_short")
        ),
        "description": _normalize_field(item.get("description")),
        "brand": _normalize_field(item.get("brand")),
        "price_current": (
            float(price_current)
            if isinstance(price_current, (int, float, str)) and str(price_current)
            else None
        ),
        "price_old": (
            float(price_old)
            if isinstance(price_old, (int, float, str)) and str(price_old)
            else None
        ),
        "discount_percent": (
            float(discount_percent)
            if isinstance(discount_percent, (int, float, str)) and str(discount_percent)
            else None
        ),
        "currency": _normalize_field(item.get("currency")),
        "rating_avg": (
            float(item.get("rating"))
            if item.get("rating") is not None
            else item.get("rating_avg")
        ),
        "rating_count": item.get("rating_count"),
        "unit": _normalize_field(item.get("unit")),
        "weight_value": weight_value,
        "weight_unit": _normalize_field(weight_unit),
        "url": _normalize_field(item.get("url")),
        "images_json": json.dumps(images, ensure_ascii=False) if images is not None else None,
        "category_json": json.dumps(category, ensure_ascii=False) if category is not None else None,
        "updated_at": ts,
    }


def _product_row_to_item(row: sqlite3.Row) -> dict[str, object]:
    weight = None
    if row["weight_value"] is not None and row["weight_unit"]:
        weight = f"{row['weight_value']} {row['weight_unit']}"
    category = None
    if row["category_json"]:
        try:
            category = json.loads(row["category_json"])
        except json.JSONDecodeError:
            category = row["category_json"]
    return {
        "id": row["id"],
        "xml_id": row["xml_id"],
        "name": row["name"],
        "price": row["price_current"],
        "rating": row["rating_avg"],
        "unit": row["unit"],
        "weight": weight,
        "url": row["url"],
        "category": category,
        "updated_at": row["updated_at"],
    }


def _product_details_from_row(
    row: sqlite3.Row,
    props: list[sqlite3.Row],
) -> dict[str, object]:
    weight = None
    if row["weight_value"] is not None and row["weight_unit"]:
        weight = f"{row['weight_value']} {row['weight_unit']}"
    category = None
    if row["category_json"]:
        try:
            category = json.loads(row["category_json"])
        except json.JSONDecodeError:
            category = row["category_json"]
    properties: list[dict[str, object]] = []
    if props:
        for prop in props:
            properties.append({"name": prop["name"], "value": prop["value"]})
    else:
        if row["composition"]:
            properties.append({"name": "Состав", "value": row["composition"]})
        if row["nutrition"]:
            properties.append({"name": "Пищевая ценность", "value": row["nutrition"]})
        if row["storage_conditions"]:
            properties.append({"name": "Условия хранения", "value": row["storage_conditions"]})
    return {
        "id": row["id"],
        "xml_id": row["xml_id"],
        "name": row["name"],
        "description": row["description_full"] or row["description_short"],
        "price": row["price_current"],
        "rating": row["rating_avg"],
        "unit": row["unit"],
        "weight": weight,
        "url": row["url"],
        "category": category,
        "properties": properties,
        "updated_at": row["updated_at"],
    }


def _should_insert_price_history(
    conn: sqlite3.Connection,
    product_id: int,
    price_current: float | None,
    price_old: float | None,
    discount_percent: float | None,
) -> bool:
    row = conn.execute(
        """
        SELECT price_current, price_old, discount_percent
        FROM price_history
        WHERE product_id = ?
        ORDER BY ts DESC
        LIMIT 1
        """,
        (product_id,),
    ).fetchone()
    if row is None:
        return True
    last_current, last_old, last_discount = row
    return not (
        _same_value(last_current, price_current)
        and _same_value(last_old, price_old)
        and _same_value(last_discount, discount_percent)
    )


def _same_value(a: float | None, b: float | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) < 1e-9
