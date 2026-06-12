#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from fastmcp import Client
from fastmcp.mcp_config import MCPConfig

BASE_URL = "https://vkusvill.ru"
GOODS_URL = f"{BASE_URL}/goods/"
MCP_URL = "https://mcp001.vkusvill.ru/mcp"
USER_AGENT = "Mozilla/5.0"
HTTP_RETRIES = 4
MCP_RETRIES = 4
HTTP_TIMEOUT_SECONDS = 30
PAGE_SIZE_HINT = 24
DEFAULT_RATE_LIMIT_SLEEP_SECONDS = 3600.0
FOOD_CATEGORY_PATTERNS = (
    "готовая еда",
    "сладост",
    "десерт",
    "заморож",
    "напит",
    "молоч",
    "яйц",
    "рыб",
    "икра",
    "морепродукт",
    "овощ",
    "фрукт",
    "ягод",
    "зелень",
    "мяс",
    "птиц",
    "говядин",
    "телят",
    "куриц",
    "свинин",
    "индейк",
    "баранин",
    "бакале",
    "сыр",
    "кафе",
    "колбас",
    "сосиск",
    "деликатес",
    "орех",
    "чипс",
    "снек",
    "снэк",
    "консерв",
    "мёд",
    "мед",
    "варенье",
    "джем",
    "пюре",
    "хлеб",
    "выпеч",
    "растительные масла",
    "масла, соусы",
    "соус",
    "спец",
    "сахар",
    "соль",
    "особое питание",
    "постное",
    "вегетариан",
    "веган",
    "детское питание",
    "каши",
    "спортивное питание",
    "продукты без глютена",
    "продукты без лактозы",
    "пиво",
    "алкоголь",
)
NUTRITION_COLUMNS = {
    "protein_per_100g": "REAL",
    "fat_per_100g": "REAL",
    "carbs_per_100g": "REAL",
    "kcal_per_100g": "REAL",
}


@dataclass(frozen=True)
class RuntimeConfig:
    db_path: Path
    mcp_url: str
    catalog_sleep: float
    mcp_sleep: float
    jitter: float
    commit_every: int
    max_errors: int
    max_pages: int
    rate_limit_sleep: float


class RateLimitedError(RuntimeError):
    pass


@dataclass(frozen=True)
class Category:
    url: str
    name: str


@dataclass(frozen=True)
class SyncRun:
    id: int
    started_at: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = unescape(str(value)).replace("\u00a0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text or None


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    match = re.search(r"\d+(?:[.,]\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0).replace(",", "."))


def same_number(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return abs(left - right) < 1e-9


def clean_slug_from_url(url: str | None) -> str | None:
    if not url:
        return None
    path = urlparse(url).path.rstrip("/")
    if not path:
        return None
    return path.rsplit("/", 1)[-1]


def append_page(url: str, page: int) -> str:
    if page <= 1:
        return url
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    query["PAGEN_1"] = str(page)
    return urlunparse(parsed._replace(query=urlencode(query)))


def fetch_html(url: str) -> str:
    last_error: Exception | None = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                return response.read().decode("utf-8", errors="ignore")
        except (TimeoutError, URLError) as exc:
            last_error = exc
            time.sleep(min(30.0, attempt * attempt))
    raise RuntimeError(f"HTTP failed for {url}: {last_error}")


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS products (
          id INTEGER PRIMARY KEY,
          xml_id INTEGER NOT NULL,
          name TEXT NOT NULL,
          slug TEXT,
          description_short TEXT,
          description_full TEXT,
          description TEXT,
          composition TEXT,
          nutrition TEXT,
          storage_conditions TEXT,
          brand TEXT,
          price_current REAL,
          price_old REAL,
          discount_percent REAL,
          currency TEXT,
          rating_avg REAL,
          rating_count INTEGER,
          unit TEXT,
          weight_value REAL,
          weight_unit TEXT,
          url TEXT,
          images_json TEXT,
          category_json TEXT,
          updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS price_history (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          product_id INTEGER NOT NULL,
          price_current REAL,
          price_old REAL,
          discount_percent REAL,
          ts TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS product_properties (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          product_id INTEGER NOT NULL,
          name TEXT NOT NULL,
          value TEXT,
          updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS product_discounts (
          product_id INTEGER NOT NULL,
          discount_type TEXT NOT NULL,
          price_current REAL,
          price_old REAL,
          discount_percent REAL,
          discount_info TEXT,
          synced_at TEXT NOT NULL,
          PRIMARY KEY (product_id, discount_type)
        );

        CREATE TABLE IF NOT EXISTS sync_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          started_at TEXT NOT NULL,
          finished_at TEXT,
          status TEXT NOT NULL,
          catalog_seen INTEGER DEFAULT 0,
          details_ok INTEGER DEFAULT 0,
          details_errors INTEGER DEFAULT 0,
          discounts_seen INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS sync_errors (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id INTEGER NOT NULL,
          stage TEXT NOT NULL,
          product_id INTEGER,
          url TEXT,
          error TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        """
    )
    ensure_columns(conn)
    conn.commit()


def ensure_columns(conn: sqlite3.Connection) -> None:
    product_columns = {row[1] for row in conn.execute("PRAGMA table_info(products)").fetchall()}
    wanted = {
        "description_short": "TEXT",
        "description_full": "TEXT",
        "composition": "TEXT",
        "nutrition": "TEXT",
        "storage_conditions": "TEXT",
        "first_seen_at": "TEXT",
        "last_seen_at": "TEXT",
        "last_catalog_run_id": "INTEGER",
        "details_synced_at": "TEXT",
        "active": "INTEGER DEFAULT 1",
        **NUTRITION_COLUMNS,
    }
    for name, column_type in wanted.items():
        if name not in product_columns:
            conn.execute(f"ALTER TABLE products ADD COLUMN {name} {column_type}")


def start_run(conn: sqlite3.Connection) -> SyncRun:
    started_at = utc_now()
    conn.execute(
        "INSERT INTO sync_runs (started_at, status) VALUES (?, ?)",
        (started_at, "running"),
    )
    conn.commit()
    run_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    return SyncRun(id=run_id, started_at=started_at)


def finish_run(conn: sqlite3.Connection, run: SyncRun, status: str) -> None:
    conn.execute(
        "UPDATE sync_runs SET finished_at = ?, status = ? WHERE id = ?",
        (utc_now(), status, run.id),
    )
    conn.commit()


def record_error(
    conn: sqlite3.Connection,
    run: SyncRun,
    stage: str,
    error: Exception | str,
    product_id: int | None = None,
    url: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO sync_errors (run_id, stage, product_id, url, error, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (run.id, stage, product_id, url, str(error), utc_now()),
    )


def discover_categories() -> list[Category]:
    soup = BeautifulSoup(fetch_html(GOODS_URL), "html.parser")
    result: list[Category] = []
    seen: set[str] = set()
    for link in soup.select("a[href]"):
        href = link.get("href") or ""
        if not is_catalog_href(href):
            continue
        url = urljoin(BASE_URL, href)
        if url in seen:
            continue
        seen.add(url)
        result.append(Category(url=url, name=normalize_text(link.get_text(" ", strip=True)) or url))
    return result


def is_catalog_href(href: str) -> bool:
    if not href.startswith("/goods/"):
        return False
    if href.rstrip("/") == "/goods":
        return False
    return not href.endswith(".html")


def max_page_from_soup(soup: BeautifulSoup) -> int:
    max_page = 1
    for link in soup.select('a[href*="PAGEN_1"]'):
        href = link.get("href") or ""
        query = dict(parse_qsl(urlparse(href).query))
        page = query.get("PAGEN_1")
        if page and page.isdigit():
            max_page = max(max_page, int(page))
    return max_page


def parse_catalog_page(
    html: str, source_url: str, run: SyncRun
) -> tuple[list[dict[str, Any]], int]:
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for card in soup.select(".ProductCard"):
        item = parse_card(card, source_url, run)
        if item:
            items.append(item)
    return items, max_page_from_soup(soup)


def parse_card(card: Any, source_url: str, run: SyncRun) -> dict[str, Any] | None:
    product_id = parse_int(card.get("data-id"))
    if product_id is None:
        sku = card.select_one('meta[itemprop="sku"]')
        product_id = parse_int(sku.get("content") if sku else None)
    if product_id is None:
        return None
    xml_id = parse_int(card.get("data-xmlid")) or product_id
    link = pick_product_link(card)
    image = card.select_one('img[itemprop="image"], .ProductCard__imageImg, img')
    name = normalize_text(image.get("alt") if image else None)
    if not name:
        name_node = card.select_one('[itemprop="name"]')
        name = normalize_text(name_node.get_text(" ", strip=True) if name_node else None)
    if not name and link:
        name = normalize_text(link.get_text(" ", strip=True))
    if not name:
        name = f"Product {product_id}"

    price_current = price_from_card(card)
    price_old = old_price_from_card(card)
    if price_old is not None and price_current is not None and price_old <= price_current:
        price_old = price_current
        discount_percent = 0.0
    elif price_old and price_current:
        discount_percent = round((1 - price_current / price_old) * 100, 2)
    else:
        discount_percent = None

    weight_value, weight_unit = parse_weight_from_name(
        link.get_text(" ", strip=True) if link else name
    )
    category_path = parse_category_path(card)
    url = urljoin(BASE_URL, link.get("href")) if link and link.get("href") else None
    now = utc_now()
    return {
        "id": product_id,
        "xml_id": xml_id,
        "name": name,
        "slug": clean_slug_from_url(url),
        "description_short": text_from_selector(card, '[itemprop="description"]'),
        "description": text_from_selector(card, '[itemprop="description"]'),
        "brand": text_from_selector(card, '[itemprop="brand"] [itemprop="name"]'),
        "price_current": price_current,
        "price_old": price_old,
        "discount_percent": discount_percent,
        "currency": attr_from_selector(card, '[itemprop="priceCurrency"]', "content") or "RUB",
        "rating_avg": to_float(text_from_selector(card, '[itemprop="ratingValue"]')),
        "rating_count": parse_int(attr_from_selector(card, '[itemprop="reviewCount"]', "content")),
        "unit": parse_unit(card),
        "weight_value": weight_value,
        "weight_unit": weight_unit,
        "url": url,
        "images_json": json.dumps(
            [image.get("src")] if image and image.get("src") else [], ensure_ascii=False
        ),
        "category_json": json.dumps(category_path, ensure_ascii=False),
        "updated_at": now,
        "first_seen_at": now,
        "last_seen_at": now,
        "last_catalog_run_id": run.id,
        "active": 1,
        "source_url": source_url,
    }


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def pick_product_link(card: Any) -> Any | None:
    links = card.select('a[href^="/goods/"], a[href*="vkusvill.ru/goods/"]')
    for link in links:
        href = link.get("href") or ""
        if re.search(r"-\d+/?$", urlparse(href).path.rstrip("/")):
            return link
    return links[0] if links else None


def text_from_selector(node: Any, selector: str) -> str | None:
    found = node.select_one(selector)
    return normalize_text(found.get_text(" ", strip=True) if found else None)


def attr_from_selector(node: Any, selector: str, attr: str) -> str | None:
    found = node.select_one(selector)
    return normalize_text(found.get(attr) if found else None)


def price_from_card(card: Any) -> float | None:
    content = attr_from_selector(card, '[itemprop="price"]', "content")
    if content:
        return to_float(content)
    return to_float(
        text_from_selector(card, ".js-datalayer-catalog-list-price, .ProductCard__price")
    )


def old_price_from_card(card: Any) -> float | None:
    return to_float(text_from_selector(card, ".js-datalayer-catalog-list-price-old"))


def parse_unit(card: Any) -> str | None:
    text = text_from_selector(card, ".ProductCard__price")
    if not text:
        return None
    match = re.search(r"/\s*([A-Za-zА-Яа-я.]+)", text)
    return normalize_text(match.group(1)) if match else None


def parse_weight_from_name(text: str | None) -> tuple[float | None, str | None]:
    if not text:
        return None, None
    match = re.search(r"(?<!\d)(\d+(?:[.,]\d+)?)\s*(кг|г|л|мл|шт)\b", text.lower())
    if not match:
        return None, None
    return float(match.group(1).replace(",", ".")), match.group(2)


def parse_category_path(card: Any) -> list[str]:
    node = card.select_one(".js-datalayer-catalog-list-category")
    text = normalize_text(node.get_text(" ", strip=True) if node else None)
    if not text:
        return []
    return [part.strip() for part in text.split("//") if part.strip()]


def upsert_catalog_item(conn: sqlite3.Connection, item: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO products (
          id, xml_id, name, slug, description_short, description,
          brand, price_current, price_old, discount_percent, currency,
          rating_avg, rating_count, unit, weight_value, weight_unit, url,
          images_json, category_json, updated_at, first_seen_at, last_seen_at,
          last_catalog_run_id, active
        )
        VALUES (
          :id, :xml_id, :name, :slug, :description_short, :description,
          :brand, :price_current, :price_old, :discount_percent, :currency,
          :rating_avg, :rating_count, :unit, :weight_value, :weight_unit, :url,
          :images_json, :category_json, :updated_at, :first_seen_at, :last_seen_at,
          :last_catalog_run_id, :active
        )
        ON CONFLICT(id) DO UPDATE SET
          xml_id=excluded.xml_id,
          name=COALESCE(excluded.name, products.name),
          slug=COALESCE(excluded.slug, products.slug),
          description_short=COALESCE(excluded.description_short, products.description_short),
          description=COALESCE(excluded.description, products.description),
          brand=COALESCE(excluded.brand, products.brand),
          price_current=COALESCE(excluded.price_current, products.price_current),
          price_old=excluded.price_old,
          discount_percent=excluded.discount_percent,
          currency=COALESCE(excluded.currency, products.currency),
          rating_avg=COALESCE(excluded.rating_avg, products.rating_avg),
          rating_count=COALESCE(excluded.rating_count, products.rating_count),
          unit=COALESCE(excluded.unit, products.unit),
          weight_value=COALESCE(excluded.weight_value, products.weight_value),
          weight_unit=COALESCE(excluded.weight_unit, products.weight_unit),
          url=COALESCE(excluded.url, products.url),
          images_json=COALESCE(excluded.images_json, products.images_json),
          category_json=COALESCE(excluded.category_json, products.category_json),
          updated_at=excluded.updated_at,
          first_seen_at=COALESCE(products.first_seen_at, excluded.first_seen_at),
          last_seen_at=excluded.last_seen_at,
          last_catalog_run_id=excluded.last_catalog_run_id,
          active=1
        """,
        item,
    )


def record_price_history(conn: sqlite3.Connection, item: dict[str, Any]) -> None:
    row = conn.execute(
        """
        SELECT price_current, price_old, discount_percent
        FROM price_history
        WHERE product_id = ?
        ORDER BY ts DESC
        LIMIT 1
        """,
        (item["id"],),
    ).fetchone()
    if row and all(
        (
            same_number(row[0], item["price_current"]),
            same_number(row[1], item["price_old"]),
            same_number(row[2], item["discount_percent"]),
        )
    ):
        return
    conn.execute(
        """
        INSERT INTO price_history (product_id, price_current, price_old, discount_percent, ts)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            item["id"],
            item["price_current"],
            item["price_old"],
            item["discount_percent"],
            item["updated_at"],
        ),
    )


def sync_catalog(conn: sqlite3.Connection, run: SyncRun, config: RuntimeConfig) -> set[int]:
    categories = discover_categories()
    print(f"catalog categories={len(categories)}")
    seen_ids: set[int] = set()
    for index, category in enumerate(categories, start=1):
        try:
            page_one = fetch_html(category.url)
            items, max_page = parse_catalog_page(page_one, category.url, run)
            max_page = min(max_page, config.max_pages)
            saved = save_catalog_items(conn, items, seen_ids)
            print(
                f"catalog {index}/{len(categories)} {category.url} "
                f"page=1/{max_page} items={len(items)} saved={saved} seen={len(seen_ids)}"
            )
            sleep_sync(config.catalog_sleep, config.jitter)
            for page in range(2, max_page + 1):
                url = append_page(category.url, page)
                page_items, _ = parse_catalog_page(fetch_html(url), url, run)
                saved += save_catalog_items(conn, page_items, seen_ids)
                if page % 10 == 0 or page == max_page:
                    print(
                        f"catalog {index}/{len(categories)} {category.url} "
                        f"page={page}/{max_page} seen={len(seen_ids)}"
                    )
                sleep_sync(config.catalog_sleep, config.jitter)
        except Exception as exc:
            record_error(conn, run, "catalog", exc, url=category.url)
            conn.commit()
            print(f"catalog error {category.url}: {exc}")
    mark_inactive_products(conn, run)
    conn.execute(
        "UPDATE sync_runs SET catalog_seen = ? WHERE id = ?",
        (len(seen_ids), run.id),
    )
    conn.commit()
    return seen_ids


def save_catalog_items(
    conn: sqlite3.Connection,
    items: list[dict[str, Any]],
    seen_ids: set[int],
) -> int:
    saved = 0
    for item in items:
        if item["id"] in seen_ids:
            continue
        upsert_catalog_item(conn, item)
        record_price_history(conn, item)
        seen_ids.add(item["id"])
        saved += 1
    conn.commit()
    return saved


def mark_inactive_products(conn: sqlite3.Connection, run: SyncRun) -> None:
    conn.execute(
        """
        UPDATE products
        SET active = 0
        WHERE last_catalog_run_id IS NULL OR last_catalog_run_id <> ?
        """,
        (run.id,),
    )
    conn.execute(
        "UPDATE products SET active = 1 WHERE last_catalog_run_id = ?",
        (run.id,),
    )


def sleep_sync(base: float, jitter: float) -> None:
    time.sleep(base + random.uniform(0, jitter))


def parse_mcp_price(price: Any) -> tuple[float | None, float | None, float | None, str, str | None]:
    if not isinstance(price, dict):
        return None, None, None, "RUB", None
    return (
        to_float(price.get("current")),
        to_float(price.get("old")),
        to_float(price.get("discount_percent")),
        normalize_text(price.get("currency")) or "RUB",
        normalize_text(price.get("discount_info")),
    )


def parse_mcp_weight(weight: Any) -> tuple[float | None, str | None]:
    if not isinstance(weight, dict):
        return None, None
    return to_float(weight.get("value")), normalize_text(weight.get("unit"))


def parse_mcp_rating(rating: Any) -> tuple[float | None, int | None]:
    if not isinstance(rating, dict):
        return None, None
    return to_float(rating.get("average")), parse_int(rating.get("count"))


def extract_mcp_properties(
    properties: list[dict[str, Any]],
) -> tuple[str | None, str | None, str | None]:
    composition = None
    nutrition = None
    storage = None
    for item in properties:
        name = normalize_text(item.get("name"))
        value = normalize_text(item.get("value"))
        if not name or not value:
            continue
        lower = name.lower()
        if composition is None and "состав" in lower:
            composition = value
        if nutrition is None and ("пищевая" in lower or "энергетическая" in lower):
            nutrition = value
        if storage is None and "услов" in lower and "хран" in lower:
            storage = value
    return composition, nutrition, storage


def parse_nutrition_metrics(text: str | None) -> dict[str, float | None]:
    if not text:
        return empty_nutrition_metrics()
    normalized = normalize_text(re.sub(r"<br\s*/?>", " ", text, flags=re.I))
    if not normalized:
        return empty_nutrition_metrics()
    return {
        "protein_per_100g": pick_nutrition_value(normalized, r"белк\w*"),
        "fat_per_100g": pick_nutrition_value(normalized, r"жир\w*"),
        "carbs_per_100g": pick_nutrition_value(normalized, r"углевод\w*"),
        "kcal_per_100g": pick_kcal_value(normalized),
    }


def empty_nutrition_metrics() -> dict[str, float | None]:
    return {name: None for name in NUTRITION_COLUMNS}


def pick_nutrition_value(text: str, label_pattern: str) -> float | None:
    match = re.search(
        rf"{label_pattern}[^0-9]{{0,80}}([0-9]+(?:[.,][0-9]+)?)\s*г",
        text,
        flags=re.I,
    )
    return float(match.group(1).replace(",", ".")) if match else None


def pick_kcal_value(text: str) -> float | None:
    match = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*к?кал", text, flags=re.I)
    return float(match.group(1).replace(",", ".")) if match else None


def nutrition_text_from_metrics(metrics: dict[str, float | None]) -> str | None:
    protein = metrics.get("protein_per_100g")
    fat = metrics.get("fat_per_100g")
    carbs = metrics.get("carbs_per_100g")
    kcal = metrics.get("kcal_per_100g")
    if all(value is None for value in (protein, fat, carbs, kcal)):
        return None
    parts = []
    if protein is not None:
        parts.append(f"белки {protein:g} г")
    if fat is not None:
        parts.append(f"жиры {fat:g} г")
    if carbs is not None:
        parts.append(f"углеводы {carbs:g} г")
    text = ", ".join(parts)
    if kcal is not None:
        return f"{text}; {kcal:g} ккал" if text else f"{kcal:g} ккал"
    return text or None


def backfill_nutrition_metrics(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT id, nutrition
        FROM products
        WHERE nutrition IS NOT NULL
          AND length(trim(nutrition)) > 0
          AND (
            protein_per_100g IS NULL OR fat_per_100g IS NULL
            OR carbs_per_100g IS NULL OR kcal_per_100g IS NULL
          )
        """
    ).fetchall()
    for product_id, nutrition in rows:
        metrics = parse_nutrition_metrics(nutrition)
        conn.execute(
            """
            UPDATE products
            SET protein_per_100g = COALESCE(protein_per_100g, ?),
                fat_per_100g = COALESCE(fat_per_100g, ?),
                carbs_per_100g = COALESCE(carbs_per_100g, ?),
                kcal_per_100g = COALESCE(kcal_per_100g, ?)
            WHERE id = ?
            """,
            (
                metrics["protein_per_100g"],
                metrics["fat_per_100g"],
                metrics["carbs_per_100g"],
                metrics["kcal_per_100g"],
                product_id,
            ),
        )
    conn.commit()


def map_mcp_item(
    raw_item: dict[str, Any],
    run: SyncRun,
    from_details: bool = False,
) -> dict[str, Any] | None:
    product_id = parse_int(raw_item.get("id"))
    xml_id = parse_int(raw_item.get("xml_id") or raw_item.get("xmlId")) or product_id
    if product_id is None or xml_id is None:
        return None
    price_current, price_old, discount_percent, currency, _ = parse_mcp_price(raw_item.get("price"))
    weight_value, weight_unit = parse_mcp_weight(raw_item.get("weight"))
    rating_avg, rating_count = parse_mcp_rating(raw_item.get("rating"))
    composition, nutrition, storage = extract_mcp_properties(raw_item.get("properties") or [])
    nutrition_metrics = parse_nutrition_metrics(nutrition)
    return {
        "id": product_id,
        "xml_id": xml_id,
        "name": normalize_text(raw_item.get("name")),
        "slug": normalize_text(raw_item.get("slug")),
        "description_short": normalize_text(raw_item.get("description")),
        "description_full": normalize_text(raw_item.get("description")),
        "description": normalize_text(raw_item.get("description")),
        "composition": composition,
        "nutrition": nutrition,
        **nutrition_metrics,
        "storage_conditions": storage,
        "brand": normalize_text(raw_item.get("brand")),
        "price_current": price_current,
        "price_old": price_old,
        "discount_percent": discount_percent,
        "currency": currency,
        "rating_avg": rating_avg,
        "rating_count": rating_count,
        "unit": normalize_text(raw_item.get("unit")),
        "weight_value": weight_value,
        "weight_unit": weight_unit,
        "url": normalize_text(raw_item.get("url")),
        "images_json": json.dumps(raw_item.get("images") or [], ensure_ascii=False),
        "category_json": json.dumps(raw_item.get("category") or [], ensure_ascii=False),
        "updated_at": utc_now(),
        "details_synced_at": utc_now() if from_details else None,
    }


def upsert_mcp_item(conn: sqlite3.Connection, item: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO products (
          id, xml_id, name, slug, description_short, description_full, description,
          composition, nutrition, protein_per_100g, fat_per_100g, carbs_per_100g,
          kcal_per_100g, storage_conditions, brand, price_current, price_old,
          discount_percent, currency, rating_avg, rating_count, unit, weight_value,
          weight_unit, url, images_json, category_json, updated_at, details_synced_at
        )
        VALUES (
          :id, :xml_id, :name, :slug, :description_short, :description_full, :description,
          :composition, :nutrition, :protein_per_100g, :fat_per_100g, :carbs_per_100g,
          :kcal_per_100g, :storage_conditions, :brand, :price_current, :price_old,
          :discount_percent, :currency, :rating_avg, :rating_count, :unit, :weight_value,
          :weight_unit, :url, :images_json, :category_json, :updated_at, :details_synced_at
        )
        ON CONFLICT(id) DO UPDATE SET
          xml_id=excluded.xml_id,
          name=COALESCE(excluded.name, products.name),
          slug=COALESCE(excluded.slug, products.slug),
          description_short=COALESCE(excluded.description_short, products.description_short),
          description_full=COALESCE(excluded.description_full, products.description_full),
          description=COALESCE(excluded.description, products.description),
          composition=COALESCE(excluded.composition, products.composition),
          nutrition=COALESCE(excluded.nutrition, products.nutrition),
          protein_per_100g=COALESCE(excluded.protein_per_100g, products.protein_per_100g),
          fat_per_100g=COALESCE(excluded.fat_per_100g, products.fat_per_100g),
          carbs_per_100g=COALESCE(excluded.carbs_per_100g, products.carbs_per_100g),
          kcal_per_100g=COALESCE(excluded.kcal_per_100g, products.kcal_per_100g),
          storage_conditions=COALESCE(excluded.storage_conditions, products.storage_conditions),
          brand=COALESCE(excluded.brand, products.brand),
          price_current=COALESCE(excluded.price_current, products.price_current),
          price_old=excluded.price_old,
          discount_percent=excluded.discount_percent,
          currency=COALESCE(excluded.currency, products.currency),
          rating_avg=COALESCE(excluded.rating_avg, products.rating_avg),
          rating_count=COALESCE(excluded.rating_count, products.rating_count),
          unit=COALESCE(excluded.unit, products.unit),
          weight_value=COALESCE(excluded.weight_value, products.weight_value),
          weight_unit=COALESCE(excluded.weight_unit, products.weight_unit),
          url=COALESCE(excluded.url, products.url),
          images_json=COALESCE(excluded.images_json, products.images_json),
          category_json=COALESCE(excluded.category_json, products.category_json),
          updated_at=excluded.updated_at,
          details_synced_at=COALESCE(excluded.details_synced_at, products.details_synced_at)
        """,
        item,
    )


def replace_properties(
    conn: sqlite3.Connection,
    product_id: int,
    properties: list[dict[str, Any]],
    synced_at: str,
) -> None:
    if not properties:
        return
    conn.execute("DELETE FROM product_properties WHERE product_id = ?", (product_id,))
    for prop in properties:
        name = normalize_text(prop.get("name"))
        if not name:
            continue
        conn.execute(
            """
            INSERT INTO product_properties (product_id, name, value, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (product_id, name, normalize_text(prop.get("value")), synced_at),
        )


def upsert_discount(
    conn: sqlite3.Connection,
    product_id: int,
    discount_type: str,
    price: dict[str, Any],
    synced_at: str,
) -> None:
    price_current, price_old, discount_percent, _, discount_info = parse_mcp_price(price)
    conn.execute(
        """
        INSERT INTO product_discounts (
          product_id, discount_type, price_current, price_old,
          discount_percent, discount_info, synced_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(product_id, discount_type) DO UPDATE SET
          price_current=excluded.price_current,
          price_old=excluded.price_old,
          discount_percent=excluded.discount_percent,
          discount_info=excluded.discount_info,
          synced_at=excluded.synced_at
        """,
        (
            product_id,
            discount_type,
            price_current,
            price_old,
            discount_percent,
            discount_info,
            synced_at,
        ),
    )


async def call_mcp_json(client: Client, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    last_error = ""
    for attempt in range(1, MCP_RETRIES + 1):
        result = await client.call_tool(tool_name, args)
        raw = result.content[0].text if result.content else ""
        data = json.loads(raw) if raw else {}
        if data.get("ok"):
            return data
        last_error = raw[:500]
        if is_rate_limited(data):
            raise RateLimitedError(last_error)
        if not is_retryable_mcp_error(data):
            break
        await asyncio.sleep(min(30.0, attempt * attempt))
    raise RuntimeError(f"{tool_name} failed: {last_error}")


def is_retryable_mcp_error(data: dict[str, Any]) -> bool:
    error = data.get("error")
    return bool(data.get("retryable")) or (isinstance(error, dict) and bool(error.get("retryable")))


def is_rate_limited(data: dict[str, Any]) -> bool:
    error = data.get("error")
    nested_code = error.get("code") if isinstance(error, dict) else None
    nested_status = error.get("http_status") if isinstance(error, dict) else None
    return (
        data.get("code") == "rate_limited" or nested_code == "rate_limited" or nested_status == 429
    )


async def sync_discount_type(
    client: Client,
    conn: sqlite3.Connection,
    run: SyncRun,
    config: RuntimeConfig,
    discount_type: str,
) -> set[int]:
    ids: set[int] = set()
    page = 1
    while True:
        data = await call_mcp_json(
            client,
            "vkusvill_products_discount",
            {"type": discount_type, "page": page, "sort": "price_asc", "vvonly": 0},
        )
        payload = data.get("data") or {}
        items = payload.get("items") or []
        synced_at = utc_now()
        for raw_item in items:
            item = map_mcp_item(raw_item, run)
            if not item:
                continue
            upsert_mcp_item(conn, item)
            record_price_history(conn, item)
            replace_properties(conn, item["id"], raw_item.get("properties") or [], synced_at)
            upsert_discount(conn, item["id"], discount_type, raw_item.get("price") or {}, synced_at)
            ids.add(item["id"])
        conn.commit()
        meta = payload.get("meta") or {}
        print(
            f"discount:{discount_type} page={page}/{meta.get('pages')} "
            f"items={len(items)} total={len(ids)}"
        )
        if not meta.get("has_more"):
            break
        page += 1
        await async_sleep(config.mcp_sleep, config.jitter)
    prune_stale_discounts(conn, discount_type, ids)
    conn.commit()
    return ids


async def sync_discounts(conn: sqlite3.Connection, run: SyncRun, config: RuntimeConfig) -> set[int]:
    discount_ids: set[int] = set()
    async with Client(MCPConfig(mcpServers={"vkusvill": {"url": config.mcp_url}})) as client:
        for discount_type in ("card", "quantity"):
            discount_ids.update(await sync_discount_type(client, conn, run, config, discount_type))
    conn.execute(
        "UPDATE sync_runs SET discounts_seen = ? WHERE id = ?",
        (len(discount_ids), run.id),
    )
    conn.commit()
    return discount_ids


def prune_stale_discounts(
    conn: sqlite3.Connection, discount_type: str, active_ids: set[int]
) -> None:
    if not active_ids:
        conn.execute("DELETE FROM product_discounts WHERE discount_type = ?", (discount_type,))
        return
    placeholders = ",".join("?" for _ in active_ids)
    conn.execute(
        f"""
        DELETE FROM product_discounts
        WHERE discount_type = ? AND product_id NOT IN ({placeholders})
        """,
        (discount_type, *active_ids),
    )


async def sync_details(
    conn: sqlite3.Connection, run: SyncRun, config: RuntimeConfig, mode: str
) -> None:
    ids = load_detail_ids(conn, run, mode)
    print(f"details mode={mode} ids={len(ids)}")
    ok_count = 0
    error_count = 0
    index = 0
    while index < len(ids):
        product_id = ids[index]
        try:
            await sync_product_details(conn, run, config, product_id)
            ok_count += 1
            index += 1
        except RateLimitedError:
            print(
                f"details rate_limited sleep={config.rate_limit_sleep}s at {index + 1}/{len(ids)}"
            )
            await asyncio.sleep(config.rate_limit_sleep)
            continue
        except Exception as exc:
            error_count += 1
            record_error(conn, run, "details", exc, product_id=product_id)
            conn.commit()
            print(f"details id={product_id} error={exc}")
            index += 1
            if error_count >= config.max_errors:
                print("details max errors reached")
                break
        if index % config.commit_every == 0:
            conn.commit()
            print(f"details {index}/{len(ids)} ok={ok_count} errors={error_count}")
        await async_sleep(config.mcp_sleep, config.jitter)
    conn.commit()
    conn.execute(
        "UPDATE sync_runs SET details_ok = ?, details_errors = ? WHERE id = ?",
        (ok_count, error_count, run.id),
    )
    conn.commit()
    print(f"details done ok={ok_count} errors={error_count}")


def sync_site_details(
    conn: sqlite3.Connection, run: SyncRun, config: RuntimeConfig, mode: str
) -> None:
    ids = load_detail_ids(conn, run, mode)
    print(f"site_details mode={mode} ids={len(ids)}")
    ok_count = 0
    error_count = 0
    for index, product_id in enumerate(ids, start=1):
        try:
            sync_product_site_details(conn, product_id)
            ok_count += 1
        except Exception as exc:
            error_count += 1
            record_error(conn, run, "site_details", exc, product_id=product_id)
            print(f"site_details id={product_id} error={exc}")
            if error_count >= config.max_errors:
                print("site_details max errors reached")
                break
        if index % config.commit_every == 0:
            conn.commit()
            print(f"site_details {index}/{len(ids)} ok={ok_count} errors={error_count}")
        sleep_sync(config.catalog_sleep, config.jitter)
    conn.commit()
    conn.execute(
        "UPDATE sync_runs SET details_ok = ?, details_errors = ? WHERE id = ?",
        (ok_count, error_count, run.id),
    )
    conn.commit()
    print(f"site_details done ok={ok_count} errors={error_count}")


def sync_product_site_details(conn: sqlite3.Connection, product_id: int) -> None:
    row = conn.execute("SELECT url FROM products WHERE id = ?", (product_id,)).fetchone()
    if not row or not row[0]:
        raise RuntimeError("missing product url")
    details = parse_site_details(fetch_html(row[0]))
    update_site_details(conn, product_id, details)


def parse_site_details(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    metrics = extract_site_energy_metrics(soup)
    composition = text_from_selector(soup, "._sostav")
    storage = extract_named_site_desc(soup, "условия хранения")
    description = extract_product_description(soup)
    return {
        "description_full": description,
        "composition": composition,
        "nutrition": nutrition_text_from_metrics(metrics),
        "storage_conditions": storage,
        **metrics,
        "details_synced_at": utc_now(),
        "updated_at": utc_now(),
    }


def extract_site_energy_metrics(soup: BeautifulSoup) -> dict[str, float | None]:
    metrics = empty_nutrition_metrics()
    for item in soup.select(".VV23_DetailProdPageAccordion__EnergyItem"):
        value = to_float(text_from_selector(item, ".VV23_DetailProdPageAccordion__EnergyValue"))
        label = (
            text_from_selector(item, ".VV23_DetailProdPageAccordion__EnergyDesc") or ""
        ).lower()
        if value is None:
            continue
        if "ккал" in label:
            metrics["kcal_per_100g"] = value
        elif "белк" in label:
            metrics["protein_per_100g"] = value
        elif "жир" in label:
            metrics["fat_per_100g"] = value
        elif "углев" in label:
            metrics["carbs_per_100g"] = value
    return metrics


def extract_named_site_desc(soup: BeautifulSoup, title_part: str) -> str | None:
    for item in soup.select(".VV23_DetailProdPageInfoDescItem"):
        title = text_from_selector(item, ".VV23_DetailProdPageInfoDescItem__Title")
        if title_part in (title or "").lower():
            return text_from_selector(item, ".VV23_DetailProdPageInfoDescItem__Desc")
    return None


def extract_product_description(soup: BeautifulSoup) -> str | None:
    for item in soup.select(".VV23_DetailProdPageInfoDescItem__Desc"):
        text = normalize_text(item.get_text(" ", strip=True))
        if text and "белки" not in text.lower() and len(text) > 40:
            return text
    description = soup.select_one('meta[name="description"]')
    return normalize_text(description.get("content") if description else None)


def update_site_details(
    conn: sqlite3.Connection,
    product_id: int,
    details: dict[str, Any],
) -> None:
    conn.execute(
        """
        UPDATE products
        SET description_full = COALESCE(?, description_full),
            description = COALESCE(description, ?),
            composition = COALESCE(?, composition),
            nutrition = COALESCE(?, nutrition),
            storage_conditions = COALESCE(?, storage_conditions),
            protein_per_100g = COALESCE(?, protein_per_100g),
            fat_per_100g = COALESCE(?, fat_per_100g),
            carbs_per_100g = COALESCE(?, carbs_per_100g),
            kcal_per_100g = COALESCE(?, kcal_per_100g),
            details_synced_at = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            details["description_full"],
            details["description_full"],
            details["composition"],
            details["nutrition"],
            details["storage_conditions"],
            details["protein_per_100g"],
            details["fat_per_100g"],
            details["carbs_per_100g"],
            details["kcal_per_100g"],
            details["details_synced_at"],
            details["updated_at"],
            product_id,
        ),
    )


async def sync_product_details(
    conn: sqlite3.Connection,
    run: SyncRun,
    config: RuntimeConfig,
    product_id: int,
) -> None:
    async with Client(MCPConfig(mcpServers={"vkusvill": {"url": config.mcp_url}})) as client:
        data = await call_mcp_json(client, "vkusvill_product_details", {"id": product_id})
    raw_item = data.get("data") or {}
    item = map_mcp_item(raw_item, run, from_details=True)
    if not item:
        raise RuntimeError("empty details item")
    upsert_mcp_item(conn, item)
    record_price_history(conn, item)
    replace_properties(conn, item["id"], raw_item.get("properties") or [], utc_now())


def load_detail_ids(conn: sqlite3.Connection, run: SyncRun, mode: str) -> list[int]:
    if mode == "none":
        return []
    if mode == "seen":
        query = "SELECT id FROM products WHERE last_catalog_run_id = ? ORDER BY id"
        return [int(row[0]) for row in conn.execute(query, (run.id,)).fetchall()]
    if mode == "active":
        return [
            int(row[0])
            for row in conn.execute("SELECT id FROM products WHERE active = 1 ORDER BY id")
        ]
    if mode == "pending":
        rows = conn.execute(
            """
            SELECT id
            FROM products
            WHERE active = 1 AND details_synced_at IS NULL
            ORDER BY id
            """
        ).fetchall()
        return [int(row[0]) for row in rows]
    if mode == "food":
        return load_food_detail_ids(conn, "active = 1")
    if mode == "food_pending":
        return load_food_detail_ids(conn, "active = 1 AND details_synced_at IS NULL")
    if mode == "food_missing":
        return load_food_detail_ids(
            conn,
            "active = 1 AND (nutrition IS NULL OR trim(nutrition) = '')",
        )
    if mode == "food_errors":
        error_ids = load_error_ids(conn)
        return filter_food_ids(conn, error_ids)
    if mode == "missing":
        rows = conn.execute(
            """
            SELECT id
            FROM products
            WHERE active = 1 AND (nutrition IS NULL OR trim(nutrition) = '')
            ORDER BY id
            """
        ).fetchall()
        return [int(row[0]) for row in rows]
    if mode == "errors":
        return load_error_ids(conn)
    raise ValueError(f"Unknown details mode: {mode}")


def load_food_detail_ids(conn: sqlite3.Connection, where_clause: str) -> list[int]:
    rows = conn.execute(
        f"SELECT id, name, category_json FROM products WHERE {where_clause} ORDER BY id"
    ).fetchall()
    return [int(row[0]) for row in rows if is_food_product(row[1], row[2])]


def load_error_ids(conn: sqlite3.Connection) -> list[int]:
    rows = conn.execute(
        """
        SELECT DISTINCT product_id
        FROM sync_errors
        WHERE product_id IS NOT NULL AND stage = 'details'
        ORDER BY product_id
        """
    ).fetchall()
    return [int(row[0]) for row in rows]


def filter_food_ids(conn: sqlite3.Connection, product_ids: list[int]) -> list[int]:
    if not product_ids:
        return []
    result = []
    for product_id in product_ids:
        row = conn.execute(
            "SELECT name, category_json FROM products WHERE id = ?",
            (product_id,),
        ).fetchone()
        if row and is_food_product(row[0], row[1]):
            result.append(product_id)
    return result


def is_food_product(name: str | None, category_json: str | None) -> bool:
    haystack = f"{name or ''} {category_text(category_json)}".lower()
    return any(pattern in haystack for pattern in FOOD_CATEGORY_PATTERNS)


def category_text(category_json: str | None) -> str:
    if not category_json:
        return ""
    try:
        category_data = json.loads(category_json)
    except json.JSONDecodeError:
        return category_json
    values = []
    for item in category_data if isinstance(category_data, list) else []:
        if isinstance(item, dict):
            values.append(str(item.get("name") or ""))
        else:
            values.append(str(item))
    return " ".join(values)


async def async_sleep(base: float, jitter: float) -> None:
    await asyncio.sleep(base + random.uniform(0, jitter))


def summarize(conn: sqlite3.Connection) -> dict[str, Any]:
    queries = {
        "products": "SELECT COUNT(*) FROM products",
        "active": "SELECT COUNT(*) FROM products WHERE active = 1",
        "priced": "SELECT COUNT(*) FROM products WHERE active = 1 AND price_current IS NOT NULL",
        "nutrition": (
            "SELECT COUNT(*) FROM products "
            "WHERE active = 1 AND nutrition IS NOT NULL AND length(trim(nutrition)) > 0"
        ),
        "nutrition_metrics": (
            "SELECT COUNT(*) FROM products "
            "WHERE active = 1 AND protein_per_100g IS NOT NULL AND kcal_per_100g IS NOT NULL"
        ),
        "discounts": "SELECT COUNT(*) FROM product_discounts",
        "errors": "SELECT COUNT(*) FROM sync_errors",
    }
    result = {name: conn.execute(query).fetchone()[0] for name, query in queries.items()}
    result["discount_types"] = conn.execute(
        "SELECT discount_type, COUNT(*) FROM product_discounts GROUP BY discount_type"
    ).fetchall()
    return result


async def run(args: argparse.Namespace) -> None:
    config = RuntimeConfig(
        db_path=Path(args.db),
        mcp_url=args.mcp_url,
        catalog_sleep=args.catalog_sleep,
        mcp_sleep=args.mcp_sleep,
        jitter=args.jitter,
        commit_every=args.commit_every,
        max_errors=args.max_errors,
        max_pages=args.max_pages,
        rate_limit_sleep=args.rate_limit_sleep,
    )
    conn = sqlite3.connect(config.db_path)
    try:
        ensure_schema(conn)
        backfill_nutrition_metrics(conn)
        run_record = start_run(conn)
        print(f"sync_run id={run_record.id} started_at={run_record.started_at}")
        if args.catalog:
            sync_catalog(conn, run_record, config)
        if args.discounts:
            await sync_discounts(conn, run_record, config)
        if args.details != "none":
            if args.details_source == "site":
                sync_site_details(conn, run_record, config, args.details)
            else:
                await sync_details(conn, run_record, config, args.details)
        finish_run(conn, run_record, "done")
        print("summary", json.dumps(summarize(conn), ensure_ascii=False))
    except Exception:
        try:
            finish_run(conn, run_record, "failed")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nightly VkusVill catalog and MCP sync")
    parser.add_argument("--db", required=True)
    parser.add_argument("--mcp-url", default=MCP_URL)
    parser.add_argument("--catalog", action="store_true")
    parser.add_argument("--discounts", action="store_true")
    parser.add_argument("--details-source", choices=["mcp", "site"], default="mcp")
    parser.add_argument(
        "--details",
        choices=[
            "none",
            "seen",
            "active",
            "pending",
            "missing",
            "errors",
            "food",
            "food_pending",
            "food_missing",
            "food_errors",
        ],
        default="none",
    )
    parser.add_argument("--catalog-sleep", type=float, default=0.08)
    parser.add_argument("--mcp-sleep", type=float, default=0.15)
    parser.add_argument("--jitter", type=float, default=0.05)
    parser.add_argument("--commit-every", type=int, default=50)
    parser.add_argument("--max-errors", type=int, default=500)
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--rate-limit-sleep", type=float, default=DEFAULT_RATE_LIMIT_SLEEP_SECONDS)
    return parser.parse_args()


def main() -> None:
    try:
        asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)


if __name__ == "__main__":
    main()
