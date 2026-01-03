from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from vkusvillbot.db import Database
from vkusvillbot.embeddings_client import EmbeddingsClient
from vkusvillbot.vector_index import FaissVectorIndex


@dataclass(frozen=True)
class SortSpec:
    field: str
    direction: str = "desc"


def _normalize_category_tokens(categories: list[str] | str | None) -> list[str]:
    if not categories:
        return []
    if isinstance(categories, str):
        raw = re.split(r"[;,/|]", categories)
        return [part.strip().lower() for part in raw if part.strip()]
    return [str(token).strip().lower() for token in categories if str(token).strip()]


def _match_categories(category: object | None, tokens: list[str]) -> bool:
    if not tokens:
        return True
    if category is None:
        return False
    categories: list[str] = []
    if isinstance(category, list):
        categories = [str(item).lower() for item in category]
    else:
        categories = [str(category).lower()]
    for token in tokens:
        for cat in categories:
            if token in cat:
                return True
    return False


def _extract_nutrition_metrics(text: str | None) -> dict[str, float | None]:
    if not text:
        return {"protein": None, "fat": None, "carbs": None, "kcal": None}
    lower = str(text).replace("\u00a0", " ").lower()

    def pick(pattern: str) -> float | None:
        match = re.search(pattern, lower)
        if not match:
            return None
        try:
            return float(match.group(1).replace(",", "."))
        except ValueError:
            return None

    return {
        "protein": pick(r"белк\w*[^0-9]*([0-9]+[.,]?[0-9]*)"),
        "fat": pick(r"жир\w*[^0-9]*([0-9]+[.,]?[0-9]*)"),
        "carbs": pick(r"углевод\w*[^0-9]*([0-9]+[.,]?[0-9]*)"),
        "kcal": pick(r"([0-9]+[.,]?[0-9]*)\s*к?кал"),
    }


def _normalize_weight(weight_value: Any, weight_unit: Any) -> tuple[float | None, str | None]:
    if weight_value is None:
        return None, None
    try:
        value = float(weight_value)
    except (TypeError, ValueError):
        return None, None
    unit = str(weight_unit).strip().lower() if weight_unit is not None else None
    if not unit:
        return value, None
    unit = unit.replace(".", "")
    if unit in {"г", "гр", "g"}:
        return value, "g"
    if unit in {"кг", "kg"}:
        return value, "kg"
    if unit in {"мл", "ml"}:
        return value, "ml"
    if unit in {"л", "l"}:
        return value, "l"
    if unit in {"шт", "уп"}:
        return value, unit
    return value, unit


def _price_metrics(
    price: float | None,
    weight_value: Any,
    weight_unit: Any,
) -> dict[str, float | None]:
    if price is None:
        return {"price_per_kg": None, "price_per_l": None, "price_per_100": None}
    value, unit = _normalize_weight(weight_value, weight_unit)
    if value is None or unit is None:
        return {"price_per_kg": None, "price_per_l": None, "price_per_100": None}
    if unit == "g":
        kg = value / 1000.0 if value else None
        price_per_kg = (price / kg) if kg else None
        price_per_100 = (price / value * 100.0) if value else None
        return {
            "price_per_kg": price_per_kg,
            "price_per_l": None,
            "price_per_100": price_per_100,
        }
    if unit == "kg":
        kg = value if value else None
        price_per_kg = (price / kg) if kg else None
        price_per_100 = (price / (kg * 10.0)) if kg else None
        return {
            "price_per_kg": price_per_kg,
            "price_per_l": None,
            "price_per_100": price_per_100,
        }
    if unit == "ml":
        liters = value / 1000.0 if value else None
        price_per_l = (price / liters) if liters else None
        price_per_100 = (price / value * 100.0) if value else None
        return {
            "price_per_kg": None,
            "price_per_l": price_per_l,
            "price_per_100": price_per_100,
        }
    if unit == "l":
        liters = value if value else None
        price_per_l = (price / liters) if liters else None
        price_per_100 = (price / (liters * 10.0)) if liters else None
        return {
            "price_per_kg": None,
            "price_per_l": price_per_l,
            "price_per_100": price_per_100,
        }
    return {"price_per_kg": None, "price_per_l": None, "price_per_100": None}


def _parse_filter_expr(expr: str | None) -> list[tuple[str, str, float]]:
    if not expr:
        return []
    text = expr.lower()
    conditions = re.split(r"\band\b|&&|,", text)
    parsed: list[tuple[str, str, float]] = []
    for cond in conditions:
        cond = cond.strip()
        if not cond:
            continue
        match = re.search(
            r"(protein|fat|carbs|kcal|price|rating|price_per_kg|price_per_l|price_per_100)"
            r"\s*(<=|>=|=|<|>)\s*([0-9]+[.,]?[0-9]*)",
            cond,
        )
        if not match:
            continue
        field, op, value = match.groups()
        try:
            parsed.append((field, op, float(value.replace(",", "."))))
        except ValueError:
            continue
    return parsed


def _compare(value: float, op: str, target: float) -> bool:
    if op == ">":
        return value > target
    if op == ">=":
        return value >= target
    if op == "<":
        return value < target
    if op == "<=":
        return value <= target
    if op == "=":
        return abs(value - target) < 1e-9
    return False


def _sort_items(items: list[dict[str, Any]], sort: list[SortSpec]) -> None:
    if not sort:
        sort = [
            SortSpec(field="lex_match", direction="desc"),
            SortSpec(field="similarity", direction="desc"),
        ]

    def field_value(item: dict[str, Any], field: str) -> float | None:
        value = item.get(field)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            if isinstance(value, bool):
                return float(value)
            return None

    def key(item: dict[str, Any]) -> tuple[tuple[int, float], ...]:
        keys: list[tuple[int, float]] = []
        for spec in sort:
            direction = (spec.direction or "desc").lower()
            v = field_value(item, spec.field)
            missing = 1 if v is None else 0
            if v is None:
                v_adj = 0.0
            else:
                v_adj = v if direction == "asc" else -v
            keys.append((missing, v_adj))
        return tuple(keys)

    items.sort(key=key)


class ProductRetriever:
    def __init__(
        self,
        db: Database,
        embeddings: EmbeddingsClient,
        index: FaissVectorIndex,
        candidate_pool: int = 200,
        fts_boost: bool = True,
    ) -> None:
        self.db = db
        self.embeddings = embeddings
        self.index = index
        self.candidate_pool = max(50, int(candidate_pool))
        self.fts_boost = bool(fts_boost)

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        categories: list[str] | str | None = None,
        filter_expr: str | None = None,
        sort: list[SortSpec] | None = None,
        include_missing: bool = False,
    ) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        requested = max(1, int(limit))
        offset = max(0, int(offset))
        k = max(self.candidate_pool, offset + requested * 5)

        emb = await self.embeddings.embed([query])
        vector = np.asarray(emb[0], dtype="float32")
        hits = self.index.search(vector, k=k)

        scores = {hit.id: hit.score for hit in hits}
        ordered_ids = [hit.id for hit in hits]
        if not ordered_ids:
            return []

        items = self.db.get_products_by_ids(ordered_ids)
        by_id = {int(item["id"]): item for item in items if item.get("id") is not None}
        ordered: list[dict[str, Any]] = []
        for pid in ordered_ids:
            item = by_id.get(pid)
            if not item:
                continue
            item = dict(item)
            item["similarity"] = float(scores.get(pid, 0.0))
            ordered.append(item)

        if self.fts_boost:
            candidate_ids = [
                int(i["id"]) for i in ordered if i.get("id") is not None
            ]
            match_ids = self.db.fts_match_ids(query, candidate_ids)
        else:
            match_ids = set()

        cat_tokens = _normalize_category_tokens(categories)
        filters = _parse_filter_expr(filter_expr)

        filtered: list[dict[str, Any]] = []
        for item in ordered:
            pid = int(item["id"])
            item["lex_match"] = 1.0 if pid in match_ids else 0.0

            if cat_tokens and not _match_categories(item.get("category"), cat_tokens):
                continue

            metrics = _extract_nutrition_metrics(item.get("nutrition"))
            protein = metrics.get("protein")
            fat = metrics.get("fat")
            carbs = metrics.get("carbs")
            kcal = metrics.get("kcal")
            item["protein"] = protein
            item["fat"] = fat
            item["carbs"] = carbs
            item["kcal"] = kcal
            item["protein_per_100g"] = protein
            item["fat_per_100g"] = fat
            item["carbs_per_100g"] = carbs
            item["kcal_per_100g"] = kcal

            price_metrics = _price_metrics(
                item.get("price"),
                item.get("weight_value"),
                item.get("weight_unit"),
            )
            item.update(price_metrics)

            if filters and not _apply_filters(item, filters, include_missing=include_missing):
                continue

            filtered.append(item)

        _sort_items(filtered, sort or [])

        if offset:
            filtered = filtered[offset:]
        return filtered[:requested]


def _apply_filters(
    item: dict[str, Any],
    filters: list[tuple[str, str, float]],
    include_missing: bool,
) -> bool:
    for field, op, target in filters:
        value = item.get(field)
        if value is None:
            if include_missing:
                continue
            return False
        try:
            if not _compare(float(value), op, target):
                return False
        except (TypeError, ValueError):
            return False
    return True
