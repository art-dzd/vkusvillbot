from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from pydantic import BaseModel, ValidationError

from vkusvillbot.db import Database
from vkusvillbot.mcp_client import MCPError, VkusvillMCP
from vkusvillbot.models import UserProfile
from vkusvillbot.product_retriever import ProductRetriever, SortSpec
from vkusvillbot.prompts import build_system_prompt

logger = logging.getLogger(__name__)
dialog_logger = logging.getLogger("dialog")


class LLMClient(Protocol):
    async def chat(self, messages: list[dict[str, str]], temperature: float = 0.4) -> str:  # noqa: D401
        """Return model response as text."""


class CartItem(BaseModel):
    xml_id: int
    q: float


class ToolCall(BaseModel):
    action: str
    tool: str
    args: dict[str, Any]
    reason: str | None = None


class FinalAnswer(BaseModel):
    action: str
    answer: str
    cart_items: list[CartItem] | None = None
    cart_link: str | None = None
    follow_up: str | None = None


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("JSON not found in LLM output")
    return json.loads(match.group(0))


def parse_llm_output(text: str) -> ToolCall | FinalAnswer:
    payload = _extract_json(text)
    action = payload.get("action")
    if action == "tool_call":
        return ToolCall.model_validate(payload)
    if action == "final":
        return FinalAnswer.model_validate(payload)
    raise ValueError(f"Unknown action: {action}")


def compact_search(data: dict[str, Any], limit: int) -> dict[str, Any]:
    items = data.get("data", {}).get("items", [])[:limit]
    trimmed: list[dict[str, Any]] = []
    for item in items:
        trimmed.append(
            {
                "id": item.get("id"),
                "xml_id": item.get("xml_id"),
                "name": item.get("name"),
                "price": item.get("price"),
                "rating": item.get("rating"),
                "unit": item.get("unit"),
                "weight": item.get("weight"),
                "url": item.get("url"),
                "category": item.get("category"),
            }
        )
    return {
        "ok": data.get("ok"),
        "data": {"items": trimmed, "meta": data.get("data", {}).get("meta")},
    }


def compact_details(data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("data", {})
    return {
        "ok": data.get("ok"),
        "data": {
            "id": payload.get("id"),
            "xml_id": payload.get("xml_id"),
            "name": payload.get("name"),
            "description": payload.get("description"),
            "price": payload.get("price"),
            "rating": payload.get("rating"),
            "unit": payload.get("unit"),
            "weight": payload.get("weight"),
            "url": payload.get("url"),
            "properties": payload.get("properties"),
        },
    }


@dataclass
class SgrConfig:
    max_steps: int = 12
    max_items_per_search: int = 10
    temperature: float = 0.4
    history_messages: int = 8
    local_fresh_hours: int = 24
    use_mcp_refresh: bool = True


class SgrAgent:
    def __init__(
        self,
        mcp: VkusvillMCP,
        llm: LLMClient,
        db: Database,
        retriever: ProductRetriever | None,
        config: SgrConfig,
        profile: UserProfile,
    ) -> None:
        self.mcp = mcp
        self.llm = llm
        self.db = db
        self.retriever = retriever
        self.config = config
        self.profile = profile

    async def run(
        self,
        user_text: str,
        history: list[dict[str, str]] | None = None,
        user_id: int | None = None,
        progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        async def safe_progress(line: str) -> None:
            if not progress:
                return
            try:
                await progress(line)
            except Exception:  # noqa: BLE001
                return

        async def render_final(final: FinalAnswer) -> str:
            answer = final.answer
            if final.cart_items and not final.cart_link:
                try:
                    cart_items = [item.model_dump() for item in final.cart_items]
                    cart_data = await self.mcp.cart(cart_items)
                    cart_link = cart_data.get("data", {}).get("link")
                    if cart_link:
                        answer = f"{answer}\n**Корзина:** [Открыть корзину]({cart_link})"
                except MCPError as exc:
                    answer = f"{answer}\nНе удалось создать корзину: {exc}"
            if final.follow_up:
                answer = f"{answer}\n{final.follow_up}"
            return answer

        system_prompt = build_system_prompt(self.profile)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        await safe_progress("🧠 Думаю…")

        for step in range(self.config.max_steps):
            await safe_progress(f"Шаг {step + 1}/{self.config.max_steps}: запрос к LLM")
            llm_text = await self.llm.chat(messages, temperature=self.config.temperature)
            if user_id is not None:
                dialog_logger.info("LLM_RAW user_id=%s step=%s: %s", user_id, step, llm_text)
            try:
                parsed = parse_llm_output(llm_text)
            except (ValueError, ValidationError) as exc:
                logger.warning("LLM output parse error: %s", exc)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Ответ должен быть валидным JSON по схеме. "
                            "Повтори ответ строго в JSON без текста вокруг."
                        ),
                    }
                )
                await safe_progress(f"Шаг {step + 1}: ошибка JSON, прошу повторить")
                continue

            messages.append({"role": "assistant", "content": llm_text})

            if isinstance(parsed, FinalAnswer):
                await safe_progress("✅ Финализирую ответ")
                return await render_final(parsed)

            tool = parsed.tool
            args = parsed.args
            await safe_progress(
                (
                    f"Шаг {step + 1}/{self.config.max_steps}: инструмент {tool} "
                    f"{_format_tool_args(args)}"
                )
            )
            try:
                if tool == "vkusvill_products_search":
                    query = str(args.get("q", "") or "")
                    page = int(args.get("page", 1))
                    limit = self.config.max_items_per_search
                    data = await self.mcp.search(query, page)
                    mcp_items = data.get("data", {}).get("items", []) if data else []
                    if mcp_items:
                        self.db.upsert_products_from_mcp(mcp_items)
                    compact = compact_search(data, limit)
                elif tool == "vkusvill_product_details":
                    product_id = int(args.get("id"))
                    data = await self.mcp.details(product_id)
                    details = data.get("data", {}) if data else {}
                    if details:
                        self.db.update_product_details_from_mcp(details)
                    compact = compact_details(data)
                elif tool == "vkusvill_cart_link_create":
                    data = await self.mcp.cart(args.get("products", []))
                    compact = {"ok": data.get("ok"), "data": data.get("data")}
                elif tool == "local_products_search":
                    query = str(args.get("q", "") or "")
                    page = int(args.get("page", 1))
                    limit = self.config.max_items_per_search
                    offset = max(0, (page - 1) * limit)
                    items = self.db.search_products(
                        query,
                        limit=limit,
                        offset=offset,
                        categories=args.get("categories"),
                    )
                    compact = {
                        "ok": True,
                        "data": {"items": items, "source": "local"},
                    }
                elif tool == "local_semantic_search":
                    if not self.retriever:
                        compact = {"ok": False, "error": "semantic_search_not_configured"}
                    else:
                        query = str(args.get("q", "") or "")
                        page = int(args.get("page", 1))
                        limit = int(args.get("limit", self.config.max_items_per_search))
                        offset = max(0, (page - 1) * limit)
                        sort = _parse_sort(args.get("sort"))
                        items = await self.retriever.semantic_search(
                            query,
                            limit=limit,
                            offset=offset,
                            categories=args.get("categories"),
                            filter_expr=args.get("filter_expr"),
                            sort=sort,
                            include_missing=bool(args.get("include_missing", False)),
                        )
                        compact = {
                            "ok": True,
                            "data": {"items": items, "source": "local_faiss"},
                        }
                elif tool == "local_product_details":
                    product_id = int(args.get("id"))
                    local_details = self.db.get_product_details(product_id)
                    if local_details:
                        compact = {
                            "ok": True,
                            "data": _strip_internal_fields(local_details),
                        }
                    else:
                        compact = {"ok": False, "error": "not_found"}
                elif tool == "local_nutrition_query":
                    compact = {
                        "ok": True,
                        "data": {
                            "items": self.db.nutrition_query(
                                query=args.get("q"),
                                page=int(args.get("page", 1)),
                                limit=int(args.get("limit", self.config.max_items_per_search)),
                                categories=args.get("categories"),
                                filter_expr=args.get("filter_expr"),
                                min_protein=_to_float(args.get("min_protein")),
                                max_protein=_to_float(args.get("max_protein")),
                                min_fat=_to_float(args.get("min_fat")),
                                max_fat=_to_float(args.get("max_fat")),
                                min_carbs=_to_float(args.get("min_carbs")),
                                max_carbs=_to_float(args.get("max_carbs")),
                                min_kcal=_to_float(args.get("min_kcal")),
                                max_kcal=_to_float(args.get("max_kcal")),
                                sort_by=str(args.get("sort_by") or "protein"),
                                order=str(args.get("order") or "desc"),
                                sort=args.get("sort"),
                                include_missing=bool(args.get("include_missing", False)),
                            ),
                            "source": "local",
                        },
                    }
                else:
                    compact = {"ok": False, "error": "unknown_tool"}

                await safe_progress(_summarize_tool_result(tool, compact))
                if user_id is not None:
                    dialog_logger.info(
                        "TOOL_RESULT user_id=%s tool=%s: %s",
                        user_id,
                        tool,
                        json.dumps(compact, ensure_ascii=False),
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": f"TOOL_RESULT {tool}: {json.dumps(compact, ensure_ascii=False)}",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                await safe_progress(f"⚠️ Ошибка инструмента {tool}: {exc}")
                if user_id is not None:
                    dialog_logger.info(
                        "TOOL_ERROR user_id=%s tool=%s: %s",
                        user_id,
                        tool,
                        exc,
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": f"TOOL_ERROR {tool}: {exc}",
                    }
                )
                continue

        # Фоллбек: последняя попытка сформировать final (без дополнительных tool_call).
        try:
            await safe_progress("⚠️ Лимит шагов исчерпан, последняя попытка финализации")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Лимит шагов исчерпан. Сформируй ОКОНЧАТЕЛЬНЫЙ ответ действием final. "
                        "Запрещено вызывать инструменты. Верни строго JSON по схеме final."
                    ),
                }
            )
            llm_text = await self.llm.chat(messages, temperature=self.config.temperature)
            if user_id is not None:
                dialog_logger.info("LLM_RAW user_id=%s step=%s: %s", user_id, "grace", llm_text)
            try:
                parsed = parse_llm_output(llm_text)
            except (ValueError, ValidationError) as exc:
                logger.warning("Grace LLM output parse error: %s", exc)
            else:
                messages.append({"role": "assistant", "content": llm_text})
                if isinstance(parsed, FinalAnswer):
                    await safe_progress("✅ Финализирую ответ")
                    return await render_final(parsed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Grace final attempt failed: %s", exc)

        return "Не успел завершить запрос. Попробуйте уточнить или повторить."


def _format_tool_args(args: dict[str, Any]) -> str:
    if not args:
        return ""
    if "q" in args and args.get("q"):
        q = str(args.get("q"))[:80]
        return f"(q={q!r})"
    if "id" in args and args.get("id") is not None:
        return f"(id={args.get('id')})"
    if "products" in args and isinstance(args.get("products"), list):
        return f"(products={len(args.get('products'))})"
    return ""


def _summarize_tool_result(tool: str, compact: dict[str, Any]) -> str:
    if not compact.get("ok"):
        return f"↳ Результат {tool}: ошибка ({compact.get('error')})"

    data = compact.get("data") or {}
    if tool in {"vkusvill_products_search", "local_products_search", "local_semantic_search"}:
        items = data.get("items") or []
        source = data.get("source") or "mcp"
        return f"↳ Найдено: {len(items)} (source={source})"

    if tool in {"vkusvill_product_details", "local_product_details"}:
        name = (data.get("name") or "").strip()
        return f"↳ Детали: {name[:80] or 'ok'}"

    if tool == "vkusvill_cart_link_create":
        link = (data.get("link") or "").strip()
        return f"↳ Корзина: {'ссылка готова' if link else 'ok'}"

    if tool == "local_nutrition_query":
        items = data.get("items") or []
        return f"↳ Подборка: {len(items)}"

    return "↳ OK"


def _normalize_weight(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        weight_val = value.get("value") or value.get("weight")
        unit = value.get("unit")
        if weight_val is None:
            return str(unit) if unit else None
        return f"{weight_val} {unit}".strip() if unit else str(weight_val)
    return str(value)


def _mcp_item_to_compact(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "xml_id": item.get("xml_id"),
        "name": item.get("name"),
        "price": item.get("price") or item.get("price_current"),
        "rating": item.get("rating") or item.get("rating_avg"),
        "unit": item.get("unit"),
        "weight": _normalize_weight(item.get("weight")),
        "url": item.get("url"),
        "category": item.get("category"),
    }


def _strip_internal_fields(item: dict[str, Any]) -> dict[str, Any]:
    clean = dict(item)
    clean.pop("updated_at", None)
    return clean


def _merge_items(
    local_items: list[dict[str, Any]],
    mcp_items: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for item in mcp_items:
        compact = _mcp_item_to_compact(item)
        if compact.get("id") is None:
            continue
        merged[int(compact["id"])] = compact
    for item in local_items:
        if item.get("id") is None:
            continue
        pid = int(item["id"])
        if pid not in merged:
            merged[pid] = _strip_internal_fields(item)
    result = list(merged.values())
    return result[:limit]


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_sort(raw: Any) -> list[SortSpec] | None:
    if not raw:
        return None
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        spec: list[SortSpec] = []
        for part in parts:
            tokens = part.split()
            field = tokens[0]
            direction = tokens[1] if len(tokens) > 1 else "desc"
            spec.append(SortSpec(field=field, direction=direction))
        return spec or None
    if isinstance(raw, list):
        spec = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            field = str(item.get("field") or "").strip()
            if not field:
                continue
            direction = str(item.get("dir") or item.get("direction") or "desc")
            spec.append(SortSpec(field=field, direction=direction))
        return spec or None
    return None
