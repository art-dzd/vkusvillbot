from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from vkusvillbot.db import Database
from vkusvillbot.mcp_client import MCPError, VkusvillMCP
from vkusvillbot.models import UserProfile
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
    max_steps: int = 8
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
        config: SgrConfig,
        profile: UserProfile,
    ) -> None:
        self.mcp = mcp
        self.llm = llm
        self.db = db
        self.config = config
        self.profile = profile
        self._mcp_search_cache: set[str] = set()

    async def run(
        self,
        user_text: str,
        history: list[dict[str, str]] | None = None,
        user_id: int | None = None,
    ) -> str:
        system_prompt = build_system_prompt(self.profile)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        for step in range(self.config.max_steps):
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
                continue

            messages.append({"role": "assistant", "content": llm_text})

            if isinstance(parsed, FinalAnswer):
                answer = parsed.answer
                if parsed.cart_items and not parsed.cart_link:
                    try:
                        cart_items = [item.model_dump() for item in parsed.cart_items]
                        cart_data = await self.mcp.cart(cart_items)
                        cart_link = cart_data.get("data", {}).get("link")
                        if cart_link:
                            answer = (
                                f"{answer}\n**Корзина:** [Открыть корзину]({cart_link})"
                            )
                    except MCPError as exc:
                        answer = f"{answer}\nНе удалось создать корзину: {exc}"
                if parsed.follow_up:
                    answer = f"{answer}\n{parsed.follow_up}"
                return answer

            tool = parsed.tool
            args = parsed.args
            try:
                if tool == "vkusvill_products_search":
                    query = str(args.get("q", "") or "")
                    page = int(args.get("page", 1))
                    limit = self.config.max_items_per_search
                    offset = max(0, (page - 1) * limit)

                    local_items = self.db.search_products(query, limit=limit, offset=offset)
                    needs_mcp = self.config.use_mcp_refresh and query not in self._mcp_search_cache
                    mcp_items: list[dict[str, Any]] = []
                    mcp_meta: dict[str, Any] | None = None
                    if needs_mcp:
                        data = await self.mcp.search(query, page)
                        mcp_items = data.get("data", {}).get("items", []) if data else []
                        mcp_meta = data.get("data", {}).get("meta")
                        self.db.upsert_products_from_mcp(mcp_items)
                        self._mcp_search_cache.add(query)

                    merged_items = _merge_items(local_items, mcp_items, limit)
                    compact = {
                        "ok": True,
                        "data": {
                            "items": merged_items,
                            "meta": mcp_meta,
                            "source": "hybrid" if mcp_items else "local",
                        },
                    }
                elif tool == "vkusvill_product_details":
                    product_id = int(args.get("id"))
                    local_details = self.db.get_product_details(product_id)
                    needs_details = True
                    if local_details:
                        updated_at = (
                            local_details.get("updated_at")
                            if isinstance(local_details, dict)
                            else None
                        )
                        needs_details = self.db.is_stale(
                            str(updated_at) if updated_at else None,
                            self.config.local_fresh_hours,
                        )
                        if not local_details.get("properties"):
                            needs_details = True
                    if self.config.use_mcp_refresh and needs_details:
                        data = await self.mcp.details(product_id)
                        details = data.get("data", {}) if data else {}
                        self.db.update_product_details_from_mcp(details)
                        local_details = self.db.get_product_details(product_id)
                    if local_details:
                        compact = {
                            "ok": True,
                            "data": _strip_internal_fields(local_details),
                        }
                    else:
                        data = await self.mcp.details(product_id)
                        compact = compact_details(data)
                elif tool == "vkusvill_cart_link_create":
                    data = await self.mcp.cart(args.get("products", []))
                    compact = {"ok": data.get("ok"), "data": data.get("data")}
                elif tool == "local_products_search":
                    query = str(args.get("q", "") or "")
                    page = int(args.get("page", 1))
                    limit = self.config.max_items_per_search
                    offset = max(0, (page - 1) * limit)
                    items = self.db.search_products(query, limit=limit, offset=offset)
                    compact = {
                        "ok": True,
                        "data": {"items": items, "source": "local"},
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
                elif tool == "local_top_protein":
                    limit = int(args.get("limit", 5))
                    items = self.db.get_top_protein(limit=limit)
                    compact = {"ok": True, "data": {"items": items, "source": "local"}}
                else:
                    compact = {"ok": False, "error": "unknown_tool"}

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

        return "Не успел завершить запрос. Попробуйте уточнить или повторить."


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
