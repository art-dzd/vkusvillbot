from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from vkusvillbot.mcp_client import MCPError, VkusvillMCP
from vkusvillbot.models import UserProfile
from vkusvillbot.prompts import build_system_prompt

logger = logging.getLogger(__name__)


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


class SgrAgent:
    def __init__(
        self,
        mcp: VkusvillMCP,
        llm: LLMClient,
        config: SgrConfig,
        profile: UserProfile,
    ) -> None:
        self.mcp = mcp
        self.llm = llm
        self.config = config
        self.profile = profile

    async def run(self, user_text: str) -> str:
        system_prompt = build_system_prompt(self.profile)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        for step in range(self.config.max_steps):
            llm_text = await self.llm.chat(messages, temperature=self.config.temperature)
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
                    data = await self.mcp.search(args.get("q", ""), int(args.get("page", 1)))
                    compact = compact_search(data, self.config.max_items_per_search)
                elif tool == "vkusvill_product_details":
                    data = await self.mcp.details(int(args.get("id")))
                    compact = compact_details(data)
                elif tool == "vkusvill_cart_link_create":
                    data = await self.mcp.cart(args.get("products", []))
                    compact = {"ok": data.get("ok"), "data": data.get("data")}
                else:
                    compact = {"ok": False, "error": "unknown_tool"}

                messages.append(
                    {
                        "role": "user",
                        "content": f"TOOL_RESULT {tool}: {json.dumps(compact, ensure_ascii=False)}",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                messages.append(
                    {
                        "role": "user",
                        "content": f"TOOL_ERROR {tool}: {exc}",
                    }
                )
                continue

        return "Не успел завершить запрос. Попробуйте уточнить или повторить."
