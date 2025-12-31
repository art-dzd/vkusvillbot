from __future__ import annotations

import json
from typing import Any

from fastmcp import Client
from fastmcp.mcp_config import MCPConfig


class MCPError(RuntimeError):
    pass


class VkusvillMCP:
    def __init__(self, url: str) -> None:
        self.url = url
        self._client: Client | None = None

    async def connect(self) -> None:
        self._client = Client(MCPConfig(mcpServers={"vkusvill": {"url": self.url}}))
        await self._client.__aenter__()

    async def close(self) -> None:
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None

    def _parse_response(self, raw: Any) -> dict[str, Any]:
        if not raw:
            raise MCPError("Пустой ответ MCP")
        data = json.loads(raw)
        if not data.get("ok"):
            raise MCPError(f"Ошибка MCP: {raw}")
        return data

    async def _call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if not self._client:
            raise MCPError("MCP клиент не инициализирован")
        res = await self._client.call_tool(name, args)
        raw = res.content[0].text if res.content else ""
        return self._parse_response(raw)

    async def search(self, query: str, page: int = 1) -> dict[str, Any]:
        return await self._call("vkusvill_products_search", {"q": query, "page": page})

    async def details(self, product_id: int) -> dict[str, Any]:
        return await self._call("vkusvill_product_details", {"id": product_id})

    async def cart(self, products: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._call("vkusvill_cart_link_create", {"products": products})
