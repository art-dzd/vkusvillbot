from __future__ import annotations

from typing import Any, Protocol

import httpx


class EmbeddingsError(RuntimeError):
    pass


class EmbeddingsClient(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]:  # noqa: D401
        """Return embeddings for given texts."""


class OpenRouterEmbeddingsClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        referer: str,
        title: str,
        proxy_url: str | None = None,
        endpoint: str = "https://openrouter.ai/api/v1/embeddings",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.referer = referer
        self.title = title
        self.proxy_url = proxy_url
        self.endpoint = endpoint
        self.timeout = timeout

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise EmbeddingsError("OpenRouter API key is empty")
        if not self.model:
            raise EmbeddingsError("OpenRouter embeddings model is not configured")
        if not texts:
            return []

        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-Title"] = _encode_header_value(self.title)

        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }

        async with httpx.AsyncClient(proxy=self.proxy_url, timeout=self.timeout) as client:
            resp = await client.post(self.endpoint, headers=headers, json=payload)

        if resp.status_code >= 400:
            raise EmbeddingsError(f"OpenRouter embeddings error: {resp.status_code} {resp.text}")

        data = resp.json()
        try:
            items = data["data"]
            indexed = sorted(items, key=lambda x: int(x.get("index", 0)))
            return [item["embedding"] for item in indexed]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingsError("Malformed OpenRouter embeddings response") from exc


def _encode_header_value(value: str) -> str | bytes:
    try:
        value.encode("ascii")
        return value
    except UnicodeEncodeError:
        return value.encode("utf-8")

