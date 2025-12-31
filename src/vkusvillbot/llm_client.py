from __future__ import annotations

from typing import Any

import httpx


class LLMError(RuntimeError):
    pass


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        referer: str,
        title: str,
        provider_order: str,
        proxy_url: str | None = None,
        endpoint: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.referer = referer
        self.title = title
        self.provider_order = provider_order
        self.proxy_url = proxy_url
        self.endpoint = endpoint
        self.timeout = timeout

    async def chat(self, messages: list[dict[str, str]], temperature: float = 0.4) -> str:
        if not self.api_key:
            raise LLMError("OpenRouter API key is empty")
        if not self.model:
            raise LLMError("OpenRouter model is not configured")

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
            "messages": messages,
            "temperature": temperature,
        }
        if self.provider_order:
            order = [p.strip() for p in self.provider_order.split(",") if p.strip()]
            payload["provider"] = {"order": order}

        async with httpx.AsyncClient(proxy=self.proxy_url, timeout=self.timeout) as client:
            resp = await client.post(self.endpoint, headers=headers, json=payload)

        if resp.status_code >= 400:
            raise LLMError(f"OpenRouter error: {resp.status_code} {resp.text}")

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError("Malformed OpenRouter response") from exc


def _encode_header_value(value: str) -> str | bytes:
    try:
        value.encode("ascii")
        return value
    except UnicodeEncodeError:
        return value.encode("utf-8")
