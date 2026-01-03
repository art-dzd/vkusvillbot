from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


class TelegramAPIError(RuntimeError):
    pass


class TelegramAPI:
    def __init__(
        self,
        token: str,
        *,
        endpoint: str = "https://api.telegram.org",
        timeout: float = 10.0,
    ) -> None:
        if not token:
            raise TelegramAPIError("Пустой токен Telegram")
        self._base_url = f"{endpoint}/bot{token}"
        self._timeout = timeout

    async def call(self, method: str, payload: dict[str, Any]) -> Any:
        url = f"{self._base_url}/{method}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload)

        if resp.status_code >= 400:
            raise TelegramAPIError(f"Telegram API HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        if not data.get("ok"):
            raise TelegramAPIError(str(data.get("description") or "Telegram API error"))
        return data.get("result")

    async def get_me(self) -> dict[str, Any]:
        result = await self.call("getMe", {})
        if not isinstance(result, dict):
            raise TelegramAPIError("Некорректный ответ getMe")
        return result

    async def send_message_draft(
        self,
        *,
        chat_id: int,
        draft_id: int,
        text: str,
        message_thread_id: int | None = None,
        parse_mode: str | None = None,
        entities: list[dict[str, Any]] | None = None,
    ) -> bool:
        payload: dict[str, Any] = {"chat_id": chat_id, "draft_id": draft_id, "text": text}
        if message_thread_id is not None:
            payload["message_thread_id"] = message_thread_id
        if parse_mode is not None:
            payload["parse_mode"] = parse_mode
        if entities is not None:
            payload["entities"] = entities
        result = await self.call("sendMessageDraft", payload)
        return bool(result)


@dataclass
class DraftProgress:
    api: TelegramAPI
    chat_id: int
    draft_id: int
    message_thread_id: int | None = None
    enabled: bool = True
    min_interval_s: float = 0.9
    max_lines: int = 18
    max_chars: int = 3900

    _lines: list[str] | None = None
    _last_sent_ts: float = 0.0
    _last_text: str = ""

    async def set(self, text: str) -> None:
        if not self.enabled:
            return
        self._lines = [text]
        await self.flush(force=True)

    async def add(self, line: str) -> None:
        if not self.enabled:
            return
        if self._lines is None:
            self._lines = []
        self._lines.append(line)
        if len(self._lines) > self.max_lines:
            self._lines = self._lines[-self.max_lines :]
        await self.flush()

    async def flush(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        if self._lines is None:
            return

        now = time.monotonic()
        if not force and (now - self._last_sent_ts) < self.min_interval_s:
            return

        text = "\n".join(self._lines).strip()
        if not text:
            text = "…"
        if len(text) > self.max_chars:
            text = "…\n" + text[-self.max_chars + 2 :]

        if text == self._last_text and not force:
            return

        try:
            await self.api.send_message_draft(
                chat_id=self.chat_id,
                draft_id=self.draft_id,
                message_thread_id=self.message_thread_id,
                text=text[:4096],
            )
        except TelegramAPIError:
            self.enabled = False
            return
        self._last_text = text
        self._last_sent_ts = now
