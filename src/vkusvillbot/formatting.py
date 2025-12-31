from __future__ import annotations

import re
from typing import Final

try:
    from telegramify_markdown import markdownify  # type: ignore
except Exception:  # pragma: no cover - fallback when dependency is missing
    markdownify = None

_SPECIAL_CHARS: Final = r"\\_\*\[\]\(\)~`>#+\-=|{}.!"
_SPECIAL_RE: Final = re.compile(f"([{_SPECIAL_CHARS}])")


def escape_markdown_v2(text: str) -> str:
    if not text:
        return ""
    # Escape backslash first to avoid double escaping.
    text = text.replace("\\", "\\\\")
    return _SPECIAL_RE.sub(r"\\\1", text)


def to_telegram_markdown(text: str) -> str:
    if not text:
        return ""
    if markdownify is None:
        return escape_markdown_v2(text)
    try:
        return markdownify(text)
    except Exception:
        return escape_markdown_v2(text)
