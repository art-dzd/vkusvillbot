from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UserProfile:
    city: str | None
    diet_notes: str | None
