from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class User:
    id: int
    tg_id: int
    city: str | None
    diet_notes: str | None


class Database:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()

    def close(self) -> None:
        self.conn.close()

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              tg_id INTEGER UNIQUE,
              city TEXT,
              diet_notes TEXT,
              created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS sessions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              last_intent TEXT,
              last_context TEXT,
              updated_at TEXT DEFAULT (datetime('now')),
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()

    def get_or_create_user(self, tg_id: int) -> User:
        row = self.conn.execute(
            "SELECT id, tg_id, city, diet_notes FROM users WHERE tg_id = ?",
            (tg_id,),
        ).fetchone()
        if row:
            return User(*row)
        self.conn.execute(
            "INSERT INTO users (tg_id, city, diet_notes) VALUES (?, ?, ?)",
            (tg_id, "Moscow", None),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id, tg_id, city, diet_notes FROM users WHERE tg_id = ?",
            (tg_id,),
        ).fetchone()
        return User(*row)

    def update_user_city(self, tg_id: int, city: str) -> None:
        self.conn.execute(
            "UPDATE users SET city = ? WHERE tg_id = ?",
            (city, tg_id),
        )
        self.conn.commit()

    def update_user_diet_notes(self, tg_id: int, diet_notes: str) -> None:
        self.conn.execute(
            "UPDATE users SET diet_notes = ? WHERE tg_id = ?",
            (diet_notes, tg_id),
        )
        self.conn.commit()

    def save_session(self, user_id: int, last_intent: str, last_context: dict) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        context_json = json.dumps(last_context, ensure_ascii=False)
        row = self.conn.execute(
            "SELECT id FROM sessions WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row:
            self.conn.execute(
                """
                UPDATE sessions
                SET last_intent = ?, last_context = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (last_intent, context_json, ts, user_id),
            )
        else:
            self.conn.execute(
                """
                INSERT INTO sessions (user_id, last_intent, last_context, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, last_intent, context_json, ts),
            )
        self.conn.commit()
