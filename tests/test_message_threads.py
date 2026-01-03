import sqlite3

from vkusvillbot.db import Database


def test_history_scoped_by_thread_id(tmp_path) -> None:
    db_path = tmp_path / "test.db"

    # Эмулируем старую схему без thread_id (миграция должна добавить колонку).
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          tg_id INTEGER UNIQUE,
          city TEXT,
          diet_notes TEXT,
          created_at TEXT
        );

        CREATE TABLE messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          created_at TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO users (tg_id, city, diet_notes, created_at) VALUES (?, ?, ?, ?)",
        (1, "Moscow", None, "2025-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    db = Database(str(db_path))
    try:
        cols = {row[1] for row in db.conn.execute("PRAGMA table_info(messages)").fetchall()}
        assert "thread_id" in cols

        user = db.get_or_create_user(1)
        db.save_message(user.id, "user", "hello t1", thread_id=100)
        db.save_message(user.id, "assistant", "ans t1", thread_id=100)
        db.save_message(user.id, "user", "hello t2", thread_id=200)
        db.save_message(user.id, "assistant", "ans t2", thread_id=200)
        db.save_message(user.id, "user", "hello none", thread_id=None)

        hist1 = db.get_recent_messages(user.id, limit=10, thread_id=100)
        assert [m["content"] for m in hist1] == ["hello t1", "ans t1"]

        hist2 = db.get_recent_messages(user.id, limit=10, thread_id=200)
        assert [m["content"] for m in hist2] == ["hello t2", "ans t2"]

        hist_none = db.get_recent_messages(user.id, limit=10, thread_id=None)
        assert [m["content"] for m in hist_none] == ["hello none"]
    finally:
        db.close()

