"""
SQLite schema definitions and migration entrypoint for Pratik AI.

`init_db()` is idempotent — safe to call on every application startup.
"""

from __future__ import annotations

from src.database.connection import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT 'New Chat',
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        tool_name TEXT,
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at REAL NOT NULL,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_conversation
        ON messages(conversation_id, created_at);
    """,
    """
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        content TEXT NOT NULL,
        source_conversation_id TEXT,
        created_at REAL NOT NULL,
        last_accessed_at REAL,
        importance REAL NOT NULL DEFAULT 0.5
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        content, conversation_id UNINDEXED, message_id UNINDEXED,
        content='messages', content_rowid='rowid'
    );
    """,
]

_FTS_TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, content, conversation_id, message_id)
        VALUES (new.rowid, new.content, new.conversation_id, new.id);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, content, conversation_id, message_id)
        VALUES ('delete', old.rowid, old.content, old.conversation_id, old.id);
    END;
    """,
]


def init_db() -> None:
    """Create all tables/indexes/triggers if they do not already exist."""
    conn = get_connection()
    with conn:
        for statement in _SCHEMA_STATEMENTS:
            conn.execute(statement)
        for trigger in _FTS_TRIGGERS:
            conn.execute(trigger)
    logger.info("SQLite schema initialized/verified.")
