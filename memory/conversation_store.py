"""
Persistence for chat sessions and messages (chat history + search).

This backs the /history and /search endpoints, and is what gives Pratik AI
multi-turn memory within and across sessions.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from src.database.connection import get_connection
from src.models.chat_models import Conversation, Message
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationStore:
    """CRUD operations for conversations and their messages."""

    def __init__(self) -> None:
        self._conn = get_connection()

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------
    def create_conversation(self, title: str = "New Chat") -> Conversation:
        conv = Conversation(title=title)
        with self._conn:
            self._conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (conv.id, conv.title, conv.created_at, conv.updated_at),
            )
        logger.info("Created conversation %s", conv.id)
        return conv

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        row = self._conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        return Conversation(**dict(row)) if row else None

    def list_conversations(self, limit: int = 50) -> list[Conversation]:
        rows = self._conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [Conversation(**dict(row)) for row in rows]

    def touch_conversation(self, conversation_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (time.time(), conversation_id),
            )

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, time.time(), conversation_id),
            )

    def delete_conversation(self, conversation_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
            )
            self._conn.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
        logger.info("Deleted conversation %s", conversation_id)

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------
    def add_message(self, message: Message) -> Message:
        with self._conn:
            self._conn.execute(
                "INSERT INTO messages "
                "(id, conversation_id, role, content, tool_name, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    message.id,
                    message.conversation_id,
                    message.role,
                    message.content,
                    message.tool_name,
                    json.dumps(message.metadata),
                    message.created_at,
                ),
            )
        self.touch_conversation(message.conversation_id)
        return message

    def get_messages(self, conversation_id: str, limit: int = 200) -> list[Message]:
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? "
            "ORDER BY created_at ASC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def get_recent_turns(self, conversation_id: str, max_turns: int = 12) -> list[Message]:
        """Return the last N messages, oldest-first, for prompt construction."""
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (conversation_id, max_turns),
        ).fetchall()
        messages = [self._row_to_message(row) for row in rows]
        return list(reversed(messages))

    # ------------------------------------------------------------------
    # Search (full text)
    # ------------------------------------------------------------------
    def search_messages(self, query: str, limit: int = 20) -> list[Message]:
        safe_query = query.replace('"', '""')
        rows = self._conn.execute(
            "SELECT m.* FROM messages_fts f "
            "JOIN messages m ON m.rowid = f.rowid "
            f'WHERE messages_fts MATCH \'"{safe_query}"\' '
            "ORDER BY m.created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_message(row) for row in rows]

    @staticmethod
    def _row_to_message(row) -> Message:
        data = dict(row)
        data["metadata"] = json.loads(data.get("metadata") or "{}")
        return Message(**data)
