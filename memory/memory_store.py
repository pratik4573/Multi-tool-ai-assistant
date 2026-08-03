"""
Long-term memory storage (SQLite).

Stores discrete memory items (facts, preferences, goals, names, reminders,
projects) about the user and retrieves the most relevant ones for a given
query using keyword-overlap scoring.
"""

from __future__ import annotations

import re
import time
from collections import Counter

import config
from src.database.connection import get_connection
from src.models.memory_models import (
    MemoryItem,
    MemoryKind,
    MemorySearchResult,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[a-zA-Z0-9']+")


def _tokenize(text: str) -> Counter:
    return Counter(w.lower() for w in _WORD_RE.findall(text))


class MemoryStore:
    """CRUD + relevance search over long-term memory items."""

    INVALID_MEMORY_PATTERNS = (
        "not provided",
        "unknown",
        "not mentioned",
        "personal context",
        "context unavailable",
        "assistant",
        "cannot determine",
        "no information",
    )

    # FIX: previously this used plain substring matching (`x in lower`),
    # which caused false positives like "assistant" matching inside
    # "Voice Assistants", silently discarding valid memories. This
    # compiled regex only matches the pattern as a whole word/phrase,
    # with word boundaries on both ends.
    _INVALID_PATTERN_RE = re.compile(
        r"\b(" + "|".join(re.escape(p) for p in INVALID_MEMORY_PATTERNS) + r")\b",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self._conn = get_connection()

    # ------------------------------------------------------------------
    # Add Memory
    # ------------------------------------------------------------------

    def add_memory(
        self,
        kind: MemoryKind,
        content: str,
        source_conversation_id: str | None = None,
        importance: float = 0.5,
    ) -> MemoryItem:

        item = MemoryItem(
            kind=kind,
            content=content,
            source_conversation_id=source_conversation_id,
            importance=importance,
        )

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO memories
                (
                    id,
                    kind,
                    content,
                    source_conversation_id,
                    created_at,
                    last_accessed_at,
                    importance
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.kind,
                    item.content,
                    item.source_conversation_id,
                    item.created_at,
                    item.last_accessed_at,
                    item.importance,
                ),
            )

        logger.info("Stored memory [%s]: %s", kind, content[:80])

        return item

    # ------------------------------------------------------------------
    # List Memories
    # ------------------------------------------------------------------

    def list_memories(
        self,
        kind: str | None = None,
        limit: int = 200,
    ) -> list[MemoryItem]:

        if kind:

            rows = self._conn.execute(
                """
                SELECT *
                FROM memories
                WHERE kind=?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (kind, limit),
            ).fetchall()

        else:

            rows = self._conn.execute(
                """
                SELECT *
                FROM memories
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [MemoryItem(**dict(row)) for row in rows]

    # ------------------------------------------------------------------
    # Delete Memory
    # ------------------------------------------------------------------

    def delete_memory(self, memory_id: str) -> None:

        with self._conn:
            self._conn.execute(
                "DELETE FROM memories WHERE id=?",
                (memory_id,),
            )

    # ------------------------------------------------------------------
    # Search Memories
    # ------------------------------------------------------------------

    def search_relevant(
        self,
        query: str,
        top_k: int = config.MEMORY_TOP_K,
    ) -> list[MemorySearchResult]:

        query_tokens = _tokenize(query)

        if not query_tokens:
            return []

        all_memories = self.list_memories(limit=2000)

        scored: list[MemorySearchResult] = []

        for mem in all_memories:

            lower = mem.content.lower()

            # FIX: word-boundary match instead of naive substring match.
            # Previously `"assistant" in lower` matched inside words like
            # "Assistants", silently dropping valid memories that just
            # happened to contain that substring.
            if self._INVALID_PATTERN_RE.search(lower):
                continue

            mem_tokens = _tokenize(mem.content)

            overlap = sum((query_tokens & mem_tokens).values())

            if overlap == 0:
                continue

            recency_boost = (
                1.0
                / (
                    1.0
                    + (time.time() - mem.created_at)
                    / (60 * 60 * 24 * 30)
                )
            )

            score = (
                overlap
                + 0.5 * mem.importance
                + 0.25 * recency_boost
            )

            scored.append(
                MemorySearchResult(
                    memory=mem,
                    score=score,
                )
            )

        scored.sort(
            key=lambda r: r.score,
            reverse=True,
        )

        # FIX: for small corpora, don't hard-truncate to top_k when it
        # would drop otherwise-valid memories that simply had thin
        # keyword overlap. If the whole scored set is at or below the
        # configured full-retrieval threshold, return everything instead
        # of an arbitrary slice.
        full_threshold = getattr(config, "MEMORY_FULL_RETRIEVAL_THRESHOLD", 30)

        if len(scored) <= full_threshold:
            results = scored
        else:
            results = scored[:top_k]

        if results:

            now = time.time()

            with self._conn:

                for result in results:

                    self._conn.execute(
                        """
                        UPDATE memories
                        SET last_accessed_at=?
                        WHERE id=?
                        """,
                        (
                            now,
                            result.memory.id,
                        ),
                    )

        return results