"""
Uses the LLM itself to extract durable user memories.

Only stores real user facts.
"""

from __future__ import annotations

import json
import re

import config
from src.llm.ollama_client import OllamaClient
from src.memory.memory_store import MemoryStore
from src.utils.logger import get_logger

logger = get_logger(__name__)

_EXTRACTION_PROMPT = """
Extract ONLY durable facts ABOUT THE USER.

Never invent information.

Never store:
- "not provided"
- "unknown"
- "not mentioned"
- "no information"
- "assistant said"
- "personal context"
- "context unavailable"

Return ONLY valid JSON.

[
  {{
    "kind": "fact",
    "content": "..."
  }}
]

User:
{user_text}

Assistant:
{assistant_text}
"""

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class MemoryExtractor:

    INVALID_PATTERNS = (
        "not provided",
        "not mentioned",
        "unknown",
        "no information",
        "personal context",
        "context unavailable",
        "assistant",
        "i don't know",
        "cannot determine",
    )

    # FIX: previously validity was checked with `pattern in lower`, a
    # plain substring test. That matched "assistant" inside words like
    # "Assistants" (e.g. "Voice Assistants"), "assistant professor", or
    # "assistant development" and silently discarded legitimate memories
    # that had nothing to do with a hallucinated "assistant said ..."
    # placeholder. This compiled regex requires the pattern to appear as
    # a whole word/phrase (word boundaries on both sides), so it only
    # catches the actual placeholder phrases, not substrings buried
    # inside unrelated words.
    _INVALID_PATTERN_RE = re.compile(
        r"\b(" + "|".join(re.escape(p) for p in INVALID_PATTERNS) + r")\b",
        re.IGNORECASE,
    )

    def __init__(self, llm_client=None):
        self._llm = llm_client or OllamaClient()
        self._store = MemoryStore()

    def _is_valid(self, text: str) -> bool:
        text = text.strip()

        if len(text) < 4:
            return False

        lower = text.lower()

        # FIX: word-boundary regex instead of naive substring `in` check.
        if self._INVALID_PATTERN_RE.search(lower):
            return False

        return True

    def extract_and_store(
        self,
        user_text: str,
        assistant_text: str,
        conversation_id: str,
    ) -> int:

        if not config.MEMORY_EXTRACTION_ENABLED:
            return 0

        prompt = _EXTRACTION_PROMPT.format(
            user_text=user_text,
            assistant_text=assistant_text,
        )

        try:
            raw = self._llm.simple_completion(
                prompt,
                temperature=0,
            )
        except Exception as exc:
            logger.warning("Memory extraction failed: %s", exc)
            return 0

        items = self._parse_items(raw)

        stored = 0

        for item in items:

            if not isinstance(item, dict):
                continue

            kind = item.get("kind", "").strip()
            content = item.get("content", "").strip()

            if kind not in config.MEMORY_KINDS:
                continue

            if not self._is_valid(content):
                logger.info("Skipping invalid memory: %s", content)
                continue

            self._store.add_memory(
                kind=kind,
                content=content,
                source_conversation_id=conversation_id,
            )

            stored += 1

        return stored

    @staticmethod
    def _parse_items(raw: str):

        raw = raw.strip()

        try:
            data = json.loads(raw)
            return data if isinstance(data, list) else []

        except Exception:
            pass

        match = _JSON_ARRAY_RE.search(raw)

        if not match:
            logger.debug("No JSON array found in LLM output.")
            return []

        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, list) else []

        except Exception as exc:
            logger.debug("Failed to parse extracted JSON: %s", exc)
            return []