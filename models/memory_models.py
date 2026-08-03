"""Domain models for long-term memory."""

from __future__ import annotations

import time
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.utils.text_utils import new_id

MemoryKind = Literal["fact", "preference", "goal", "name", "reminder", "project"]


class MemoryItem(BaseModel):
    """A single unit of long-term memory about the user."""

    id: str = Field(default_factory=lambda: new_id("mem_"))
    kind: MemoryKind
    content: str
    source_conversation_id: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    last_accessed_at: Optional[float] = None
    importance: float = 0.5  # 0-1, used to break ties when trimming old memories


class MemoryCreateRequest(BaseModel):
    kind: MemoryKind
    content: str
    importance: float = 0.5


class MemorySearchResult(BaseModel):
    memory: MemoryItem
    score: float
