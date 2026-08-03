"""Domain models for chat: messages, conversations, and API payloads."""

from __future__ import annotations

import time
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from src.utils.text_utils import new_id

Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """A single message inside a conversation."""

    id: str = Field(default_factory=lambda: new_id("msg_"))
    conversation_id: str
    role: Role
    content: str
    created_at: float = Field(default_factory=time.time)
    tool_name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    """A chat session — a container of ordered messages."""

    id: str = Field(default_factory=lambda: new_id("conv_"))
    title: str = "New Chat"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    use_rag: bool = True
    use_memory: bool = True


class ChatResponse(BaseModel):
    conversation_id: str
    message: Message
    tool_calls: list[str] = Field(default_factory=list)
    retrieved_context: list[str] = Field(default_factory=list)


class VoiceChatResponse(BaseModel):
    conversation_id: str
    transcript: str
    reply_text: str
    reply_audio_path: str
