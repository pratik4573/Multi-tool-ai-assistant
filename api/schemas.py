"""Pydantic request/response schemas for the FastAPI layer.

Domain models in src/models already cover most shapes; this module adds
API-specific request payloads that don't belong in the domain layer.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from src.models.memory_models import MemoryCreateRequest  # re-exported for routes


class NewConversationRequest(BaseModel):
    title: str = "New Chat"


class RenameConversationRequest(BaseModel):
    title: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 20


class ToolExecuteRequest(BaseModel):
    name: str
    arguments: dict = {}


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


class ErrorResponse(BaseModel):
    detail: str
