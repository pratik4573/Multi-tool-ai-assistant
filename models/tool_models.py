"""Domain models for the tool-calling subsystem."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ToolSpec(BaseModel):
    """JSON-schema style description of a tool, sent to the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    name: str
    output: str
    success: bool = True
