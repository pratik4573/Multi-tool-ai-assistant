"""GET/POST /memory — long-term memory inspection and management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.memory.memory_store import MemoryStore
from src.models.memory_models import MemoryCreateRequest, MemoryItem, MemorySearchResult

router = APIRouter(tags=["memory"])

_memory_store = MemoryStore()


@router.get("/memory", response_model=list[MemoryItem])
def list_memories(kind: str | None = None, limit: int = 200) -> list[MemoryItem]:
    return _memory_store.list_memories(kind=kind, limit=limit)


@router.post("/memory", response_model=MemoryItem)
def create_memory(request: MemoryCreateRequest) -> MemoryItem:
    return _memory_store.add_memory(
        kind=request.kind, content=request.content, importance=request.importance
    )


@router.get("/memory/search", response_model=list[MemorySearchResult])
def search_memories(query: str, top_k: int = 5) -> list[MemorySearchResult]:
    return _memory_store.search_relevant(query, top_k=top_k)


@router.delete("/memory/{memory_id}")
def delete_memory(memory_id: str) -> dict:
    _memory_store.delete_memory(memory_id)
    return {"deleted": memory_id}
