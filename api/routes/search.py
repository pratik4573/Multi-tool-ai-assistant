"""GET /search — search personal knowledge (RAG) and/or chat history."""

from __future__ import annotations

from fastapi import APIRouter, Query

from src.memory.conversation_store import ConversationStore
from src.rag.retriever import Retriever

router = APIRouter(tags=["search"])

_retriever: Retriever | None = None
_conversations = ConversationStore()


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


@router.get("/search")
def search(
    query: str = Query(..., min_length=1),
    scope: str = Query("all", pattern="^(all|documents|history)$"),
    limit: int = Query(10, ge=1, le=50),
) -> dict:
    result: dict = {"query": query, "scope": scope}

    if scope in ("all", "documents"):
        result["document_matches"] = get_retriever().retrieve(query, top_k=limit)

    if scope in ("all", "history"):
        messages = _conversations.search_messages(query, limit=limit)
        result["message_matches"] = [m.model_dump() for m in messages]

    return result
