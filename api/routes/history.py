"""GET/DELETE /history — conversation and chat-history management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import NewConversationRequest, RenameConversationRequest
from src.memory.conversation_store import ConversationStore
from src.models.chat_models import Conversation, Message

router = APIRouter(tags=["history"])

_conversations = ConversationStore()


@router.get("/history/conversations", response_model=list[Conversation])
def list_conversations(limit: int = 50) -> list[Conversation]:
    return _conversations.list_conversations(limit=limit)


@router.post("/history/conversations", response_model=Conversation)
def new_conversation(request: NewConversationRequest) -> Conversation:
    return _conversations.create_conversation(title=request.title)


@router.get("/history/conversations/{conversation_id}", response_model=list[Message])
def get_conversation_messages(conversation_id: str) -> list[Message]:
    if _conversations.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return _conversations.get_messages(conversation_id)


@router.patch("/history/conversations/{conversation_id}", response_model=Conversation)
def rename_conversation(conversation_id: str, request: RenameConversationRequest) -> Conversation:
    if _conversations.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    _conversations.rename_conversation(conversation_id, request.title)
    return _conversations.get_conversation(conversation_id)


@router.delete("/history/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict:
    if _conversations.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    _conversations.delete_conversation(conversation_id)
    return {"deleted": conversation_id}
