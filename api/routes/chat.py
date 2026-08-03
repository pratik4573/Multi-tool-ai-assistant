"""POST /chat — text-based chat turn."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.llm.orchestrator import ChatOrchestrator
from src.memory.conversation_store import ConversationStore
from src.models.chat_models import ChatRequest, ChatResponse
from src.utils.exceptions import PratikAIError
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["chat"])

_orchestrator: ChatOrchestrator | None = None
_conversations = ConversationStore()


def get_orchestrator() -> ChatOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ChatOrchestrator()
    return _orchestrator


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id
    if not conversation_id:
        conversation_id = _conversations.create_conversation().id
    elif _conversations.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")

    try:
        assistant_message, tool_calls, retrieved_context = get_orchestrator().handle_turn(
            conversation_id=conversation_id,
            user_text=request.message,
            use_rag=request.use_rag,
            use_memory=request.use_memory,
        )
    except PratikAIError as exc:
        logger.exception("Chat turn failed.")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(
        conversation_id=conversation_id,
        message=assistant_message,
        tool_calls=tool_calls,
        retrieved_context=retrieved_context,
    )
