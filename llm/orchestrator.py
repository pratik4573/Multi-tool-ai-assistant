"""
The orchestrator ties everything together for a single chat turn.
"""

from __future__ import annotations

import json

import config
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import build_system_prompt
from src.memory.conversation_store import ConversationStore
from src.memory.memory_extractor import MemoryExtractor
from src.memory.memory_store import MemoryStore
from src.models.chat_models import Message
from src.rag.retriever import Retriever
from src.tools.registry import TOOL_SPECS, run_tool
from src.utils.exceptions import (
    LLMError,
    ToolExecutionError,
    ToolNotFoundError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChatOrchestrator:
    def __init__(
        self,
        llm_client: OllamaClient | None = None,
        retriever: Retriever | None = None,
        memory_store: MemoryStore | None = None,
        conversation_store: ConversationStore | None = None,
        memory_extractor: MemoryExtractor | None = None,
    ) -> None:
        self._llm = llm_client or OllamaClient()
        self._retriever = retriever
        self._memory_store = memory_store or MemoryStore()
        self._conversations = conversation_store or ConversationStore()
        self._extractor = memory_extractor or MemoryExtractor(self._llm)

    def _get_retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever()
        return self._retriever

    def handle_turn(
        self,
        conversation_id: str,
        user_text: str,
        use_rag: bool = True,
        use_memory: bool = True,
    ) -> tuple[Message, list[str], list[str]]:

        # --------------------------
        # Retrieve RAG context
        # --------------------------

        retrieved_context: list[str] = []

        if use_rag:
            try:
                retrieved_context = self._get_retriever().retrieve(user_text)
            except Exception:
                logger.exception(
                    "RAG retrieval failed; continuing without RAG context."
                )
                retrieved_context = []

        # --------------------------
        # Retrieve memories
        # --------------------------

        relevant_memories: list[str] = []

        if use_memory:

            results = self._memory_store.search_relevant(user_text)

            for r in results:

                text = r.memory.content.strip()

                # Ignore placeholder memories.
                if "not provided" in text.lower():
                    continue

                # Ignore memories already contained in retrieved docs.
                duplicate = False

                for chunk in retrieved_context:
                    if text.lower() in chunk.lower():
                        duplicate = True
                        break

                if not duplicate:
                    relevant_memories.append(text)

        system_prompt = build_system_prompt(
            retrieved_context,
            relevant_memories,
        )

        logger.info("=" * 80)
        logger.info("Retrieved Context = %s", retrieved_context)
        logger.info("Retrieved Memories = %s", relevant_memories)
        logger.info("System Prompt:\n%s", system_prompt)
        logger.info("=" * 80)

        history = self._conversations.get_recent_turns(
            conversation_id,
            max_turns=12,
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        for msg in history:
            if msg.role in ("user", "assistant"):
                messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        user_message = Message(
            conversation_id=conversation_id,
            role="user",
            content=user_text,
        )

        self._conversations.add_message(user_message)

        tool_calls_made: list[str] = []

        final_text = self._run_tool_loop(
            messages,
            tool_calls_made,
        )

        assistant_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=final_text,
        )

        self._conversations.add_message(assistant_message)

        if config.MEMORY_EXTRACTION_ENABLED:
            try:
                self._extractor.extract_and_store(
                    user_text,
                    final_text,
                    conversation_id,
                )
            except Exception:
                logger.exception(
                    "Memory extraction failed; continuing."
                )

        return (
            assistant_message,
            tool_calls_made,
            retrieved_context,
        )

    def _run_tool_loop(
        self,
        messages: list[dict],
        tool_calls_made: list[str],
    ) -> str:

        try:

            for _ in range(config.LLM_MAX_TOOL_HOPS):

                message = self._llm.chat(
                    messages,
                    tools=TOOL_SPECS,
                )

                messages.append(message)

                tool_calls = message.get("tool_calls")

                if not tool_calls:
                    return (message.get("content") or "").strip()

                for call in tool_calls:

                    fn_name = call["function"]["name"]

                    raw_args = call["function"].get(
                        "arguments",
                        {},
                    )

                    args = (
                        raw_args
                        if isinstance(raw_args, dict)
                        else json.loads(raw_args)
                    )

                    logger.info(
                        "Calling tool '%s' with args %s",
                        fn_name,
                        args,
                    )

                    tool_calls_made.append(fn_name)

                    try:
                        result = run_tool(fn_name, args)

                    except (
                        ToolExecutionError,
                        ToolNotFoundError,
                    ) as exc:

                        result = f"Error: {exc}"

                    messages.append(
                        {
                            "role": "tool",
                            "content": result,
                        }
                    )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Please provide the final answer."
                    ),
                }
            )

            final_message = self._llm.chat(
                messages,
                tools=None,
            )

            return (
                final_message.get("content") or ""
            ).strip()

        except LLMError as exc:

            logger.warning(
                "LLM unavailable: %s",
                exc,
            )

            return (
                "I'm currently unable to reach the local language model. "
                "Please make sure Ollama is running."
            )