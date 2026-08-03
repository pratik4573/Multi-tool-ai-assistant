"""
Low-level HTTP client for Ollama's OpenAI-compatible /api/chat endpoint.

Kept separate from the orchestration logic (prompt building, RAG/memory
injection, tool loop) so it can be unit-tested or swapped independently.
"""

from __future__ import annotations

import requests

import config
from src.utils.exceptions import LLMError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaClient:
    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        self.host = host or config.OLLAMA_HOST
        self.model = model or config.OLLAMA_MODEL

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = config.LLM_TEMPERATURE,
    ) -> dict:
        """Send a chat request and return the raw assistant message dict."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools

        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=config.LLM_REQUEST_TIMEOUT_SEC,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(
                f"Failed to reach Ollama at {self.host}. Is 'ollama serve' running "
                f"and is model '{self.model}' pulled? ({exc})"
            ) from exc

        data = response.json()
        message = data.get("message")
        if message is None:
            raise LLMError(f"Unexpected Ollama response shape: {data}")
        return message

    def simple_completion(self, prompt: str, temperature: float = 0.2) -> str:
        """Convenience helper for single-shot, tool-free completions
        (used by the memory extractor)."""
        message = self.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            temperature=temperature,
        )
        return message.get("content", "").strip()
