"""
Abstract TTS interface. Every text-to-speech engine (Piper today, XTTS v2
later for voice cloning) implements this so callers never depend on a
specific engine's API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TTSBase(ABC):
    """Common interface for all text-to-speech engines."""

    @abstractmethod
    def speak_to_file(self, text: str, output_path: str | None = None) -> str:
        """Synthesize `text` to a WAV file on disk and return its path."""
        raise NotImplementedError
