"""
Speech-to-text via faster-whisper (CTranslate2 Whisper implementation).
Fully offline once the model weights are cached locally.
"""

from __future__ import annotations

import numpy as np

import config
from src.utils.exceptions import SpeechError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SpeechToText:
    """Wraps faster-whisper for microphone-array and file-based transcription."""

    def __init__(self) -> None:
        from faster_whisper import WhisperModel

        logger.info("Loading Whisper model '%s' (%s/%s)...",
                    config.WHISPER_MODEL_SIZE, config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE)
        try:
            self._model = WhisperModel(
                config.WHISPER_MODEL_SIZE,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
            )
        except Exception as exc:  # noqa: BLE001
            raise SpeechError(f"Failed to load Whisper model: {exc}") from exc
        logger.info("Whisper model ready.")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a mono float32 numpy array sampled at config.SAMPLE_RATE."""
        if audio.size == 0:
            return ""

        segments, _info = self._model.transcribe(
            audio,
            language=config.WHISPER_LANGUAGE,
            vad_filter=True,
            beam_size=config.WHISPER_BEAM_SIZE,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()

    def transcribe_file(self, path: str) -> str:
        """Transcribe an audio file from disk (used by the /voice API endpoint)."""
        segments, _info = self._model.transcribe(
            path,
            language=config.WHISPER_LANGUAGE,
            vad_filter=True,
            beam_size=config.WHISPER_BEAM_SIZE,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()
