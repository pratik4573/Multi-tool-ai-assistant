"""
XTTS v2 (voice cloning) TTS engine — reserved for future use.

This stub exists so the project structure and config.TTS_ENGINE switch are
ready today; implementing this class (using the `TTS` / `coqui-tts` package
and config.XTTS_MODEL_NAME / config.XTTS_SPEAKER_WAV) is all that is needed
to enable voice cloning without touching any other module.
"""

from __future__ import annotations

from src.tts.base import TTSBase
from src.utils.exceptions import ConfigurationError


class XTTSVoiceCloneTTS(TTSBase):
    def __init__(self) -> None:
        raise ConfigurationError(
            "XTTS v2 support is not implemented yet. Set PRATIK_TTS_ENGINE=piper "
            "in your .env, or implement src/tts/xtts_tts.py to enable voice cloning."
        )

    def speak_to_file(self, text: str, output_path: str | None = None) -> str:
        raise NotImplementedError("XTTS v2 backend is not yet implemented.")
