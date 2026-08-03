"""TTS engine factory — returns the configured TTSBase implementation."""

from __future__ import annotations

import config
from src.tts.base import TTSBase


def get_tts_engine() -> TTSBase:
    if config.TTS_ENGINE == "xtts":
        from src.tts.xtts_tts import XTTSVoiceCloneTTS
        return XTTSVoiceCloneTTS()

    from src.tts.piper_tts import PiperTTS
    return PiperTTS()
