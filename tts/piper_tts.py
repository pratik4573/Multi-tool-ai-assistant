"""
Offline text-to-speech via Piper.

Requires a Piper voice (.onnx + .onnx.json pair) placed at the paths
configured in config.py. Download voices from:
https://github.com/rhasspy/piper/blob/master/VOICES.md
"""

from __future__ import annotations

import time
import wave
from pathlib import Path

import config
from src.tts.base import TTSBase
from src.utils.exceptions import ConfigurationError, SpeechError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PiperTTS(TTSBase):
    def __init__(self) -> None:
        from piper import PiperVoice

        model_path = Path(config.PIPER_MODEL_PATH)
        config_path = Path(config.PIPER_CONFIG_PATH)

        if not model_path.exists() or not config_path.exists():
            raise ConfigurationError(
                f"Piper voice files not found. Expected '{model_path}' and "
                f"'{config_path}'. Download a voice from the Piper voices list "
                "and place both files under the configured models directory."
            )

        logger.info("Loading Piper voice from %s...", model_path)
        try:
            self._voice = PiperVoice.load(str(model_path), config_path=str(config_path))
        except Exception as exc:  # noqa: BLE001
            raise SpeechError(f"Failed to load Piper voice: {exc}") from exc
        logger.info("Piper voice ready.")

    def speak_to_file(self, text: str, output_path: str | None = None) -> str:
        if not text.strip():
            raise SpeechError("Cannot synthesize empty text.")

        output_path = output_path or str(config.TTS_OUTPUT_DIR / f"tts_{int(time.time() * 1000)}.wav")

        try:
            with wave.open(output_path, "wb") as wav_file:
                self._voice.synthesize(text, wav_file)
        except Exception as exc:  # noqa: BLE001
            raise SpeechError(f"Piper synthesis failed: {exc}") from exc

        return output_path
