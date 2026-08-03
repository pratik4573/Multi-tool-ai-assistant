"""
Pratik AI — application entrypoint.

Running `python main.py` starts the FastAPI backend (serving /chat, /voice,
/upload, /search, /history, /memory, /tools). The Streamlit UI is a separate
process started with `streamlit run frontend/streamlit_app.py`, and a raw
CLI voice loop is available via `python main.py --voice-cli` for quick
terminal-only testing without the API/UI.
"""

from __future__ import annotations

import argparse
import socket
import sys

import uvicorn

import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_available_port(preferred_port: int, host: str | None = None) -> int:
    """Return a free TCP port, falling back to an ephemeral one when needed."""
    bind_host = host or config.API_HOST
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, preferred_port))
        except OSError:
            sock.bind((bind_host, 0))
        return sock.getsockname()[1]


def run_api() -> None:
    port = get_available_port(config.API_PORT)
    if port != config.API_PORT:
        logger.warning("Port %s is busy; starting server on %s instead.", config.API_PORT, port)
    config.API_PORT = port
    config.STREAMLIT_API_BASE_URL = f"http://localhost:{port}"
    uvicorn.run("src.api.server:app", host=config.API_HOST, port=port, reload=False)


def run_voice_cli() -> None:
    """A minimal terminal voice loop, useful for quick offline testing
    without starting the API/UI (mirrors the original prototype's main.py)."""
    import traceback

    from src.database.schema import init_db
    from src.llm.orchestrator import ChatOrchestrator
    from src.memory.conversation_store import ConversationStore
    from src.speech.audio_io import VoiceRecorder, play_wav
    from src.stt.faster_whisper_stt import SpeechToText
    from src.tts import get_tts_engine

    print("=" * 60)
    print(f"{config.APP_NAME} — Voice CLI")
    print("=" * 60)

    try:
        init_db()
        conversations = ConversationStore()
        conversation_id = conversations.create_conversation(title="Voice CLI Session").id

        recorder = VoiceRecorder()
        print("✓ Recorder + VAD loaded")
        stt = SpeechToText()
        print("✓ Whisper loaded")
        orchestrator = ChatOrchestrator()
        print("✓ LLM + RAG + memory loaded")
        tts = get_tts_engine()
        print("✓ TTS loaded")
        print("\nAssistant ready! Say 'exit' or 'quit' to stop.\n")

        while True:
            audio = recorder.record_utterance()
            if audio.size == 0:
                continue

            user_text = stt.transcribe(audio)
            if not user_text:
                print("Nothing understood.\n")
                continue

            print(f"\n🧑 You: {user_text}")

            if user_text.lower().strip() in ("exit", "quit", "goodbye", "stop listening"):
                print("Bye!")
                break

            assistant_message, tool_calls, _context = orchestrator.handle_turn(
                conversation_id=conversation_id, user_text=user_text
            )
            print(f"\n🤖 {config.APP_NAME}: {assistant_message.content}")
            if tool_calls:
                print(f"   (used tools: {', '.join(tool_calls)})")

            wav_path = tts.speak_to_file(assistant_message.content)
            try:
                play_wav(wav_path)
            except Exception:
                print("\nAudio playback failed:")
                traceback.print_exc()

    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception:
        print("\nFatal error:")
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{config.APP_NAME} entrypoint")
    parser.add_argument(
        "--voice-cli",
        action="store_true",
        help="Run a terminal-only voice loop instead of the FastAPI server.",
    )
    args = parser.parse_args()

    if args.voice_cli:
        run_voice_cli()
    else:
        run_api()


if __name__ == "__main__":
    main()
