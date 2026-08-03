"""POST /voice — voice-in, voice-out chat turn over HTTP."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

import config
from src.memory.conversation_store import ConversationStore
from src.models.chat_models import VoiceChatResponse
from src.stt.faster_whisper_stt import SpeechToText
from src.tts import get_tts_engine
from src.utils.exceptions import PratikAIError, SpeechError
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["voice"])

_orchestrator: "ChatOrchestrator" | None = None
_conversations = ConversationStore()


def get_orchestrator() -> "ChatOrchestrator":
    global _orchestrator
    if _orchestrator is None:
        from src.llm.orchestrator import ChatOrchestrator

        _orchestrator = ChatOrchestrator()
    return _orchestrator

# STT/TTS engines are loaded lazily on first request since they are heavy.
_stt: SpeechToText | None = None
_tts = None


def _get_stt() -> SpeechToText:
    global _stt
    if _stt is None:
        _stt = SpeechToText()
    return _stt


def _get_tts():
    global _tts
    if _tts is None:
        _tts = get_tts_engine()
    return _tts


@router.post("/voice", response_model=VoiceChatResponse)
async def voice_chat(
    audio: UploadFile = File(...),
    conversation_id: str | None = Form(default=None),
    use_rag: bool = Form(default=True),
    use_memory: bool = Form(default=True),
) -> VoiceChatResponse:
    if not conversation_id:
        conversation_id = _conversations.create_conversation().id
    elif _conversations.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")

    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=config.UPLOADS_DIR) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name

    try:
        transcript = _get_stt().transcribe_file(tmp_path)
    except SpeechError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not transcript:
        raise HTTPException(status_code=422, detail="Could not understand any speech in the audio.")

    try:
        assistant_message, _tool_calls, _context = get_orchestrator().handle_turn(
            conversation_id=conversation_id,
            user_text=transcript,
            use_rag=use_rag,
            use_memory=use_memory,
        )
        audio_path = _get_tts().speak_to_file(assistant_message.content)
    except PratikAIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return VoiceChatResponse(
        conversation_id=conversation_id,
        transcript=transcript,
        reply_text=assistant_message.content,
        reply_audio_path=audio_path,
    )


@router.post("/voice/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)) -> dict:
    """
    Transcribe an audio recording only — no conversation turn, no LLM call,
    no TTS reply. Used by the frontend to let the user review/edit a
    transcript before deciding whether to send it as a chat message.
    """
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=config.UPLOADS_DIR) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name

    try:
        transcript = _get_stt().transcribe_file(tmp_path)
    except SpeechError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not transcript:
        raise HTTPException(status_code=422, detail="Could not understand any speech in the audio.")

    return {"transcript": transcript}


@router.get("/voice/audio/{filename}")
def get_voice_audio(filename: str) -> FileResponse:
    path = config.TTS_OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(str(path), media_type="audio/wav")