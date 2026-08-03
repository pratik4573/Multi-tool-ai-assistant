"""
Pratik AI — Streamlit frontend entrypoint.

Run with:
    streamlit run frontend/streamlit_app.py

This UI is a thin client over the FastAPI backend
(src/api/server.py). All business logic (LLM orchestration,
RAG, memory, tools, speech) lives in the backend.
"""

from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import requests
import streamlit as st

import config
from frontend.components.chat_window import render_chat_history
from frontend.components.input_bar import render_input_bar
from frontend.components.sidebar import render_sidebar
from frontend.state import init_session_state

API_BASE = config.STREAMLIT_API_BASE_URL

st.set_page_config(
    page_title=config.APP_NAME,
    page_icon="🧠",
    layout="wide",
)

init_session_state()


def _ensure_conversation() -> str:
    """Create a conversation if one doesn't already exist."""
    if st.session_state.conversation_id:
        return st.session_state.conversation_id

    response = requests.post(
        f"{API_BASE}/history/conversations",
        json={"title": "New Chat"},
        timeout=10,
    )
    response.raise_for_status()

    conversation = response.json()
    conversation_id = conversation["id"]

    st.session_state.conversation_id = conversation_id
    return conversation_id


def _send_text_message(text: str) -> None:
    """Send a text message to the backend."""

    if not st.session_state.conversation_id:
        _ensure_conversation()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": text,
        }
    )

    st.session_state.is_generating = True
    st.session_state.stop_requested = False

    with st.chat_message("assistant"):
        placeholder = st.empty()

        try:
            response = requests.post(
                f"{API_BASE}/chat",
                json={
                    "conversation_id": st.session_state.conversation_id,
                    "message": text,
                    "use_rag": True,
                    "use_memory": True,
                },
                timeout=config.LLM_REQUEST_TIMEOUT_SEC,
            )

            response.raise_for_status()

            data = response.json()

            st.session_state.conversation_id = data["conversation_id"]

            reply = data["message"]["content"]

            retrieved_context = data.get("retrieved_context", [])

            placeholder.markdown(reply)

        except Exception as exc:
            reply = f"⚠️ {exc}"
            retrieved_context = []
            placeholder.error(reply)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply,
            "retrieved_context": retrieved_context,
        }
    )

    st.session_state.is_generating = False


def _build_audio_files_payload(uploaded_audio_file, recorded_audio):
    """
    Build the `files` dict for the /voice request depending on the audio
    source (uploaded file vs. microphone recording).

    Returns:
        files: dict | None
    """

    if uploaded_audio_file is not None:
        return {
            "audio": (
                uploaded_audio_file.name,
                uploaded_audio_file.getvalue(),
                uploaded_audio_file.type,
            )
        }

    if recorded_audio is not None:
        audio_bytes = recorded_audio.get("bytes")
        if not audio_bytes:
            return None

        return {
            "audio": (
                "voice.wav",
                audio_bytes,
                "audio/wav",
            )
        }

    return None


def _send_voice_message(uploaded_audio_file=None, recorded_audio=None) -> None:
    """
    Send a voice message (from file upload or microphone recording)
    to the backend and play the voice reply.
    """

    if not st.session_state.conversation_id:
        _ensure_conversation()

    st.session_state.is_generating = True
    st.session_state.stop_requested = False

    with st.chat_message("assistant"):
        placeholder = st.empty()

        try:
            files = _build_audio_files_payload(uploaded_audio_file, recorded_audio)

            if files is None:
                placeholder.error("⚠️ Unable to send audio.")
                return

            form_data = {
                "conversation_id": st.session_state.conversation_id,
                "use_rag": "true",
                "use_memory": "true",
            }

            response = requests.post(
                f"{API_BASE}/voice",
                files=files,
                data=form_data,
                timeout=config.LLM_REQUEST_TIMEOUT_SEC,
            )

            if response.status_code != 200:
                try:
                    error_detail = response.json().get("detail", response.text)
                except Exception:
                    error_detail = response.text

                placeholder.error(f"⚠️ {error_detail}")
                return

            data = response.json()

            st.session_state.conversation_id = data["conversation_id"]

            transcript = data["transcript"]
            reply = data["reply_text"]

            # Show the transcribed user message
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": transcript,
                }
            )

            placeholder.markdown(reply)

            # Store assistant reply
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": reply,
                }
            )

            # Play TTS audio if available
            audio_path = data.get("reply_audio_path")

            if audio_path:
                filename = os.path.basename(audio_path)

                audio_response = requests.get(
                    f"{API_BASE}/voice/audio/{filename}",
                    timeout=30,
                )

                if audio_response.status_code == 200:
                    st.audio(audio_response.content)

        except Exception as exc:
            placeholder.error(f"⚠️ Unable to send audio. ({exc})")

        finally:
            st.session_state.is_generating = False


def _transcribe_audio(recorded_audio) -> str | None:
    """
    Send a microphone recording to the backend for transcription only
    (no chat turn is created). The transcript is meant to be dropped into
    the text input box so the user can review/edit before sending.
    """

    audio_bytes = recorded_audio.get("bytes")
    if not audio_bytes:
        st.error("⚠️ No audio captured — check mic permissions in your browser.")
        return None

    try:
        response = requests.post(
            f"{API_BASE}/voice/transcribe",
            files={"audio": ("voice.wav", audio_bytes, "audio/wav")},
            timeout=config.LLM_REQUEST_TIMEOUT_SEC,
        )
        response.raise_for_status()
        return response.json()["transcript"]

    except Exception as exc:
        st.error(f"⚠️ Transcription failed: {exc}")
        return None


def main() -> None:
    render_sidebar()

    st.markdown(f"### {config.APP_NAME}")
    st.caption("Your offline-first personal AI assistant.")

    render_chat_history()

    typed_text, uploaded_audio, recorded_audio = render_input_bar()

    if recorded_audio and not st.session_state.is_generating:
        transcript = _transcribe_audio(recorded_audio)
        if transcript:
            st.session_state.draft_text = transcript
        st.rerun()

    elif typed_text and not st.session_state.is_generating:
        _send_text_message(typed_text)
        st.rerun()

    elif uploaded_audio and not st.session_state.is_generating:
        _send_voice_message(uploaded_audio_file=uploaded_audio)
        st.rerun()


if __name__ == "__main__":
    main()