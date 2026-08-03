"""Sidebar: New Chat, Previous Chats, Upload Documents, Settings."""

from __future__ import annotations

import requests
import streamlit as st

import config
from frontend.state import reset_conversation

API_BASE = config.STREAMLIT_API_BASE_URL


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(f"## 🧠 {config.APP_NAME}")

        if st.button("➕ New Chat", use_container_width=True):
            reset_conversation()
            st.rerun()

        st.divider()
        st.markdown("### Previous Chats")
        _render_previous_chats()

        st.divider()
        st.markdown("### Upload Documents")
        _render_upload()

        st.divider()
        with st.expander("⚙️ Settings"):
            _render_settings()


def _render_previous_chats() -> None:
    try:
        response = requests.get(f"{API_BASE}/history/conversations", timeout=10)
        response.raise_for_status()
        conversations = response.json()
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Could not load chat history ({exc}).")
        return

    if not conversations:
        st.caption("No previous chats yet.")
        return

    for conv in conversations[:20]:
        label = conv["title"] or "Untitled chat"
        if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
            st.session_state.conversation_id = conv["id"]
            _load_messages(conv["id"])
            st.rerun()


def _load_messages(conversation_id: str) -> None:
    try:
        response = requests.get(
            f"{API_BASE}/history/conversations/{conversation_id}", timeout=10
        )
        response.raise_for_status()
        messages = response.json()
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] in ("user", "assistant")
        ]
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load conversation: {exc}")


def _render_upload() -> None:
    uploaded_file = st.file_uploader(
        "Add a personal document",
        type=["md", "txt", "pdf"],
        label_visibility="collapsed",
        key="doc_uploader",
    )
    if uploaded_file is not None and st.button("Index document", use_container_width=True):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{API_BASE}/upload", files=files, timeout=120)
            response.raise_for_status()
            data = response.json()
            st.success(data["message"])
        except Exception as exc:  # noqa: BLE001
            st.error(f"Upload failed: {exc}")


def _render_settings() -> None:
    st.text_input("API base URL", value=API_BASE, disabled=True)
    st.text_input("LLM model", value=config.OLLAMA_MODEL, disabled=True)
    st.text_input("Whisper model", value=config.WHISPER_MODEL_SIZE, disabled=True)
    st.text_input("TTS engine", value=config.TTS_ENGINE, disabled=True)
    st.caption("Edit config.py or .env to change these values.")
