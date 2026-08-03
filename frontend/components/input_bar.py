"""Bottom input bar: message box, upload, mic recorder, and stop-generation button."""
from __future__ import annotations
import streamlit as st
from streamlit_mic_recorder import mic_recorder
import config

API_BASE = config.STREAMLIT_API_BASE_URL


def render_input_bar():
    """
    Returns:
        typed_text: str | None   -> text submitted via Enter/Send
        uploaded_audio_file: UploadedFile | None
        recorded_audio: dict | None (raw bytes from mic_recorder, only when
                                      a NEW recording just finished)
    """
    st.session_state.setdefault("draft_text", "")
    st.session_state.setdefault("chat_box", "")

    # If a transcript (or any other caller) queued up draft_text, push it into
    # the actual widget key BEFORE the widget is instantiated. Once a keyed
    # widget has rendered once, Streamlit ignores `value=` on later reruns and
    # reads from session_state[key] instead — so this sync step is required,
    # not optional.
    if st.session_state.draft_text:
        st.session_state.chat_box = st.session_state.draft_text
        st.session_state.draft_text = ""

    typed_text = None
    uploaded_audio_file = None
    recorded_audio = None

    # --- Text input (form so Enter submits, and so we can prefill it) ---
    with st.form(key="message_form", clear_on_submit=True):
        col_text, col_send = st.columns([6, 1])
        with col_text:
            box_value = st.text_input(
                "Message",
                key="chat_box",
                label_visibility="collapsed",
                placeholder="Message Pratik AI...",
            )
        with col_send:
            submitted = st.form_submit_button("Send")

        if submitted and box_value.strip():
            typed_text = box_value.strip()
            # clear_on_submit=True already clears chat_box for us

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("📎 Upload Audio", expanded=False):
            file = st.file_uploader(
                "Upload a voice message",
                type=["wav", "mp3", "m4a", "webm", "ogg", "flac"],
                key="audio_uploader",
            )
            if file is not None:
                if st.button("Send uploaded audio", key="send_uploaded_audio"):
                    uploaded_audio_file = file

    with col2:
        with st.expander("🎤 Record Voice", expanded=False):
            recorded_audio = mic_recorder(
                start_prompt="🎤 Start recording",
                stop_prompt="⏹ Stop recording",
                just_once=True,
                key="mic",
            )

    if st.session_state.get("is_generating"):
        if st.button("⏹ Stop Generation", key="stop_generation"):
            st.session_state.stop_requested = True

    return typed_text, uploaded_audio_file, recorded_audio