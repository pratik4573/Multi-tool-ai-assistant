"""Streamlit session_state helpers for Pratik AI's frontend."""

from __future__ import annotations

import streamlit as st


def init_session_state() -> None:
    defaults = {
        "conversation_id": None,
        "messages": [],       # list[dict]: {"role", "content"}
        "is_generating": False,
        "stop_requested": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_conversation() -> None:
    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.session_state.is_generating = False
    st.session_state.stop_requested = False
