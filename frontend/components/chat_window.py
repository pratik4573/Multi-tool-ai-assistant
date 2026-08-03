"""Main chat window: message history with Markdown, code blocks, and copy button."""

from __future__ import annotations

import streamlit as st


def render_chat_history() -> None:
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_copy_button(message["content"], key=f"copy_{i}")
                if message.get("retrieved_context"):
                    with st.expander("📎 Retrieved context"):
                        for i, chunk in enumerate(message["retrieved_context"], start=1):
                            st.markdown(f"**Chunk {i}**")
                            st.info(chunk)
                            


def _render_copy_button(text: str, key: str) -> None:
    # Streamlit has no native clipboard API, so a tiny HTML/JS snippet is used
    # to implement the copy button without any extra Python dependency.
    escaped = text.replace("`", "\\`").replace("</", "<\\/")
    st.components.v1.html(
        f"""
        <button onclick="navigator.clipboard.writeText(`{escaped}`)"
                style="font-size:12px;padding:2px 8px;border-radius:6px;
                       border:1px solid #ccc;background:#f5f5f5;cursor:pointer;">
            📋 Copy
        </button>
        """,
        height=32,
    )
