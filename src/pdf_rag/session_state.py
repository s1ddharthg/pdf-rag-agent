"""Streamlit session-state initialization helpers."""

import streamlit as st


def init_session_state() -> None:
    """Ensure all session-state keys used by the app exist with sane defaults."""
    defaults = {
        "messages": [],
        "pdfs": {},
        "active_pdfs": [],
        "vector_db": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
