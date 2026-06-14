import sys
import warnings
from pathlib import Path

if __package__ in (None, ""):
    # Allow this module to be run directly (e.g. `streamlit run src/pdf_rag/app.py`)
    # by registering it under its proper package path before using relative imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.pdf_rag"

import streamlit as st

from .config import PAGE_CONFIG
from .logging_config import configure_logging
from .ollama_client import get_available_models
from .session_state import init_session_state
from .ui.chat import render_chat
from .ui.pdf_viewer import render_pdf_viewer
from .ui.sidebar import render_sidebar
from .ui.styles import apply_custom_styles, render_header
from .ui.upload import render_pdf_source

# Suppress torch warning.
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.classes.*")

configure_logging()


def main() -> None:
    st.set_page_config(**PAGE_CONFIG)
    apply_custom_styles()
    render_header()

    init_session_state()

    available_models = get_available_models()

    col1, col2 = st.columns([1.5, 2], gap="large")

    selected_model = None
    if available_models:
        selected_model = col2.selectbox(
            "🧠 Model",
            available_models,
            key="model_select",
            help="Pick an Ollama model that's available locally on your system.",
        )
    else:
        col2.warning("No local Ollama models found. Pull a model with `ollama pull <model>`.")

    render_sidebar()
    render_pdf_source(col1)
    render_pdf_viewer(col1)
    render_chat(col2, selected_model)
