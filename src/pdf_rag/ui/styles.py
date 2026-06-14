import streamlit as st

_CUSTOM_CSS = """
<style>
    .app-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 1.1rem 1.5rem;
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 1.25rem;
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.55rem;
        font-weight: 700;
    }
    .app-header p {
        margin: 0.3rem 0 0;
        opacity: 0.9;
        font-size: 0.92rem;
    }

    .section-label {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.35rem;
        color: inherit;
    }

    .source-pill {
        display: inline-block;
        background-color: rgba(79, 70, 229, 0.12);
        color: #6366f1;
        border-radius: 999px;
        padding: 0.15rem 0.7rem;
        margin: 0.15rem 0.3rem 0.15rem 0;
        font-size: 0.8rem;
        font-weight: 600;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    .stButton > button {
        border-radius: 8px;
    }

    div[data-testid="stFileUploaderDropzone"] {
        border-radius: 12px;
    }
</style>
"""


def apply_custom_styles() -> None:
    """Inject the application's shared CSS once per render."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def render_header() -> None:
    """Render the gradient page header."""
    st.markdown(
        """
        <div class="app-header">
            <h1>🎈 PDF RAG Application</h1>
            <p>Chat with your PDFs using locally-hosted Ollama models — fully offline retrieval-augmented Q&A.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source_pills(sources_by_pdf: dict) -> None:
    """Render a row of pill-shaped badges summarizing retrieved sources."""
    pills = "".join(
        f'<span class="source-pill">📄 {name} · {count} chunk{"s" if count != 1 else ""}</span>'
        for name, count in sources_by_pdf.items()
    )
    st.markdown(pills, unsafe_allow_html=True)
