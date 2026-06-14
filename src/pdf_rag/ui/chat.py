"""Chat interface: history rendering, prompt handling and source display."""

import logging
from typing import Dict, List, Optional

import streamlit as st

from ..rag_chain import process_question_multi_pdf
from .styles import render_source_pills

logger = logging.getLogger(__name__)


def _group_sources(sources: List[Dict]) -> Dict[str, int]:
    sources_by_pdf: Dict[str, int] = {}
    for src in sources:
        pdf_name = src.get("pdf_name", "Unknown")
        sources_by_pdf[pdf_name] = sources_by_pdf.get(pdf_name, 0) + 1
    return sources_by_pdf


def _render_sources(sources: List[Dict]) -> None:
    """Render a grouped, per-PDF chunk-count summary of retrieved sources."""
    st.divider()
    st.caption("📚 Sources")
    render_source_pills(_group_sources(sources))


def render_chat(col2, selected_model: Optional[str]) -> None:
    """Render the chat history and handle new prompts."""
    with col2:
        st.markdown('<div class="section-label">💬 Chat</div>', unsafe_allow_html=True)

        message_container = st.container(height=500, border=True)

        if not st.session_state["messages"]:
            with message_container:
                st.markdown(
                    "##### 👋 Ask me anything about your documents\n"
                    "Upload one or more PDFs and ask a question "
                    "to get a source-cited answer."
                )

        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

                if message["role"] == "assistant" and "sources" in message:
                    _render_sources(message["sources"])

        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[Thinking...]"):
                        if st.session_state.get("pdfs"):
                            response, sources = process_question_multi_pdf(
                                prompt,
                                st.session_state["pdfs"],
                                selected_model,
                            )
                            st.markdown(response)

                            if sources:
                                _render_sources(sources)
                        else:
                            st.warning("Please upload PDF files first.")
                            response = None
                            sources = None

                if response:
                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": response,
                            "sources": sources,
                        }
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if not st.session_state.get("pdfs"):
                st.warning("Upload PDF files to begin chat...")
