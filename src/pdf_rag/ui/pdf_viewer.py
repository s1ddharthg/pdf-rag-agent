"""Tabbed PDF page viewer and vector-DB cleanup controls."""

import streamlit as st

from ..vector_store import delete_pdf, delete_vector_db


def render_pdf_viewer(col1) -> None:
    """Render a tabbed PDF viewer for all loaded PDFs."""
    col1.markdown('<div class="section-label">🖼️ Document Viewer</div>', unsafe_allow_html=True)

    if st.session_state.get("pdfs") and st.session_state.get("active_pdfs"):
        zoom_level = col1.slider(
            "🔍 Zoom Level",
            min_value=100,
            max_value=1000,
            value=700,
            step=50,
            key="zoom_slider",
        )

        active_pdfs = st.session_state["active_pdfs"]
        tab_labels = [
            f"📄 {st.session_state['pdfs'][pdf_id]['name']}"
            for pdf_id in active_pdfs
            if pdf_id in st.session_state["pdfs"]
        ]

        with col1:
            tabs = st.tabs(tab_labels)
            for tab, pdf_id in zip(tabs, active_pdfs):
                if pdf_id not in st.session_state["pdfs"]:
                    continue

                pdf_data = st.session_state["pdfs"][pdf_id]

                with tab:
                    info_col, action_col = st.columns([4, 1])
                    with info_col:
                        st.caption(
                            f"🕒 {pdf_data['upload_timestamp'].strftime('%Y-%m-%d %H:%M')} · "
                            f"🧩 {pdf_data['doc_count']} chunks · "
                            f"📃 {len(pdf_data['pages'])} pages"
                        )
                    with action_col:
                        if st.button("🗑️ Remove", key=f"remove_{pdf_id}", use_container_width=True):
                            delete_pdf(pdf_id)
                            st.rerun()

                    with st.container(height=410, border=True):
                        for page_idx, page_image in enumerate(pdf_data["pages"]):
                            st.caption(f"Page {page_idx + 1}")
                            st.image(page_image, width=zoom_level)
    else:
        col1.info("Upload PDF files to view them here.")

    with col1.expander("⚙️ Advanced"):
        delete_collection = st.button(
            "⚠️ Delete collection",
            type="secondary",
            key="delete_button",
            use_container_width=True,
        )

        if delete_collection:
            delete_vector_db(st.session_state["vector_db"])
