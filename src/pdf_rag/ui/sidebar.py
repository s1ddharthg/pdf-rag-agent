"""Sidebar UI: loaded-PDF summary and management controls."""

import streamlit as st

from ..vector_store import delete_all_pdfs, delete_pdf


def render_sidebar() -> None:
    """Render the sidebar with loaded PDF stats and delete controls."""
    with st.sidebar:
        st.subheader("📚 Loaded PDFs")

        if st.session_state.get("pdfs"):
            total_pdfs = len(st.session_state["pdfs"])
            total_chunks = sum(
                pdf["doc_count"] for pdf in st.session_state["pdfs"].values()
            )

            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("PDFs", total_pdfs)
            metric_col2.metric("Chunks", total_chunks)
            st.divider()

            for pdf_id in st.session_state["active_pdfs"]:
                pdf_data = st.session_state["pdfs"][pdf_id]

                with st.expander(f"📄 {pdf_data['name']}", expanded=False):
                    st.caption(f"🧩 {pdf_data['doc_count']} chunks · 📃 {len(pdf_data['pages'])} pages")
                    st.caption(f"🕒 Uploaded {pdf_data['upload_timestamp'].strftime('%Y-%m-%d %H:%M')}")

                    if st.button("🗑️ Delete", key=f"delete_{pdf_id}", use_container_width=True):
                        delete_pdf(pdf_id)
                        st.rerun()

            st.divider()
            if st.button("🗑️ Delete All PDFs", use_container_width=True):
                delete_all_pdfs()
                st.rerun()
        else:
            st.info("No PDFs loaded yet. Upload a file to get started.")
