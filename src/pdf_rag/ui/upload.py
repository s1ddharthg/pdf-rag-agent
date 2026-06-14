import logging

import streamlit as st

from ..pdf_processing import generate_pdf_id, process_and_store_pdf

logger = logging.getLogger(__name__)


def render_pdf_source(col1) -> None:
    """Render the file uploader, ingesting any new PDFs."""
    col1.markdown('<div class="section-label">📥 Add Documents</div>', unsafe_allow_html=True)

    file_uploads = col1.file_uploader(
        "Drag and drop PDF files here, or click to browse",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if file_uploads:
        for file_upload in file_uploads:
            pdf_id = generate_pdf_id(file_upload)

            if pdf_id not in st.session_state.get("pdfs", {}):
                with st.spinner(f"📖 Processing {file_upload.name}..."):
                    process_and_store_pdf(file_upload, pdf_id)
                st.toast(f"Added {file_upload.name}", icon="✅")
