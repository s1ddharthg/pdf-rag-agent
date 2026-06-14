"""PDF ingestion: loading, chunking, embedding and persisting documents."""

import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import Any, List

import pdfplumber
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, PERSIST_DIRECTORY

logger = logging.getLogger(__name__)


def generate_pdf_id(file_upload) -> str:
    """Generate a unique ID for an uploaded PDF."""
    timestamp = datetime.now().isoformat()
    return f"pdf_{abs(hash(file_upload.name + timestamp))}"


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages


def process_and_store_pdf(file_upload, pdf_id: str) -> None:
    """Process a single PDF and store its vector DB and pages in session state."""
    logger.info(f"Processing PDF: {file_upload.name} with ID: {pdf_id}")

    # Create temp directory and save the upload to disk.
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file_upload.name)

    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")

    try:
        # Load and chunk.
        loader = PyPDFLoader(path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(data)
        logger.info(f"Document split into {len(chunks)} chunks")

        # Add metadata to each chunk.
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "pdf_id": pdf_id,
                    "pdf_name": file_upload.name,
                    "chunk_index": i,
                    "source_file": file_upload.name,
                }
            )

        # Create vector DB with a unique collection.
        collection_name = f"pdf_{abs(hash(file_upload.name + pdf_id))}"
        logger.info(f"Creating vector DB with collection name: {collection_name}")

        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=collection_name,
        )
        logger.info("Vector DB created with persistent storage")

        # Extract PDF pages for the viewer.
        with pdfplumber.open(file_upload) as pdf:
            pdf_pages = [page.to_image().original for page in pdf.pages]
        logger.info(f"Extracted {len(pdf_pages)} pages from PDF")

        # Store in session state.
        st.session_state["pdfs"][pdf_id] = {
            "name": file_upload.name,
            "vector_db": vector_db,
            "pages": pdf_pages,
            "file_upload": file_upload,
            "collection_name": collection_name,
            "upload_timestamp": datetime.now(),
            "doc_count": len(chunks),
        }
        st.session_state["active_pdfs"].append(pdf_id)
        logger.info(f"PDF stored in session state with {len(chunks)} chunks")
    finally:
        shutil.rmtree(temp_dir)
        logger.info(f"Temporary directory {temp_dir} removed")
