"""Vector store creation and lifecycle management."""

import logging
import os
import shutil
import tempfile
from typing import Optional

import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, PERSIST_DIRECTORY

logger = logging.getLogger(__name__)


def create_vector_db(file_upload) -> Chroma:
    """
    Create a vector database from an uploaded PDF file.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        loader = UnstructuredPDFLoader(path)
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=f"pdf_{hash(file_upload.name)}",  # Unique collection name per file
    )
    logger.info("Vector DB created with persistent storage")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db


def delete_pdf(pdf_id: str) -> None:
    """Delete a single PDF and its vector collection from session state."""
    if pdf_id in st.session_state["pdfs"]:
        pdf_data = st.session_state["pdfs"][pdf_id]
        logger.info(f"Deleting PDF: {pdf_data['name']} (ID: {pdf_id})")

        try:
            pdf_data["vector_db"].delete_collection()
            logger.info(f"Deleted collection: {pdf_data['collection_name']}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

        del st.session_state["pdfs"][pdf_id]
        st.session_state["active_pdfs"].remove(pdf_id)

        st.success(f"Deleted {pdf_data['name']}")


def delete_all_pdfs() -> None:
    """Delete all loaded PDFs and their vector collections."""
    logger.info("Deleting all PDFs")
    for pdf_id in list(st.session_state["pdfs"].keys()):
        delete_pdf(pdf_id)
    st.session_state["pdfs"] = {}
    st.session_state["active_pdfs"] = []


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.

    Args:
        vector_db (Optional[Chroma]): The vector database to be deleted.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        try:
            vector_db.delete_collection()

            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_upload", None)
            st.session_state.pop("vector_db", None)

            st.success("Collection and temporary files deleted successfully.")
            logger.info("Vector DB and related session state cleared")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")
            logger.error(f"Error deleting collection: {e}")
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")
