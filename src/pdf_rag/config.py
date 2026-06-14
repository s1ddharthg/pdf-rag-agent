"""Application-wide configuration and constants."""

import os

# Set protobuf environment variable to avoid error messages.
# This might cause some issues with latency but it's a tradeoff.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Directory where Chroma persists its vector collections.
PERSIST_DIRECTORY = os.path.join("data", "vectors")

# Embedding / chunking configuration.
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 7500
CHUNK_OVERLAP = 100

# Retrieval configuration.
RETRIEVAL_K = 3
MAX_CONTEXT_CHUNKS = 10

# Streamlit page configuration.
PAGE_CONFIG = {
    "page_title": "PDF RAG Application",
    "page_icon": ":)",
    "layout": "wide",
    "initial_sidebar_state": "collapsed",
}

# Prompt used to generate alternative phrasings of a user question
# for multi-query retrieval.
MULTI_QUERY_PROMPT_TEMPLATE = """You are an AI language model assistant. Your task is to generate 2
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}"""
