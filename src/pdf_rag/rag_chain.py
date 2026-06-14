"""Question-answering chains over one or more PDF vector stores."""

import logging
from typing import Dict, List, Tuple

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from .config import MAX_CONTEXT_CHUNKS, MULTI_QUERY_PROMPT_TEMPLATE, RETRIEVAL_K

logger = logging.getLogger(__name__)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=MULTI_QUERY_PROMPT_TEMPLATE,
)

MULTI_PDF_ANSWER_TEMPLATE = """Answer the question based ONLY on the following context from multiple PDF documents.
Each section is marked with its source document.

When answering:
1. Cite the source document name for each piece of information
2. If information comes from multiple sources, mention all relevant sources
3. If sources contradict, note the discrepancy and cite both

Context:
{context}

Question: {question}

Answer (include source citations):"""

SINGLE_PDF_ANSWER_TEMPLATE = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""


def process_question_multi_pdf(
    question: str,
    pdfs_dict: Dict[str, Dict],
    selected_model: str,
) -> Tuple[str, List[Dict]]:
    """Query across multiple PDFs with source attribution."""
    logger.info(f"Processing question across {len(pdfs_dict)} PDFs: {question}")

    llm = ChatOllama(model=selected_model)

    # Retrieve from ALL PDF collections.
    all_retrieved_docs = []
    for pdf_id, pdf_data in pdfs_dict.items():
        vector_db = pdf_data["vector_db"]

        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
            llm,
            prompt=QUERY_PROMPT,
        )

        try:
            docs = retriever.get_relevant_documents(question)
            logger.info(f"Retrieved {len(docs)} documents from {pdf_data['name']}")
            for doc in docs:
                if "pdf_name" not in doc.metadata:
                    doc.metadata["pdf_name"] = pdf_data["name"]
                if "pdf_id" not in doc.metadata:
                    doc.metadata["pdf_id"] = pdf_id
            all_retrieved_docs.extend(docs)
        except Exception as e:
            logger.warning(f"Error retrieving from {pdf_data['name']}: {e}")

    logger.info(f"Total documents retrieved: {len(all_retrieved_docs)}")

    # Format context with source labels.
    context_parts = []
    for doc in all_retrieved_docs[:MAX_CONTEXT_CHUNKS]:
        source = doc.metadata.get("pdf_name", "Unknown")
        context_parts.append(f"[Source: {source}]\n{doc.page_content}\n")

    formatted_context = "\n---\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template(MULTI_PDF_ANSWER_TEMPLATE)
    chain = (
        {"context": lambda x: formatted_context, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Generated response with source attribution")

    source_details = [
        {
            "pdf_name": doc.metadata.get("pdf_name"),
            "pdf_id": doc.metadata.get("pdf_id"),
            "chunk_index": doc.metadata.get("chunk_index", 0),
        }
        for doc in all_retrieved_docs[:MAX_CONTEXT_CHUNKS]
    ]

    return response, source_details


def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using a single vector database and selected language model.

    Args:
        question (str): The user's question.
        vector_db (Chroma): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    logger.info(f"Processing question: {question} using model: {selected_model}")

    llm = ChatOllama(model=selected_model)

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT,
    )

    prompt = ChatPromptTemplate.from_template(SINGLE_PDF_ANSWER_TEMPLATE)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response
