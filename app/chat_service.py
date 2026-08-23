"""Chat service to interface Streamlit with the existing RAG pipeline."""

import logging

from src.rag.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def get_rag_pipeline() -> RAGPipeline:
    """
    Initialize and return the RAG pipeline.
    This creates the necessary Retriever, VectorStore, and caching mechanisms
    required by the underlying framework.
    """
    return RAGPipeline(model="fine-tuned")


def get_document_count(pipeline: RAGPipeline) -> int:
    """
    Get the total number of indexed document chunks from the vector store.
    """
    try:
        return pipeline.retriever.vector_store.count()
    except Exception as exc:
        logger.error("Failed to get document count: %s", exc)
        return 0
