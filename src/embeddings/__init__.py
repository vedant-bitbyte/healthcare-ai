"""Embedding and vector store package for Healthcare AI RAG."""

from .embedding_model import (
    DEFAULT_MODEL_NAME,
    EmbeddingModelError,
    embed_text,
    embed_texts,
    get_embedding_model,
)
from .vector_store import (
    DEFAULT_CHUNKS_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIR,
    VectorStore,
    VectorStoreError,
    load_chunk_records,
    load_chunks_from_directory,
)

__all__ = [
    "DEFAULT_CHUNKS_DIR",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_PERSIST_DIR",
    "EmbeddingModelError",
    "VectorStore",
    "VectorStoreError",
    "embed_text",
    "embed_texts",
    "get_embedding_model",
    "load_chunk_records",
    "load_chunks_from_directory",
]
