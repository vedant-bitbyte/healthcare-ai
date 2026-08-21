"""Embedding package for Healthcare AI RAG."""

from .embedding_model import (
    DEFAULT_MODEL_NAME,
    EmbeddingModelError,
    embed_text,
    embed_texts,
    get_embedding_model,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "EmbeddingModelError",
    "embed_text",
    "embed_texts",
    "get_embedding_model",
]
