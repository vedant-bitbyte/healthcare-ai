"""Singleton wrapper for the SentenceTransformers embedding model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingModelError(Exception):
    """Raised when the embedding model cannot be loaded or used."""


class _EmbeddingModelSingleton:
    """Thread-safe singleton holder for a single SentenceTransformer instance."""

    _instance: _EmbeddingModelSingleton | None = None
    _model: SentenceTransformer | None = None

    def __new__(cls) -> _EmbeddingModelSingleton:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self, model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
        """Load and cache the embedding model on first access."""
        if self._model is None:
            logger.info("Loading embedding model: %s", model_name)
            try:
                self._model = SentenceTransformer(model_name)
            except Exception as exc:
                logger.exception("Failed to load embedding model '%s'", model_name)
                raise EmbeddingModelError(
                    f"Could not load embedding model '{model_name}': {exc}"
                ) from exc
            logger.info("Embedding model loaded successfully")
        return self._model


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Return the shared SentenceTransformer model instance.

    Args:
        model_name: Hugging Face model name for SentenceTransformers.

    Returns:
        Loaded SentenceTransformer model.
    """
    return _EmbeddingModelSingleton().get_model(model_name)


def embed_text(text: str) -> list[float]:
    """Embed a single text string into a dense vector.

    Args:
        text: Input text to embed.

    Returns:
        Embedding vector as a list of floats.

    Raises:
        ValueError: If the input text is empty.
        EmbeddingModelError: If embedding generation fails.
    """
    if not text.strip():
        raise ValueError("Cannot embed empty text")

    model = get_embedding_model()

    try:
        embedding: NDArray[np.float32] = model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        logger.exception("Failed to embed text")
        raise EmbeddingModelError(f"Embedding failed: {exc}") from exc

    return embedding.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple text strings in a single batch.

    Args:
        texts: List of input texts to embed.

    Returns:
        List of embedding vectors.

    Raises:
        ValueError: If the input list is empty or contains only blank text.
        EmbeddingModelError: If embedding generation fails.
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts")

    valid_texts = [text for text in texts if text.strip()]
    if not valid_texts:
        raise ValueError("All provided texts are empty")

    model = get_embedding_model()

    try:
        embeddings: NDArray[np.float32] = model.encode(
            valid_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        logger.exception("Failed to embed %d text(s)", len(valid_texts))
        raise EmbeddingModelError(f"Batch embedding failed: {exc}") from exc

    return embeddings.tolist()
