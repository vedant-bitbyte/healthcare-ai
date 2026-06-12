"""Semantic retrieval over the healthcare document vector store."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..embeddings.embedding_model import EmbeddingModelError, embed_text
from ..embeddings.vector_store import VectorStore

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 15


class RetrievalError(Exception):
    """Raised when a retrieval query fails."""


@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieved document chunk."""

    text: str
    chunk_id: int
    source: str
    distance: float


class Retriever:
    """Query the ChromaDB vector store and return ranked chunk matches."""

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        """Initialize the retriever.

        Args:
            vector_store: Optional vector store instance. A default store is
                created when none is provided.
        """
        self._vector_store = vector_store or VectorStore()

    @property
    def vector_store(self) -> VectorStore:
        """Return the underlying vector store."""
        return self._vector_store

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievalResult]:
        """Search the vector database for chunks similar to the query.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of results to return.

        Returns:
            Ranked list of retrieval results, best match first.

        Raises:
            ValueError: If the query is empty or ``top_k`` is invalid.
            RetrievalError: If the search fails.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        logger.info("Retrieving top %d result(s) for query", top_k)

        try:
            query_embedding = embed_text(query)
        except EmbeddingModelError as exc:
            raise RetrievalError(f"Failed to embed query: {exc}") from exc

        collection = self._vector_store.get_collection()
        document_count = collection.count()

        if document_count == 0:
            logger.warning("Vector store is empty; no results to retrieve")
            return []

        n_results = min(top_k, document_count)

        try:
            results: dict[str, Any] = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.exception("ChromaDB query failed")
            raise RetrievalError(f"Vector search failed: {exc}") from exc

        parsed = self._parse_query_results(results)

        logger.info("Retrieved %d result(s)", len(parsed))
        return parsed

    @staticmethod
    def _parse_query_results(results: dict[str, Any]) -> list[RetrievalResult]:
        """Convert raw ChromaDB query output into ``RetrievalResult`` objects."""
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        parsed: list[RetrievalResult] = []

        for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
            if document is None or metadata is None:
                continue

            parsed.append(
                RetrievalResult(
                    text=document,
                    chunk_id=int(metadata.get("chunk_id", -1)),
                    source=str(metadata.get("source", "unknown")),
                    distance=float(distance),
                )
            )

        return parsed
