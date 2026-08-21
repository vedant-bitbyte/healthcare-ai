"""Semantic retrieval over the healthcare document vector store."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..embeddings.embedding_model import EmbeddingModelError, embed_text
from src.vectorstore.chromadb_store import VectorStore
from .query_router import QueryCategory, QueryRouter

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
        self._query_router = QueryRouter()

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

        categories, preferred_sources = self._resolve_routing(query)
        self._log_routing_decision(categories, preferred_sources)

        try:
            query_embedding = embed_text(query)
        except EmbeddingModelError as exc:
            raise RetrievalError(f"Failed to embed query: {exc}") from exc

        collection = self._vector_store.get_collection()
        document_count = collection.count()

        if document_count == 0:
            logger.warning("Vector store is empty; no results to retrieve")
            return []

        where_filter, searchable_count = self._build_source_filter(
            collection,
            preferred_sources,
            document_count,
        )

        if searchable_count == 0:
            logger.warning("No searchable chunks available after routing filters")
            return []

        n_results = min(top_k, searchable_count)

        try:
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter is not None:
                query_kwargs["where"] = where_filter

            results: dict[str, Any] = collection.query(**query_kwargs)
        except Exception as exc:
            logger.exception("ChromaDB query failed")
            raise RetrievalError(f"Vector search failed: {exc}") from exc

        parsed = self._parse_query_results(results)

        logger.info("Retrieved %d result(s)", len(parsed))
        return parsed

    def _resolve_routing(self, query: str) -> tuple[list[QueryCategory], list[str]]:
        """Detect categories and preferred sources for a query."""
        categories = self._query_router.detect_categories(query)
        preferred_sources = self._preferred_sources_from_categories(categories)
        return categories, preferred_sources

    def _preferred_sources_from_categories(
        self,
        categories: list[QueryCategory],
    ) -> list[str]:
        """Build a de-duplicated preferred source list from detected categories."""
        preferred_sources: list[str] = []
        seen: set[str] = set()

        for category in categories:
            for source in self._query_router.get_sources_for_category(category):
                if source not in seen:
                    seen.add(source)
                    preferred_sources.append(source)

        return preferred_sources

    def _log_routing_decision(
        self,
        categories: list[QueryCategory],
        preferred_sources: list[str],
    ) -> None:
        """Log routing metadata before vector search."""
        category_labels = ", ".join(category.value for category in categories) or "none"
        source_labels = ", ".join(preferred_sources) or "none"

        logger.info("Query categories: %s", category_labels)
        logger.info("Preferred sources: %s", source_labels)

    def _build_source_filter(
        self,
        collection: Any,
        preferred_sources: list[str],
        document_count: int,
    ) -> tuple[dict[str, Any] | None, int]:
        """Build a ChromaDB metadata filter and count searchable chunks."""
        if not preferred_sources:
            logger.info("Filtered chunks available: %d", document_count)
            return None, document_count

        where_filter = {"source": {"$in": preferred_sources}}
        filtered_count = self._count_filtered_chunks(collection, where_filter)
        logger.info("Filtered chunks available: %d", filtered_count)
        return where_filter, filtered_count

    @staticmethod
    def _count_filtered_chunks(collection: Any, where_filter: dict[str, Any]) -> int:
        """Count chunks that match a ChromaDB metadata filter."""
        try:
            filtered = collection.get(where=where_filter, include=[])
        except Exception as exc:
            logger.exception("Failed to count filtered chunks")
            raise RetrievalError(f"Could not count filtered chunks: {exc}") from exc

        return len(filtered.get("ids", []))

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
