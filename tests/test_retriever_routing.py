"""Unit tests for QueryRouter integration in Retriever."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.query_router import QueryCategory
from src.retrieval.retriever import Retriever


@pytest.fixture
def mock_collection() -> MagicMock:
    """Return a mocked ChromaDB collection."""
    collection = MagicMock()
    collection.count.return_value = 100
    collection.get.return_value = {"ids": ["id-1", "id-2", "id-3"]}
    collection.query.return_value = {
        "documents": [["workforce chunk text"]],
        "metadatas": [[{"chunk_id": 1, "source": "rhs_2020.csv"}]],
        "distances": [[0.12]],
    }
    return collection


@pytest.fixture
def retriever(mock_collection: MagicMock) -> Retriever:
    """Return a retriever with mocked vector store and embedding."""
    vector_store = MagicMock()
    vector_store.get_collection.return_value = mock_collection

    retriever = Retriever(vector_store=vector_store)

    with patch("src.retrieval.retriever.embed_text", return_value=[0.1, 0.2, 0.3]):
        yield retriever


class TestRetrieverRouting:
    """Tests for source-aware retrieval routing."""

    def test_retrieve_applies_source_filter_for_workforce_query(
        self,
        retriever: Retriever,
        mock_collection: MagicMock,
    ) -> None:
        with patch("src.retrieval.retriever.embed_text", return_value=[0.1, 0.2, 0.3]):
            results = retriever.retrieve("doctor shortage in Bihar", top_k=5)

        mock_collection.get.assert_called_once_with(
            where={"source": {"$in": ["rhs_2020.csv"]}},
            include=[],
        )
        mock_collection.query.assert_called_once()
        query_kwargs = mock_collection.query.call_args.kwargs
        assert query_kwargs["where"] == {"source": {"$in": ["rhs_2020.csv"]}}
        assert len(results) == 1
        assert results[0].source == "rhs_2020.csv"

    def test_retrieve_searches_full_collection_without_routing_match(
        self,
        retriever: Retriever,
        mock_collection: MagicMock,
    ) -> None:
        with patch("src.retrieval.retriever.embed_text", return_value=[0.1, 0.2, 0.3]):
            retriever.retrieve("general hospital information", top_k=5)

        mock_collection.get.assert_not_called()
        query_kwargs = mock_collection.query.call_args.kwargs
        assert "where" not in query_kwargs

    def test_retrieve_policy_query_filters_multiple_sources(
        self,
        retriever: Retriever,
        mock_collection: MagicMock,
    ) -> None:
        with patch("src.retrieval.retriever.embed_text", return_value=[0.1, 0.2, 0.3]):
            retriever.retrieve("Ayushman Bharat budget policy", top_k=5)

        query_kwargs = mock_collection.query.call_args.kwargs
        assert query_kwargs["where"] == {
            "source": {
                "$in": [
                    "National_Health_Policy_2017.pdf",
                    "Ayushman_Bharat_Guidelines.pdf",
                ]
            }
        }

    def test_retrieve_returns_empty_when_filtered_collection_is_empty(
        self,
        retriever: Retriever,
        mock_collection: MagicMock,
    ) -> None:
        mock_collection.get.return_value = {"ids": []}

        with patch("src.retrieval.retriever.embed_text", return_value=[0.1, 0.2, 0.3]):
            results = retriever.retrieve("doctor workforce", top_k=5)

        assert results == []
        mock_collection.query.assert_not_called()

    def test_public_retriever_api_unchanged(self) -> None:
        retriever = Retriever()
        assert hasattr(retriever, "retrieve")
        assert hasattr(retriever, "vector_store")

    def test_resolve_routing_detects_categories(self, retriever: Retriever) -> None:
        categories, preferred_sources = retriever._resolve_routing("maternal health pregnancy")
        assert QueryCategory.MATERNAL_HEALTH in categories
        assert preferred_sources == ["NFHS-5_National_Report.pdf"]
