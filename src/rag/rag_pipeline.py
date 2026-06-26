"""End-to-end RAG pipeline."""

from __future__ import annotations

import logging
from typing import Any

from ..retrieval.retriever import RetrievalError, Retriever
from .llm_client import DEFAULT_MODEL, LLMError, generate_response
from .prompt_builder import build_prompt

logger = logging.getLogger(__name__)

DEFAULT_RAG_TOP_K = 15


class RAGPipelineError(Exception):
    """Raised when RAG pipeline fails."""


class RAGPipeline:
    """Main RAG pipeline."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        model: str = DEFAULT_MODEL,
        top_k: int = DEFAULT_RAG_TOP_K,
    ) -> None:
        self._retriever = retriever or Retriever()
        self._model = model
        self._top_k = top_k

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    def run(self, question: str) -> dict[str, Any]:
        """
        Execute RAG pipeline.
        """

        if not question.strip():
            raise ValueError("Question cannot be empty")

        logger.info("Running RAG pipeline")

        try:
            retrieved_chunks = self._retriever.retrieve(
                question,
                top_k=self._top_k,
            )

        except RetrievalError as exc:
            raise RAGPipelineError(
                f"Retrieval failed: {exc}"
            ) from exc

        sources = _extract_sources(retrieved_chunks)

        logger.info(
            "Retrieved %d chunks from %d sources",
            len(retrieved_chunks),
            len(sources),
        )

        prompt = build_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        try:
            answer = generate_response(
                prompt=prompt,
                model=self._model,
            )

        except LLMError as exc:
            raise RAGPipelineError(
                f"Generation failed: {exc}"
            ) from exc

        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
        }

        logger.info("RAG pipeline completed")

        return result


def _extract_sources(
    retrieved_chunks: list[Any],
) -> list[str]:
    """
    Extract unique sources.
    """

    seen: set[str] = set()
    sources: list[str] = []

    for chunk in retrieved_chunks:
        source = str(chunk.source)

        if source not in seen:
            seen.add(source)
            sources.append(source)

    return sources


def run_rag_pipeline(
    question: str,
    model: str = DEFAULT_MODEL,
    top_k: int = DEFAULT_RAG_TOP_K,
) -> dict[str, Any]:
    """Run the RAG pipeline for a single question.

    Args:
        question: User healthcare question.
        model: Ollama model name for answer generation.
        top_k: Number of chunks to retrieve.

    Returns:
        RAG result with ``question``, ``answer``, ``sources``, and
        ``retrieved_chunks``.
    """
    pipeline = RAGPipeline(model=model, top_k=top_k)
    return pipeline.run(question)