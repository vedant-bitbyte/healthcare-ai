"""Build grounded prompts for healthcare RAG generation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = """
You are an Indian Healthcare Policy Assistant.

STRICT RULES:

1. Answer ONLY using the retrieved context.
2. Never use your own knowledge.
3. Never invent facts, statistics, policies, state names, or recommendations.
4. If the answer is not explicitly supported by the context, reply:

"The available healthcare documents do not contain sufficient evidence to answer this question."

5. Prefer evidence from:
   - rhs_2020.csv
   - Health Ministry Annual Reports
   - NFHS Reports
   - National Health Policy

6. Structure responses as:

Answer:
...

Supporting Evidence:
...

Sources Used:
...

7. Be concise and factual.
"""


def _format_context_block(index: int, chunk: RetrievalResult) -> str:
    """Format retrieved chunk for prompt."""

    return (
        f"\n[Source: {chunk.source}]\n"
        f"{chunk.text}\n"
    )


def build_prompt(
    question: str,
    retrieved_chunks: list[RetrievalResult],
) -> str:
    """
    Build prompt for RAG generation.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if retrieved_chunks:
        context_text = "\n".join(
            _format_context_block(i, chunk)
            for i, chunk in enumerate(retrieved_chunks, start=1)
        )

        logger.info(
            "Built prompt with %d context chunks",
            len(retrieved_chunks),
        )

    else:
        context_text = (
            "No relevant healthcare documents were retrieved."
        )

        logger.warning(
            "Prompt built without retrieved context"
        )

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

Retrieved Context:

{context_text}

User Question:

{question}

Answer:
"""

    return prompt