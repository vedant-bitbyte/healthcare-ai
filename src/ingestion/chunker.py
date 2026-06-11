"""Split cleaned document text into overlapping chunks."""

from __future__ import annotations

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks using a recursive character splitter.

    Args:
        text: Cleaned document text to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of text chunks. Returns an empty list if input text is empty.
    """
    if not text.strip():
        logger.warning("Received empty text for chunking")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = splitter.split_text(text)

    logger.info(
        "Created %d chunk(s) (size=%d, overlap=%d)",
        len(chunks),
        chunk_size,
        chunk_overlap,
    )

    return chunks
