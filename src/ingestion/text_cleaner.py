"""Clean and normalize raw text extracted from documents."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Normalize whitespace and line breaks in extracted document text.

    Processing steps:
        1. Normalize all line endings to ``\\n``.
        2. Strip trailing whitespace from each line.
        3. Collapse runs of spaces and tabs within a line to a single space.
        4. Collapse three or more consecutive blank lines to a single blank line.
        5. Strip leading and trailing whitespace from the full document.

    Args:
        text: Raw text extracted from a document.

    Returns:
        Cleaned text ready for chunking.
    """
    if not text:
        logger.warning("Received empty text for cleaning")
        return ""

    original_length = len(text)

    # Normalize Windows (\r\n) and old Mac (\r) line endings to Unix (\n).
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing whitespace on each line.
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))

    # Collapse repeated spaces/tabs within a line (preserve newlines).
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Collapse three or more consecutive blank lines into one blank line.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    cleaned = cleaned.strip()

    logger.info(
        "Text cleaned: %d -> %d character(s)",
        original_length,
        len(cleaned),
    )

    return cleaned
