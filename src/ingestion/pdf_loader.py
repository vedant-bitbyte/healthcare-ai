"""Load and extract text from PDF documents."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)


class PDFLoadError(Exception):
    """Raised when a PDF cannot be loaded or parsed."""


def load_pdf_text(pdf_path: str | Path) -> str:
    """Load a PDF file and return its full extracted text.

    Args:
        pdf_path: Path to the PDF file on disk.

    Returns:
        A single string containing all page text joined with newlines.

    Raises:
        PDFLoadError: If the file is missing, not a PDF, or cannot be parsed.
        FileNotFoundError: If the PDF path does not exist.
    """
    path = Path(pdf_path)

    if not path.exists():
        logger.error("PDF file not found: %s", path)
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        logger.error("Invalid file type (expected .pdf): %s", path)
        raise PDFLoadError(f"Expected a PDF file, got: {path.suffix}")

    logger.info("Loading PDF: %s", path.name)

    try:
        loader = PyPDFLoader(str(path))
        documents = loader.load()
    except Exception as exc:
        logger.exception("Failed to load PDF: %s", path)
        raise PDFLoadError(f"Could not load PDF '{path.name}': {exc}") from exc

    if not documents:
        logger.warning("PDF contains no extractable text: %s", path.name)
        return ""

    page_texts = [doc.page_content for doc in documents if doc.page_content.strip()]
    full_text = "\n".join(page_texts)

    logger.info(
        "Successfully loaded PDF '%s' (%d page(s), %d character(s))",
        path.name,
        len(documents),
        len(full_text),
    )

    return full_text
