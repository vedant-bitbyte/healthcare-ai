"""End-to-end document ingestion pipeline for PDF files."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .chunker import chunk_text
from .pdf_loader import PDFLoadError, load_pdf_text
from .text_cleaner import clean_text

logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for CLI and pipeline execution.

    Args:
        level: Logging level name (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_LOG_DATE_FORMAT,
        stream=sys.stdout,
        force=True,
    )


def validate_pdf_path(pdf_path: str | Path) -> Path:
    """Validate that the input path refers to an existing PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Resolved absolute path to the PDF.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path is not a file or does not have a ``.pdf`` extension.
    """
    path = Path(pdf_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    return path


def validate_input_directory(input_dir: str | Path) -> Path:
    """Validate that the input path refers to an existing directory.

    Args:
        input_dir: Path to a folder containing PDF files.

    Returns:
        Resolved absolute path to the directory.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If the path exists but is not a directory.
    """
    path = Path(input_dir).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Input directory not found: {path}")

    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    return path


def discover_pdf_files(input_dir: Path) -> list[Path]:
    """Find all PDF files in a directory (non-recursive).

    Args:
        input_dir: Validated directory to scan.

    Returns:
        Sorted list of PDF file paths. Empty if no PDFs are found.
    """
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", input_dir)

    return pdf_files


def resolve_output_path(
    pdf_path: Path,
    output_path: str | Path | None = None,
) -> Path:
    """Resolve the JSON output path, creating the default folder when needed.

    When ``output_path`` is omitted, chunks are written to::

        data/processed/<pdf_stem>_chunks.json

    Args:
        pdf_path: Validated path to the source PDF.
        output_path: Optional explicit output file or directory path.

    Returns:
        Resolved path where chunk JSON will be saved.
    """
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return (DEFAULT_OUTPUT_DIR / f"{pdf_path.stem}_chunks.json").resolve()

    resolved = Path(output_path).resolve()

    if resolved.is_dir() or resolved.suffix == "":
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved / f"{pdf_path.stem}_chunks.json"

    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _build_chunk_records(chunks: list[str], source_filename: str) -> list[dict[str, Any]]:
    """Convert raw text chunks into structured records for JSON export."""
    return [
        {
            "chunk_id": index,
            "source": source_filename,
            "text": chunk,
        }
        for index, chunk in enumerate(chunks, start=1)
    ]


def _save_chunks_json(chunks: list[dict[str, Any]], output_path: Path) -> None:
    """Write chunk records to a JSON file with pretty formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    logger.info("Saved %d chunk(s) to %s", len(chunks), output_path)


def run_ingestion_pipeline(
    pdf_path: str | Path,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    """Run the full ingestion pipeline: load, clean, chunk, and save.

    Args:
        pdf_path: Path to the source PDF file.
        output_path: Path where the JSON chunk file will be written.

    Returns:
        A list of chunk records in the format::

            [
                {
                    "chunk_id": 1,
                    "source": "filename.pdf",
                    "text": "chunk content"
                }
            ]

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        PDFLoadError: If the PDF cannot be loaded or parsed.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)

    logger.info("Starting ingestion pipeline for '%s'", pdf_path.name)

    raw_text = load_pdf_text(pdf_path)
    cleaned_text = clean_text(raw_text)
    text_chunks = chunk_text(cleaned_text)
    chunk_records = _build_chunk_records(text_chunks, pdf_path.name)

    _save_chunks_json(chunk_records, output_path)

    logger.info(
        "Ingestion pipeline complete for '%s' -> '%s'",
        pdf_path.name,
        output_path,
    )

    return chunk_records


def run_batch_ingestion_pipeline(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Process every PDF in a directory through the ingestion pipeline.

    Each PDF is saved as its own JSON file under ``data/processed/`` by default::

        data/processed/<filename>_chunks.json

    Args:
        input_dir: Directory containing PDF files to process.
        output_dir: Optional output directory for all JSON files.

    Returns:
        A summary dict with keys ``succeeded``, ``failed``, and ``total_chunks``.
    """
    input_dir = validate_input_directory(input_dir)
    pdf_files = discover_pdf_files(input_dir)

    summary: dict[str, Any] = {
        "succeeded": [],
        "failed": [],
        "total_chunks": 0,
    }

    if not pdf_files:
        return summary

    logger.info("Batch ingestion started: %d PDF file(s) in %s", len(pdf_files), input_dir)

    for pdf_file in pdf_files:
        try:
            validated_pdf = validate_pdf_path(pdf_file)
            output_path = resolve_output_path(validated_pdf, output_dir)
            chunk_records = run_ingestion_pipeline(validated_pdf, output_path)

            summary["succeeded"].append(
                {
                    "pdf": str(validated_pdf),
                    "output": str(output_path),
                    "chunks": len(chunk_records),
                }
            )
            summary["total_chunks"] += len(chunk_records)
        except (FileNotFoundError, PDFLoadError, ValueError) as exc:
            logger.error("Failed to process '%s': %s", pdf_file.name, exc)
            summary["failed"].append({"pdf": str(pdf_file), "error": str(exc)})
        except Exception:
            logger.exception("Unexpected error processing '%s'", pdf_file.name)
            summary["failed"].append(
                {"pdf": str(pdf_file), "error": "Unexpected error (see logs)"}
            )

    logger.info(
        "Batch ingestion complete: %d succeeded, %d failed, %d total chunk(s)",
        len(summary["succeeded"]),
        len(summary["failed"]),
        summary["total_chunks"],
    )

    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF(s): extract text, clean, chunk, and save as JSON.",
    )
    parser.add_argument(
        "input_path",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Path to a PDF file or a folder of PDFs. "
            f"Use --batch to process all PDFs in {DEFAULT_INPUT_DIR}."
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help=f"Process all PDF files in {DEFAULT_INPUT_DIR}.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSON file (single PDF) or output directory (batch). "
            f"Defaults to {DEFAULT_OUTPUT_DIR}/<filename>_chunks.json"
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the document ingestion pipeline.

    Args:
        argv: Optional argument list for testing. Uses ``sys.argv`` when omitted.

    Returns:
        Process exit code (``0`` on success, ``1`` on failure).
    """
    args = _parse_args(argv)

    try:
        configure_logging(args.log_level)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.batch and args.input_path:
        logger.error("Use either --batch or an input path, not both.")
        return 1

    if args.batch:
        return _run_batch_cli(DEFAULT_INPUT_DIR, args.output)

    if args.input_path is None:
        logger.error(
            "No input provided. Pass a PDF path, a folder path, or use --batch."
        )
        return 1

    input_path = Path(args.input_path).resolve()

    if input_path.is_dir():
        return _run_batch_cli(input_path, args.output)

    return _run_single_cli(input_path, args.output)


def _run_single_cli(pdf_path: Path, output: str | None) -> int:
    """Process a single PDF from the CLI."""
    try:
        validated_pdf = validate_pdf_path(pdf_path)
        output_path = resolve_output_path(validated_pdf, output)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Input PDF: %s", validated_pdf)
    logger.info("Output JSON: %s", output_path)

    try:
        chunk_records = run_ingestion_pipeline(validated_pdf, output_path)
    except (FileNotFoundError, PDFLoadError) as exc:
        logger.error("Ingestion failed: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error during ingestion")
        return 1

    logger.info("Successfully processed %d chunk(s)", len(chunk_records))
    return 0


def _run_batch_cli(input_dir: Path, output: str | None) -> int:
    """Process all PDFs in a directory from the CLI."""
    if output and Path(output).suffix.lower() == ".json":
        logger.error(
            "Batch mode requires --output to be a directory, not a single .json file."
        )
        return 1

    try:
        validated_dir = validate_input_directory(input_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Input directory: %s", validated_dir)
    logger.info("Output directory: %s", output or DEFAULT_OUTPUT_DIR)

    summary = run_batch_ingestion_pipeline(validated_dir, output)

    if not summary["succeeded"] and not summary["failed"]:
        logger.warning("No PDF files were processed.")
        return 1

    if summary["failed"]:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
