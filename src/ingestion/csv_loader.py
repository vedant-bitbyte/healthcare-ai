"""Load healthcare CSV data and convert rows into natural-language chunk records."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = Path("data/raw/rhs_2020.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/rhs_chunks.json")
DEFAULT_SOURCE_NAME = "rhs_2020.csv"

STATE_COLUMN = "State/UT"

METRIC_COLUMNS: list[tuple[str, str]] = [
    ("SubCenters", "SubCenters"),
    ("PHCs", "PHCs"),
    ("CHCs", "CHCs"),
    ("ANM/Health_Worker_Female", "ANM/health workers"),
    ("Doctors", "doctors"),
    ("Specialists", "specialists"),
    ("Radiographers", "radiographers"),
    ("Pharmacists", "pharmacists"),
    ("LabTechnicians", "lab technicians"),
    ("NursingStaff", "nursing staff"),
]

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class CSVLoadError(Exception):
    """Raised when a CSV file cannot be loaded or processed."""


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for CLI execution."""
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


def _parse_numeric_value(value: Any) -> int | None:
    """Convert a CSV cell value to an integer, ignoring missing or N/A values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.upper() in {"N APP", "N/A", "NA", "NOT AVAILABLE"}:
        return None

    try:
        return int(float(text))
    except ValueError:
        logger.warning("Skipping non-numeric value: %r", value)
        return None


def _format_metric_phrase(label: str, value: int) -> str:
    """Format a single metric as a natural-language phrase fragment."""
    return f"{value} {label}"


def _join_phrases(phrases: list[str]) -> str:
    """Join phrase fragments using commas and a final 'and'."""
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def row_to_statement(row: pd.Series) -> str | None:
    """Convert one CSV row into a natural-language healthcare statement.

    Example::

        "Bihar has 9112 SubCenters, 1702 PHCs, 57 CHCs, 15656 ANM/health workers,
        1745 doctors, 124 specialists, 3 radiographers, 492 pharmacists,
        438 lab technicians and 1346 nursing staff."

    Args:
        row: A pandas Series representing one state/UT record.

    Returns:
        Natural-language statement, or ``None`` if the row has no usable data.
    """
    state = str(row.get(STATE_COLUMN, "")).strip()
    if not state:
        logger.warning("Skipping row with missing state/UT name")
        return None

    phrases: list[str] = []
    for column, label in METRIC_COLUMNS:
        value = _parse_numeric_value(row.get(column))
        if value is not None:
            phrases.append(_format_metric_phrase(label, value))

    if not phrases:
        logger.warning("Skipping row for '%s': no valid metric values", state)
        return None

    return f"{state} has {_join_phrases(phrases)}."


def load_rhs_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load the RHS CSV file into a pandas DataFrame.

    Args:
        csv_path: Path to the RHS CSV file.

    Returns:
        Loaded DataFrame with expected healthcare columns.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        CSVLoadError: If the file cannot be read or is missing required columns.
    """
    path = Path(csv_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if path.suffix.lower() != ".csv":
        raise CSVLoadError(f"Expected a .csv file, got: {path.suffix}")

    logger.info("Loading CSV: %s", path.name)

    try:
        dataframe = pd.read_csv(path)
    except Exception as exc:
        logger.exception("Failed to read CSV: %s", path)
        raise CSVLoadError(f"Could not read CSV '{path.name}': {exc}") from exc

    required_columns = [STATE_COLUMN, *[column for column, _ in METRIC_COLUMNS]]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise CSVLoadError(f"Missing required columns: {', '.join(missing_columns)}")

    logger.info("Loaded %d row(s) from %s", len(dataframe), path.name)
    return dataframe


def build_chunk_records(
    dataframe: pd.DataFrame,
    source: str = DEFAULT_SOURCE_NAME,
) -> list[dict[str, Any]]:
    """Convert CSV rows into ingestion-compatible chunk records.

    Args:
        dataframe: RHS healthcare DataFrame.
        source: Source filename stored in each chunk record.

    Returns:
        List of chunk dictionaries with ``chunk_id``, ``source``, and ``text``.
    """
    records: list[dict[str, Any]] = []
    chunk_id = 1

    for _, row in dataframe.iterrows():
        statement = row_to_statement(row)
        if statement is None:
            continue

        records.append(
            {
                "chunk_id": chunk_id,
                "source": source,
                "text": statement,
            }
        )
        chunk_id += 1

    logger.info("Built %d chunk record(s) from CSV data", len(records))
    return records


def save_chunk_records(
    records: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Save chunk records to a JSON file.

    Args:
        records: Chunk records to persist.
        output_path: Destination JSON file path.

    Raises:
        CSVLoadError: If the file cannot be written.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(records, file, indent=2, ensure_ascii=False)
    except OSError as exc:
        logger.exception("Failed to write chunk JSON: %s", path)
        raise CSVLoadError(f"Could not write output file '{path}': {exc}") from exc

    logger.info("Saved %d chunk(s) to %s", len(records), path)


def run_csv_loader(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    source: str = DEFAULT_SOURCE_NAME,
) -> list[dict[str, Any]]:
    """Load RHS CSV data and save natural-language chunks as JSON.

    Args:
        csv_path: Path to the input CSV file.
        output_path: Path for the output JSON chunk file.
        source: Source label stored in each chunk record.

    Returns:
        List of generated chunk records.
    """
    dataframe = load_rhs_csv(csv_path)
    records = build_chunk_records(dataframe, source=source)
    save_chunk_records(records, output_path)
    return records


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert RHS CSV rows into natural-language JSON chunks.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV_PATH),
        help=f"Input CSV path (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH})",
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
    """CLI entry point for the CSV loader."""
    args = _parse_args(argv)

    try:
        configure_logging(args.log_level)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    csv_path = Path(args.csv)

    try:
        records = run_csv_loader(
            csv_path=csv_path,
            output_path=args.output,
            source=csv_path.name,
        )
    except (FileNotFoundError, CSVLoadError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error during CSV loading")
        return 1

    if not records:
        logger.error("No chunk records were generated")
        return 1

    logger.info("CSV loader complete: %d chunk(s) written", len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
