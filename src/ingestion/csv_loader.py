"""Load healthcare CSV data and convert rows into structured chunk records."""

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

CSV_COLUMNS: list[str] = [
    "SubCenters",
    "PHCs",
    "CHCs",
    "ANM/Health_Worker_Female",
    "Doctors",
    "Specialists",
    "Radiographers",
    "Pharmacists",
    "LabTechnicians",
    "NursingStaff",
]

INFRASTRUCTURE_FIELDS: list[tuple[str, str]] = [
    ("SubCenters", "SubCenters"),
    ("PHCs", "PHCs"),
    ("CHCs", "CHCs"),
]

WORKFORCE_FIELDS: list[tuple[str, str]] = [
    ("Doctors", "Doctors"),
    ("Specialists", "Specialists"),
    ("Pharmacists", "Pharmacists"),
    ("LabTechnicians", "Lab Technicians"),
    ("NursingStaff", "Nursing Staff"),
]

PURPOSE_TEXT = (
    "Purpose:\n"
    "This information can be used for healthcare workforce planning,\n"
    "resource allocation, specialist shortage identification,\n"
    "infrastructure development and budget recommendations."
)

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


def _format_metric_line(label: str, value: int | None) -> str:
    """Format a single metric as a bullet line for structured profiles."""
    display_value = str(value) if value is not None else "Not available"
    return f"- {label}: {display_value}"


def row_to_statement(row: pd.Series) -> str | None:
    """Convert one CSV row into a retrieval-optimized healthcare profile.

    Args:
        row: A pandas Series representing one state/UT record.

    Returns:
        Structured healthcare profile text, or ``None`` if the row has no usable data.
    """
    state = str(row.get(STATE_COLUMN, "")).strip()
    if not state:
        logger.warning("Skipping row with missing state/UT name")
        return None

    infrastructure_values = {
        column: _parse_numeric_value(row.get(column))
        for column, _ in INFRASTRUCTURE_FIELDS
    }
    workforce_values = {
        column: _parse_numeric_value(row.get(column))
        for column, _ in WORKFORCE_FIELDS
    }

    if not any(value is not None for value in infrastructure_values.values()) and not any(
        value is not None for value in workforce_values.values()
    ):
        logger.warning("Skipping row for '%s': no valid metric values", state)
        return None

    infrastructure_lines = [
        _format_metric_line(label, infrastructure_values[column])
        for column, label in INFRASTRUCTURE_FIELDS
    ]
    workforce_lines = [
        _format_metric_line(label, workforce_values[column])
        for column, label in WORKFORCE_FIELDS
    ]

    return (
        f"Healthcare workforce profile for {state}\n\n"
        f"Infrastructure:\n"
        f"{chr(10).join(infrastructure_lines)}\n\n"
        f"Workforce:\n"
        f"{chr(10).join(workforce_lines)}\n\n"
        f"{PURPOSE_TEXT}"
    )


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

    required_columns = [STATE_COLUMN, *CSV_COLUMNS]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise CSVLoadError(f"Missing required columns: {', '.join(missing_columns)}")

    logger.info("Loaded %d row(s) from %s", len(dataframe), path.name)
    return dataframe


def build_chunk_records(
    dataframe: pd.DataFrame,
    source: str = DEFAULT_SOURCE_NAME,
) -> list[dict[str, Any]]:
    """Convert CSV rows into retrieval-optimized chunk records."""
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
    """Load RHS CSV data and save structured healthcare profile chunks as JSON."""
    dataframe = load_rhs_csv(csv_path)
    records = build_chunk_records(dataframe, source=source)
    save_chunk_records(records, output_path)
    return records


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert RHS CSV rows into structured healthcare profile JSON chunks.",
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
