"""Batch evaluation runner for Healthcare AI RAG models."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.rag import RAGPipelineError, run_rag_pipeline

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS_PATH = Path("evaluation/evaluation_questions.csv")
DEFAULT_RESULTS_DIR = Path("evaluation")
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

REQUIRED_QUESTION_COLUMNS = {"category", "question"}


class EvaluationError(Exception):
    """Raised when evaluation setup or export fails."""


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for evaluation runs."""
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


def model_to_results_filename(model: str) -> str:
    """Convert an Ollama model name to a results filename slug.

    Examples:
        phi3:mini -> results_phi3.csv
        gemma3:4b -> results_gemma3.csv
    """
    slug = model.split(":")[0].strip().lower()
    slug = re.sub(r"[^\w.-]", "_", slug)
    if not slug:
        raise ValueError(f"Invalid model name for output file: {model}")
    return f"results_{slug}.csv"


def resolve_output_path(results_dir: Path, model: str) -> Path:
    """Resolve the CSV output path for a model evaluation run."""
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / model_to_results_filename(model)


def load_evaluation_questions(questions_path: str | Path) -> pd.DataFrame:
    """Load evaluation questions from a CSV file.

    Args:
        questions_path: Path to the evaluation questions CSV.

    Returns:
        DataFrame containing at least ``category`` and ``question`` columns.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        EvaluationError: If required columns are missing.
    """
    path = Path(questions_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Evaluation questions file not found: {path}")

    try:
        dataframe = pd.read_csv(path)
    except Exception as exc:
        raise EvaluationError(f"Could not read evaluation questions CSV: {exc}") from exc

    missing_columns = REQUIRED_QUESTION_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise EvaluationError(
            f"Missing required columns in evaluation CSV: {', '.join(sorted(missing_columns))}"
        )

    dataframe = dataframe.dropna(subset=["question"])
    dataframe["question"] = dataframe["question"].astype(str).str.strip()
    dataframe = dataframe[dataframe["question"] != ""]

    if dataframe.empty:
        raise EvaluationError("No valid evaluation questions found in CSV")

    logger.info("Loaded %d evaluation question(s) from %s", len(dataframe), path.name)
    return dataframe


def _format_sources(sources: list[str]) -> str:
    """Serialize source filenames for CSV export."""
    return "; ".join(sources)


def _evaluate_single_question(
    question: str,
    category: str,
    model: str,
) -> dict[str, Any]:
    """Run RAG for one question and capture latency."""
    start_time = time.perf_counter()

    result = run_rag_pipeline(question=question, model=model)
    latency_seconds = time.perf_counter() - start_time

    return {
        "Question": question,
        "Category": category,
        "Answer": result["answer"],
        "Sources": _format_sources(result["sources"]),
        "Latency (seconds)": round(latency_seconds, 3),
    }


def evaluate_model(
    model: str,
    questions_path: str | Path = DEFAULT_QUESTIONS_PATH,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    """Evaluate a RAG model against all questions in the evaluation CSV.

    Args:
        model: Ollama model name passed to the RAG pipeline.
        questions_path: Path to evaluation questions CSV.
        results_dir: Directory where results CSV will be written.

    Returns:
        DataFrame containing evaluation results.

    Raises:
        EvaluationError: If setup fails or no questions succeed.
    """
    questions_df = load_evaluation_questions(questions_path)
    output_path = resolve_output_path(Path(results_dir), model)
    total_questions = len(questions_df)

    logger.info("Starting evaluation for model '%s'", model)
    logger.info("Output file: %s", output_path)

    results: list[dict[str, Any]] = []
    failed_count = 0

    for question_index, (_, row) in enumerate(questions_df.iterrows(), start=1):
        question = str(row["question"]).strip()
        category = str(row.get("category", "")).strip()

        logger.info(
            "Evaluating question %d/%d [%s]: %s",
            question_index,
            total_questions,
            category or "uncategorized",
            question,
        )

        try:
            record = _evaluate_single_question(
                question=question,
                category=category,
                model=model,
            )
            results.append(record)
            logger.info(
                "Completed question %d/%d in %.3f seconds",
                question_index,
                total_questions,
                record["Latency (seconds)"],
            )
        except (RAGPipelineError, ValueError) as exc:
            failed_count += 1
            logger.error(
                "Skipping question %d/%d due to error: %s",
                question_index,
                total_questions,
                exc,
            )
        except Exception:
            failed_count += 1
            logger.exception(
                "Skipping question %d/%d due to unexpected error",
                question_index,
                total_questions,
            )

    if not results:
        raise EvaluationError(
            f"All {total_questions} evaluation question(s) failed for model '{model}'"
        )

    results_df = pd.DataFrame(results)

    try:
        results_df.to_csv(output_path, index=False, encoding="utf-8")
    except OSError as exc:
        raise EvaluationError(f"Could not write results CSV: {exc}") from exc

    logger.info(
        "Evaluation complete for model '%s': %d succeeded, %d failed, saved to %s",
        model,
        len(results),
        failed_count,
        output_path,
    )

    return results_df


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a Healthcare AI RAG model against benchmark questions.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ollama model name (e.g. phi3:mini, gemma3:4b).",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=str(DEFAULT_QUESTIONS_PATH),
        help=f"Path to evaluation questions CSV (default: {DEFAULT_QUESTIONS_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Directory for results CSV files (default: {DEFAULT_RESULTS_DIR}).",
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
    """CLI entry point for model evaluation."""
    args = _parse_args(argv)

    try:
        configure_logging(args.log_level)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        evaluate_model(
            model=args.model,
            questions_path=args.questions,
            results_dir=args.output_dir,
        )
    except (FileNotFoundError, EvaluationError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error during model evaluation")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
