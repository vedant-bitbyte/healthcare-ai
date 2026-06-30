"""Generate a manual evaluation template for Healthcare RAG model comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_QUESTIONS_PATH = Path("evaluation/evaluation_questions.csv")
DEFAULT_PHI3_RESULTS_PATH = Path("evaluation/results_phi3.csv")
DEFAULT_GEMMA3_RESULTS_PATH = Path("evaluation/results_gemma3.csv")
DEFAULT_OUTPUT_PATH = Path("evaluation/manual_evaluation_template.csv")

REQUIRED_QUESTION_COLUMNS = {"id", "category", "question"}
REQUIRED_RESULT_COLUMNS = {
    "Question",
    "Category",
    "Answer",
    "Sources",
    "Latency (seconds)",
}

# Output columns in the exact order required for manual evaluation.
TEMPLATE_COLUMNS: list[str] = [
    "Question ID",
    "Category",
    "Question",
    "Phi3 Answer",
    "Phi3 Sources",
    "Phi3 Latency (seconds)",
    "Phi3 Answer Length (words)",
    "Gemma3 Answer",
    "Gemma3 Sources",
    "Gemma3 Latency (seconds)",
    "Gemma3 Answer Length (words)",
    "Phi3 Faithfulness (1-5)",
    "Phi3 Completeness (1-5)",
    "Phi3 Conciseness (1-5)",
    "Phi3 Readability (1-5)",
    "Phi3 Overall Score (1-5)",
    "Gemma3 Faithfulness (1-5)",
    "Gemma3 Completeness (1-5)",
    "Gemma3 Conciseness (1-5)",
    "Gemma3 Readability (1-5)",
    "Gemma3 Overall Score (1-5)",
    "Winner",
    "Comments",
]

MANUAL_SCORE_COLUMNS = TEMPLATE_COLUMNS[11:21]
EMPTY_TRAILING_COLUMNS = TEMPLATE_COLUMNS[21:]


class ManualEvaluationTemplateError(Exception):
    """Raised when template generation fails due to invalid input data."""


def count_answer_words(answer: Any) -> int | str:
    """Return word count for an answer, or blank if the answer is missing."""
    if pd.isna(answer):
        return ""
    text = str(answer).strip()
    if not text:
        return ""
    return len(text.split())


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file and translate low-level errors into meaningful messages."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ManualEvaluationTemplateError(f"Input file is empty: {path}") from exc
    except OSError as exc:
        raise ManualEvaluationTemplateError(f"Could not read CSV file: {path}") from exc


def load_evaluation_questions(questions_path: str | Path) -> pd.DataFrame:
    """Load evaluation questions and preserve their original order."""
    path = Path(questions_path).resolve()
    questions_df = _read_csv(path)

    missing_columns = REQUIRED_QUESTION_COLUMNS - set(questions_df.columns)
    if missing_columns:
        raise ManualEvaluationTemplateError(
            f"{path.name} is missing required columns: {sorted(missing_columns)}"
        )

    if questions_df.empty:
        raise ManualEvaluationTemplateError(f"No questions found in {path.name}")

    ordered_questions = questions_df.copy()
    ordered_questions["question"] = ordered_questions["question"].astype(str).str.strip()
    ordered_questions = ordered_questions[ordered_questions["question"] != ""]

    if ordered_questions.empty:
        raise ManualEvaluationTemplateError(f"No valid questions found in {path.name}")

    return ordered_questions


def load_model_results(results_path: str | Path, model_label: str) -> pd.DataFrame:
    """Load and validate model evaluation results keyed by question text."""
    path = Path(results_path).resolve()
    results_df = _read_csv(path)

    missing_columns = REQUIRED_RESULT_COLUMNS - set(results_df.columns)
    if missing_columns:
        raise ManualEvaluationTemplateError(
            f"{path.name} is missing required columns: {sorted(missing_columns)}"
        )

    if results_df.empty:
        raise ManualEvaluationTemplateError(f"No results found in {path.name}")

    normalized = results_df.copy()
    normalized["Question"] = normalized["Question"].astype(str).str.strip()
    normalized = normalized[normalized["Question"] != ""]

    duplicate_questions = normalized["Question"].duplicated(keep=False)
    if duplicate_questions.any():
        duplicated_values = sorted(normalized.loc[duplicate_questions, "Question"].unique())
        print(
            f"Warning: {model_label} results contain duplicate questions; "
            f"using the first match for: {duplicated_values}",
            file=sys.stderr,
        )
        normalized = normalized.drop_duplicates(subset=["Question"], keep="first")

    return normalized.set_index("Question", drop=False)


def _lookup_result(
    results_by_question: pd.DataFrame,
    question: str,
    model_label: str,
) -> dict[str, Any]:
    """Fetch one model's output for a question, returning blanks if missing."""
    if question not in results_by_question.index:
        print(
            f"Warning: No {model_label} result found for question: {question}",
            file=sys.stderr,
        )
        return {
            "answer": "",
            "sources": "",
            "latency": "",
            "answer_length": "",
        }

    row = results_by_question.loc[question]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    answer = row["Answer"]
    return {
        "answer": "" if pd.isna(answer) else answer,
        "sources": "" if pd.isna(row["Sources"]) else row["Sources"],
        "latency": "" if pd.isna(row["Latency (seconds)"]) else row["Latency (seconds)"],
        "answer_length": count_answer_words(answer),
    }


def build_manual_evaluation_template(
    questions_df: pd.DataFrame,
    phi3_results_df: pd.DataFrame,
    gemma3_results_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge question metadata and both model outputs into one evaluation sheet."""
    rows: list[dict[str, Any]] = []

    for _, question_row in questions_df.iterrows():
        question_id = question_row["id"]
        category = question_row["category"]
        question = question_row["question"]

        phi3 = _lookup_result(phi3_results_df, question, "Phi3")
        gemma3 = _lookup_result(gemma3_results_df, question, "Gemma3")

        row = {
            "Question ID": question_id,
            "Category": category,
            "Question": question,
            "Phi3 Answer": phi3["answer"],
            "Phi3 Sources": phi3["sources"],
            "Phi3 Latency (seconds)": phi3["latency"],
            "Phi3 Answer Length (words)": phi3["answer_length"],
            "Gemma3 Answer": gemma3["answer"],
            "Gemma3 Sources": gemma3["sources"],
            "Gemma3 Latency (seconds)": gemma3["latency"],
            "Gemma3 Answer Length (words)": gemma3["answer_length"],
        }

        # Manual scoring and comparison columns are intentionally left blank.
        for column in MANUAL_SCORE_COLUMNS + EMPTY_TRAILING_COLUMNS:
            row[column] = ""

        rows.append(row)

    template_df = pd.DataFrame(rows, columns=TEMPLATE_COLUMNS)
    return template_df


def save_template(template_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Write the manual evaluation template to CSV."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        template_df.to_csv(path, index=False, encoding="utf-8")
    except OSError as exc:
        raise ManualEvaluationTemplateError(
            f"Could not write manual evaluation template: {exc}"
        ) from exc

    return path


def create_manual_evaluation_template(
    questions_path: str | Path = DEFAULT_QUESTIONS_PATH,
    phi3_results_path: str | Path = DEFAULT_PHI3_RESULTS_PATH,
    gemma3_results_path: str | Path = DEFAULT_GEMMA3_RESULTS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Load inputs, build the template, save it, and print a summary."""
    questions_df = load_evaluation_questions(questions_path)
    phi3_results_df = load_model_results(phi3_results_path, "Phi3")
    gemma3_results_df = load_model_results(gemma3_results_path, "Gemma3")

    template_df = build_manual_evaluation_template(
        questions_df=questions_df,
        phi3_results_df=phi3_results_df,
        gemma3_results_df=gemma3_results_df,
    )

    saved_path = save_template(template_df, output_path)

    print(f"Loaded {len(questions_df)} evaluation questions")
    print("Loaded Phi3 results")
    print("Loaded Gemma3 results")
    print(f"Created {saved_path.name} successfully")
    print(f"Rows: {len(template_df)}")

    return template_df


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a manual evaluation template CSV for comparing "
            "Phi3 Mini and Gemma3 4B Healthcare RAG outputs."
        ),
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=str(DEFAULT_QUESTIONS_PATH),
        help=f"Path to evaluation questions CSV (default: {DEFAULT_QUESTIONS_PATH}).",
    )
    parser.add_argument(
        "--phi3-results",
        type=str,
        default=str(DEFAULT_PHI3_RESULTS_PATH),
        help=f"Path to Phi3 results CSV (default: {DEFAULT_PHI3_RESULTS_PATH}).",
    )
    parser.add_argument(
        "--gemma3-results",
        type=str,
        default=str(DEFAULT_GEMMA3_RESULTS_PATH),
        help=(
            f"Path to Gemma3 results CSV (default: {DEFAULT_GEMMA3_RESULTS_PATH})."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)

    try:
        create_manual_evaluation_template(
            questions_path=args.questions,
            phi3_results_path=args.phi3_results,
            gemma3_results_path=args.gemma3_results,
            output_path=args.output,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ManualEvaluationTemplateError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print("Error: Unexpected failure while creating manual evaluation template.", file=sys.stderr)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
