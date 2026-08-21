"""Compare evaluation results across Healthcare AI RAG models."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_PHI3_RESULTS = Path("evaluation/results_phi3.csv")
DEFAULT_GEMMA3_RESULTS = Path("evaluation/results_gemma3.csv")
DEFAULT_REPORTS_DIR = Path("evaluation/reports")
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

REQUIRED_RESULT_COLUMNS = {
    "Question",
    "Category",
    "Answer",
    "Sources",
    "Latency (seconds)",
}

LATENCY_COLUMN = "Latency (seconds)"
SOURCE_SEPARATOR = "; "

OVERALL_LATENCY_METRICS = (
    "avg_latency",
    "median_latency",
    "min_latency",
    "max_latency",
)
OVERALL_ANSWER_METRICS = (
    "avg_answer_length_words",
    "avg_sources",
)


class CompareModelsError(Exception):
    """Raised when model comparison setup or export fails."""


@dataclass(frozen=True)
class ModelResults:
    """Loaded evaluation results for a single model."""

    name: str
    dataframe: pd.DataFrame


@dataclass(frozen=True)
class ComparisonOutputs:
    """Paths to generated comparison artifacts."""

    metrics_csv: Path
    report_md: Path
    latency_plot: Path
    answer_length_plot: Path
    category_latency_plot: Path
    category_answer_length_plot: Path


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for comparison runs."""
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


def model_name_from_path(results_path: Path) -> str:
    """Derive a display name from a results CSV filename."""
    stem = results_path.stem
    if stem.startswith("results_"):
        return stem.removeprefix("results_")
    return stem


def load_results(results_path: str | Path) -> ModelResults:
    """Load and validate an evaluation results CSV.

    Args:
        results_path: Path to a model results CSV file.

    Returns:
        ModelResults containing the model name and validated DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        CompareModelsError: If required columns are missing or the file is empty.
    """
    path = Path(results_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Results file not found: {path}")

    try:
        dataframe = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise CompareModelsError(f"Results file is empty: {path}") from exc
    except OSError as exc:
        raise CompareModelsError(f"Could not read results CSV: {exc}") from exc

    missing_columns = REQUIRED_RESULT_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise CompareModelsError(
            f"Results file {path.name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise CompareModelsError(f"No evaluation rows found in {path.name}")

    enriched = _enrich_results(dataframe)
    model_name = model_name_from_path(path)
    logger.info(
        "Loaded %d result row(s) for model '%s' from %s",
        len(enriched),
        model_name,
        path.name,
    )
    return ModelResults(name=model_name, dataframe=enriched)


def count_answer_words(answer: Any) -> int:
    """Count words in an answer string."""
    if pd.isna(answer):
        return 0
    return len(str(answer).split())


def count_sources(sources: Any) -> int:
    """Count the number of cited sources in a results row."""
    if pd.isna(sources):
        return 0

    source_text = str(sources).strip()
    if not source_text:
        return 0

    return len([part for part in source_text.split(SOURCE_SEPARATOR) if part.strip()])


def _enrich_results(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add derived metrics used for aggregation and plotting."""
    enriched = dataframe.copy()
    enriched["answer_length_words"] = enriched["Answer"].map(count_answer_words)
    enriched["source_count"] = enriched["Sources"].map(count_sources)
    return enriched


def compute_overall_metrics(dataframe: pd.DataFrame) -> dict[str, float]:
    """Compute aggregate latency and answer metrics for one model."""
    latency = dataframe[LATENCY_COLUMN]
    return {
        "avg_latency": float(latency.mean()),
        "median_latency": float(latency.median()),
        "min_latency": float(latency.min()),
        "max_latency": float(latency.max()),
        "avg_answer_length_words": float(dataframe["answer_length_words"].mean()),
        "avg_sources": float(dataframe["source_count"].mean()),
    }


def compute_category_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Compute category-level latency and answer length metrics."""
    grouped = (
        dataframe.groupby("Category", sort=True)
        .agg(
            avg_latency=(LATENCY_COLUMN, "mean"),
            avg_answer_length_words=("answer_length_words", "mean"),
            question_count=("Question", "count"),
        )
        .reset_index()
    )
    return grouped


def build_metrics_dataframe(models: list[ModelResults]) -> pd.DataFrame:
    """Build a long-format metrics table for all models."""
    rows: list[dict[str, Any]] = []

    for model in models:
        overall = compute_overall_metrics(model.dataframe)
        for metric_name, value in overall.items():
            rows.append(
                {
                    "model": model.name,
                    "scope": "overall",
                    "category": "",
                    "metric": metric_name,
                    "value": round(value, 3),
                }
            )

        category_metrics = compute_category_metrics(model.dataframe)
        for _, category_row in category_metrics.iterrows():
            rows.append(
                {
                    "model": model.name,
                    "scope": "category",
                    "category": category_row["Category"],
                    "metric": "avg_latency",
                    "value": round(float(category_row["avg_latency"]), 3),
                }
            )
            rows.append(
                {
                    "model": model.name,
                    "scope": "category",
                    "category": category_row["Category"],
                    "metric": "avg_answer_length_words",
                    "value": round(float(category_row["avg_answer_length_words"]), 3),
                }
            )

    return pd.DataFrame(rows)


def save_metrics_csv(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Write comparison metrics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        metrics_df.to_csv(output_path, index=False, encoding="utf-8")
    except OSError as exc:
        raise CompareModelsError(f"Could not write metrics CSV: {exc}") from exc
    logger.info("Saved comparison metrics to %s", output_path)


def _format_metric_table(
    models: list[ModelResults],
    metric_names: tuple[str, ...],
    title: str,
) -> str:
    """Render a markdown table for selected overall metrics."""
    header = "| Metric | " + " | ".join(model.name for model in models) + " |"
    separator = "| --- | " + " | ".join("---" for _ in models) + " |"
    lines = [f"### {title}", "", header, separator]

    for metric_name in metric_names:
        values = []
        for model in models:
            value = compute_overall_metrics(model.dataframe)[metric_name]
            values.append(f"{value:.3f}")
        label = metric_name.replace("_", " ")
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.append("")
    return "\n".join(lines)


def _format_category_table(
    models: list[ModelResults],
    value_column: str,
    title: str,
) -> str:
    """Render a markdown table for category-level metrics."""
    categories = sorted(
        {
            category
            for model in models
            for category in model.dataframe["Category"].unique()
        }
    )
    header = "| Category | " + " | ".join(model.name for model in models) + " |"
    separator = "| --- | " + " | ".join("---" for _ in models) + " |"
    lines = [f"### {title}", "", header, separator]

    for category in categories:
        values = []
        for model in models:
            category_metrics = compute_category_metrics(model.dataframe)
            match = category_metrics.loc[category_metrics["Category"] == category]
            if match.empty:
                values.append("N/A")
            else:
                values.append(f"{float(match.iloc[0][value_column]):.3f}")
        lines.append(f"| {category} | " + " | ".join(values) + " |")

    lines.append("")
    return "\n".join(lines)


def build_report_markdown(
    models: list[ModelResults],
    outputs: ComparisonOutputs,
) -> str:
    """Build a markdown comparison report."""
    model_names = ", ".join(model.name for model in models)
    sections = [
        "# Model Comparison Report",
        "",
        f"Comparison of evaluation results for: **{model_names}**.",
        "",
        "## Overall Metrics",
        "",
        _format_metric_table(models, OVERALL_LATENCY_METRICS, "Latency (seconds)"),
        _format_metric_table(models, OVERALL_ANSWER_METRICS, "Answer Quality"),
        "## Category Metrics",
        "",
        _format_category_table(models, "avg_latency", "Average Latency by Category"),
        _format_category_table(
            models,
            "avg_answer_length_words",
            "Average Answer Length by Category (words)",
        ),
        "## Generated Artifacts",
        "",
        f"- Metrics CSV: `{outputs.metrics_csv.as_posix()}`",
        f"- Latency plot: `{outputs.latency_plot.as_posix()}`",
        f"- Answer length plot: `{outputs.answer_length_plot.as_posix()}`",
        f"- Category latency plot: `{outputs.category_latency_plot.as_posix()}`",
        (
            "- Category answer length plot: "
            f"`{outputs.category_answer_length_plot.as_posix()}`"
        ),
        "",
    ]
    return "\n".join(sections)


def save_report_markdown(report_text: str, output_path: Path) -> None:
    """Write the comparison report to markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(report_text, encoding="utf-8")
    except OSError as exc:
        raise CompareModelsError(f"Could not write comparison report: {exc}") from exc
    logger.info("Saved comparison report to %s", output_path)


def _model_overall_values(
    models: list[ModelResults],
    metric_names: tuple[str, ...],
) -> dict[str, list[float]]:
    """Collect overall metric values keyed by metric name."""
    values: dict[str, list[float]] = {metric: [] for metric in metric_names}
    for model in models:
        overall = compute_overall_metrics(model.dataframe)
        for metric in metric_names:
            values[metric].append(overall[metric])
    return values


def plot_latency(models: list[ModelResults], output_path: Path) -> None:
    """Plot overall latency metrics for each model."""
    metric_names = OVERALL_LATENCY_METRICS
    metric_values = _model_overall_values(models, metric_names)
    model_names = [model.name for model in models]

    x_positions = range(len(metric_names))
    bar_width = 0.35
    offsets = [-bar_width / 2, bar_width / 2]

    fig, axis = plt.subplots(figsize=(10, 6))
    for index, model_name in enumerate(model_names):
        values = [metric_values[metric][index] for metric in metric_names]
        positions = [x + offsets[index] for x in x_positions]
        axis.bar(positions, values, width=bar_width, label=model_name)

    axis.set_title("Overall Latency Comparison")
    axis.set_xlabel("Metric")
    axis.set_ylabel("Seconds")
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels([name.replace("_", " ") for name in metric_names], rotation=15)
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved latency plot to %s", output_path)


def plot_answer_length(models: list[ModelResults], output_path: Path) -> None:
    """Plot overall answer length and source count for each model."""
    metric_names = OVERALL_ANSWER_METRICS
    metric_values = _model_overall_values(models, metric_names)
    model_names = [model.name for model in models]

    x_positions = range(len(metric_names))
    bar_width = 0.35
    offsets = [-bar_width / 2, bar_width / 2]

    fig, axis = plt.subplots(figsize=(10, 6))
    for index, model_name in enumerate(model_names):
        values = [metric_values[metric][index] for metric in metric_names]
        positions = [x + offsets[index] for x in x_positions]
        axis.bar(positions, values, width=bar_width, label=model_name)

    axis.set_title("Overall Answer Length and Source Count")
    axis.set_xlabel("Metric")
    axis.set_ylabel("Average Value")
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels([name.replace("_", " ") for name in metric_names], rotation=15)
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved answer length plot to %s", output_path)


def _plot_category_comparison(
    models: list[ModelResults],
    value_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot a grouped bar chart for a category-level metric."""
    categories = sorted(
        {
            category
            for model in models
            for category in model.dataframe["Category"].unique()
        }
    )
    model_names = [model.name for model in models]
    x_positions = range(len(categories))
    bar_width = 0.35
    offsets = [-bar_width / 2, bar_width / 2]

    fig, axis = plt.subplots(figsize=(12, 6))
    for index, model in enumerate(models):
        category_metrics = compute_category_metrics(model.dataframe).set_index("Category")
        values = [
            float(category_metrics.loc[category, value_column])
            if category in category_metrics.index
            else 0.0
            for category in categories
        ]
        positions = [x + offsets[index] for x in x_positions]
        axis.bar(positions, values, width=bar_width, label=model_names[index])

    axis.set_title(title)
    axis.set_xlabel("Category")
    axis.set_ylabel(ylabel)
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(categories, rotation=20, ha="right")
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot to %s", output_path)


def plot_category_latency(models: list[ModelResults], output_path: Path) -> None:
    """Plot category-wise average latency for each model."""
    _plot_category_comparison(
        models=models,
        value_column="avg_latency",
        title="Average Latency by Category",
        ylabel="Seconds",
        output_path=output_path,
    )


def plot_category_answer_length(models: list[ModelResults], output_path: Path) -> None:
    """Plot category-wise average answer length for each model."""
    _plot_category_comparison(
        models=models,
        value_column="avg_answer_length_words",
        title="Average Answer Length by Category",
        ylabel="Words",
        output_path=output_path,
    )


def compare_models(
    phi3_results_path: str | Path = DEFAULT_PHI3_RESULTS,
    gemma3_results_path: str | Path = DEFAULT_GEMMA3_RESULTS,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> ComparisonOutputs:
    """Compare two model evaluation result files and write reports and plots.

    Args:
        phi3_results_path: Path to phi3 evaluation results CSV.
        gemma3_results_path: Path to gemma3 evaluation results CSV.
        reports_dir: Directory for metrics, report, and plot outputs.

    Returns:
        ComparisonOutputs with paths to all generated artifacts.
    """
    reports_path = Path(reports_dir).resolve()
    outputs = ComparisonOutputs(
        metrics_csv=reports_path / "comparison_metrics.csv",
        report_md=reports_path / "comparison_report.md",
        latency_plot=reports_path / "latency.png",
        answer_length_plot=reports_path / "answer_length.png",
        category_latency_plot=reports_path / "category_latency.png",
        category_answer_length_plot=reports_path / "category_answer_length.png",
    )

    models = [
        load_results(phi3_results_path),
        load_results(gemma3_results_path),
    ]

    metrics_df = build_metrics_dataframe(models)
    save_metrics_csv(metrics_df, outputs.metrics_csv)

    report_text = build_report_markdown(models, outputs)
    save_report_markdown(report_text, outputs.report_md)

    plot_latency(models, outputs.latency_plot)
    plot_answer_length(models, outputs.answer_length_plot)
    plot_category_latency(models, outputs.category_latency_plot)
    plot_category_answer_length(models, outputs.category_answer_length_plot)

    logger.info(
        "Model comparison complete for '%s' vs '%s'",
        models[0].name,
        models[1].name,
    )
    return outputs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Healthcare AI RAG model evaluation results.",
    )
    parser.add_argument(
        "--phi3-results",
        type=str,
        default=str(DEFAULT_PHI3_RESULTS),
        help=f"Path to phi3 results CSV (default: {DEFAULT_PHI3_RESULTS}).",
    )
    parser.add_argument(
        "--gemma3-results",
        type=str,
        default=str(DEFAULT_GEMMA3_RESULTS),
        help=f"Path to gemma3 results CSV (default: {DEFAULT_GEMMA3_RESULTS}).",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(DEFAULT_REPORTS_DIR),
        help=f"Directory for comparison outputs (default: {DEFAULT_REPORTS_DIR}).",
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
    """CLI entry point for model comparison."""
    args = _parse_args(argv)

    try:
        configure_logging(args.log_level)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        compare_models(
            phi3_results_path=args.phi3_results,
            gemma3_results_path=args.gemma3_results,
            reports_dir=args.reports_dir,
        )
    except (FileNotFoundError, CompareModelsError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error during model comparison")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
