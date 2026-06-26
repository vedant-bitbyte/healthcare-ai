"""Unit tests for evaluate_model helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from evaluate_model import (
    EvaluationError,
    evaluate_model,
    load_evaluation_questions,
    model_to_results_filename,
    resolve_output_path,
)
from src.rag import RAGPipelineError


def test_model_to_results_filename() -> None:
    assert model_to_results_filename("phi3:mini") == "results_phi3.csv"
    assert model_to_results_filename("gemma3:4b") == "results_gemma3.csv"


def test_resolve_output_path(tmp_path: Path) -> None:
    output_path = resolve_output_path(tmp_path, "phi3:mini")
    assert output_path == tmp_path / "results_phi3.csv"


def test_load_evaluation_questions(tmp_path: Path) -> None:
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text(
        "id,category,question\n1,workforce,Which states have the fewest specialists?\n",
        encoding="utf-8",
    )

    dataframe = load_evaluation_questions(csv_path)
    assert len(dataframe) == 1
    assert dataframe.iloc[0]["category"] == "workforce"


def test_load_evaluation_questions_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text("id,question\n1,Sample?\n", encoding="utf-8")

    with pytest.raises(EvaluationError, match="Missing required columns"):
        load_evaluation_questions(csv_path)


def test_evaluate_model_skips_failed_questions_and_continues(tmp_path: Path) -> None:
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text(
        "id,category,question\n"
        "1,workforce,Question one?\n"
        "2,policy,Question two?\n",
        encoding="utf-8",
    )

    side_effects = [
        {
            "question": "Question one?",
            "answer": "Answer one",
            "sources": ["rhs_2020.csv"],
            "retrieved_chunks": [],
        },
        RAGPipelineError("Generation failed"),
    ]

    with patch("evaluate_model.run_rag_pipeline", side_effect=side_effects):
        results_df = evaluate_model(
            model="phi3:mini",
            questions_path=csv_path,
            results_dir=tmp_path,
        )

    assert len(results_df) == 1
    assert results_df.iloc[0]["Answer"] == "Answer one"
    assert (tmp_path / "results_phi3.csv").exists()
