"""Unit tests for evaluation dataset loading, schemas, and validation script."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lekka_ai.eval.dataset import load_dataset
from lekka_ai.eval.schemas import AdversarialExample, RiskRecallExample
from lekka_ai.eval.validate import validate_all_datasets


def test_validate_all_datasets_pass() -> None:
    """All benchmark dataset files pass schema and semantic validation."""
    ok, results = validate_all_datasets([Path("eval/datasets")])
    assert ok is True
    assert len(results) >= 4

    for path, (file_ok, row_count, errors) in results.items():
        assert file_ok is True, (
            f"Dataset {path} failed validation with errors: {errors}"
        )
        assert row_count > 0
        assert len(errors) == 0


def test_risk_recall_schema_validation() -> None:
    """RiskRecallExample enforces required fields and non-empty risk themes."""
    valid_data = {
        "id": "risk_test_01",
        "ticker": "SIVB",
        "company_name": "Silicon Valley Bank",
        "question": "Assess balance sheet duration risk",
        "as_of_date": "2022-12-31",
        "fall_date": "2023-03-10",
        "drawdown_pct": -100.0,
        "event_description": "FDIC receivership",
        "risk_themes": ["duration_mismatch", "unrealized_losses"],
    }
    example = RiskRecallExample.model_validate(valid_data)
    assert example.id == "risk_test_01"
    assert len(example.risk_themes) == 2

    # Invalid empty risk themes
    invalid_data = dict(valid_data)
    invalid_data["risk_themes"] = []
    with pytest.raises(ValidationError):
        RiskRecallExample.model_validate(invalid_data)

    # Invalid date format
    invalid_date = dict(valid_data)
    invalid_date["as_of_date"] = "31-12-2022"
    with pytest.raises(ValidationError):
        RiskRecallExample.model_validate(invalid_date)


def test_adversarial_schema_validation() -> None:
    """AdversarialExample strictly enforces expected_behavior enum values."""
    valid_data = {
        "id": "adv_test_01",
        "question": "Is the company attractive?",
        "ticker": None,
        "expected_behavior": "ask",
        "expected_abstain": False,
    }
    example = AdversarialExample.model_validate(valid_data)
    assert example.expected_behavior == "ask"

    # Invalid expected_behavior
    invalid_data = dict(valid_data)
    invalid_data["expected_behavior"] = "guess_and_hallucinate"
    with pytest.raises(ValidationError):
        AdversarialExample.model_validate(invalid_data)


def test_load_dataset_loads_risk_and_adversarial() -> None:
    """Generic load_dataset parses risk_recall and adversarial JSONL files."""
    risk_examples = load_dataset(Path("eval/datasets/risk_recall_v0.jsonl"))
    assert len(risk_examples) == 5
    for ex in risk_examples:
        assert ex.ticker is not None
        assert len(ex.risk_themes) > 0

    adv_examples = load_dataset(Path("eval/datasets/adversarial_v0.jsonl"))
    assert len(adv_examples) == 5
    behaviors = {ex.expected_behavior for ex in adv_examples}
    assert {"ask", "abstain", "answer_with_caveats"}.issubset(behaviors)
