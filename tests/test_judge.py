"""Unit tests for rubric-based Gemini judge and inter-rater agreement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lekka_ai.eval.judge import (
    GeminiRubricJudge,
    JudgeEvaluationRecord,
    JudgeEvaluationRequest,
    compute_agreement_report,
    compute_cohen_kappa,
    interpret_cohen_kappa,
    load_judge_sample_csv,
    normalize_verdict_string,
    run_judge_validation,
)
from lekka_ai.llm.base import LLMResponse, Usage
from lekka_ai.llm.fake import FakeLLM

# ---------------------------------------------------------------------------
# Cohen's Kappa & Agreement Calculation Tests
# ---------------------------------------------------------------------------


def test_normalize_verdict_string() -> None:
    """Test normalization of various verdict representations."""
    for val in ("pass", "PASS", "yes", "true", "supported", "1", "1.0"):
        assert normalize_verdict_string(val) == "pass"

    for val in ("fail", "FAIL", "no", "false", "unsupported", "0", "0.0"):
        assert normalize_verdict_string(val) == "fail"

    assert normalize_verdict_string(None) == "fail"


def test_cohen_kappa_perfect_agreement() -> None:
    """Two identical sets of ratings yield kappa = 1.0."""
    r1 = ["pass", "pass", "fail", "fail"]
    r2 = ["pass", "pass", "fail", "fail"]
    kappa, p_obs, p_exp = compute_cohen_kappa(r1, r2)
    assert kappa == 1.0
    assert p_obs == 1.0
    assert p_exp == 0.5


def test_cohen_kappa_systematic_disagreement() -> None:
    """Two perfectly inverted sets of ratings yield kappa = -1.0."""
    r1 = ["pass", "fail", "pass", "fail"]
    r2 = ["fail", "pass", "fail", "pass"]
    kappa, p_obs, p_exp = compute_cohen_kappa(r1, r2)
    assert kappa == -1.0
    assert p_obs == 0.0
    assert p_exp == 0.5


def test_cohen_kappa_chance_agreement() -> None:
    """Independent ratings at chance baseline yield kappa near 0.0."""
    r1 = ["pass", "pass", "pass", "pass"]
    r2 = ["fail", "fail", "fail", "fail"]
    kappa, p_obs, p_exp = compute_cohen_kappa(r1, r2)
    assert kappa == 0.0
    assert p_obs == 0.0


def test_cohen_kappa_unequal_length_raises() -> None:
    """Unequal length inputs raise ValueError."""
    with pytest.raises(ValueError, match="identical length"):
        compute_cohen_kappa(["pass"], ["pass", "fail"])


def test_interpret_cohen_kappa() -> None:
    """Validate qualitative interpretation tiers for kappa scores."""
    assert "Almost Perfect" in interpret_cohen_kappa(0.85)
    assert "Substantial" in interpret_cohen_kappa(0.70)
    assert "Moderate" in interpret_cohen_kappa(0.50)
    assert "Fair" in interpret_cohen_kappa(0.30)
    assert "Slight" in interpret_cohen_kappa(0.10)
    assert "Poor" in interpret_cohen_kappa(-0.20)


# ---------------------------------------------------------------------------
# GeminiRubricJudge with FakeLLM Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_evaluates_pass_verdict() -> None:
    """Judge parses structured pass output from LLM."""
    mock_payload = {
        "verdict": "pass",
        "score": 0.95,
        "reasoning": "The revenue figure matches Form 10-K reported net sales.",
        "key_discrepancies": [],
        "confidence": 1.0,
    }
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                text=json.dumps(mock_payload),
                usage=Usage(input_tokens=100, output_tokens=30),
            )
        ]
    )
    judge = GeminiRubricJudge(client=fake_llm)

    req = JudgeEvaluationRequest(
        id="sample_01",
        question="What was Apple revenue in 2024?",
        candidate_answer="Apple reported $391B net sales.",
        reference="Apple 10-K reported $391,035M net sales.",
    )
    res = await judge.evaluate(req)
    assert res.verdict == "pass"
    assert res.score == 0.95
    assert "revenue figure matches" in res.reasoning
    assert res.key_discrepancies == []


@pytest.mark.asyncio
async def test_judge_evaluates_fail_verdict() -> None:
    """Judge parses structured fail output from LLM with discrepancies."""
    mock_payload = {
        "verdict": "fail",
        "score": 0.1,
        "reasoning": "Reported margin contradicts disclosure.",
        "key_discrepancies": ["Margin was 20% in filing, answer claimed 48%"],
        "confidence": 0.9,
    }
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                text=json.dumps(mock_payload),
                usage=Usage(input_tokens=100, output_tokens=30),
            )
        ]
    )
    judge = GeminiRubricJudge(client=fake_llm)

    req = JudgeEvaluationRequest(
        id="sample_02",
        question="What was Azure margin?",
        candidate_answer="Azure had 48% margin.",
        reference="Filing reports 20% margin.",
    )
    res = await judge.evaluate(req)
    assert res.verdict == "fail"
    assert res.score == 0.1
    assert len(res.key_discrepancies) == 1


@pytest.mark.asyncio
async def test_judge_fallback_on_unstructured_text() -> None:
    """Judge parses markdown codeblock or text fallback gracefully."""
    raw_markdown = """```json
    {
      "verdict": "pass",
      "score": 1.0,
      "reasoning": "Fully grounded in filing."
    }
    ```"""
    fake_llm = FakeLLM(responses=[LLMResponse(text=raw_markdown)])
    judge = GeminiRubricJudge(client=fake_llm)

    req = JudgeEvaluationRequest(
        id="sample_03",
        question="Q?",
        candidate_answer="A",
    )
    res = await judge.evaluate(req)
    assert res.verdict == "pass"
    assert res.score == 1.0


# ---------------------------------------------------------------------------
# Agreement Report Generation Tests
# ---------------------------------------------------------------------------


def test_compute_agreement_report() -> None:
    """Compute agreement report with confusion matrix and markdown table."""
    records = [
        JudgeEvaluationRecord(
            id="s1",
            question="What was Q1 revenue?",
            candidate_answer="Rev $100M",
            human_label="pass",
            judge_verdict="pass",
            score=1.0,
            reasoning="Exact match",
        ),
        JudgeEvaluationRecord(
            id="s2",
            question="What was margin?",
            candidate_answer="Margin 80%",
            human_label="fail",
            judge_verdict="fail",
            score=0.0,
            reasoning="Fabricated figure",
        ),
        JudgeEvaluationRecord(
            id="s3",
            question="What was capex?",
            candidate_answer="Capex $5B",
            human_label="pass",
            judge_verdict="fail",
            score=0.2,
            reasoning="Disagreed",
        ),
    ]

    report = compute_agreement_report(records)
    assert report.total_samples == 3
    # 2 agreed out of 3 = 66.67%
    assert round(report.accuracy, 2) == 0.67
    assert report.confusion_matrix["pass"]["pass"] == 1
    assert report.confusion_matrix["fail"]["fail"] == 1
    assert report.confusion_matrix["pass"]["fail"] == 1
    assert "| `s1` |" in report.markdown_table
    assert "✅" in report.markdown_table
    assert "❌" in report.markdown_table


def test_load_judge_sample_csv() -> None:
    """Load default judge sample CSV and verify expected row count."""
    csv_path = Path("eval/labels/judge_sample.csv")
    assert csv_path.exists()
    requests = load_judge_sample_csv(csv_path)
    assert len(requests) == 10
    assert requests[0].id == "sample_01"
    assert "Apple" in requests[0].question


@pytest.mark.asyncio
async def test_run_judge_validation_pipeline(tmp_path: Path) -> None:
    """Run full validation pipeline on CSV with FakeLLM and generate report."""
    csv_path = Path("eval/labels/judge_sample.csv")

    # Generate mock responses for all 10 sample items matching human labels
    mock_responses = []
    # 6 pass and 4 fail in judge_sample.csv
    # Samples: 1(pass), 2(fail), 3(pass), 4(pass), 5(fail),
    # 6(pass), 7(fail), 8(pass), 9(fail), 10(pass)
    expected_labels = [
        "pass",
        "fail",
        "pass",
        "pass",
        "fail",
        "pass",
        "fail",
        "pass",
        "fail",
        "pass",
    ]
    for lbl in expected_labels:
        payload = {
            "verdict": lbl,
            "score": 1.0 if lbl == "pass" else 0.0,
            "reasoning": f"Automated audit verdict: {lbl}",
            "key_discrepancies": [] if lbl == "pass" else ["Discrepancy found"],
        }
        mock_responses.append(LLMResponse(text=json.dumps(payload)))

    fake_llm = FakeLLM(responses=mock_responses)
    judge = GeminiRubricJudge(client=fake_llm)

    output_file = tmp_path / "judge_agreement_report.md"
    report = await run_judge_validation(
        csv_path=csv_path,
        judge=judge,
        output_path=output_file,
    )

    assert report.total_samples == 10
    # Perfect agreement
    assert report.accuracy == 1.0
    assert report.cohen_kappa == 1.0
    assert output_file.exists()
    saved_text = output_file.read_text(encoding="utf-8")
    assert "LLM Judge Agreement Validation Report" in saved_text
    assert "100.00%" in saved_text
