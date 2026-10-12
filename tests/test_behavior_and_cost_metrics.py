"""Unit tests for Step 84 behavioral and cost evaluation metrics."""

from __future__ import annotations

import json

import pytest

from lekka_ai.eval.dataset import EvalExample
from lekka_ai.eval.metrics import (
    DEFAULT_METRICS,
    abstention_precision,
    abstention_quality,
    abstention_quality_metric,
    abstention_recall,
    brier_score,
    calibration_metric,
    compute_abstention_quality,
    compute_calibration,
    compute_cost_breakdown,
    extract_flagged_themes,
    extract_used_claim_ids,
    latency_metric,
    risk_recall,
    risk_recall_metric,
    tokens_in_metric,
    tokens_out_metric,
    tokens_total_metric,
    tool_calls_metric,
    unnecessary_agent_calls,
    unnecessary_agent_calls_details,
    unnecessary_agent_calls_metric,
    usd_estimate_metric,
)
from lekka_ai.eval.metrics.risk import (
    RiskRecallResult,
    evaluate_risk_recall_llm,
    evaluate_risk_recall_rule_based,
)
from lekka_ai.eval.variant import RunResult
from lekka_ai.llm.base import LLMResponse, Usage
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.schemas.claims import Claim, ClaimType, Evidence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_claim(
    claim_id: str,
    text: str = "Test claim",
    produced_by: str = "fundamental",
) -> Claim:
    """Helper creating a test Claim model."""
    return Claim(
        claim_id=claim_id,
        text=text,
        type=ClaimType.FACT,
        confidence=1.0,
        produced_by=produced_by,
        evidence=[
            Evidence(
                source_type="sec_filing",
                document_id="DOC_01",
                tool_call_id="tc_01",
                raw_value={"value": 100.0},
            )
        ],
    )


# ---------------------------------------------------------------------------
# Risk Recall Tests
# ---------------------------------------------------------------------------


def test_extract_flagged_themes_from_markdown() -> None:
    """Extract flagged risk themes from markdown Ranked Risks section."""
    markdown = """
    ## Executive Summary
    Apple delivered solid performance.

    ## Ranked Risks
    1. Foreign exchange volatility and currency headwinds
    2. Concentrated iPhone product revenue reliance
    3. Regulatory antitrust scrutiny in App Store

    ## Catalysts
    AI integration on device.
    """
    themes = extract_flagged_themes(markdown)
    assert len(themes) == 3
    assert "Foreign exchange volatility" in themes[0]
    assert "Concentrated iPhone" in themes[1]
    assert "Regulatory antitrust" in themes[2]


def test_extract_flagged_themes_from_run_result_claims() -> None:
    """Extract risk themes from claims produced by risk agent."""
    c1 = _make_claim("c1", "Revenue grew 10%", produced_by="fundamental")
    c2 = _make_claim(
        "c2",
        "Severe commercial real estate default risk",
        produced_by="risk",
    )
    c3 = _make_claim(
        "c3",
        "Uninsured deposit flight exposure",
        produced_by="risk_agent",
    )

    res = RunResult(
        example_id="ex_risk",
        variant="test",
        question="What risks exist?",
        claims=[c1, c2, c3],
    )
    themes = extract_flagged_themes(res)
    assert len(themes) == 2
    assert "real estate" in themes[0]
    assert "deposit flight" in themes[1]


def test_risk_recall_rule_based_matching() -> None:
    """Evaluate risk recall matching against expected labeled themes."""
    flagged = [
        "Unrealized held-to-maturity securities losses",
        "Concentrated tech venture deposit base",
    ]
    labeled = [
        "Unrealized HTM losses",
        "Venture startup deposit concentration",
        "Crypto asset contagion",
    ]

    res: RiskRecallResult = evaluate_risk_recall_rule_based(flagged, labeled)
    assert res.total_labeled == 3
    assert res.matched_count == 2
    assert res.recall == round(2 / 3, 4)
    assert res.matches[0].matched is True
    assert res.matches[1].matched is True
    assert res.matches[2].matched is False


@pytest.mark.asyncio
async def test_risk_recall_llm_judge() -> None:
    """Test LLM judge evaluation using FakeLLM returning rubric JSON."""
    llm_payload = {
        "matches": [
            {
                "theme": "CRE debt maturities",
                "matched": True,
                "matched_by": "Upcoming office mortgage roll-off",
                "reasoning": "Underlying debt refinancing risk captured.",
            },
            {
                "theme": "Liquidity freeze",
                "matched": False,
                "matched_by": None,
                "reasoning": "Omitted liquidity crunch.",
            },
        ]
    }
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                text=json.dumps(llm_payload),
                usage=Usage(input_tokens=50, output_tokens=50),
            )
        ]
    )

    flagged = ["Upcoming office mortgage roll-off"]
    labeled = ["CRE debt maturities", "Liquidity freeze"]

    res = await evaluate_risk_recall_llm(flagged, labeled, judge=fake_llm)
    assert res.total_labeled == 2
    assert res.matched_count == 1
    assert res.recall == 0.5
    assert res.matches[0].matched is True
    assert res.matches[1].matched is False


def test_risk_recall_empty_labeled_returns_one() -> None:
    """When labeled risk themes list is empty, risk recall defaults to 1.0."""
    assert risk_recall(["Risk A"], []) == 1.0


def test_risk_recall_metric_adapter() -> None:
    """Validate MetricFunc adapter for risk recall."""
    c = _make_claim("c1", "Severe liquidity crunch", produced_by="risk")
    res = RunResult(
        example_id="risk_01",
        variant="test",
        question="Q?",
        claims=[c],
    )
    ex = EvalExample(
        id="risk_01",
        question="Q?",
        risk_themes=["Severe liquidity crunch"],
    )
    assert risk_recall_metric(res, ex) == 1.0


# ---------------------------------------------------------------------------
# Calibration Tests
# ---------------------------------------------------------------------------


def test_brier_score_calculation() -> None:
    """Compute Brier score on calibrated and miscalibrated predictions."""
    # (0.9 - 1.0)^2 = 0.01; (0.1 - 0.0)^2 = 0.01 -> mean = 0.01
    well_calibrated = [(0.9, 1.0), (0.1, 0.0)]
    assert brier_score(well_calibrated) == 0.01

    # (0.9 - 0.0)^2 = 0.81; (0.1 - 1.0)^2 = 0.81 -> mean = 0.81
    miscalibrated = [(0.9, 0.0), (0.1, 1.0)]
    assert brier_score(miscalibrated) == 0.81


def test_compute_calibration_bins_and_ece() -> None:
    """Verify reliability binning and expected calibration error (ECE)."""
    # 5 bins of width 0.2: [0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
    records = [
        (0.1, 0.0),  # bin 0
        (0.3, 0.0),  # bin 1
        (0.5, 1.0),  # bin 2
        (0.7, 1.0),  # bin 3
        (0.9, 1.0),  # bin 4
    ]
    res = compute_calibration(records, n_bins=5)
    assert res.count == 5
    assert len(res.bins) == 5
    assert res.brier_score == round(
        (0.01 + 0.09 + 0.25 + 0.09 + 0.01) / 5,
        4,
    )
    assert res.ece >= 0.0
    assert res.mce >= 0.0


def test_compute_calibration_empty_records() -> None:
    """Empty record sequence returns zeroed CalibrationResult."""
    res = compute_calibration([])
    assert res.count == 0
    assert res.brier_score == 0.0
    assert res.ece == 0.0


def test_calibration_metric_adapter() -> None:
    """Evaluate single-result calibration metric adapter."""
    res = RunResult(
        example_id="ex_calib",
        variant="test",
        question="Q?",
        answer="Answer text",
        confidence_score=0.8,
        metrics={"accuracy": 1.0},
    )
    ex = EvalExample(id="ex_calib", question="Q?")
    # 1.0 - (0.8 - 1.0)^2 = 1.0 - 0.04 = 0.96
    assert calibration_metric(res, ex) == 0.96


# ---------------------------------------------------------------------------
# Abstention Quality Tests
# ---------------------------------------------------------------------------


def test_compute_abstention_quality_confusion_matrix() -> None:
    """Validate precision, recall, and F1 calculations for abstention quality."""
    records = [
        (True, True),  # TP: expected to abstain and did
        (True, False),  # FP: expected to answer but abstained
        (False, True),  # FN: expected to abstain but answered
        (False, False),  # TN: expected to answer and did
    ]
    res = compute_abstention_quality(records)
    assert res.total == 4
    assert res.tp == 1
    assert res.fp == 1
    assert res.fn == 1
    assert res.tn == 1
    # Precision: 1 / (1 + 1) = 0.50
    assert res.precision == 0.5
    # Recall: 1 / (1 + 1) = 0.50
    assert res.recall == 0.5
    # F1: 0.50
    assert res.f1 == 0.5
    assert res.accuracy == 0.5


def test_abstention_quality_helpers() -> None:
    """Verify precision and recall helper functions."""
    records = [(True, True), (False, False)]
    assert abstention_precision(records) == 1.0
    assert abstention_recall(records) == 1.0
    assert abstention_quality(records) == 1.0


def test_abstention_quality_metric_run_result() -> None:
    """Single item abstention quality returns 1.0 when decision matches."""
    res = RunResult(
        example_id="adv_01",
        variant="test",
        question="Unanswerable prompt",
        answer="I have insufficient evidence to answer.",
        is_abstention=True,
    )
    ex = EvalExample(
        id="adv_01",
        question="Unanswerable prompt",
        expected_abstain=True,
    )
    assert abstention_quality_metric(res, ex) == 1.0


# ---------------------------------------------------------------------------
# Cost and Operational Efficiency Tests
# ---------------------------------------------------------------------------


def test_extract_used_claim_ids() -> None:
    """Extract cited claim IDs from bracketed report text and structures."""
    report = "Revenue was $100M [claim_01]. Margin was 20% [claim_02, claim_03]."
    cited = extract_used_claim_ids(report)
    assert cited == {"claim_01", "claim_02", "claim_03"}

    dict_citations = {"sentence 1": ["claim_01", "claim_02"]}
    assert extract_used_claim_ids(dict_citations) == {"claim_01", "claim_02"}


def test_unnecessary_agent_calls_detection() -> None:
    """Identify dispatched agents whose claims were never cited in report."""
    c_fund = _make_claim("c_f1", "Revenue $100M", produced_by="fundamental")
    c_val = _make_claim("c_v1", "DCF value $150", produced_by="valuation")
    c_mkt = _make_claim("c_m1", "Beta is 1.2", produced_by="market")
    c_news = _make_claim("c_n1", "CEO interviewed", produced_by="news")

    report = (
        "The company generated strong revenue [c_f1] and shares appear cheap [c_v1]."
    )
    claims = [c_fund, c_val, c_mkt, c_news]

    count, unused, used = unnecessary_agent_calls_details(claims, report)
    # market and news claims were never cited in the report
    assert count == 2
    assert sorted(unused) == ["market", "news"]
    assert sorted(used) == ["fundamental", "valuation"]
    assert unnecessary_agent_calls(claims, report) == 2


def test_compute_cost_breakdown_and_resource_metrics() -> None:
    """Validate resource metric functions and CostBreakdown summary model."""
    c1 = _make_claim("c1", "Revenue $100M", produced_by="fundamental")
    res = RunResult(
        example_id="ex_cost",
        variant="test",
        question="Q?",
        answer="Revenue was $100M [c1].",
        claims=[c1],
        cost=0.0125,
        tokens_in=1200,
        tokens_out=300,
        tokens_total=1500,
        latency_ms=1850.5,
        tool_calls=4,
    )
    ex = EvalExample(id="ex_cost", question="Q?")

    assert tool_calls_metric(res, ex) == 4
    assert tokens_total_metric(res, ex) == 1500
    assert tokens_in_metric(res, ex) == 1200
    assert tokens_out_metric(res, ex) == 300
    assert latency_metric(res, ex) == 1850.5
    assert usd_estimate_metric(res, ex) == 0.0125
    assert unnecessary_agent_calls_metric(res, ex) == 0

    breakdown = compute_cost_breakdown(res)
    assert breakdown.tool_calls == 4
    assert breakdown.tokens_total == 1500
    assert breakdown.usd_estimate == 0.0125
    assert breakdown.unnecessary_agent_calls == 0
    assert breakdown.used_agents == ["fundamental"]


# ---------------------------------------------------------------------------
# DEFAULT_METRICS Registry Integration
# ---------------------------------------------------------------------------


def test_default_metrics_registry_coverage() -> None:
    """Verify all Step 84 behavioral and cost metrics exist in DEFAULT_METRICS."""
    expected_metrics = [
        "risk_recall",
        "calibration",
        "abstention_quality",
        "usd_estimate",
        "tokens",
        "tokens_total",
        "tokens_in",
        "tokens_out",
        "tool_calls",
        "latency_ms",
        "unnecessary_agent_calls",
    ]
    for m in expected_metrics:
        assert m in DEFAULT_METRICS, f"Missing metric {m} in DEFAULT_METRICS"
