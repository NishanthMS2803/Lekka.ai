"""Tests for deterministic confidence scoring methodology and report abstention path."""

from __future__ import annotations

import re
from datetime import date, timedelta

import pytest

from lekka_ai.report.confidence import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    ConfidenceBreakdown,
    calculate_confidence,
    generate_abstention_report,
    synthesize_report_with_confidence,
)
from lekka_ai.schemas.claims import Claim, ClaimType, Evidence


@pytest.fixture
def high_evidence_claims() -> list[Claim]:
    """Return comprehensive multi-agent grounded claims across all core roles."""
    today = date.today()
    return [
        Claim(
            claim_id="claim_rev_001",
            text="Apple reported quarterly revenue of $85.8B for Q3 2024.",
            type=ClaimType.FACT,
            confidence=1.0,
            produced_by="fundamental",
            evidence=[
                Evidence(
                    source_type="quarterly_financials",
                    document_id="AAPL",
                    tool_call_id="tc_001",
                    as_of_date=today.isoformat(),
                    raw_value={"revenue": 85780000000.0},
                )
            ],
        ),
        Claim(
            claim_id="claim_mkt_002",
            text="Market price closed at $220.00 with annualized volatility of 18.5%.",
            type=ClaimType.FACT,
            confidence=1.0,
            produced_by="market",
            evidence=[
                Evidence(
                    source_type="market_data",
                    document_id="AAPL",
                    tool_call_id="tc_002",
                    as_of_date=today.isoformat(),
                    raw_value={"price": 220.0},
                )
            ],
        ),
        Claim(
            claim_id="claim_val_003",
            text="Discounted cash flow model indicates an intrinsic value of $210.00.",
            type=ClaimType.CALCULATION,
            confidence=0.9,
            produced_by="valuation",
            derived_by="run_dcf",
            evidence=[
                Evidence(
                    source_type="quant",
                    tool_call_id="tc_003",
                    as_of_date=today.isoformat(),
                    raw_value={"dcf": 210.0},
                )
            ],
        ),
        Claim(
            claim_id="claim_risk_004",
            text="Risk assessment indicates maximum drawdown exposure of 22.0%.",
            type=ClaimType.FACT,
            confidence=0.9,
            produced_by="risk",
            evidence=[
                Evidence(
                    source_type="risk_stats",
                    document_id="AAPL",
                    tool_call_id="tc_004",
                    as_of_date=today.isoformat(),
                    raw_value={"mdd": 0.22},
                )
            ],
        ),
        Claim(
            claim_id="claim_news_005",
            text="Antitrust scrutiny in the EU App Store remains active.",
            type=ClaimType.BACKGROUND,
            confidence=0.85,
            produced_by="news",
            evidence=[
                Evidence(
                    source_type="news",
                    document_id="AAPL",
                    tool_call_id="tc_005",
                    as_of_date=today.isoformat(),
                    raw_value="EU regulatory update",
                )
            ],
        ),
    ]


@pytest.fixture
def low_evidence_fixture() -> list[Claim]:
    """Low-evidence fixture: outdated dates, no verified facts, narrative background."""
    old_date = (date.today() - timedelta(days=700)).isoformat()
    return [
        Claim(
            claim_id="claim_vague_001",
            text="The company has a historical brand presence in consumer electronics.",
            type=ClaimType.BACKGROUND,
            confidence=0.45,
            produced_by="news",
            evidence=[
                Evidence(
                    source_type="news",
                    as_of_date=old_date,
                    raw_value="archived narrative",
                )
            ],
        ),
        Claim(
            claim_id="claim_contr_002",
            text="Rumors claim market share was 45%.",
            type=ClaimType.FACT,
            confidence=0.40,
            produced_by="news",
            contradicted_by=["claim_vague_001"],
            evidence=[
                Evidence(
                    source_type="news",
                    tool_call_id="tc_contr_002",
                    as_of_date=old_date,
                )
            ],
        ),
    ]


def test_confidence_rules_scoring_factors(
    high_evidence_claims: list[Claim],
) -> None:
    """Verify deterministic confidence calculation weights and components."""
    breakdown = calculate_confidence(
        high_evidence_claims,
        threshold=0.60,
    )
    assert isinstance(breakdown, ConfidenceBreakdown)
    assert breakdown.confidence_score >= 0.70
    assert breakdown.is_abstention is False
    assert breakdown.share_facts_verified == 1.0
    assert breakdown.unresolved_conflicts_count == 0
    assert breakdown.evidence_recency_score == 1.0
    assert breakdown.agent_coverage_score == 1.0
    assert breakdown.limits_hit_count == 0
    assert breakdown.missing_evidence == []


def test_conflict_penalty_deduction(high_evidence_claims: list[Claim]) -> None:
    """Unresolved conflicts apply a deterministic penalty of -0.15 per conflict."""
    clean_score = calculate_confidence(high_evidence_claims).confidence_score

    # Add contradictory claims
    conflicted_claims = list(high_evidence_claims) + [
        Claim(
            claim_id="claim_rev_conflict",
            text="Revenue was actually $95.0B in Q3 2024.",
            type=ClaimType.FACT,
            confidence=0.5,
            contradicted_by=["claim_rev_001"],
            evidence=[
                Evidence(
                    source_type="quarterly_financials",
                    tool_call_id="tc_c",
                )
            ],
        )
    ]

    breakdown_conflicted = calculate_confidence(
        conflicted_claims,
        open_questions=["Reconcile conflict on revenue between agents."],
    )
    assert breakdown_conflicted.unresolved_conflicts_count >= 1
    assert breakdown_conflicted.confidence_score < clean_score
    # Conflict penalty reflected in factors summary
    assert breakdown_conflicted.factors_summary["conflict_penalty"] <= -0.15


def test_limits_hit_penalty_deduction(high_evidence_claims: list[Claim]) -> None:
    """Limits hit apply deterministic penalties of -0.10 each."""
    clean_score = calculate_confidence(high_evidence_claims).confidence_score

    breakdown_limits = calculate_confidence(
        high_evidence_claims,
        limits_hit=["iteration_limit_exceeded", "token_limit_exceeded"],
    )
    assert breakdown_limits.limits_hit_count == 2
    assert breakdown_limits.confidence_score < clean_score
    assert breakdown_limits.factors_summary["limits_penalty"] == -0.20


def test_configurable_threshold_behavior(
    high_evidence_claims: list[Claim],
) -> None:
    """Configuring higher threshold triggers abstention when score falls short."""
    # Score is around ~0.85; threshold of 0.95 forces abstention
    strict_breakdown = calculate_confidence(
        high_evidence_claims,
        threshold=0.95,
    )
    assert strict_breakdown.is_abstention is True

    # Standard threshold allows normal synthesis
    standard_breakdown = calculate_confidence(
        high_evidence_claims,
        threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    )
    assert standard_breakdown.is_abstention is False


def test_low_evidence_fixture_triggers_abstention(
    low_evidence_fixture: list[Claim],
) -> None:
    """Low-evidence fixture fails threshold and is flagged for abstention."""
    breakdown = calculate_confidence(low_evidence_fixture, threshold=0.60)
    assert breakdown.is_abstention is True
    assert breakdown.confidence_score < 0.60
    assert len(breakdown.missing_evidence) > 0


def test_low_evidence_fixture_produces_abstention_report(
    low_evidence_fixture: list[Claim],
) -> None:
    """CRITICAL VERIFY: Low-evidence fixture produces an abstention report.

    The report MUST:
    1. Explicitly state 'insufficient evidence'.
    2. List what is missing.
    3. Give NO scenario numbers (no price targets or DCF values).
    """
    breakdown = calculate_confidence(low_evidence_fixture, threshold=0.60)
    report_md = generate_abstention_report("XYZ", breakdown)

    # 1. Must explicitly state 'insufficient evidence'
    assert "insufficient evidence" in report_md.lower()
    assert "Executive Summary: Insufficient Evidence" in report_md

    # 2. Must list what is missing
    assert "Missing Evidence and Deficiencies" in report_md
    for missing_item in breakdown.missing_evidence:
        assert missing_item in report_md

    # 3. Must give NO scenario numbers
    # Ensure valuation scenario section explicitly states no numbers
    assert "No valuation scenarios or target prices can be derived" in report_md
    assert "Scenario numbers" in report_md and "withheld" in report_md

    # Regex test: no dollar amounts anywhere in the valuation scenario text
    scenario_section_match = re.search(
        r"## Valuation Scenarios(.*?)(?=##|\Z)", report_md, re.DOTALL
    )
    assert scenario_section_match is not None
    scenario_text = scenario_section_match.group(1)
    assert "$" not in scenario_text
    assert not re.search(r"\b\d+\.\d+\b", scenario_text)


@pytest.mark.asyncio
async def test_synthesize_report_with_confidence_abstention_path(
    low_evidence_fixture: list[Claim],
) -> None:
    """Integration: synthesize_report_with_confidence takes abstention path."""
    report, breakdown = await synthesize_report_with_confidence(
        claims=low_evidence_fixture,
        ticker="LOWEV",
        threshold=0.60,
    )

    assert breakdown.is_abstention is True
    assert report.validation_passed is True  # Abstention report is valid
    assert "insufficient evidence" in report.markdown.lower()
    assert report.sentence_to_claims == {}
    assert report.claims_used == []

    # Ensure no scenario numbers appear in the markdown report
    assert "No valuation scenarios" in report.markdown
