"""Tests for Claim and Evidence Pydantic schemas and epistemic validation rules."""

import pytest
from pydantic import ValidationError

from lekka_ai.schemas.claims import Claim, ClaimType, Evidence


def test_valid_fact_claim_with_evidence() -> None:
    """Fact claim with evidence containing a tool_call_id is valid."""
    ev = Evidence(
        source_type="sec_filing",
        document_id="0000320193-23-000106",
        section="Item 8",
        tool_call_id="call_sec_financials_01",
        as_of_date="2023-09-30",
        raw_value={"revenue": 383285000000},
    )
    claim = Claim(
        claim_id="claim_rev_2023",
        text="Apple reported $383.3B in fiscal 2023 revenue.",
        type=ClaimType.FACT,
        evidence=[ev],
        produced_by="FundamentalAnalyst",
        confidence=1.0,
    )
    assert claim.claim_id == "claim_rev_2023"
    assert claim.type == ClaimType.FACT
    assert len(claim.evidence) == 1
    assert claim.evidence[0].tool_call_id == "call_sec_financials_01"


def test_fact_claim_without_evidence_rejected() -> None:
    """Fact claim without evidence raises ValidationError."""
    with pytest.raises(ValidationError, match="require at least one Evidence"):
        Claim(
            text="Apple reported $383.3B in revenue.",
            type="fact",
            evidence=[],
            confidence=0.9,
        )


def test_fact_claim_without_tool_call_id_rejected() -> None:
    """Fact claim whose evidence has no tool_call_id raises ValidationError."""
    ev_no_tool = Evidence(
        source_type="sec_filing",
        document_id="0000320193-23-000106",
        tool_call_id=None,
    )
    with pytest.raises(ValidationError, match="with tool_call_id"):
        Claim(
            text="Apple reported $383.3B in revenue.",
            type="fact",
            evidence=[ev_no_tool],
            confidence=0.9,
        )

    ev_empty_tool = Evidence(
        source_type="sec_filing",
        tool_call_id="   ",
    )
    with pytest.raises(ValidationError, match="with tool_call_id"):
        Claim(
            text="Apple reported $383.3B in revenue.",
            type="fact",
            evidence=[ev_empty_tool],
            confidence=0.9,
        )


def test_valid_calculation_claim_with_derived_by() -> None:
    """Calculation claim with derived_by formula/rule is valid."""
    claim = Claim(
        claim_id="claim_op_margin",
        text="Apple operating margin for FY23 was 29.8%.",
        type="calculation",
        derived_by="operating_income / revenue",
        produced_by="QuantAuditor",
        confidence=0.95,
    )
    assert claim.type == ClaimType.CALCULATION
    assert claim.derived_by == "operating_income / revenue"


def test_valid_calculation_claim_with_evidence() -> None:
    """Calculation claim with supporting evidence is valid."""
    ev = Evidence(
        source_type="quant_tool",
        tool_call_id="call_risk_stats_123",
        raw_value={"volatility": 0.22},
    )
    claim = Claim(
        claim_id="claim_volatility",
        text="Annualized volatility is 22%.",
        type=ClaimType.CALCULATION,
        evidence=[ev],
        confidence=0.9,
    )
    assert claim.type == ClaimType.CALCULATION
    assert len(claim.evidence) == 1


def test_calculation_claim_without_derivation_or_evidence_rejected() -> None:
    """Calculation claim lacking both derived_by and evidence raises ValidationError."""
    with pytest.raises(
        ValidationError, match="require either 'derived_by' .* or supporting 'evidence'"
    ):
        Claim(
            text="Apple operating margin was 29.8%.",
            type="calculation",
            derived_by=None,
            evidence=[],
            confidence=0.9,
        )


def test_valid_interpretation_claim_with_depends_on() -> None:
    """Interpretation claim referencing prerequisite claim IDs is valid."""
    claim = Claim(
        claim_id="claim_thesis_1",
        text="Apple possesses significant pricing power in consumer electronics.",
        type="interpretation",
        depends_on=["claim_rev_2023", "claim_op_margin"],
        produced_by="SynthesisAgent",
        confidence=0.85,
    )
    assert claim.type == ClaimType.INTERPRETATION
    assert "claim_rev_2023" in claim.depends_on
    assert "claim_op_margin" in claim.depends_on


def test_interpretation_claim_without_depends_on_rejected() -> None:
    """Interpretation claim with empty depends_on raises ValidationError."""
    with pytest.raises(
        ValidationError,
        match="require at least one prerequisite claim ID in 'depends_on'",
    ):
        Claim(
            text="Apple has durable pricing power.",
            type="interpretation",
            depends_on=[],
            confidence=0.8,
        )


def test_valid_background_claim() -> None:
    """Background claim requires no special evidence or dependencies."""
    claim = Claim(
        claim_id="claim_bg_1",
        text="Apple Inc. designs and manufactures smartphones and personal computers.",
        type="background",
        confidence=0.99,
    )
    assert claim.type == ClaimType.BACKGROUND
    assert claim.evidence == []
    assert claim.depends_on == []


def test_invalid_confidence_rejected() -> None:
    """Confidence outside [0.0, 1.0] raises ValidationError."""
    with pytest.raises(ValidationError):
        Claim(
            text="Apple is a tech company.",
            type="background",
            confidence=-0.1,
        )

    with pytest.raises(ValidationError):
        Claim(
            text="Apple is a tech company.",
            type="background",
            confidence=1.05,
        )


def test_invalid_claim_type_rejected() -> None:
    """Invalid claim type outside supported enum raises ValidationError."""
    with pytest.raises(ValidationError, match="Invalid claim type"):
        Claim(
            text="Some statement",
            type="opinion",
            confidence=0.5,
        )


def test_evidence_date_parsing() -> None:
    """Evidence model formats dates and timestamps cleanly."""
    ev = Evidence(
        source_type="market_data",
        retrieved_at="2024-01-15T12:00:00Z",
        as_of_date="2024-01-15",
    )
    assert ev.as_of_date == "2024-01-15"
    assert ev.retrieved_at == "2024-01-15T12:00:00Z"
