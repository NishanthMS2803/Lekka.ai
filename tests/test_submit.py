"""Tests for SubmitAnswer schema, epistemic rules, and typed answer extensions."""

from collections.abc import Generator

import pytest
from pydantic import BaseModel, Field, ValidationError

from lekka_ai.schemas.claims import Claim, ClaimType, Evidence
from lekka_ai.schemas.submit import (
    AnswerExtension,
    AnswerStatus,
    SubmitAnswer,
    clear_answer_extensions,
    export_submit_answer_schema,
    register_answer_extension,
)


@pytest.fixture(autouse=True)
def clean_extensions() -> Generator[None, None, None]:
    """Ensure extension registry is reset before and after each test."""
    clear_answer_extensions()
    yield
    clear_answer_extensions()


def _make_dummy_claim(claim_id: str = "claim_1") -> Claim:
    """Helper to generate a valid fact claim with tool evidence."""
    return Claim(
        claim_id=claim_id,
        text="Apple generated $383.3B revenue in FY23.",
        type=ClaimType.FACT,
        evidence=[
            Evidence(
                source_type="sec_filing",
                tool_call_id="call_sec_financials_01",
                raw_value=383285000000,
            )
        ],
        confidence=1.0,
    )


def test_valid_completed_submit_answer() -> None:
    """Test valid completed SubmitAnswer with claims, assumptions, and rationale."""
    claim = _make_dummy_claim()
    ans = SubmitAnswer(
        task_id="task_123",
        agent_type="fundamental_analyst",
        status=AnswerStatus.COMPLETED,
        claims=[claim],
        assumptions={"discount_rate": 0.09, "terminal_growth": 0.02},
        open_questions=[],
        confidence=0.92,
        confidence_rationale="Supported by audited SEC 10-K company facts.",
        limits_hit=[],
    )

    assert ans.task_id == "task_123"
    assert ans.status == AnswerStatus.COMPLETED
    assert len(ans.claims) == 1
    assert ans.assumptions["discount_rate"] == 0.09
    assert ans.confidence == 0.92


def test_valid_partial_submit_answer() -> None:
    """Test partial SubmitAnswer with claims, open questions, and limits hit."""
    claim = _make_dummy_claim()
    ans = SubmitAnswer(
        task_id="task_456",
        agent_type="filing_researcher",
        status="partial",
        claims=[claim],
        open_questions=["Missing Segment breakdown for Q4 2023"],
        confidence=0.75,
        confidence_rationale="Item 7 retrieved but segment detail unavailable.",
        limits_hit=["rate_limit_exceeded"],
    )

    assert ans.status == AnswerStatus.PARTIAL
    assert len(ans.open_questions) == 1
    assert "rate_limit_exceeded" in ans.limits_hit


def test_insufficient_evidence_with_zero_claims_allowed() -> None:
    """insufficient_evidence status explicitly allows zero claims."""
    ans = SubmitAnswer(
        task_id="task_789",
        agent_type="filing_researcher",
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        claims=[],  # Zero claims allowed
        assumptions={},
        open_questions=["No filings found for ticker XYZ"],
        confidence=0.1,
        confidence_rationale="No 10-K filings available before cutoff date.",
        limits_hit=["no_filings_found"],
    )

    assert ans.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert ans.claims == []
    assert ans.confidence == 0.1


def test_completed_answer_with_zero_claims_rejected() -> None:
    """Completed answer with zero claims raises ValidationError."""
    with pytest.raises(ValidationError, match="require at least one grounded Claim"):
        SubmitAnswer(
            task_id="task_error_1",
            agent_type="fundamental_analyst",
            status="completed",
            claims=[],  # Invalid for completed
            confidence=0.9,
            confidence_rationale="Everything looks fine.",
        )


def test_invalid_status_rejected() -> None:
    """Invalid status outside {completed, partial, insufficient_evidence} rejected."""
    claim = _make_dummy_claim()
    with pytest.raises(ValidationError, match="Invalid status"):
        SubmitAnswer(
            task_id="task_error_2",
            agent_type="analyst",
            status="finished",  # Invalid status string
            claims=[claim],
            confidence=0.9,
            confidence_rationale="Done",
        )


def test_invalid_confidence_rejected() -> None:
    """Confidence outside [0.0, 1.0] raises ValidationError."""
    claim = _make_dummy_claim()
    with pytest.raises(ValidationError):
        SubmitAnswer(
            task_id="task_error_3",
            agent_type="analyst",
            status="completed",
            claims=[claim],
            confidence=1.2,  # > 1.0
            confidence_rationale="Super confident",
        )


def test_typed_answer_extension_hook() -> None:
    """Per-agent-type typed answer extension hook parses and validates payload."""

    class QuantAuditorExtension(AnswerExtension):
        dcf_fair_value: float = Field(..., description="Calculated DCF fair value")
        wacc: float = Field(..., description="Discount rate used")

    register_answer_extension("quant_auditor", QuantAuditorExtension)

    claim = _make_dummy_claim()
    payload = {
        "task_id": "task_quant_1",
        "agent_type": "quant_auditor",
        "status": "completed",
        "claims": [claim],
        "confidence": 0.95,
        "confidence_rationale="
        "Grounded in verified historical cash flows and WACC calculation.": "Verified",
        "confidence_rationale": "Grounded in verified cash flows.",
        "extension": {
            "dcf_fair_value": 182.5,
            "wacc": 0.085,
        },
    }

    ans = SubmitAnswer.model_validate(payload)
    assert isinstance(ans.extension, QuantAuditorExtension)
    assert ans.extension.dcf_fair_value == 182.5
    assert ans.extension.wacc == 0.085


def test_typed_answer_extension_invalid_payload_rejected() -> None:
    """Invalid typed extension payload raises ValidationError."""

    class FilingSummaryExtension(BaseModel):
        accession_number: str
        fiscal_year: int

    register_answer_extension("filing_researcher", FilingSummaryExtension)

    claim = _make_dummy_claim()
    with pytest.raises(ValidationError):
        SubmitAnswer(
            task_id="task_fail_ext",
            agent_type="filing_researcher",
            status=AnswerStatus.COMPLETED,
            claims=[claim],
            confidence=0.9,
            confidence_rationale="Done",
            extension={"accession_number": "12345"},  # Missing required fiscal_year
        )


def test_export_submit_answer_schema() -> None:
    """export_submit_answer_schema returns valid JSON Schema definition."""
    schema = export_submit_answer_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "task_id" in schema["properties"]
    assert "status" in schema["properties"]
    assert "claims" in schema["properties"]
    assert "confidence" in schema["properties"]
    assert "extension" in schema["properties"]


def test_assumptions_list_conversion() -> None:
    """Assumptions passed as a list of strings are converted to a keyed dictionary."""
    claim = _make_dummy_claim()
    ans = SubmitAnswer(
        task_id="task_assumptions_list",
        agent_type="analyst",
        status="completed",
        claims=[claim],
        assumptions=["Constant tax rate of 21%", "Capex grows at 5% annually"],  # type: ignore[arg-type]
        confidence=0.9,
        confidence_rationale="Standard assumptions used.",
    )
    assert isinstance(ans.assumptions, dict)
    assert "assumption_1" in ans.assumptions
    assert ans.assumptions["assumption_1"] == "Constant tax rate of 21%"
