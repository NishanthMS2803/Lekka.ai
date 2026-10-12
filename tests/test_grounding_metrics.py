"""Unit tests for grounding evaluation metrics with hand-made claim sets."""

from __future__ import annotations

from lekka_ai.eval.dataset import EvalExample
from lekka_ai.eval.metrics import (
    DEFAULT_METRICS,
    background_claim_rate,
    citation_precision,
    interpretation_validity,
    unsupported_fact_rate,
)
from lekka_ai.eval.variant import RunResult
from lekka_ai.schemas.claims import Claim, ClaimType, Evidence
from lekka_ai.verify.support import ClaimSupportCheck, SupportVerdict


def _make_fact_claim(
    claim_id: str,
    text: str = "Empirical metric fact",
    tool_call_id: str | None = "tc_01",
    raw_value: float | dict[str, float] | None = 100.0,
    has_evidence: bool = True,
) -> Claim:
    """Helper to build a small hand-made Fact claim."""
    ev = []
    if has_evidence:
        ev.append(
            Evidence(
                source_type="sec_filing",
                document_id="DOC_01",
                tool_call_id=tool_call_id,
                raw_value=(
                    {"value": raw_value} if isinstance(raw_value, float) else raw_value
                ),
            )
        )
    return Claim(
        claim_id=claim_id,
        text=text,
        type=ClaimType.FACT,
        confidence=1.0,
        produced_by="fundamental",
        evidence=ev,
    )


def _make_background_claim(
    claim_id: str,
    text: str = "Market context narrative",
) -> Claim:
    """Helper to build a small hand-made Background claim."""
    return Claim(
        claim_id=claim_id,
        text=text,
        type=ClaimType.BACKGROUND,
        confidence=0.8,
        produced_by="analyst",
        evidence=[],
    )


def _make_interpretation_claim(
    claim_id: str,
    depends_on: list[str],
    text: str = "Derived business interpretation",
) -> Claim:
    """Helper to build a small hand-made Interpretation claim."""
    return Claim(
        claim_id=claim_id,
        text=text,
        type=ClaimType.INTERPRETATION,
        confidence=0.9,
        produced_by="analyst",
        depends_on=depends_on,
        evidence=[],
    )


# ---------------------------------------------------------------------------
# Citation Precision Tests
# ---------------------------------------------------------------------------


def test_citation_precision_full_support() -> None:
    """Citation precision is 1.0 when all cited sources have 'yes' verdicts."""
    c1 = _make_fact_claim("c1", "Revenue was $100M", tool_call_id="tc_1")
    c2 = _make_fact_claim("c2", "Margin was 20%", tool_call_id="tc_2")

    verdicts = {
        "c1": SupportVerdict.YES,
        "c2": "yes",
    }
    precision = citation_precision([c1, c2], verdicts)
    assert precision == 1.0


def test_citation_precision_partial_and_no() -> None:
    """Citation precision weights partial verdicts and counts zero for refuted."""
    c1 = _make_fact_claim("c1", "Revenue was $100M", tool_call_id="tc_1")
    c2 = _make_fact_claim("c2", "Margin was 20%", tool_call_id="tc_2")
    c3 = _make_fact_claim("c3", "Operating income $20M", tool_call_id="tc_3")

    verdicts = [
        ClaimSupportCheck(
            claim_id="c1",
            supported=SupportVerdict.YES,
            reason="Matches 10-K",
        ),
        ClaimSupportCheck(
            claim_id="c2",
            supported=SupportVerdict.PARTIAL,
            reason="Slight mismatch",
        ),
        ClaimSupportCheck(
            claim_id="c3",
            supported=SupportVerdict.NO,
            reason="Refuted",
        ),
    ]
    # (1.0 + 0.5 + 0.0) / 3 = 0.50
    precision = citation_precision([c1, c2, c3], verdicts, partial_weight=0.5)
    assert precision == 0.5


def test_citation_precision_zero_citations_returns_default() -> None:
    """When claims contain zero citations, default value is returned."""
    c_bg = _make_background_claim("bg_1")
    precision = citation_precision([c_bg])
    assert precision == 1.0


# ---------------------------------------------------------------------------
# Unsupported Fact Rate Tests
# ---------------------------------------------------------------------------


def test_unsupported_fact_rate_all_supported() -> None:
    """Unsupported fact rate is 0.0 when every fact has valid grounding evidence."""
    f1 = _make_fact_claim("f1", tool_call_id="tc_1", raw_value=100.0)
    f2 = _make_fact_claim("f2", tool_call_id="tc_2", raw_value=200.0)
    assert unsupported_fact_rate([f1, f2]) == 0.0


def test_unsupported_fact_rate_with_missing_evidence() -> None:
    """Facts with empty or missing evidence are flagged as unsupported."""
    f1 = _make_fact_claim("f1", tool_call_id="tc_1", raw_value=100.0)
    # f2 has no evidence citations
    f2 = {
        "claim_id": "f2",
        "text": "Ungrounded fact without citations",
        "type": "fact",
        "evidence": [],
    }
    # 1 supported out of 2 total facts -> unsupported rate = 0.50
    assert unsupported_fact_rate([f1, f2]) == 0.5


def test_unsupported_fact_rate_with_verifier_rejection() -> None:
    """Verifier negative verdict ('no') marks fact as unsupported."""
    f1 = _make_fact_claim("f1", tool_call_id="tc_1")
    f2 = _make_fact_claim("f2", tool_call_id="tc_2")

    verdicts = {"f1": "yes", "f2": "no"}
    # f2 has evidence but verifier verdict is explicitly 'no'
    assert unsupported_fact_rate([f1, f2], verdicts=verdicts) == 0.5


def test_unsupported_fact_rate_zero_facts_returns_zero() -> None:
    """When set contains zero fact claims, unsupported rate is 0.0."""
    bg = _make_background_claim("bg_1")
    assert unsupported_fact_rate([bg]) == 0.0


# ---------------------------------------------------------------------------
# Background Claim Rate Tests
# ---------------------------------------------------------------------------


def test_background_claim_rate_distribution() -> None:
    """Background claim rate accurately computes share of narrative claims."""
    f1 = _make_fact_claim("f1")
    f2 = _make_fact_claim("f2")
    bg1 = _make_background_claim("bg1")
    bg2 = _make_background_claim("bg2")

    # 2 background out of 4 total -> 0.50
    assert background_claim_rate([f1, f2, bg1, bg2]) == 0.5

    # 0 background out of 2 total -> 0.0
    assert background_claim_rate([f1, f2]) == 0.0

    # 2 background out of 2 total -> 1.0
    assert background_claim_rate([bg1, bg2]) == 1.0


def test_background_claim_rate_empty_returns_zero() -> None:
    """Empty claim list produces 0.0 background claim rate."""
    assert background_claim_rate([]) == 0.0


# ---------------------------------------------------------------------------
# Interpretation Validity Tests
# ---------------------------------------------------------------------------


def test_interpretation_validity_with_valid_fact_support() -> None:
    """Interpretation supported by a valid fact claim passes check."""
    f1 = _make_fact_claim("f1")
    interp1 = _make_interpretation_claim("i1", depends_on=["f1"])

    validity = interpretation_validity([f1, interp1])
    assert validity == 1.0


def test_interpretation_validity_missing_dependency() -> None:
    """Interpretation referencing non-existent claim fails check."""
    f1 = _make_fact_claim("f1")
    interp_broken = _make_interpretation_claim(
        "i_broken",
        depends_on=["phantom_claim_id"],
    )

    # 0 valid out of 1 interpretation -> 0.0
    assert interpretation_validity([f1, interp_broken]) == 0.0


def test_interpretation_validity_solely_background_fails() -> None:
    """Interpretation supported solely by background claims fails support check."""
    bg1 = _make_background_claim("bg1")
    interp_unsupported = _make_interpretation_claim("i_bg", depends_on=["bg1"])

    # Fails Check (d): background cannot be the sole support
    assert interpretation_validity([bg1, interp_unsupported]) == 0.0


def test_interpretation_validity_mixed_set() -> None:
    """Mixed set computes correct share of valid interpretations."""
    f1 = _make_fact_claim("f1")
    bg1 = _make_background_claim("bg1")

    valid_interp = _make_interpretation_claim("i_valid", depends_on=["f1", "bg1"])
    invalid_interp = _make_interpretation_claim("i_invalid", depends_on=["bg1"])

    claims = [f1, bg1, valid_interp, invalid_interp]
    # 1 valid, 1 invalid -> 0.50
    assert interpretation_validity(claims) == 0.5


def test_interpretation_validity_empty_returns_default() -> None:
    """When zero interpretations exist, default 1.0 is returned."""
    f1 = _make_fact_claim("f1")
    assert interpretation_validity([f1]) == 1.0


# ---------------------------------------------------------------------------
# MetricFunc Evaluation Harness Integration Tests
# ---------------------------------------------------------------------------


def test_metrics_registry_and_run_result_adapters() -> None:
    """Metrics adapt cleanly to RunResult and are registered in DEFAULT_METRICS."""
    assert "citation_precision" in DEFAULT_METRICS
    assert "unsupported_fact_rate" in DEFAULT_METRICS
    assert "background_claim_rate" in DEFAULT_METRICS
    assert "interpretation_validity" in DEFAULT_METRICS

    f1 = _make_fact_claim("f1", tool_call_id="tc_1")
    bg1 = _make_background_claim("bg1")
    interp1 = _make_interpretation_claim("i1", depends_on=["f1"])

    result = RunResult(
        example_id="ex_01",
        question="What was Apple revenue?",
        variant="single",
        answer="Apple grew revenue to $100M.",
        claims=[f1, bg1, interp1],
        metadata={"verifier_verdicts": {"f1": "yes"}},
    )
    example = EvalExample(id="ex_01", question="What was Apple revenue?")

    assert DEFAULT_METRICS["citation_precision"](result, example) == 1.0
    assert DEFAULT_METRICS["unsupported_fact_rate"](result, example) == 0.0
    assert DEFAULT_METRICS["background_claim_rate"](result, example) == round(1 / 3, 4)
    assert DEFAULT_METRICS["interpretation_validity"](result, example) == 1.0
