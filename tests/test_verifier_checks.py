"""Tests for deterministic verification checks (calculations, types, dates)."""

from __future__ import annotations

from lekka_ai.schemas.claims import Claim, ClaimType, Evidence
from lekka_ai.verify.checks import (
    CheckStatus,
    check_background_not_sole_support,
    check_calculation_recomputation,
    check_evidence_dates_cutoff,
    check_interpretation_supporting_ids,
    verify_deterministic_checks,
)

# ===========================================================================
# Check (a): Calculation Claims Recomputation Tests
# ===========================================================================


def test_calculation_recomputation_pass() -> None:
    """Check (a) passes when calculation matches expected value within tolerance."""
    # 1. Direct arithmetic formula
    claim_1 = Claim(
        claim_id="calc_margin_01",
        text="Operating margin is 7.8%.",
        type=ClaimType.CALCULATION,
        derived_by="29915 / 383285",
        confidence=0.95,
    )
    res_1 = check_calculation_recomputation(claim_1, tolerance=0.01)
    assert res_1.passed is True
    assert res_1.status == CheckStatus.PASSED
    assert "matches expected" in res_1.message

    # 2. Formula with variable inputs in evidence
    claim_2 = Claim(
        claim_id="calc_op_income_02",
        text="Operating margin for FY23 was 29.8%.",
        type=ClaimType.CALCULATION,
        derived_by="operating_income / revenue",
        evidence=[
            Evidence(
                source_type="sec_filing",
                tool_call_id="tc_sec_01",
                raw_value={"operating_income": 114301.0, "revenue": 383285.0},
            )
        ],
        confidence=0.96,
    )
    res_2 = check_calculation_recomputation(claim_2, tolerance=0.01)
    assert res_2.passed is True
    assert res_2.status == CheckStatus.PASSED


def test_calculation_recomputation_fail() -> None:
    """Check (a) fails when calculation deviates beyond tolerance or has errors."""
    # 1. Deviates from claimed value
    claim_wrong = Claim(
        claim_id="calc_wrong_val",
        text="Operating margin is 45.0%.",
        type=ClaimType.CALCULATION,
        derived_by="29915 / 383285",  # Evaluates to ~7.8%, not 45.0%
        confidence=0.9,
    )
    res_wrong = check_calculation_recomputation(claim_wrong, tolerance=0.01)
    assert res_wrong.passed is False
    assert res_wrong.status == CheckStatus.FAILED
    assert "deviates from stated value" in res_wrong.message

    # 2. Division by zero in formula
    claim_zero = Claim(
        claim_id="calc_zero_div",
        text="Ratio is 10.0.",
        type=ClaimType.CALCULATION,
        derived_by="100.0 / 0.0",
        confidence=0.5,
    )
    res_zero = check_calculation_recomputation(claim_zero)
    assert res_zero.passed is False
    assert res_zero.status == CheckStatus.FAILED
    assert "Division by zero" in res_zero.message


# ===========================================================================
# Check (b): Interpretation Supporting IDs Tests
# ===========================================================================


def test_interpretation_supporting_ids_pass() -> None:
    """Check (b) passes when all prerequisite claim IDs exist in repository."""
    claim = Claim(
        claim_id="interp_thesis_01",
        text="Apple possesses durable pricing power in consumer tech.",
        type=ClaimType.INTERPRETATION,
        depends_on=["fact_rev_2023", "calc_margin_2023"],
        confidence=0.88,
    )
    known_ids = {"fact_rev_2023", "calc_margin_2023", "fact_cash_2023"}

    res = check_interpretation_supporting_ids(claim, existing_claim_ids=known_ids)
    assert res.passed is True
    assert res.status == CheckStatus.PASSED
    assert "cites valid supporting claim IDs" in res.message


def test_interpretation_supporting_ids_fail() -> None:
    """Check (b) fails when cited prerequisite IDs are missing or list is empty."""
    # 1. Missing / hallucinated claim ID
    claim_missing = Claim(
        claim_id="interp_thesis_02",
        text="Apple has a dominant moat.",
        type=ClaimType.INTERPRETATION,
        depends_on=["fact_rev_2023", "ghost_claim_999"],
        confidence=0.85,
    )
    known_ids = {"fact_rev_2023"}

    res_missing = check_interpretation_supporting_ids(
        claim_missing, existing_claim_ids=known_ids
    )
    assert res_missing.passed is False
    assert res_missing.status == CheckStatus.FAILED
    assert "ghost_claim_999" in res_missing.message

    # 2. Empty depends_on dictionary input
    dict_claim_empty = {
        "claim_id": "interp_empty_deps",
        "text": "Unsupported thesis statement.",
        "type": "interpretation",
        "depends_on": [],
        "confidence": 0.5,
    }
    res_empty = check_interpretation_supporting_ids(
        dict_claim_empty, existing_claim_ids=known_ids
    )
    assert res_empty.passed is False
    assert res_empty.status == CheckStatus.FAILED
    assert "empty 'depends_on'" in res_empty.message


# ===========================================================================
# Check (c): No Evidence Date Exceeds As-Of Date Tests
# ===========================================================================


def test_evidence_dates_cutoff_pass() -> None:
    """Check (c) passes when all evidence dates are on or before as_of_date."""
    claim = Claim(
        claim_id="fact_point_in_time",
        text="Net sales for Q4 was $89.5B.",
        type=ClaimType.FACT,
        evidence=[
            Evidence(
                source_type="sec_filing",
                tool_call_id="tc_sec_01",
                as_of_date="2023-09-30",
            ),
            Evidence(
                source_type="market_data",
                tool_call_id="tc_mkt_02",
                as_of_date="2024-01-10",
            ),
        ],
        confidence=0.95,
    )

    res = check_evidence_dates_cutoff(claim, as_of_date="2024-01-15")
    assert res.passed is True
    assert res.status == CheckStatus.PASSED
    assert "on or before as_of_date cutoff" in res.message


def test_evidence_dates_cutoff_fail() -> None:
    """Check (c) fails when evidence date exceeds as_of_date cutoff (lookahead)."""
    claim_future = Claim(
        claim_id="fact_lookahead_leak",
        text="Q1 FY24 revenue was announced.",
        type=ClaimType.FACT,
        evidence=[
            Evidence(
                source_type="sec_filing",
                tool_call_id="tc_sec_01",
                as_of_date="2024-02-01",  # Future date relative to cutoff
            )
        ],
        confidence=0.95,
    )

    res = check_evidence_dates_cutoff(claim_future, as_of_date="2024-01-15")
    assert res.passed is False
    assert res.status == CheckStatus.FAILED
    assert "lookahead bias detected" in res.message


# ===========================================================================
# Check (d): Background Not Sole Support Tests
# ===========================================================================


def test_background_not_sole_support_pass() -> None:
    """Check (d) passes when interpretation is grounded in empirical facts/calcs."""
    claim_fact = Claim(
        claim_id="fact_iphone_rev",
        text="iPhone net sales were $200B.",
        type=ClaimType.FACT,
        evidence=[
            Evidence(
                source_type="sec_filing",
                tool_call_id="tc_sec_01",
            )
        ],
        confidence=0.99,
    )
    claim_bg = Claim(
        claim_id="bg_smartphones",
        text="Smartphones are essential consumer electronics devices globally.",
        type=ClaimType.BACKGROUND,
        confidence=1.0,
    )
    claim_interp = Claim(
        claim_id="interp_product_strength",
        text="Apple maintains leadership in premium consumer handsets.",
        type=ClaimType.INTERPRETATION,
        depends_on=["fact_iphone_rev", "bg_smartphones"],
        confidence=0.9,
    )

    claims_map = {
        "fact_iphone_rev": claim_fact,
        "bg_smartphones": claim_bg,
        "interp_product_strength": claim_interp,
    }

    res = check_background_not_sole_support(claim_interp, claims_by_id=claims_map)
    assert res.passed is True
    assert res.status == CheckStatus.PASSED
    assert "grounded in substantive non-background evidence" in res.message


def test_background_not_sole_support_fail() -> None:
    """Check (d) fails when interpretation rests solely on background claims."""
    claim_bg_1 = Claim(
        claim_id="bg_hardware_def",
        text="Hardware businesses sell physical consumer technology.",
        type=ClaimType.BACKGROUND,
        confidence=1.0,
    )
    claim_bg_2 = Claim(
        claim_id="bg_branding_def",
        text="Strong consumer branding enables premium pricing.",
        type=ClaimType.BACKGROUND,
        confidence=1.0,
    )
    claim_interp_sole_bg = Claim(
        claim_id="interp_ungrounded_thesis",
        text="Apple will generate high returns because strong brands win.",
        type=ClaimType.INTERPRETATION,
        depends_on=["bg_hardware_def", "bg_branding_def"],
        confidence=0.8,
    )

    claims_map = {
        "bg_hardware_def": claim_bg_1,
        "bg_branding_def": claim_bg_2,
        "interp_ungrounded_thesis": claim_interp_sole_bg,
    }

    res = check_background_not_sole_support(
        claim_interp_sole_bg, claims_by_id=claims_map
    )
    assert res.passed is False
    assert res.status == CheckStatus.FAILED
    assert "solely supported by background claims" in res.message


# ===========================================================================
# Comprehensive Deterministic Suite Test
# ===========================================================================


def test_verify_deterministic_checks_suite() -> None:
    """Verify that composite suite runner correctly executes all four checks."""
    fact_claim = Claim(
        claim_id="c_fact_01",
        text="Apple reported $383.3B in FY23 revenue.",
        type=ClaimType.FACT,
        evidence=[
            Evidence(
                source_type="sec_filing",
                tool_call_id="tc_sec",
                as_of_date="2023-09-30",
            )
        ],
        confidence=0.98,
    )

    calc_claim = Claim(
        claim_id="c_calc_01",
        text="R&D margin is 7.8%.",
        type=ClaimType.CALCULATION,
        derived_by="29915 / 383285",
        confidence=0.95,
    )

    interp_claim = Claim(
        claim_id="c_interp_01",
        text="Apple reinvests heavily into proprietary silicon architecture.",
        type=ClaimType.INTERPRETATION,
        depends_on=["c_fact_01", "c_calc_01"],
        confidence=0.9,
    )

    # All valid claims should yield all_passed=True
    report_pass = verify_deterministic_checks(
        [fact_claim, calc_claim, interp_claim],
        as_of_date="2023-12-31",
        tolerance=0.01,
    )

    assert report_pass.all_passed is True
    assert len(report_pass.failures) == 0
    assert report_pass.passed_count > 0

    # Introduce a future date leak into fact_claim to trigger failure in suite
    leaked_fact = Claim(
        claim_id="c_leaked_02",
        text="Future Q2 revenue reached $95B.",
        type=ClaimType.FACT,
        evidence=[
            Evidence(
                source_type="sec_filing",
                tool_call_id="tc_future",
                as_of_date="2024-03-31",
            )
        ],
        confidence=0.9,
    )

    report_fail = verify_deterministic_checks(
        [fact_claim, calc_claim, interp_claim, leaked_fact],
        as_of_date="2023-12-31",
    )

    assert report_fail.all_passed is False
    assert len(report_fail.failures) >= 1
    assert any(f.check_name == "evidence_dates_cutoff" for f in report_fail.failures)
