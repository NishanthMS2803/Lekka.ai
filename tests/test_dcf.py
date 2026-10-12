"""Tests for Discounted Cash Flow (DCF) valuation engine."""

import pytest

from lekka_ai.quant.dcf import (
    CashFlowPeriod,
    DCFAssumptions,
    DCFResult,
    compute_dcf,
    dcf,
    validate_dcf_inputs,
)


def test_dcf_hand_calculated_fcf_textbook_example() -> None:
    """Verify DCF output matches an exact textbook hand calculation."""
    # Hand-checked parameters:
    # Starting FCF = 100
    # 3-year forecast @ 10% annual growth:
    # Year 1 FCF = 110.0 -> PV @ 10% = 110 / 1.10 = 100.0
    # Year 2 FCF = 121.0 -> PV @ 10% = 121 / 1.21 = 100.0
    # Year 3 FCF = 133.1 -> PV @ 10% = 133.1 / 1.331 = 100.0
    # Sum PV of explicit cash flows = 300.0
    # Terminal FCF = 133.1 * 1.02 = 135.762
    # Terminal Value = 135.762 / (0.10 - 0.02) = 1697.025
    # PV of Terminal Value = 1697.025 / 1.331 = 1275.0
    # Enterprise Value = 300.0 + 1275.0 = 1575.0
    # Equity Value = 1575.0 - 75.0 = 1500.0
    # Fair Value per share = 1500.0 / 50.0 = 30.0

    res = compute_dcf(
        starting_fcf=100.0,
        fcf_growth_path=[0.10, 0.10, 0.10],
        discount_rate=0.10,
        terminal_growth=0.02,
        shares_outstanding=50.0,
        net_debt=75.0,
    )

    assert isinstance(res, DCFResult)
    assert len(res.cash_flow_table) == 3

    # Check cash flow table breakdown
    p1 = res.cash_flow_table[0]
    assert p1.year == 1
    assert pytest.approx(p1.fcf, rel=1e-6) == 110.0
    assert pytest.approx(p1.present_value, rel=1e-6) == 100.0

    p2 = res.cash_flow_table[1]
    assert p2.year == 2
    assert pytest.approx(p2.fcf, rel=1e-6) == 121.0
    assert pytest.approx(p2.present_value, rel=1e-6) == 100.0

    p3 = res.cash_flow_table[2]
    assert p3.year == 3
    assert pytest.approx(p3.fcf, rel=1e-6) == 133.1
    assert pytest.approx(p3.present_value, rel=1e-6) == 100.0

    # Check aggregate metrics matching hand calculation
    assert pytest.approx(res.pv_projected_fcf, rel=1e-6) == 300.0
    assert pytest.approx(res.terminal_fcf, rel=1e-6) == 135.762
    assert pytest.approx(res.terminal_value, rel=1e-6) == 1697.025
    assert pytest.approx(res.pv_terminal_value, rel=1e-6) == 1275.0
    assert pytest.approx(res.enterprise_value, rel=1e-6) == 1575.0
    assert pytest.approx(res.equity_value, rel=1e-6) == 1500.0
    assert pytest.approx(res.fair_value_per_share, rel=1e-6) == 30.0


def test_dcf_hand_calculated_revenue_and_margin_path() -> None:
    """Verify DCF output using revenue growth and margin path projections."""
    # Starting Revenue = 1000, 10% growth per year, 10% FCF margin
    # Results should be identical to the 100 starting FCF case above
    res = compute_dcf(
        starting_revenue=1000.0,
        revenue_growth_path=[0.10, 0.10, 0.10],
        margin_path=[0.10, 0.10, 0.10],
        discount_rate=0.10,
        terminal_growth=0.02,
        shares_outstanding=50.0,
        net_debt=75.0,
    )

    assert len(res.cash_flow_table) == 3
    p1 = res.cash_flow_table[0]
    assert pytest.approx(p1.revenue or 0.0, rel=1e-6) == 1100.0
    assert pytest.approx(p1.margin or 0.0, rel=1e-6) == 0.10
    assert pytest.approx(p1.fcf, rel=1e-6) == 110.0
    assert pytest.approx(p1.present_value, rel=1e-6) == 100.0

    assert pytest.approx(res.enterprise_value, rel=1e-6) == 1575.0
    assert pytest.approx(res.equity_value, rel=1e-6) == 1500.0
    assert pytest.approx(res.fair_value_per_share, rel=1e-6) == 30.0


def test_dcf_validation_terminal_growth_greater_than_or_equal_to_discount() -> None:
    """Validation fails if terminal growth >= discount rate."""
    with pytest.raises(ValueError, match="strictly less than discount rate"):
        compute_dcf(
            starting_fcf=100.0,
            fcf_growth_path=[0.05],
            discount_rate=0.08,
            terminal_growth=0.08,  # Equal to discount rate
            shares_outstanding=10.0,
        )

    with pytest.raises(ValueError, match="strictly less than discount rate"):
        compute_dcf(
            starting_fcf=100.0,
            fcf_growth_path=[0.05],
            discount_rate=0.08,
            terminal_growth=0.09,  # Greater than discount rate
            shares_outstanding=10.0,
        )


def test_dcf_validation_discount_rate_positive() -> None:
    """Validation fails if discount rate is zero or negative."""
    with pytest.raises(ValueError, match="strictly positive"):
        validate_dcf_inputs(
            discount_rate=0.0,
            terminal_growth=0.02,
            shares_outstanding=10.0,
            starting_fcf=100.0,
            fcf_growth_path=[0.05],
        )


def test_dcf_validation_shares_outstanding_positive() -> None:
    """Validation fails if shares outstanding is zero or negative."""
    with pytest.raises(ValueError, match="strictly positive"):
        validate_dcf_inputs(
            discount_rate=0.10,
            terminal_growth=0.02,
            shares_outstanding=0.0,
            starting_fcf=100.0,
            fcf_growth_path=[0.05],
        )


def test_dcf_validation_mismatched_revenue_margin_paths() -> None:
    """Validation fails if revenue and margin paths have differing lengths."""
    with pytest.raises(ValueError, match="must match length of margin_path"):
        validate_dcf_inputs(
            discount_rate=0.10,
            terminal_growth=0.02,
            shares_outstanding=10.0,
            starting_revenue=500.0,
            revenue_growth_path=[0.10, 0.08],
            margin_path=[0.15],  # Length mismatch
        )


def test_dcf_validation_missing_paths() -> None:
    """Validation fails if neither FCF path nor revenue/margin path is given."""
    with pytest.raises(ValueError, match="Must provide either"):
        validate_dcf_inputs(
            discount_rate=0.10,
            terminal_growth=0.02,
            shares_outstanding=10.0,
        )


def test_dcf_convenience_entrypoint_with_assumptions_object() -> None:
    """Convenience function dcf works with DCFAssumptions object."""
    assumptions = DCFAssumptions(
        starting_fcf=50.0,
        fcf_growth_path=[0.05, 0.05],
        discount_rate=0.09,
        terminal_growth=0.02,
        shares_outstanding=25.0,
        net_debt=10.0,
    )
    result = dcf(assumptions)
    assert isinstance(result, DCFResult)
    assert result.enterprise_value > 0
    assert result.fair_value_per_share > 0

    # Check dictionary serialization
    res_dict = result.to_dict()
    assert "enterprise_value" in res_dict
    assert "cash_flow_table" in res_dict
    assert len(res_dict["cash_flow_table"]) == 2


def test_dcf_mid_year_convention() -> None:
    """Mid-year discounting yields higher PV than end-of-year discounting."""
    end_of_year = compute_dcf(
        starting_fcf=100.0,
        fcf_growth_path=[0.05, 0.05],
        discount_rate=0.10,
        terminal_growth=0.02,
        shares_outstanding=10.0,
        mid_year_convention=False,
    )
    mid_year = compute_dcf(
        starting_fcf=100.0,
        fcf_growth_path=[0.05, 0.05],
        discount_rate=0.10,
        terminal_growth=0.02,
        shares_outstanding=10.0,
        mid_year_convention=True,
    )
    assert mid_year.pv_projected_fcf > end_of_year.pv_projected_fcf


def test_cash_flow_period_dict_access() -> None:
    """CashFlowPeriod supports subscription access."""
    p = CashFlowPeriod(
        year=1,
        fcf=100.0,
        discount_factor=0.9,
        present_value=90.0,
    )
    assert p["year"] == 1
    assert p["fcf"] == 100.0
