"""Tests for Reverse DCF solving for market-implied growth rates and margins."""

import pytest

from lekka_ai.quant.dcf import compute_dcf
from lekka_ai.quant.reverse_dcf import (
    ReverseDCFResult,
    reverse_dcf,
    solve_implied_growth,
    solve_implied_margin,
)


def test_reverse_dcf_round_trip_recovers_growth_rate() -> None:
    """Compute forward DCF value and confirm solver recovers original growth rate."""
    original_growth = 0.08  # 8% annual FCF growth
    starting_fcf = 100.0
    forecast_years = 5
    discount_rate = 0.10
    terminal_growth = 0.02
    shares_outstanding = 50.0
    net_debt = 100.0

    # 1. Forward DCF calculation
    forward_res = compute_dcf(
        starting_fcf=starting_fcf,
        fcf_growth_path=[original_growth] * forecast_years,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        shares_outstanding=shares_outstanding,
        net_debt=net_debt,
    )
    target_price = forward_res.fair_value_per_share
    assert target_price > 0.0

    # 2. Reverse DCF solver
    reverse_res = reverse_dcf(
        target_price=target_price,
        starting_fcf=starting_fcf,
        forecast_years=forecast_years,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        shares_outstanding=shares_outstanding,
        net_debt=net_debt,
        solve_for="growth",
    )

    assert isinstance(reverse_res, ReverseDCFResult)
    assert reverse_res.converged is True
    assert reverse_res.implied_value is not None
    # Confirm recovered growth rate matches original growth rate
    assert pytest.approx(reverse_res.implied_value, rel=1e-5) == original_growth
    assert reverse_res.fair_value_at_implied is not None
    assert pytest.approx(reverse_res.fair_value_at_implied, rel=1e-5) == target_price
    assert reverse_res.dcf_result is not None


def test_reverse_dcf_round_trip_negative_and_high_growth() -> None:
    """Verify solver recovers negative and high growth rates accurately."""
    starting_fcf = 50.0
    forecast_years = 4
    discount_rate = 0.09
    terminal_growth = 0.02
    shares_outstanding = 20.0
    net_debt = 10.0

    for test_growth in [-0.05, 0.0, 0.15, 0.30]:
        fwd = compute_dcf(
            starting_fcf=starting_fcf,
            fcf_growth_path=[test_growth] * forecast_years,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
            shares_outstanding=shares_outstanding,
            net_debt=net_debt,
        )
        price = fwd.fair_value_per_share

        rev = solve_implied_growth(
            target_price=price,
            starting_fcf=starting_fcf,
            forecast_years=forecast_years,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
            shares_outstanding=shares_outstanding,
            net_debt=net_debt,
        )
        assert rev.converged is True
        assert rev.implied_value is not None
        assert pytest.approx(rev.implied_value, abs=1e-5) == test_growth


def test_reverse_dcf_round_trip_recovers_margin() -> None:
    """Compute forward DCF value with margin and confirm solver recovers margin."""
    original_margin = 0.18  # 18% FCF margin
    starting_revenue = 1000.0
    revenue_growth_path = [0.08, 0.07, 0.06, 0.05, 0.05]
    forecast_years = len(revenue_growth_path)
    discount_rate = 0.10
    terminal_growth = 0.02
    shares_outstanding = 40.0
    net_debt = 50.0

    # 1. Forward DCF calculation
    fwd = compute_dcf(
        starting_revenue=starting_revenue,
        revenue_growth_path=revenue_growth_path,
        margin_path=[original_margin] * forecast_years,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        shares_outstanding=shares_outstanding,
        net_debt=net_debt,
    )
    price = fwd.fair_value_per_share

    # 2. Reverse DCF solving for margin
    rev = solve_implied_margin(
        target_price=price,
        starting_revenue=starting_revenue,
        revenue_growth_path=revenue_growth_path,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        shares_outstanding=shares_outstanding,
        net_debt=net_debt,
    )

    assert rev.converged is True
    assert rev.implied_value is not None
    assert pytest.approx(rev.implied_value, rel=1e-5) == original_margin
    assert rev.fair_value_at_implied is not None
    assert pytest.approx(rev.fair_value_at_implied, rel=1e-5) == price


def test_reverse_dcf_unified_entrypoint_with_margin() -> None:
    """Unified reverse_dcf function delegates correctly when solve_for='margin'."""
    rev = reverse_dcf(
        target_price=25.0,
        starting_revenue=500.0,
        revenue_growth_path=0.05,
        forecast_years=5,
        discount_rate=0.09,
        terminal_growth=0.02,
        shares_outstanding=20.0,
        net_debt=0.0,
        solve_for="margin",
    )
    assert rev.converged is True
    assert rev.implied_value is not None
    assert rev.parameter_name == "margin"


def test_reverse_dcf_invalid_inputs() -> None:
    """Reverse DCF validates target price and required parameters."""
    with pytest.raises(ValueError, match="strictly positive"):
        reverse_dcf(
            target_price=0.0,
            starting_fcf=100.0,
            discount_rate=0.10,
            terminal_growth=0.02,
            shares_outstanding=10.0,
        )

    with pytest.raises(ValueError, match="starting_revenue must be provided"):
        reverse_dcf(
            target_price=50.0,
            discount_rate=0.10,
            terminal_growth=0.02,
            shares_outstanding=10.0,
            solve_for="margin",
        )

    with pytest.raises(ValueError, match="Unknown solve_for parameter"):
        reverse_dcf(
            target_price=50.0,
            starting_fcf=100.0,
            discount_rate=0.10,
            terminal_growth=0.02,
            shares_outstanding=10.0,
            solve_for="invalid_param",
        )


def test_reverse_dcf_subscription_access() -> None:
    """ReverseDCFResult supports dict-like subscription access."""
    res = ReverseDCFResult(
        target_price=100.0,
        implied_value=0.10,
        converged=True,
        parameter_name="growth_rate",
        iterations=5,
    )
    assert res["target_price"] == 100.0
    assert res["implied_value"] == 0.10
    assert res["converged"] is True
