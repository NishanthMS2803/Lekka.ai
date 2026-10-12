"""Tests for valuation multiples and historical percentile ranking."""

import pytest

from lekka_ai.quant.multiples import (
    MultiplesSummary,
    compute_multiples,
    ev_to_ebitda,
    ev_to_sales,
    forward_pe_ratio,
    multiple_percentile,
    pe_ratio,
    peg_ratio,
    price_to_fcf,
)


def test_pe_ratio() -> None:
    """Test Price-to-Earnings computation and edge cases."""
    assert pe_ratio(150.0, 5.0) == 30.0
    assert pe_ratio(150.0, None) is None
    assert pe_ratio(150.0, 0.0) is None
    # Negative EPS returns None by default, float when allow_negative=True
    assert pe_ratio(150.0, -3.0) is None
    assert pe_ratio(150.0, -3.0, allow_negative=True) == -50.0


def test_forward_pe_ratio() -> None:
    """Test Forward P/E calculation."""
    assert forward_pe_ratio(120.0, 6.0) == 20.0
    assert forward_pe_ratio(120.0, None) is None


def test_ev_to_ebitda() -> None:
    """Test Enterprise Value to EBITDA calculation."""
    assert ev_to_ebitda(1000.0, 100.0) == 10.0
    assert ev_to_ebitda(1000.0, None) is None
    assert ev_to_ebitda(1000.0, 0.0) is None
    assert ev_to_ebitda(1000.0, -50.0) is None
    assert ev_to_ebitda(1000.0, -50.0, allow_negative=True) == -20.0


def test_ev_to_sales() -> None:
    """Test Enterprise Value to Sales calculation."""
    assert ev_to_sales(1200.0, 400.0) == 3.0
    assert ev_to_sales(1200.0, None) is None
    assert ev_to_sales(1200.0, 0.0) is None
    assert ev_to_sales(1200.0, -100.0) is None


def test_price_to_fcf() -> None:
    """Test Price to Free Cash Flow using per-share and aggregate metrics."""
    # Per-share inputs
    assert price_to_fcf(price=100.0, fcf_per_share=4.0) == 25.0
    assert price_to_fcf(price=100.0, fcf_per_share=0.0) is None
    assert price_to_fcf(price=100.0, fcf_per_share=-2.0) is None
    assert price_to_fcf(price=100.0, fcf_per_share=-2.0, allow_negative=True) == -50.0

    # Aggregate company inputs
    assert price_to_fcf(market_cap=2_000_000.0, fcf=100_000.0) == 20.0
    assert price_to_fcf(market_cap=2_000_000.0, fcf=0.0) is None


def test_peg_ratio() -> None:
    """Test PEG ratio calculation with percentage and decimal growth rates."""
    # P/E = 30, Growth = 15% -> PEG = 30 / 15 = 2.0
    # Auto-detected decimal (0.15 -> 15%)
    assert pytest.approx(peg_ratio(30.0, 0.15) or 0.0, rel=1e-5) == 2.0
    # Explicit percentage points (15.0)
    assert (
        pytest.approx(peg_ratio(30.0, 15.0, growth_as_percentage=True) or 0.0, rel=1e-5)
        == 2.0
    )

    # Missing or invalid inputs
    assert peg_ratio(None, 0.15) is None
    assert peg_ratio(30.0, None) is None
    assert peg_ratio(30.0, 0.0) is None
    assert peg_ratio(30.0, -0.05) is None
    assert peg_ratio(-10.0, 0.15) is None


def test_multiple_percentile() -> None:
    """Test percentile ranking of current multiple vs historical observations."""
    history = [10.0, 15.0, 20.0, 25.0, 30.0]

    # Current = 20.0 (exact median) -> (2 below + 0.5 equal) / 5 = 2.5 / 5 = 50.0%
    assert multiple_percentile(20.0, history) == 50.0

    # Current = 5.0 (below all) -> 0.0%
    assert multiple_percentile(5.0, history) == 0.0

    # Current = 35.0 (above all) -> 100.0%
    assert multiple_percentile(35.0, history) == 100.0

    # None and empty checks
    assert multiple_percentile(None, history) is None
    assert multiple_percentile(20.0, []) is None


def test_compute_multiples_summary() -> None:
    """Test compute_multiples aggregate helper function."""
    summary = compute_multiples(
        price=180.0,
        eps=6.0,
        forward_eps=7.2,
        enterprise_value=2700.0,
        ebitda=180.0,
        sales=900.0,
        market_cap=2700.0,
        fcf=135.0,
        earnings_growth_rate=0.20,
        historical_pe=[15.0, 20.0, 25.0, 30.0, 35.0],
    )

    assert isinstance(summary, MultiplesSummary)
    assert summary.pe == 30.0
    assert summary.forward_pe == 25.0
    assert summary.ev_ebitda == 15.0
    assert summary.ev_sales == 3.0
    assert summary.price_fcf == 20.0
    assert pytest.approx(summary.peg or 0.0, rel=1e-5) == 1.5
    assert "pe" in summary.percentiles
    assert summary["pe"] == 30.0
