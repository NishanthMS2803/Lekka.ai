"""Valuation multiples and historical percentile ranking calculations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


def pe_ratio(
    price: float,
    eps: float | None,
    *,
    allow_negative: bool = False,
) -> float | None:
    """Compute Price-to-Earnings (P/E) ratio.

    Args:
        price: Current stock price.
        eps: Diluted Earnings Per Share (EPS).
        allow_negative: Whether to return negative P/E for loss-making firms.

    Returns:
        P/E ratio, or None if EPS is None, 0, or negative (when allow_negative=False).
    """
    if eps is None or eps == 0.0:
        return None
    if eps < 0.0 and not allow_negative:
        return None
    return float(price / eps)


def forward_pe_ratio(
    price: float,
    forward_eps: float | None,
    *,
    allow_negative: bool = False,
) -> float | None:
    """Compute Forward Price-to-Earnings ratio using forward consensus EPS.

    Args:
        price: Current stock price.
        forward_eps: Consensus forward EPS estimate.
        allow_negative: Whether to return negative forward P/E.

    Returns:
        Forward P/E ratio, or None if forward_eps is missing or invalid.
    """
    return pe_ratio(price, forward_eps, allow_negative=allow_negative)


def ev_to_ebitda(
    enterprise_value: float,
    ebitda: float | None,
    *,
    allow_negative: bool = False,
) -> float | None:
    """Compute Enterprise Value to EBITDA (EV/EBITDA) multiple.

    Args:
        enterprise_value: Enterprise value (Market Cap + Debt - Cash).
        ebitda: Earnings Before Interest, Taxes, Depreciation, and Amortization.
        allow_negative: Whether to allow negative EV/EBITDA ratios.

    Returns:
        EV/EBITDA multiple, or None if EBITDA is None, zero, or negative.
    """
    if ebitda is None or ebitda == 0.0:
        return None
    if ebitda < 0.0 and not allow_negative:
        return None
    return float(enterprise_value / ebitda)


def ev_to_sales(
    enterprise_value: float,
    sales: float | None,
) -> float | None:
    """Compute Enterprise Value to Sales (EV/Sales) multiple.

    Args:
        enterprise_value: Enterprise value.
        sales: Total revenue / sales.

    Returns:
        EV/Sales multiple, or None if sales is None or <= 0.
    """
    if sales is None or sales <= 0.0:
        return None
    return float(enterprise_value / sales)


def price_to_fcf(
    price: float | None = None,
    fcf_per_share: float | None = None,
    *,
    market_cap: float | None = None,
    fcf: float | None = None,
    allow_negative: bool = False,
) -> float | None:
    """Compute Price to Free Cash Flow (P/FCF) ratio.

    Can be called either with per-share metrics (price, fcf_per_share)
    or aggregate company-wide metrics (market_cap, fcf).

    Args:
        price: Stock price per share.
        fcf_per_share: Free Cash Flow per share.
        market_cap: Total market capitalization.
        fcf: Total company Free Cash Flow.
        allow_negative: Whether to return negative P/FCF.

    Returns:
        P/FCF multiple, or None if cash flow is missing, zero, or negative.
    """
    if market_cap is not None and fcf is not None:
        if fcf == 0.0:
            return None
        if fcf < 0.0 and not allow_negative:
            return None
        return float(market_cap / fcf)

    if price is not None and fcf_per_share is not None:
        if fcf_per_share == 0.0:
            return None
        if fcf_per_share < 0.0 and not allow_negative:
            return None
        return float(price / fcf_per_share)

    return None


def peg_ratio(
    pe: float | None,
    earnings_growth_rate: float | None,
    *,
    growth_as_percentage: bool | None = None,
) -> float | None:
    """Compute Price/Earnings to Growth (PEG) ratio.

    PEG = (P/E) / (Annual Earnings Growth Rate).

    Args:
        pe: Price-to-Earnings ratio.
        earnings_growth_rate: Expected annual earnings growth rate.
        growth_as_percentage: If True, growth is treated as percentage points (e.g. 15.0
            for 15%). If False, treated as decimal (e.g. 0.15). If None, auto-detects:
            growth <= 1.0 is treated as decimal, > 1.0 as percent.

    Returns:
        PEG ratio, or None if inputs are missing, growth <= 0, or P/E <= 0.
    """
    if pe is None or earnings_growth_rate is None:
        return None
    if pe <= 0.0 or earnings_growth_rate <= 0.0:
        return None

    if growth_as_percentage is True:
        rate = earnings_growth_rate
    elif growth_as_percentage is False:
        rate = earnings_growth_rate * 100.0
    else:
        # Auto-detect: e.g. 0.15 -> 15.0%, but 15.0 -> 15.0%
        rate = (
            earnings_growth_rate * 100.0
            if earnings_growth_rate <= 1.0
            else earnings_growth_rate
        )

    if rate == 0.0:
        return None

    return float(pe / rate)


def multiple_percentile(
    current: float | None,
    history: Sequence[float],
) -> float | None:
    """Compute the percentile rank (0.0 to 100.0) of a current multiple vs history.

    Uses standard percentile ranking: percentage of historical values that are
    strictly below the current value plus half the percentage equal to current.

    Args:
        current: Current multiple value.
        history: Sequence of historical multiple observations.

    Returns:
        Percentile rank from 0.0 to 100.0, or None if current is None or history empty.
    """
    if current is None:
        return None

    valid_history = [v for v in history if v is not None and not (v != v)]
    n = len(valid_history)
    if n == 0:
        return None

    less_than = sum(1 for v in valid_history if v < current)
    equal_to = sum(1 for v in valid_history if v == current)

    rank = (less_than + 0.5 * equal_to) / float(n) * 100.0
    return float(max(0.0, min(100.0, rank)))


@dataclass
class MultiplesSummary:
    """Comprehensive summary of valuation multiples and historical percentiles."""

    pe: float | None = None
    forward_pe: float | None = None
    ev_ebitda: float | None = None
    ev_sales: float | None = None
    price_fcf: float | None = None
    peg: float | None = None
    percentiles: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


def compute_multiples(
    *,
    price: float,
    eps: float | None = None,
    forward_eps: float | None = None,
    enterprise_value: float | None = None,
    ebitda: float | None = None,
    sales: float | None = None,
    fcf: float | None = None,
    market_cap: float | None = None,
    earnings_growth_rate: float | None = None,
    historical_pe: Sequence[float] | None = None,
    historical_ev_ebitda: Sequence[float] | None = None,
) -> MultiplesSummary:
    """Compute full valuation multiples suite and historical percentiles."""
    m_pe = pe_ratio(price, eps)
    m_fwd_pe = forward_pe_ratio(price, forward_eps)
    m_ev_ebitda = (
        ev_to_ebitda(enterprise_value, ebitda) if enterprise_value is not None else None
    )
    m_ev_sales = (
        ev_to_sales(enterprise_value, sales) if enterprise_value is not None else None
    )
    m_p_fcf = price_to_fcf(market_cap=market_cap, fcf=fcf)
    m_peg = peg_ratio(m_pe, earnings_growth_rate)

    pcts: dict[str, float] = {}
    if historical_pe is not None and m_pe is not None:
        p_val = multiple_percentile(m_pe, historical_pe)
        if p_val is not None:
            pcts["pe"] = p_val

    if historical_ev_ebitda is not None and m_ev_ebitda is not None:
        p_val = multiple_percentile(m_ev_ebitda, historical_ev_ebitda)
        if p_val is not None:
            pcts["ev_ebitda"] = p_val

    return MultiplesSummary(
        pe=m_pe,
        forward_pe=m_fwd_pe,
        ev_ebitda=m_ev_ebitda,
        ev_sales=m_ev_sales,
        price_fcf=m_p_fcf,
        peg=m_peg,
        percentiles=pcts,
    )
