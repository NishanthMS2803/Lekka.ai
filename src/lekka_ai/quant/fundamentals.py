"""Fundamental financial metrics and accounting calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from lekka_ai.data.xbrl import PeriodFinancials


def yoy_growth(
    current: float | int | None,
    prior: float | int | None,
) -> float | None:
    """Compute Year-over-Year (YoY) growth rate: (current - prior) / abs(prior).

    Args:
        current: Metric value for the current period.
        prior: Metric value for the prior comparable period.

    Returns:
        Growth rate as a float (e.g. 0.15 for +15%), or None if either value is
        missing or prior is zero.
    """
    if current is None or prior is None:
        return None
    if prior == 0:
        return None
    return float((current - prior) / abs(prior))


def operating_margin(
    operating_income: float | int | None,
    revenues: float | int | None,
) -> float | None:
    """Compute operating margin: operating_income / revenues.

    Args:
        operating_income: Operating income (EBIT).
        revenues: Total revenues or net sales.

    Returns:
        Operating margin as a float, or None if inputs are missing or revenues is 0.
    """
    if operating_income is None or revenues is None or revenues == 0:
        return None
    return float(operating_income / revenues)


def net_margin(
    net_income: float | int | None,
    revenues: float | int | None,
) -> float | None:
    """Compute net profit margin: net_income / revenues.

    Args:
        net_income: Net income (loss).
        revenues: Total revenues or net sales.

    Returns:
        Net margin as a float, or None if inputs are missing or revenues is 0.
    """
    if net_income is None or revenues is None or revenues == 0:
        return None
    return float(net_income / revenues)


def free_cash_flow(
    operating_cash_flow: float | int | None,
    capex: float | int | None,
) -> float | int | None:
    """Compute free cash flow: Cash from Operations (CFO) minus Capex.

    In corporate financial statements, capex (payments to acquire PPE) is typically
    reported as a positive cash outflow magnitude. If capex is reported negative,
    its absolute value is subtracted.

    Args:
        operating_cash_flow: Cash provided by operating activities (CFO).
        capex: Payments to acquire property, plant, and equipment.

    Returns:
        Free cash flow value, or None if inputs are missing.
    """
    if operating_cash_flow is None or capex is None:
        return None
    return operating_cash_flow - abs(capex)


def net_debt(
    debt: float | int | None,
    cash_and_cash_equivalents: float | int | None,
) -> float | int | None:
    """Compute net debt: Total Debt minus Cash and Cash Equivalents.

    Args:
        debt: Total or long-term debt.
        cash_and_cash_equivalents: Cash and cash equivalents.

    Returns:
        Net debt value, or None if both inputs are missing.
    """
    if debt is None and cash_and_cash_equivalents is None:
        return None
    d_val = debt if debt is not None else 0
    c_val = cash_and_cash_equivalents if cash_and_cash_equivalents is not None else 0
    return d_val - c_val


class DerivedFinancials(BaseModel):
    """Derived fundamental and profitability metrics for a fiscal period."""

    model_config = ConfigDict(extra="allow")

    revenue_yoy: float | None = Field(
        default=None, description="Year-over-year revenue growth rate"
    )
    net_income_yoy: float | None = Field(
        default=None, description="Year-over-year net income growth rate"
    )
    operating_margin: float | None = Field(
        default=None, description="Operating margin (operating income / revenues)"
    )
    net_margin: float | None = Field(
        default=None, description="Net margin (net income / revenues)"
    )
    free_cash_flow: float | int | None = Field(
        default=None, description="Free cash flow (CFO minus capex)"
    )
    net_debt: float | int | None = Field(
        default=None, description="Net debt (Debt minus Cash & Equivalents)"
    )


def compute_derived_financials(
    current: PeriodFinancials,
    prior: PeriodFinancials | None = None,
) -> DerivedFinancials:
    """Compute derived fundamental metrics for a period relative to a prior period."""
    rev_yoy = (
        yoy_growth(current.revenues, prior.revenues) if prior is not None else None
    )
    ni_yoy = (
        yoy_growth(current.net_income, prior.net_income) if prior is not None else None
    )
    op_m = operating_margin(current.operating_income, current.revenues)
    net_m = net_margin(current.net_income, current.revenues)
    fcf = free_cash_flow(current.operating_cash_flow, current.capex)
    n_debt = net_debt(current.debt, current.cash_and_cash_equivalents)

    return DerivedFinancials(
        revenue_yoy=rev_yoy,
        net_income_yoy=ni_yoy,
        operating_margin=op_m,
        net_margin=net_m,
        free_cash_flow=fcf,
        net_debt=n_debt,
    )
