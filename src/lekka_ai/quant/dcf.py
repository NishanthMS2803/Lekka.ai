"""Discounted Cash Flow (DCF) valuation engine and modeling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CashFlowPeriod:
    """Projected cash flow for an individual forecast year."""

    year: int
    fcf: float
    discount_factor: float
    present_value: float
    revenue: float | None = None
    revenue_growth: float | None = None
    margin: float | None = None
    fcf_growth: float | None = None

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


@dataclass
class DCFAssumptions:
    """Explicit parameters and scenario assumptions for DCF valuation."""

    discount_rate: float
    terminal_growth: float
    shares_outstanding: float
    starting_fcf: float | None = None
    fcf_growth_path: list[float] | None = None
    starting_revenue: float | None = None
    revenue_growth_path: list[float] | None = None
    margin_path: list[float] | None = None
    net_debt: float = 0.0
    mid_year_convention: bool = False

    def __post_init__(self) -> None:
        validate_dcf_inputs(
            discount_rate=self.discount_rate,
            terminal_growth=self.terminal_growth,
            shares_outstanding=self.shares_outstanding,
            starting_fcf=self.starting_fcf,
            fcf_growth_path=self.fcf_growth_path,
            starting_revenue=self.starting_revenue,
            revenue_growth_path=self.revenue_growth_path,
            margin_path=self.margin_path,
        )


@dataclass
class DCFResult:
    """Comprehensive output of a DCF valuation analysis."""

    enterprise_value: float
    equity_value: float
    fair_value_per_share: float
    pv_projected_fcf: float
    terminal_value: float
    pv_terminal_value: float
    terminal_fcf: float
    cash_flow_table: list[CashFlowPeriod]
    assumptions: DCFAssumptions
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def to_dict(self) -> dict[str, Any]:
        """Convert result and cash flow table to a dictionary representation."""
        return {
            "enterprise_value": self.enterprise_value,
            "equity_value": self.equity_value,
            "fair_value_per_share": self.fair_value_per_share,
            "pv_projected_fcf": self.pv_projected_fcf,
            "terminal_value": self.terminal_value,
            "pv_terminal_value": self.pv_terminal_value,
            "terminal_fcf": self.terminal_fcf,
            "cash_flow_table": [asdict(p) for p in self.cash_flow_table],
            "assumptions": asdict(self.assumptions),
        }


def validate_dcf_inputs(
    *,
    discount_rate: float,
    terminal_growth: float,
    shares_outstanding: float,
    starting_fcf: float | None = None,
    fcf_growth_path: list[float] | None = None,
    starting_revenue: float | None = None,
    revenue_growth_path: list[float] | None = None,
    margin_path: list[float] | None = None,
) -> None:
    """Validate input parameters for discounted cash flow calculations.

    Raises:
        ValueError: If any input fails financial or mathematical validity.
    """
    if discount_rate <= 0.0:
        raise ValueError(f"Discount rate ({discount_rate}) must be strictly positive.")

    if terminal_growth >= discount_rate:
        raise ValueError(
            f"Terminal growth rate ({terminal_growth:.4f}) must be strictly less than "
            f"discount rate ({discount_rate:.4f}) for Gordon Growth Model convergence."
        )

    if shares_outstanding <= 0.0:
        raise ValueError(
            f"Shares outstanding ({shares_outstanding}) must be strictly positive."
        )

    has_fcf_path = (
        starting_fcf is not None
        and fcf_growth_path is not None
        and len(fcf_growth_path) > 0
    )
    has_rev_path = (
        starting_revenue is not None
        and revenue_growth_path is not None
        and margin_path is not None
        and len(revenue_growth_path) > 0
        and len(margin_path) > 0
    )

    if not has_fcf_path and not has_rev_path:
        raise ValueError(
            "Must provide either (starting_fcf and non-empty fcf_growth_path) "
            "or (starting_revenue, non-empty revenue_growth_path, and margin_path)."
        )

    if has_rev_path and not has_fcf_path:
        assert revenue_growth_path is not None
        assert margin_path is not None
        if len(revenue_growth_path) != len(margin_path):
            raise ValueError(
                f"Length of revenue_growth_path ({len(revenue_growth_path)}) "
                f"must match length of margin_path ({len(margin_path)})."
            )


def compute_dcf(
    *,
    starting_fcf: float | None = None,
    fcf_growth_path: list[float] | None = None,
    fcf_growth: list[float] | None = None,
    starting_revenue: float | None = None,
    revenue_growth_path: list[float] | None = None,
    margin_path: list[float] | None = None,
    discount_rate: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float = 0.0,
    mid_year_convention: bool = False,
) -> DCFResult:
    """Execute Discounted Cash Flow (DCF) model and calculate enterprise & equity value.

    Args:
        starting_fcf: Base year Free Cash Flow.
        fcf_growth_path: Projected annual FCF growth rates.
        fcf_growth: Alias for fcf_growth_path.
        starting_revenue: Base year revenue.
        revenue_growth_path: Projected annual revenue growth rates.
        margin_path: Projected annual FCF margins (FCF / Revenue).
        discount_rate: Weighted Average Cost of Capital (WACC), e.g. 0.09.
        terminal_growth: Perpetual growth rate, e.g. 0.02.
        shares_outstanding: Diluted common shares outstanding.
        net_debt: Total debt minus cash and liquid investments.
        mid_year_convention: Whether to discount using mid-period assumption.

    Returns:
        DCFResult with EV, equity value, fair value per share, and cash flow table.
    """
    resolved_fcf_path = fcf_growth_path if fcf_growth_path is not None else fcf_growth

    validate_dcf_inputs(
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        shares_outstanding=shares_outstanding,
        starting_fcf=starting_fcf,
        fcf_growth_path=resolved_fcf_path,
        starting_revenue=starting_revenue,
        revenue_growth_path=revenue_growth_path,
        margin_path=margin_path,
    )

    assumptions = DCFAssumptions(
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        shares_outstanding=shares_outstanding,
        starting_fcf=starting_fcf,
        fcf_growth_path=resolved_fcf_path,
        starting_revenue=starting_revenue,
        revenue_growth_path=revenue_growth_path,
        margin_path=margin_path,
        net_debt=net_debt,
        mid_year_convention=mid_year_convention,
    )

    cash_flow_table: list[CashFlowPeriod] = []

    if resolved_fcf_path is not None and starting_fcf is not None:
        current_fcf = starting_fcf
        for year_idx, growth in enumerate(resolved_fcf_path, start=1):
            current_fcf = current_fcf * (1.0 + growth)
            t = (year_idx - 0.5) if mid_year_convention else float(year_idx)
            df = 1.0 / ((1.0 + discount_rate) ** t)
            pv = current_fcf * df
            cash_flow_table.append(
                CashFlowPeriod(
                    year=year_idx,
                    fcf=current_fcf,
                    fcf_growth=growth,
                    discount_factor=df,
                    present_value=pv,
                )
            )
    elif (
        starting_revenue is not None
        and revenue_growth_path is not None
        and margin_path is not None
    ):
        current_revenue = starting_revenue
        for year_idx, (rev_growth, margin) in enumerate(
            zip(revenue_growth_path, margin_path, strict=True), start=1
        ):
            current_revenue = current_revenue * (1.0 + rev_growth)
            fcf = current_revenue * margin
            t = (year_idx - 0.5) if mid_year_convention else float(year_idx)
            df = 1.0 / ((1.0 + discount_rate) ** t)
            pv = fcf * df
            cash_flow_table.append(
                CashFlowPeriod(
                    year=year_idx,
                    revenue=current_revenue,
                    revenue_growth=rev_growth,
                    margin=margin,
                    fcf=fcf,
                    discount_factor=df,
                    present_value=pv,
                )
            )

    n_years = len(cash_flow_table)
    pv_projected_fcf = sum(period.present_value for period in cash_flow_table)

    final_fcf = cash_flow_table[-1].fcf
    terminal_fcf = final_fcf * (1.0 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)

    terminal_df = 1.0 / ((1.0 + discount_rate) ** float(n_years))
    pv_terminal_value = terminal_value * terminal_df

    enterprise_value = pv_projected_fcf + pv_terminal_value
    equity_value = enterprise_value - net_debt
    fair_value_per_share = equity_value / shares_outstanding

    return DCFResult(
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        fair_value_per_share=fair_value_per_share,
        pv_projected_fcf=pv_projected_fcf,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal_value,
        terminal_fcf=terminal_fcf,
        cash_flow_table=cash_flow_table,
        assumptions=assumptions,
    )


def dcf(
    assumptions: DCFAssumptions | None = None,
    *,
    starting_fcf: float | None = None,
    fcf_growth_path: list[float] | None = None,
    fcf_growth: list[float] | None = None,
    starting_revenue: float | None = None,
    revenue_growth_path: list[float] | None = None,
    margin_path: list[float] | None = None,
    discount_rate: float | None = None,
    terminal_growth: float | None = None,
    shares_outstanding: float | None = None,
    net_debt: float = 0.0,
    mid_year_convention: bool = False,
) -> DCFResult:
    """Convenience entrypoint supporting both DCFAssumptions objects and kwargs."""
    if assumptions is not None:
        return compute_dcf(
            starting_fcf=assumptions.starting_fcf,
            fcf_growth_path=assumptions.fcf_growth_path,
            starting_revenue=assumptions.starting_revenue,
            revenue_growth_path=assumptions.revenue_growth_path,
            margin_path=assumptions.margin_path,
            discount_rate=assumptions.discount_rate,
            terminal_growth=assumptions.terminal_growth,
            shares_outstanding=assumptions.shares_outstanding,
            net_debt=assumptions.net_debt,
            mid_year_convention=assumptions.mid_year_convention,
        )

    if discount_rate is None or terminal_growth is None or shares_outstanding is None:
        raise ValueError(
            "Must provide discount_rate, terminal_growth, and shares_outstanding."
        )

    return compute_dcf(
        starting_fcf=starting_fcf,
        fcf_growth_path=fcf_growth_path,
        fcf_growth=fcf_growth,
        starting_revenue=starting_revenue,
        revenue_growth_path=revenue_growth_path,
        margin_path=margin_path,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        shares_outstanding=shares_outstanding,
        net_debt=net_debt,
        mid_year_convention=mid_year_convention,
    )
