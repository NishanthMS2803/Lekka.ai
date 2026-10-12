"""Reverse DCF modeling solving for market-implied growth rate or margins."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from scipy.optimize import brentq  # type: ignore[import-untyped]

from lekka_ai.quant.dcf import DCFResult, compute_dcf


@dataclass
class ReverseDCFResult:
    """Outcome of reverse DCF root-finding optimization."""

    target_price: float
    implied_value: float | None
    converged: bool
    parameter_name: str  # "growth_rate" or "margin"
    iterations: int | None = None
    fair_value_at_implied: float | None = None
    dcf_result: DCFResult | None = None
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


def _find_bracket(
    objective: Callable[[float], float],
    low: float,
    high: float,
    *,
    steps: int = 20,
) -> tuple[float, float] | None:
    """Search for an interval [a, b] where objective(a) * objective(b) <= 0.

    If low and high do not bracket a zero, grid search the interval, and then
    attempt expanding bounds outward.
    """
    f_low = objective(low)
    f_high = objective(high)

    if f_low * f_high <= 0:
        return (low, high)

    # Grid search between low and high
    step_size = (high - low) / steps
    prev_x = low
    prev_f = f_low
    for i in range(1, steps + 1):
        curr_x = low + i * step_size
        curr_f = objective(curr_x)
        if prev_f * curr_f <= 0:
            return (prev_x, curr_x)
        prev_x, prev_f = curr_x, curr_f

    # Attempt outward bracket expansion
    expand_low = low
    expand_high = high
    for _ in range(5):
        span = expand_high - expand_low
        expand_low = max(-0.95, expand_low - span * 0.5)
        expand_high = expand_high + span * 0.5

        f_el = objective(expand_low)
        f_eh = objective(expand_high)
        if f_el * f_eh <= 0:
            return (expand_low, expand_high)

    return None


def solve_implied_growth(
    target_price: float,
    *,
    starting_fcf: float | None = None,
    starting_revenue: float | None = None,
    fixed_margin: float | None = None,
    forecast_years: int = 5,
    discount_rate: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float = 0.0,
    mid_year_convention: bool = False,
    bounds: tuple[float, float] = (-0.80, 1.50),
    tolerance: float = 1e-7,
    max_iterations: int = 100,
) -> ReverseDCFResult:
    """Solve for the constant annual growth rate implied by a given share price.

    Args:
        target_price: Market price per share to match.
        starting_fcf: Base year Free Cash Flow (for FCF path).
        starting_revenue: Base year revenue (if using revenue + margin path).
        fixed_margin: Constant FCF margin (if using revenue path).
        forecast_years: Number of explicit projection years.
        discount_rate: Weighted Average Cost of Capital (WACC).
        terminal_growth: Perpetual growth rate.
        shares_outstanding: Common shares outstanding.
        net_debt: Total debt minus cash.
        mid_year_convention: Whether to discount using mid-period convention.
        bounds: Initial lower and upper bounds for annual growth rate.
        tolerance: Convergence tolerance for brentq.
        max_iterations: Maximum iterations for root finding.

    Returns:
        ReverseDCFResult with implied constant growth rate and convergence status.
    """
    if target_price <= 0.0:
        raise ValueError(f"target_price ({target_price}) must be strictly positive.")

    if starting_fcf is None and (starting_revenue is None or fixed_margin is None):
        raise ValueError(
            "Must provide either starting_fcf or both (starting_revenue, fixed_margin)."
        )

    def objective(g: float) -> float:
        growth_path = [g] * forecast_years
        if starting_fcf is not None:
            res = compute_dcf(
                starting_fcf=starting_fcf,
                fcf_growth_path=growth_path,
                discount_rate=discount_rate,
                terminal_growth=terminal_growth,
                shares_outstanding=shares_outstanding,
                net_debt=net_debt,
                mid_year_convention=mid_year_convention,
            )
        else:
            assert starting_revenue is not None
            assert fixed_margin is not None
            margin_path = [fixed_margin] * forecast_years
            res = compute_dcf(
                starting_revenue=starting_revenue,
                revenue_growth_path=growth_path,
                margin_path=margin_path,
                discount_rate=discount_rate,
                terminal_growth=terminal_growth,
                shares_outstanding=shares_outstanding,
                net_debt=net_debt,
                mid_year_convention=mid_year_convention,
            )
        return res.fair_value_per_share - target_price

    bracket = _find_bracket(objective, bounds[0], bounds[1])
    if bracket is None:
        return ReverseDCFResult(
            target_price=target_price,
            implied_value=None,
            converged=False,
            parameter_name="growth_rate",
            message=(
                f"Could not bracket implied growth rate for target price "
                f"{target_price} within bounds {bounds}."
            ),
        )

    try:
        root, opt_res = brentq(
            objective,
            bracket[0],
            bracket[1],
            xtol=tolerance,
            maxiter=max_iterations,
            full_output=True,
            disp=False,
        )
    except Exception as exc:
        return ReverseDCFResult(
            target_price=target_price,
            implied_value=None,
            converged=False,
            parameter_name="growth_rate",
            message=f"Root finding solver failed: {exc}",
        )

    converged = bool(opt_res.converged)
    final_dcf: DCFResult | None = None
    fv_at_root: float | None = None

    if converged:
        growth_path = [float(root)] * forecast_years
        if starting_fcf is not None:
            final_dcf = compute_dcf(
                starting_fcf=starting_fcf,
                fcf_growth_path=growth_path,
                discount_rate=discount_rate,
                terminal_growth=terminal_growth,
                shares_outstanding=shares_outstanding,
                net_debt=net_debt,
                mid_year_convention=mid_year_convention,
            )
        else:
            assert starting_revenue is not None
            assert fixed_margin is not None
            final_dcf = compute_dcf(
                starting_revenue=starting_revenue,
                revenue_growth_path=growth_path,
                margin_path=[fixed_margin] * forecast_years,
                discount_rate=discount_rate,
                terminal_growth=terminal_growth,
                shares_outstanding=shares_outstanding,
                net_debt=net_debt,
                mid_year_convention=mid_year_convention,
            )
        fv_at_root = final_dcf.fair_value_per_share

    return ReverseDCFResult(
        target_price=target_price,
        implied_value=float(root) if converged else None,
        converged=converged,
        parameter_name="growth_rate",
        iterations=int(opt_res.iterations),
        fair_value_at_implied=fv_at_root,
        dcf_result=final_dcf,
        message="Optimization converged successfully."
        if converged
        else "Did not converge.",
    )


def solve_implied_margin(
    target_price: float,
    *,
    starting_revenue: float,
    revenue_growth_path: list[float] | float,
    forecast_years: int = 5,
    discount_rate: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float = 0.0,
    mid_year_convention: bool = False,
    bounds: tuple[float, float] = (-0.50, 0.90),
    tolerance: float = 1e-7,
    max_iterations: int = 100,
) -> ReverseDCFResult:
    """Solve for the constant FCF margin implied by a given share price.

    Args:
        target_price: Market price per share to match.
        starting_revenue: Base year revenue.
        revenue_growth_path: Projected revenue growth rates (list or constant float).
        forecast_years: Explicit forecast years (used if revenue_growth_path is float).
        discount_rate: Weighted Average Cost of Capital (WACC).
        terminal_growth: Perpetual growth rate.
        shares_outstanding: Common shares outstanding.
        net_debt: Total debt minus cash.
        mid_year_convention: Whether to discount using mid-period convention.
        bounds: Initial lower and upper bounds for FCF margin.
        tolerance: Convergence tolerance for brentq.
        max_iterations: Maximum iterations for root finding.

    Returns:
        ReverseDCFResult with implied constant margin and convergence status.
    """
    if target_price <= 0.0:
        raise ValueError(f"target_price ({target_price}) must be strictly positive.")

    if isinstance(revenue_growth_path, (int, float)):
        rev_path = [float(revenue_growth_path)] * forecast_years
    else:
        rev_path = list(revenue_growth_path)
        forecast_years = len(rev_path)

    def objective(m: float) -> float:
        margin_path = [m] * forecast_years
        res = compute_dcf(
            starting_revenue=starting_revenue,
            revenue_growth_path=rev_path,
            margin_path=margin_path,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
            shares_outstanding=shares_outstanding,
            net_debt=net_debt,
            mid_year_convention=mid_year_convention,
        )
        return res.fair_value_per_share - target_price

    bracket = _find_bracket(objective, bounds[0], bounds[1])
    if bracket is None:
        return ReverseDCFResult(
            target_price=target_price,
            implied_value=None,
            converged=False,
            parameter_name="margin",
            message=(
                f"Could not bracket implied margin for target price {target_price} "
                f"within bounds {bounds}."
            ),
        )

    try:
        root, opt_res = brentq(
            objective,
            bracket[0],
            bracket[1],
            xtol=tolerance,
            maxiter=max_iterations,
            full_output=True,
            disp=False,
        )
    except Exception as exc:
        return ReverseDCFResult(
            target_price=target_price,
            implied_value=None,
            converged=False,
            parameter_name="margin",
            message=f"Root finding solver failed: {exc}",
        )

    converged = bool(opt_res.converged)
    final_dcf: DCFResult | None = None
    fv_at_root: float | None = None

    if converged:
        margin_path = [float(root)] * forecast_years
        final_dcf = compute_dcf(
            starting_revenue=starting_revenue,
            revenue_growth_path=rev_path,
            margin_path=margin_path,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
            shares_outstanding=shares_outstanding,
            net_debt=net_debt,
            mid_year_convention=mid_year_convention,
        )
        fv_at_root = final_dcf.fair_value_per_share

    return ReverseDCFResult(
        target_price=target_price,
        implied_value=float(root) if converged else None,
        converged=converged,
        parameter_name="margin",
        iterations=int(opt_res.iterations),
        fair_value_at_implied=fv_at_root,
        dcf_result=final_dcf,
        message="Optimization converged successfully."
        if converged
        else "Did not converge.",
    )


def reverse_dcf(
    target_price: float,
    *,
    starting_fcf: float | None = None,
    starting_revenue: float | None = None,
    revenue_growth_path: list[float] | float | None = None,
    fixed_margin: float | None = None,
    forecast_years: int = 5,
    discount_rate: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float = 0.0,
    mid_year_convention: bool = False,
    solve_for: str = "growth",
    bounds: tuple[float, float] | None = None,
    tolerance: float = 1e-7,
    max_iterations: int = 100,
) -> ReverseDCFResult:
    """Unified entrypoint for Reverse DCF solving for implied growth or margin.

    Args:
        target_price: Current market stock price.
        starting_fcf: Base year Free Cash Flow.
        starting_revenue: Base year revenue.
        revenue_growth_path: Path or constant for revenue growth.
        fixed_margin: Fixed FCF margin if solving for growth with revenue path.
        forecast_years: Forecast period length.
        discount_rate: WACC discount rate.
        terminal_growth: Perpetual growth rate.
        shares_outstanding: Common shares outstanding.
        net_debt: Net debt (debt minus cash).
        mid_year_convention: Whether to discount mid-period.
        solve_for: "growth" (default) or "margin".
        bounds: Optimization search bounds tuple (low, high).
        tolerance: Root-finding convergence tolerance.
        max_iterations: Maximum iterations.

    Returns:
        ReverseDCFResult with implied parameter value and convergence flag.
    """
    clean_target = solve_for.strip().lower()
    if clean_target in ("growth", "growth_rate"):
        default_bounds = bounds or (-0.80, 1.50)
        return solve_implied_growth(
            target_price=target_price,
            starting_fcf=starting_fcf,
            starting_revenue=starting_revenue,
            fixed_margin=fixed_margin,
            forecast_years=forecast_years,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
            shares_outstanding=shares_outstanding,
            net_debt=net_debt,
            mid_year_convention=mid_year_convention,
            bounds=default_bounds,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    if clean_target in ("margin", "margins"):
        default_bounds = bounds or (-0.50, 0.90)
        if starting_revenue is None:
            raise ValueError(
                "starting_revenue must be provided when solving for margin."
            )
        rev_path = revenue_growth_path if revenue_growth_path is not None else 0.05
        return solve_implied_margin(
            target_price=target_price,
            starting_revenue=starting_revenue,
            revenue_growth_path=rev_path,
            forecast_years=forecast_years,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
            shares_outstanding=shares_outstanding,
            net_debt=net_debt,
            mid_year_convention=mid_year_convention,
            bounds=default_bounds,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    raise ValueError(
        f"Unknown solve_for parameter: {solve_for}. Must be 'growth' or 'margin'."
    )
