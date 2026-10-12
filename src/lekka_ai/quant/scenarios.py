"""Scenario analysis and valuation sensitivity grid modeling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from lekka_ai.quant.dcf import DCFAssumptions, DCFResult, compute_dcf


@dataclass
class ScenarioAssumption:
    """Valuation inputs and probability weighting for a single scenario."""

    name: str
    probability: float
    starting_fcf: float | None = None
    fcf_growth_path: list[float] | None = None
    starting_revenue: float | None = None
    revenue_growth_path: list[float] | None = None
    margin_path: list[float] | None = None
    discount_rate: float | None = None
    terminal_growth: float | None = None
    shares_outstanding: float | None = None
    net_debt: float = 0.0
    mid_year_convention: bool = False
    assumptions: DCFAssumptions | None = None


@dataclass
class ScenarioOutcome:
    """DCF outcome metrics for a specific named scenario."""

    name: str
    probability: float
    fair_value_per_share: float
    enterprise_value: float
    equity_value: float
    upside_downside_pct: float | None = None
    dcf_result: DCFResult | None = None

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


@dataclass
class ScenarioTableResult:
    """Summary of Bear, Base, Bull scenarios and probability-weighted valuation."""

    scenarios: list[ScenarioOutcome]
    probability_weighted_fair_value: float
    current_price: float | None = None
    probability_weighted_upside_pct: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def to_dict(self) -> dict[str, Any]:
        """Convert scenario table to dictionary."""
        return {
            "scenarios": [
                {
                    "name": s.name,
                    "probability": s.probability,
                    "fair_value_per_share": s.fair_value_per_share,
                    "enterprise_value": s.enterprise_value,
                    "equity_value": s.equity_value,
                    "upside_downside_pct": s.upside_downside_pct,
                }
                for s in self.scenarios
            ],
            "probability_weighted_fair_value": self.probability_weighted_fair_value,
            "current_price": self.current_price,
            "probability_weighted_upside_pct": self.probability_weighted_upside_pct,
        }


def _resolve_scenario_dcf(scenario: ScenarioAssumption) -> DCFResult:
    """Execute DCF for an individual scenario definition."""
    if scenario.assumptions is not None:
        assumptions = scenario.assumptions
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

    if (
        scenario.discount_rate is None
        or scenario.terminal_growth is None
        or scenario.shares_outstanding is None
    ):
        raise ValueError(
            f"Scenario '{scenario.name}' is missing required DCF parameters "
            f"(discount_rate, terminal_growth, shares_outstanding)."
        )

    return compute_dcf(
        starting_fcf=scenario.starting_fcf,
        fcf_growth_path=scenario.fcf_growth_path,
        starting_revenue=scenario.starting_revenue,
        revenue_growth_path=scenario.revenue_growth_path,
        margin_path=scenario.margin_path,
        discount_rate=scenario.discount_rate,
        terminal_growth=scenario.terminal_growth,
        shares_outstanding=scenario.shares_outstanding,
        net_debt=scenario.net_debt,
        mid_year_convention=scenario.mid_year_convention,
    )


def scenario_table(
    scenarios: Sequence[ScenarioAssumption | dict[str, Any]] | None = None,
    *,
    bear: ScenarioAssumption | dict[str, Any] | None = None,
    base: ScenarioAssumption | dict[str, Any] | None = None,
    bull: ScenarioAssumption | dict[str, Any] | None = None,
    current_price: float | None = None,
    normalize_probabilities: bool = True,
) -> ScenarioTableResult:
    """Build a multi-scenario DCF table with probability-weighted valuation.

    Accepts either an explicit list of scenarios or named bear/base/bull parameters.

    Args:
        scenarios: List of ScenarioAssumption definitions.
        bear: Bear case definition.
        base: Base case definition.
        bull: Bull case definition.
        current_price: Current market price per share for upside/downside comparison.
        normalize_probabilities: Rescale probabilities to sum to 1.0 if True.

    Returns:
        ScenarioTableResult with scenario outcomes and weighted valuation.
    """
    raw_list: list[ScenarioAssumption] = []
    if scenarios is not None:
        for s in scenarios:
            if isinstance(s, ScenarioAssumption):
                raw_list.append(s)
            elif isinstance(s, dict):
                raw_list.append(ScenarioAssumption(**s))
    else:
        named_pairs: list[tuple[str, ScenarioAssumption | dict[str, Any] | None]] = [
            ("Bear", bear),
            ("Base", base),
            ("Bull", bull),
        ]
        for default_name, item in named_pairs:
            if item is not None:
                if isinstance(item, ScenarioAssumption):
                    raw_list.append(item)
                elif isinstance(item, dict):
                    data = dict(item)
                    if "name" not in data:
                        data["name"] = default_name
                    raw_list.append(ScenarioAssumption(**data))

    if not raw_list:
        raise ValueError("Must provide at least one scenario to scenario_table.")

    total_prob = sum(s.probability for s in raw_list)
    if total_prob <= 0.0:
        raise ValueError("Sum of scenario probabilities must be strictly positive.")

    outcomes: list[ScenarioOutcome] = []
    weighted_value = 0.0

    for s in raw_list:
        prob = (
            (s.probability / total_prob) if normalize_probabilities else s.probability
        )
        dcf_res = _resolve_scenario_dcf(s)
        fv = dcf_res.fair_value_per_share
        weighted_value += prob * fv

        upside: float | None = None
        if current_price is not None and current_price > 0.0:
            upside = (fv - current_price) / current_price

        outcomes.append(
            ScenarioOutcome(
                name=s.name,
                probability=prob,
                fair_value_per_share=fv,
                enterprise_value=dcf_res.enterprise_value,
                equity_value=dcf_res.equity_value,
                upside_downside_pct=upside,
                dcf_result=dcf_res,
            )
        )

    weighted_upside: float | None = None
    if current_price is not None and current_price > 0.0:
        weighted_upside = (weighted_value - current_price) / current_price

    return ScenarioTableResult(
        scenarios=outcomes,
        probability_weighted_fair_value=weighted_value,
        current_price=current_price,
        probability_weighted_upside_pct=weighted_upside,
    )


@dataclass
class SensitivityGridResult:
    """Matrix of DCF fair values over discount rate and terminal growth rate ranges."""

    discount_rates: list[float]
    terminal_growths: list[float]
    fair_values: list[list[float | None]]
    upside_percentages: list[list[float | None]] | None = None
    current_price: float | None = None

    def get_fair_value(
        self, discount_rate: float, terminal_growth: float
    ) -> float | None:
        """Lookup fair value for specific discount rate and terminal growth."""
        try:
            r_idx = self.discount_rates.index(discount_rate)
            g_idx = self.terminal_growths.index(terminal_growth)
            return self.fair_values[r_idx][g_idx]
        except ValueError:
            return None

    def to_dict(self) -> dict[str, Any]:
        """Convert sensitivity grid to a JSON-serializable dictionary."""
        return {
            "discount_rates": self.discount_rates,
            "terminal_growths": self.terminal_growths,
            "fair_values": self.fair_values,
            "upside_percentages": self.upside_percentages,
            "current_price": self.current_price,
        }


def sensitivity_grid(
    *,
    base_assumptions: DCFAssumptions | None = None,
    starting_fcf: float | None = None,
    fcf_growth_path: list[float] | None = None,
    starting_revenue: float | None = None,
    revenue_growth_path: list[float] | None = None,
    margin_path: list[float] | None = None,
    shares_outstanding: float | None = None,
    net_debt: float = 0.0,
    mid_year_convention: bool = False,
    discount_rates: list[float],
    terminal_growths: list[float],
    current_price: float | None = None,
) -> SensitivityGridResult:
    """Compute a two-dimensional valuation matrix across WACC and terminal growth rates.

    Args:
        base_assumptions: Optional DCFAssumptions object supplying base parameters.
        starting_fcf: Base year Free Cash Flow.
        fcf_growth_path: Projected annual FCF growth rates.
        starting_revenue: Base year revenue.
        revenue_growth_path: Projected annual revenue growth rates.
        margin_path: Projected annual FCF margins.
        shares_outstanding: Diluted common shares outstanding.
        net_debt: Net debt.
        mid_year_convention: Whether to use mid-year discounting.
        discount_rates: List of WACC discount rates (rows).
        terminal_growths: List of terminal growth rates (columns).
        current_price: Optional current share price to compute upside/downside matrix.

    Returns:
        SensitivityGridResult containing the 2D fair value matrix and upside matrix.
    """
    if base_assumptions is not None:
        starting_fcf = base_assumptions.starting_fcf
        fcf_growth_path = base_assumptions.fcf_growth_path
        starting_revenue = base_assumptions.starting_revenue
        revenue_growth_path = base_assumptions.revenue_growth_path
        margin_path = base_assumptions.margin_path
        shares_outstanding = base_assumptions.shares_outstanding
        net_debt = base_assumptions.net_debt
        mid_year_convention = base_assumptions.mid_year_convention

    if shares_outstanding is None or shares_outstanding <= 0.0:
        raise ValueError("shares_outstanding must be strictly positive.")

    fair_values: list[list[float | None]] = []
    upsides: list[list[float | None]] = []

    for r in discount_rates:
        row_fv: list[float | None] = []
        row_up: list[float | None] = []
        for g in terminal_growths:
            if g >= r:
                # Gordon Growth model does not converge when g >= r
                row_fv.append(None)
                row_up.append(None)
                continue

            try:
                res = compute_dcf(
                    starting_fcf=starting_fcf,
                    fcf_growth_path=fcf_growth_path,
                    starting_revenue=starting_revenue,
                    revenue_growth_path=revenue_growth_path,
                    margin_path=margin_path,
                    discount_rate=r,
                    terminal_growth=g,
                    shares_outstanding=shares_outstanding,
                    net_debt=net_debt,
                    mid_year_convention=mid_year_convention,
                )
                fv = res.fair_value_per_share
                row_fv.append(fv)
                if current_price is not None and current_price > 0.0:
                    row_up.append((fv - current_price) / current_price)
                else:
                    row_up.append(None)
            except Exception:
                row_fv.append(None)
                row_up.append(None)

        fair_values.append(row_fv)
        upsides.append(row_up)

    upside_matrix = upsides if current_price is not None else None

    return SensitivityGridResult(
        discount_rates=discount_rates,
        terminal_growths=terminal_growths,
        fair_values=fair_values,
        upside_percentages=upside_matrix,
        current_price=current_price,
    )
