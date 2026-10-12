"""Registered agent tools for DCF, reverse DCF, multiples, and scenarios."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lekka_ai.quant.dcf import compute_dcf
from lekka_ai.quant.multiples import compute_multiples
from lekka_ai.quant.reverse_dcf import reverse_dcf
from lekka_ai.quant.scenarios import scenario_table
from lekka_ai.tools.base import Tool, default_registry

# ---------------------------------------------------------------------------
# run_dcf Tool
# ---------------------------------------------------------------------------


class RunDCFInput(BaseModel):
    """Input payload for run_dcf tool."""

    model_config = ConfigDict(extra="allow")

    ticker: str | None = Field(
        default=None, description="Stock ticker symbol (optional)"
    )
    starting_fcf: float | None = Field(
        default=None, description="Base year Free Cash Flow"
    )
    fcf_growth_path: list[float] | None = Field(
        default=None, description="Annual projected FCF growth rates"
    )
    starting_revenue: float | None = Field(
        default=None, description="Base year revenue"
    )
    revenue_growth_path: list[float] | None = Field(
        default=None, description="Annual projected revenue growth rates"
    )
    margin_path: list[float] | None = Field(
        default=None, description="Annual projected FCF margins (FCF / Revenue)"
    )
    discount_rate: float = Field(
        ..., description="Weighted Average Cost of Capital (WACC), e.g. 0.09"
    )
    terminal_growth: float = Field(..., description="Perpetual growth rate, e.g. 0.02")
    shares_outstanding: float = Field(
        ..., description="Diluted common shares outstanding"
    )
    net_debt: float = Field(
        default=0.0, description="Total debt minus cash and liquid investments"
    )
    mid_year_convention: bool = Field(
        default=False, description="Whether to discount with mid-year convention"
    )


class RunDCFOutput(BaseModel):
    """Output payload for run_dcf tool echoing assumptions and valuation table."""

    model_config = ConfigDict(extra="allow")

    # Echo inputs and assumptions
    ticker: str | None = Field(default=None, description="Ticker symbol analyzed")
    assumptions: dict[str, Any] = Field(
        ..., description="Echoed input assumptions used in DCF valuation"
    )

    # Core valuation results
    enterprise_value: float = Field(..., description="Enterprise Value")
    equity_value: float = Field(..., description="Equity Value (EV - Net Debt)")
    fair_value_per_share: float = Field(..., description="Fair value per common share")
    pv_projected_fcf: float = Field(
        ..., description="Present value of explicit forecast period cash flows"
    )
    terminal_value: float = Field(..., description="Undiscounted terminal value")
    pv_terminal_value: float = Field(..., description="Present value of terminal value")
    terminal_fcf: float = Field(..., description="Terminal year forward free cash flow")
    cash_flow_table: list[dict[str, Any]] = Field(
        ..., description="Explicit annual projected cash flows and discount factors"
    )

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class RunDCFTool(Tool[RunDCFInput, RunDCFOutput]):
    """Tool for computing discounted cash flow valuation and fair value per share."""

    name: str = "run_dcf"
    description: str = (
        "Execute a discounted cash flow (DCF) valuation given explicit cash flow "
        "assumptions, discount rate, terminal growth, shares, and net debt. "
        "Returns enterprise value, equity value, fair value, and cash flow table."
    )
    input_model: type[RunDCFInput] = RunDCFInput
    output_model: type[RunDCFOutput] = RunDCFOutput

    async def run(self, input_data: RunDCFInput) -> RunDCFOutput:
        """Run DCF calculation and return results with echoed assumptions."""
        assumptions_dict = input_data.model_dump(exclude_none=False)

        res = compute_dcf(
            starting_fcf=input_data.starting_fcf,
            fcf_growth_path=input_data.fcf_growth_path,
            starting_revenue=input_data.starting_revenue,
            revenue_growth_path=input_data.revenue_growth_path,
            margin_path=input_data.margin_path,
            discount_rate=input_data.discount_rate,
            terminal_growth=input_data.terminal_growth,
            shares_outstanding=input_data.shares_outstanding,
            net_debt=input_data.net_debt,
            mid_year_convention=input_data.mid_year_convention,
        )

        table_dicts = [
            {
                "year": p.year,
                "fcf": p.fcf,
                "discount_factor": p.discount_factor,
                "present_value": p.present_value,
                "revenue": p.revenue,
                "revenue_growth": p.revenue_growth,
                "margin": p.margin,
                "fcf_growth": p.fcf_growth,
            }
            for p in res.cash_flow_table
        ]

        return RunDCFOutput(
            ticker=input_data.ticker,
            assumptions=assumptions_dict,
            enterprise_value=res.enterprise_value,
            equity_value=res.equity_value,
            fair_value_per_share=res.fair_value_per_share,
            pv_projected_fcf=res.pv_projected_fcf,
            terminal_value=res.terminal_value,
            pv_terminal_value=res.pv_terminal_value,
            terminal_fcf=res.terminal_fcf,
            cash_flow_table=table_dicts,
        )


# ---------------------------------------------------------------------------
# run_reverse_dcf Tool
# ---------------------------------------------------------------------------


class RunReverseDCFInput(BaseModel):
    """Input payload for run_reverse_dcf tool."""

    model_config = ConfigDict(extra="allow")

    current_price: float = Field(
        ..., description="Current stock market price per share to match"
    )
    ticker: str | None = Field(default=None, description="Stock ticker symbol")
    starting_fcf: float | None = Field(
        default=None, description="Base year Free Cash Flow"
    )
    starting_revenue: float | None = Field(
        default=None, description="Base year revenue"
    )
    revenue_growth_path: list[float] | float | None = Field(
        default=None, description="Revenue growth path or constant rate"
    )
    fixed_margin: float | None = Field(
        default=None, description="Constant FCF margin if using revenue path"
    )
    forecast_years: int = Field(
        default=5, description="Explicit projection period length in years"
    )
    discount_rate: float = Field(
        ..., description="Weighted Average Cost of Capital (WACC), e.g. 0.09"
    )
    terminal_growth: float = Field(..., description="Perpetual growth rate, e.g. 0.02")
    shares_outstanding: float = Field(
        ..., description="Diluted common shares outstanding"
    )
    net_debt: float = Field(
        default=0.0, description="Total debt minus cash and liquid investments"
    )
    mid_year_convention: bool = Field(
        default=False, description="Whether to discount with mid-year convention"
    )
    solve_for: str = Field(
        default="growth",
        description="Parameter to solve for: 'growth' (default) or 'margin'",
    )


class RunReverseDCFOutput(BaseModel):
    """Output payload for run_reverse_dcf tool echoing assumptions."""

    model_config = ConfigDict(extra="allow")

    ticker: str | None = Field(default=None, description="Ticker symbol analyzed")
    assumptions: dict[str, Any] = Field(
        ..., description="Echoed input assumptions used in reverse DCF"
    )
    target_price: float = Field(..., description="Target share price solved for")
    implied_value: float | None = Field(
        default=None,
        description="Solved implied constant annual growth rate or margin",
    )
    converged: bool = Field(
        ..., description="Whether root-finding optimization converged"
    )
    parameter_name: str = Field(
        ..., description="Parameter solved ('growth_rate' or 'margin')"
    )
    iterations: int | None = Field(
        default=None, description="Number of solver iterations"
    )
    fair_value_at_implied: float | None = Field(
        default=None,
        description="Fair value per share evaluated at the implied value",
    )
    message: str | None = Field(default=None, description="Solver status message")

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class RunReverseDCFTool(Tool[RunReverseDCFInput, RunReverseDCFOutput]):
    """Tool for reverse DCF root-finding to infer market-implied growth or margin."""

    name: str = "run_reverse_dcf"
    description: str = (
        "Solve for the constant growth rate or margin implied by current market price "
        "using scipy brentq root finding. Returns implied parameter and convergence."
    )
    input_model: type[RunReverseDCFInput] = RunReverseDCFInput
    output_model: type[RunReverseDCFOutput] = RunReverseDCFOutput

    async def run(self, input_data: RunReverseDCFInput) -> RunReverseDCFOutput:
        """Run reverse DCF solver and return outcome echoing assumptions."""
        assumptions_dict = input_data.model_dump(exclude_none=False)

        res = reverse_dcf(
            target_price=input_data.current_price,
            starting_fcf=input_data.starting_fcf,
            starting_revenue=input_data.starting_revenue,
            revenue_growth_path=input_data.revenue_growth_path,
            fixed_margin=input_data.fixed_margin,
            forecast_years=input_data.forecast_years,
            discount_rate=input_data.discount_rate,
            terminal_growth=input_data.terminal_growth,
            shares_outstanding=input_data.shares_outstanding,
            net_debt=input_data.net_debt,
            mid_year_convention=input_data.mid_year_convention,
            solve_for=input_data.solve_for,
        )

        return RunReverseDCFOutput(
            ticker=input_data.ticker,
            assumptions=assumptions_dict,
            target_price=res.target_price,
            implied_value=res.implied_value,
            converged=res.converged,
            parameter_name=res.parameter_name,
            iterations=res.iterations,
            fair_value_at_implied=res.fair_value_at_implied,
            message=res.message,
        )


# ---------------------------------------------------------------------------
# compute_multiples Tool
# ---------------------------------------------------------------------------


class ComputeMultiplesInput(BaseModel):
    """Input payload for compute_multiples tool."""

    model_config = ConfigDict(extra="allow")

    price: float = Field(..., description="Current stock market price per share")
    ticker: str | None = Field(default=None, description="Stock ticker symbol")
    eps: float | None = Field(
        default=None, description="Diluted earnings per share (EPS)"
    )
    forward_eps: float | None = Field(
        default=None, description="Consensus forward earnings per share"
    )
    enterprise_value: float | None = Field(default=None, description="Enterprise value")
    ebitda: float | None = Field(default=None, description="EBITDA")
    sales: float | None = Field(default=None, description="Revenues / Total Sales")
    fcf: float | None = Field(default=None, description="Free Cash Flow")
    market_cap: float | None = Field(default=None, description="Market Capitalization")
    earnings_growth_rate: float | None = Field(
        default=None, description="Expected annual earnings growth rate"
    )
    historical_pe: list[float] | None = Field(
        default=None, description="Historical P/E ratios for percentile ranking"
    )
    historical_ev_ebitda: list[float] | None = Field(
        default=None, description="Historical EV/EBITDA ratios for percentile ranking"
    )


class ComputeMultiplesOutput(BaseModel):
    """Output payload for compute_multiples tool echoing assumptions."""

    model_config = ConfigDict(extra="allow")

    ticker: str | None = Field(default=None, description="Stock ticker symbol")
    assumptions: dict[str, Any] = Field(
        ..., description="Echoed input values used in multiples computation"
    )
    pe: float | None = Field(default=None, description="Price-to-Earnings ratio")
    forward_pe: float | None = Field(
        default=None, description="Forward Price-to-Earnings ratio"
    )
    ev_ebitda: float | None = Field(
        default=None, description="Enterprise Value to EBITDA multiple"
    )
    ev_sales: float | None = Field(
        default=None, description="Enterprise Value to Sales multiple"
    )
    price_fcf: float | None = Field(
        default=None, description="Price to Free Cash Flow multiple"
    )
    peg: float | None = Field(default=None, description="PEG ratio")
    percentiles: dict[str, float] = Field(
        default_factory=dict,
        description="Historical percentile rank (0.0 to 100.0) for multiples",
    )

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class ComputeMultiplesTool(Tool[ComputeMultiplesInput, ComputeMultiplesOutput]):
    """Tool for calculating valuation multiples suite and historical percentiles."""

    name: str = "compute_multiples"
    description: str = (
        "Compute fundamental valuation multiples: P/E, Forward P/E, EV/EBITDA, "
        "EV/Sales, P/FCF, PEG, and historical percentile ranks."
    )
    input_model: type[ComputeMultiplesInput] = ComputeMultiplesInput
    output_model: type[ComputeMultiplesOutput] = ComputeMultiplesOutput

    async def run(self, input_data: ComputeMultiplesInput) -> ComputeMultiplesOutput:
        """Compute valuation multiples and return results with echoed inputs."""
        assumptions_dict = input_data.model_dump(exclude_none=False)

        summary = compute_multiples(
            price=input_data.price,
            eps=input_data.eps,
            forward_eps=input_data.forward_eps,
            enterprise_value=input_data.enterprise_value,
            ebitda=input_data.ebitda,
            sales=input_data.sales,
            fcf=input_data.fcf,
            market_cap=input_data.market_cap,
            earnings_growth_rate=input_data.earnings_growth_rate,
            historical_pe=input_data.historical_pe,
            historical_ev_ebitda=input_data.historical_ev_ebitda,
        )

        return ComputeMultiplesOutput(
            ticker=input_data.ticker,
            assumptions=assumptions_dict,
            pe=summary.pe,
            forward_pe=summary.forward_pe,
            ev_ebitda=summary.ev_ebitda,
            ev_sales=summary.ev_sales,
            price_fcf=summary.price_fcf,
            peg=summary.peg,
            percentiles=summary.percentiles,
        )


# ---------------------------------------------------------------------------
# scenario_table Tool
# ---------------------------------------------------------------------------


class ScenarioTableInput(BaseModel):
    """Input payload for scenario_table tool."""

    model_config = ConfigDict(extra="allow")

    ticker: str | None = Field(default=None, description="Stock ticker symbol")
    current_price: float | None = Field(
        default=None,
        description="Current market price per share for upside/downside comparison",
    )
    scenarios: list[dict[str, Any]] | None = Field(
        default=None, description="List of scenario definition dictionaries"
    )
    bear: dict[str, Any] | None = Field(
        default=None, description="Bear case assumptions dictionary"
    )
    base: dict[str, Any] | None = Field(
        default=None, description="Base case assumptions dictionary"
    )
    bull: dict[str, Any] | None = Field(
        default=None, description="Bull case assumptions dictionary"
    )
    normalize_probabilities: bool = Field(
        default=True,
        description="Whether to automatically rescale probabilities to sum to 1.0",
    )


class ScenarioTableOutput(BaseModel):
    """Output payload for scenario_table tool echoing assumptions."""

    model_config = ConfigDict(extra="allow")

    ticker: str | None = Field(default=None, description="Stock ticker symbol")
    assumptions: dict[str, Any] = Field(
        ..., description="Echoed input scenarios and configuration"
    )
    scenarios: list[dict[str, Any]] = Field(
        ..., description="List of computed scenario outcomes and upsides"
    )
    probability_weighted_fair_value: float = Field(
        ..., description="Probability-weighted fair value per share"
    )
    current_price: float | None = Field(
        default=None, description="Current price compared against"
    )
    probability_weighted_upside_pct: float | None = Field(
        default=None,
        description="Percentage upside/downside of weighted value vs current price",
    )

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class ScenarioTableTool(Tool[ScenarioTableInput, ScenarioTableOutput]):
    """Tool for multi-scenario DCF valuation table and probability weighting."""

    name: str = "scenario_table"
    description: str = (
        "Generate a multi-scenario valuation table (Bear, Base, Bull) with "
        "probabilities, fair values per share, upside vs market price, and "
        "weighted valuation."
    )
    input_model: type[ScenarioTableInput] = ScenarioTableInput
    output_model: type[ScenarioTableOutput] = ScenarioTableOutput

    async def run(self, input_data: ScenarioTableInput) -> ScenarioTableOutput:
        """Compute scenario table and return outcome with echoed assumptions."""
        assumptions_dict = input_data.model_dump(exclude_none=False)

        res = scenario_table(
            scenarios=input_data.scenarios,
            bear=input_data.bear,
            base=input_data.base,
            bull=input_data.bull,
            current_price=input_data.current_price,
            normalize_probabilities=input_data.normalize_probabilities,
        )

        scenario_dicts = [
            {
                "name": s.name,
                "probability": s.probability,
                "fair_value_per_share": s.fair_value_per_share,
                "enterprise_value": s.enterprise_value,
                "equity_value": s.equity_value,
                "upside_downside_pct": s.upside_downside_pct,
            }
            for s in res.scenarios
        ]

        return ScenarioTableOutput(
            ticker=input_data.ticker,
            assumptions=assumptions_dict,
            scenarios=scenario_dicts,
            probability_weighted_fair_value=res.probability_weighted_fair_value,
            current_price=res.current_price,
            probability_weighted_upside_pct=res.probability_weighted_upside_pct,
        )


# Register default instances in the global tool registry
default_run_dcf_tool = RunDCFTool()
default_run_reverse_dcf_tool = RunReverseDCFTool()
default_compute_multiples_tool = ComputeMultiplesTool()
default_scenario_table_tool = ScenarioTableTool()

default_registry.register(default_run_dcf_tool, overwrite=True)
default_registry.register(default_run_reverse_dcf_tool, overwrite=True)
default_registry.register(default_compute_multiples_tool, overwrite=True)
default_registry.register(default_scenario_table_tool, overwrite=True)
