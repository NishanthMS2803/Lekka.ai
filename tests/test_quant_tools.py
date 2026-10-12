"""Tests for registered quant tools (DCF, reverse DCF, multiples, scenarios)."""

import pytest

from lekka_ai.tools.base import default_registry
from lekka_ai.tools.quant import (
    ComputeMultiplesOutput,
    ComputeMultiplesTool,
    RunDCFOutput,
    RunDCFTool,
    RunReverseDCFOutput,
    RunReverseDCFTool,
    ScenarioTableOutput,
    ScenarioTableTool,
)


def test_quant_tools_registration() -> None:
    """Verify that all four quant tools are registered in default_registry."""
    expected_tools = [
        "run_dcf",
        "run_reverse_dcf",
        "compute_multiples",
        "scenario_table",
    ]
    for tool_name in expected_tools:
        assert tool_name in default_registry
        tool = default_registry.get(tool_name)
        assert tool is not None
        assert tool.name == tool_name
        spec = tool.to_spec()
        assert spec.name == tool_name
        assert "properties" in spec.json_schema


@pytest.mark.asyncio
async def test_run_dcf_tool_end_to_end() -> None:
    """Test run_dcf tool through registry with textbook fixtures and echoed inputs."""
    tool = default_registry.get("run_dcf")
    assert isinstance(tool, RunDCFTool)

    payload = {
        "ticker": "AAPL",
        "starting_fcf": 100.0,
        "fcf_growth_path": [0.10, 0.10, 0.10],
        "discount_rate": 0.10,
        "terminal_growth": 0.02,
        "shares_outstanding": 50.0,
        "net_debt": 75.0,
    }

    result: RunDCFOutput = await tool.execute(payload)

    assert isinstance(result, RunDCFOutput)
    assert result.ticker == "AAPL"

    # Verify input echo
    assert result.assumptions["starting_fcf"] == 100.0
    assert result.assumptions["discount_rate"] == 0.10
    assert result.assumptions["terminal_growth"] == 0.02
    assert result.assumptions["shares_outstanding"] == 50.0
    assert result.assumptions["net_debt"] == 75.0

    # Verify textbook DCF calculation
    assert pytest.approx(result.pv_projected_fcf, rel=1e-5) == 300.0
    assert pytest.approx(result.enterprise_value, rel=1e-5) == 1575.0
    assert pytest.approx(result.equity_value, rel=1e-5) == 1500.0
    assert pytest.approx(result.fair_value_per_share, rel=1e-5) == 30.0
    assert len(result.cash_flow_table) == 3


@pytest.mark.asyncio
async def test_run_reverse_dcf_tool_end_to_end() -> None:
    """Test run_reverse_dcf tool recovering implied growth and echoing assumptions."""
    tool = default_registry.get("run_reverse_dcf")
    assert isinstance(tool, RunReverseDCFTool)

    payload = {
        "ticker": "MSFT",
        "current_price": 30.0,
        "starting_fcf": 100.0,
        "forecast_years": 3,
        "discount_rate": 0.10,
        "terminal_growth": 0.02,
        "shares_outstanding": 50.0,
        "net_debt": 75.0,
        "solve_for": "growth",
    }

    result: RunReverseDCFOutput = await tool.execute(payload)

    assert isinstance(result, RunReverseDCFOutput)
    assert result.ticker == "MSFT"
    assert result.converged is True

    # Confirm echo
    assert result.assumptions["current_price"] == 30.0
    assert result.assumptions["forecast_years"] == 3
    assert result.assumptions["solve_for"] == "growth"

    # Target price 30.0 corresponds to 10% annual growth
    assert result.implied_value is not None
    assert pytest.approx(result.implied_value, rel=1e-4) == 0.10
    assert result.fair_value_at_implied is not None
    assert pytest.approx(result.fair_value_at_implied, rel=1e-4) == 30.0


@pytest.mark.asyncio
async def test_compute_multiples_tool_end_to_end() -> None:
    """Test compute_multiples tool computing multiples and echoing assumptions."""
    tool = default_registry.get("compute_multiples")
    assert isinstance(tool, ComputeMultiplesTool)

    payload = {
        "ticker": "GOOGL",
        "price": 150.0,
        "eps": 6.0,
        "forward_eps": 7.5,
        "enterprise_value": 1800.0,
        "ebitda": 120.0,
        "sales": 600.0,
        "market_cap": 1800.0,
        "fcf": 90.0,
        "earnings_growth_rate": 0.25,
        "historical_pe": [15.0, 20.0, 25.0, 30.0, 35.0],
    }

    result: ComputeMultiplesOutput = await tool.execute(payload)

    assert isinstance(result, ComputeMultiplesOutput)
    assert result.ticker == "GOOGL"

    # Confirm echo
    assert result.assumptions["price"] == 150.0
    assert result.assumptions["eps"] == 6.0
    assert result.assumptions["forward_eps"] == 7.5

    # Multiples check
    assert result.pe == 25.0
    assert result.forward_pe == 20.0
    assert result.ev_ebitda == 15.0
    assert result.ev_sales == 3.0
    assert result.price_fcf == 20.0
    # PEG = 25 / 25% = 1.0
    assert pytest.approx(result.peg or 0.0, rel=1e-5) == 1.0
    assert "pe" in result.percentiles


@pytest.mark.asyncio
async def test_scenario_table_tool_end_to_end() -> None:
    """Test scenario_table tool probability-weighted valuation and echoes."""
    tool = default_registry.get("scenario_table")
    assert isinstance(tool, ScenarioTableTool)

    payload = {
        "ticker": "AMZN",
        "current_price": 28.0,
        "bear": {
            "name": "Bear",
            "probability": 0.25,
            "starting_fcf": 100.0,
            "fcf_growth_path": [0.03, 0.03, 0.03],
            "discount_rate": 0.10,
            "terminal_growth": 0.015,
            "shares_outstanding": 50.0,
            "net_debt": 50.0,
        },
        "base": {
            "name": "Base",
            "probability": 0.50,
            "starting_fcf": 100.0,
            "fcf_growth_path": [0.08, 0.08, 0.08],
            "discount_rate": 0.10,
            "terminal_growth": 0.02,
            "shares_outstanding": 50.0,
            "net_debt": 50.0,
        },
        "bull": {
            "name": "Bull",
            "probability": 0.25,
            "starting_fcf": 100.0,
            "fcf_growth_path": [0.15, 0.15, 0.15],
            "discount_rate": 0.10,
            "terminal_growth": 0.025,
            "shares_outstanding": 50.0,
            "net_debt": 50.0,
        },
    }

    result: ScenarioTableOutput = await tool.execute(payload)

    assert isinstance(result, ScenarioTableOutput)
    assert result.ticker == "AMZN"
    assert result.current_price == 28.0

    # Confirm echo
    assert "bear" in result.assumptions
    assert "base" in result.assumptions
    assert "bull" in result.assumptions

    # Result checks
    assert len(result.scenarios) == 3
    assert result.probability_weighted_fair_value > 0.0
    assert result.probability_weighted_upside_pct is not None
