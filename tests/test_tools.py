"""Tests for Tool framework, EchoTool, and ToolRegistry."""

import pytest
from pydantic import ValidationError

from lekka_ai.tools import EchoInput, EchoOutput, EchoTool, ToolRegistry


@pytest.mark.asyncio
async def test_tool_schema_export() -> None:
    """to_spec() generates a valid ToolSpec from the Pydantic input schema."""
    tool = EchoTool()
    spec = tool.to_spec()

    assert spec.name == "echo"
    assert spec.description == "Echo back the provided message string."
    assert spec.json_schema.get("type") == "object"
    assert "message" in spec.json_schema.get("properties", {})
    assert "message" in spec.json_schema.get("required", [])


@pytest.mark.asyncio
async def test_tool_execution_success() -> None:
    """Tool executes successfully with both Pydantic input and raw dictionary."""
    tool = EchoTool()

    # Direct model run
    result1 = await tool.run(EchoInput(message="hello world"))
    assert isinstance(result1, EchoOutput)
    assert result1.echoed == "hello world"

    # Execution with dictionary parsing
    result2 = await tool.execute({"message": "test dictionary"})
    assert isinstance(result2, EchoOutput)
    assert result2.echoed == "test dictionary"


@pytest.mark.asyncio
async def test_tool_input_validation_error() -> None:
    """Passing invalid input to execute() raises Pydantic ValidationError."""
    tool = EchoTool()

    with pytest.raises(ValidationError) as exc_info:
        await tool.execute({})

    errors = exc_info.value.errors()
    assert any(err["loc"] == ("message",) for err in errors)


def test_tool_registry_operations() -> None:
    """ToolRegistry supports registration, retrieval, listing, and errors."""
    registry = ToolRegistry()
    tool = EchoTool()

    assert len(registry.list()) == 0

    registry.register(tool)
    assert len(registry.list()) == 1
    assert registry.list()[0] is tool

    retrieved = registry.get("echo")
    assert retrieved is tool

    # Duplicate registration
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)

    # Missing tool lookup
    with pytest.raises(KeyError, match="not found"):
        registry.get("nonexistent_tool")


@pytest.mark.asyncio
async def test_get_market_data_as_of_clamp() -> None:
    """Verify as_of_date clamps series and drops all future rows."""
    from lekka_ai.data.market import CsvFixtureProvider
    from lekka_ai.tools import GetMarketDataInput, GetMarketDataTool

    provider = CsvFixtureProvider()
    tool = GetMarketDataTool(provider=provider)

    # 1. Unclamped query covers all 9 trading days up to 2024-01-12
    full_output = await tool.run(
        GetMarketDataInput(
            ticker="AAPL",
            start="2024-01-02",
            end="2024-01-12",
        )
    )
    assert full_output.summary.total_bars == 9
    assert full_output.summary.date == "2024-01-12"

    # 2. Clamped query with as_of_date="2024-01-05" drops all subsequent rows
    clamped_output = await tool.run(
        GetMarketDataInput(
            ticker="AAPL",
            start="2024-01-02",
            end="2024-01-12",
            as_of_date="2024-01-05",
        )
    )

    # Primary verification: absolutely no rows exist after as_of_date
    assert len(clamped_output.compact_series) == 4
    for bar in clamped_output.compact_series:
        assert bar.date <= "2024-01-05", f"Leaked future row with date {bar.date}"

    # Verify summary reflects clamped endpoint
    assert clamped_output.summary.total_bars == 4
    assert clamped_output.summary.date == "2024-01-05"
    assert clamped_output.summary.latest_close == pytest.approx(181.18, rel=1e-3)
    assert clamped_output.latest_close == pytest.approx(181.18, rel=1e-3)
    assert "AAPL" in clamped_output.series_reference
    assert "4_bars" in clamped_output.series_reference


@pytest.mark.asyncio
async def test_get_market_data_weekend_as_of_clamp() -> None:
    """Verify as_of_date on weekend/holiday clamps to latest available prior bar."""
    from lekka_ai.data.market import CsvFixtureProvider
    from lekka_ai.tools import GetMarketDataInput, GetMarketDataTool

    provider = CsvFixtureProvider()
    tool = GetMarketDataTool(provider=provider)

    # 2024-01-06 is a Saturday; next trading day in fixture is 2024-01-08
    output = await tool.run(
        GetMarketDataInput(
            ticker="AAPL",
            start="2024-01-02",
            end="2024-01-12",
            as_of_date="2024-01-06",
        )
    )

    assert len(output.compact_series) == 4
    for bar in output.compact_series:
        assert bar.date <= "2024-01-06"
    assert output.summary.date == "2024-01-05"


@pytest.mark.asyncio
async def test_get_market_data_registered_in_default_registry() -> None:
    """get_market_data tool is automatically registered in default_registry."""
    from lekka_ai.tools import default_registry

    tool = default_registry.get("get_market_data")
    assert tool is not None
    assert tool.name == "get_market_data"
    assert "get_market_data" in default_registry


@pytest.mark.asyncio
async def test_get_market_data_schema_export() -> None:
    """get_market_data tool exports schema with required parameters and as_of_date."""
    from lekka_ai.tools import GetMarketDataTool

    tool = GetMarketDataTool()
    spec = tool.to_spec()
    assert spec.name == "get_market_data"
    props = spec.json_schema.get("properties", {})
    assert "ticker" in props
    assert "start" in props
    assert "end" in props
    assert "as_of_date" in props
    assert set(spec.json_schema.get("required", [])) == {"ticker", "start", "end"}


@pytest.mark.asyncio
async def test_compute_risk_stats_fixtures() -> None:
    """Verify compute_risk_stats produces risk metrics with AAPL/SPY fixtures."""
    from lekka_ai.data.market import CsvFixtureProvider
    from lekka_ai.tools import ComputeRiskStatsInput, ComputeRiskStatsTool

    provider = CsvFixtureProvider()
    tool = ComputeRiskStatsTool(provider=provider)

    output = await tool.run(
        ComputeRiskStatsInput(
            ticker="AAPL",
            benchmark="SPY",
            as_of_date="2024-01-12",
        )
    )

    # 1. Echoed inputs
    assert output.ticker == "AAPL"
    assert output.benchmark == "SPY"
    assert output.as_of_date == "2024-01-12"
    assert output.start_date == "2024-01-02"
    assert output.end_date == "2024-01-12"
    assert output.total_bars == 9

    # 2. Risk & performance metrics
    assert output.returns == pytest.approx(0.001508, abs=1e-3)
    assert output.benchmark_returns == pytest.approx(0.008526, abs=1e-3)
    assert output.volatility == pytest.approx(0.1768, abs=1e-3)
    assert output.drawdown == pytest.approx(-0.024025, abs=1e-3)
    assert output.max_drawdown == output.drawdown
    assert output.drawdown_peak_date == "2024-01-02"
    assert output.drawdown_trough_date == "2024-01-05"

    # 3. Benchmark relative metrics
    assert output.beta == pytest.approx(1.549, abs=1e-2)
    assert output.correlation == pytest.approx(0.923, abs=1e-2)
    assert output.relative_strength == pytest.approx(-0.0070, abs=1e-3)

    # 4. Trend classification
    assert output.trend_label in ("BULLISH", "BEARISH", "NEUTRAL")
    assert output.trend == output.trend_label
    assert len(output.moving_averages) > 0


@pytest.mark.asyncio
async def test_compute_risk_stats_point_in_time_clamp() -> None:
    """Verify compute_risk_stats clamps to as_of_date avoiding lookahead."""
    from lekka_ai.data.market import CsvFixtureProvider
    from lekka_ai.tools import ComputeRiskStatsInput, ComputeRiskStatsTool

    provider = CsvFixtureProvider()
    tool = ComputeRiskStatsTool(provider=provider)

    output = await tool.run(
        ComputeRiskStatsInput(
            ticker="AAPL",
            benchmark="SPY",
            as_of_date="2024-01-05",
        )
    )

    assert output.end_date == "2024-01-05"
    assert output.total_bars == 4
    # On Jan 5, AAPL suffered a 2.4% drop
    assert output.returns == pytest.approx(-0.024025, abs=1e-3)
    assert output.trend_label == "BEARISH"


@pytest.mark.asyncio
async def test_compute_risk_stats_registered() -> None:
    """compute_risk_stats is registered in default_registry and exports spec."""
    from lekka_ai.tools import default_registry

    tool = default_registry.get("compute_risk_stats")
    assert tool is not None
    assert tool.name == "compute_risk_stats"
    spec = tool.to_spec()
    assert spec.name == "compute_risk_stats"
    props = spec.json_schema.get("properties", {})
    assert "ticker" in props
    assert "as_of_date" in props
    assert "benchmark" in props
