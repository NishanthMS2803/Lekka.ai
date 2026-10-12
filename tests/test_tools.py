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


@pytest.mark.asyncio
async def test_list_filings_fixture_and_as_of_filter() -> None:
    """Verify list_filings parses saved fixture JSON and filters by as_of_date."""
    import json
    from pathlib import Path

    import httpx

    from lekka_ai.data.http import create_http_client
    from lekka_ai.data.sec import SecClient
    from lekka_ai.tools import ListFilingsInput, ListFilingsTool

    fixture_path = Path(__file__).parent / "fixtures" / "AAPL_submissions.json"
    with open(fixture_path) as f:
        aapl_fixture_data = json.load(f)

    tickers_data = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "company_tickers.json" in url_str:
            return httpx.Response(200, json=tickers_data, request=request)
        if "CIK0000320193.json" in url_str:
            return httpx.Response(200, json=aapl_fixture_data, request=request)
        return httpx.Response(404, request=request)

    transport = httpx.MockTransport(handler)
    http_client = create_http_client(transport=transport)
    sec_client = SecClient(http_client=http_client)
    tool = ListFilingsTool(sec_client=sec_client)

    # 1. Unclamped query for 10-K, 10-Q, 8-K
    output_full = await tool.run(
        ListFilingsInput(
            ticker="AAPL",
            form_types=["10-K", "10-Q", "8-K"],
            limit=10,
        )
    )

    assert output_full.ticker == "AAPL"
    assert output_full.cik == "0000320193"
    assert output_full.company_name == "Apple Inc."
    assert len(output_full.filings) > 0
    # Ensure form 4 was excluded
    for filing in output_full.filings:
        assert filing.form in ("10-K", "10-Q", "8-K")
        assert filing.filing_date is not None
        assert filing.accession_number is not None
        assert filing.primary_document_url.startswith(
            "https://www.sec.gov/Archives/edgar/data/320193/"
        )

    # 2. Clamped query with as_of_date="2024-12-31"
    output_clamped = await tool.run(
        ListFilingsInput(
            ticker="AAPL",
            form_types=["10-K", "10-Q", "8-K"],
            as_of_date="2024-12-31",
            limit=10,
        )
    )

    # Primary verification: absolutely no filings after 2024-12-31
    assert len(output_clamped.filings) > 0
    for filing in output_clamped.filings:
        assert filing.filing_date <= "2024-12-31"

    # Only 2023 and 2024 10-K filings exist in fixture before 2024-12-31
    dates = [f.filing_date for f in output_clamped.filings]
    assert all(d <= "2024-12-31" for d in dates)
    assert "2024-11-01" in dates
    assert "2023-11-03" in dates


@pytest.mark.asyncio
async def test_list_filings_registered() -> None:
    """list_filings is registered in default_registry and exports schema."""
    from lekka_ai.tools import default_registry

    tool = default_registry.get("list_filings")
    assert tool is not None
    assert tool.name == "list_filings"
    spec = tool.to_spec()
    assert spec.name == "list_filings"
    props = spec.json_schema.get("properties", {})
    assert "ticker" in props
    assert "form_types" in props
    assert "limit" in props
    assert "as_of_date" in props


@pytest.mark.asyncio
async def test_get_financials_tool_fixture_and_derived_metrics() -> None:
    """Verify get_financials returns facts matching hand calculations."""
    import json
    from pathlib import Path

    import httpx

    from lekka_ai.data.http import create_http_client
    from lekka_ai.data.sec import SecClient
    from lekka_ai.tools import GetFinancialsInput, GetFinancialsTool

    facts_path = Path(__file__).parent / "fixtures" / "AAPL_company_facts.json"
    with open(facts_path, encoding="utf-8") as f:
        aapl_facts_data = json.load(f)

    tickers_data = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "company_tickers.json" in url_str:
            return httpx.Response(200, json=tickers_data, request=request)
        if "companyfacts" in url_str:
            return httpx.Response(200, json=aapl_facts_data, request=request)
        return httpx.Response(404, request=request)

    transport = httpx.MockTransport(handler)
    http_client = create_http_client(transport=transport)
    sec_client = SecClient(http_client=http_client)
    tool = GetFinancialsTool(sec_client=sec_client)

    # 1. Query for recent 4 annual periods
    output = await tool.run(
        GetFinancialsInput(ticker="AAPL", periods=4, period_type="annual")
    )

    assert output.ticker == "AAPL"
    assert output.cik == "0000320193"
    assert output.company_name == "Apple Inc."
    assert len(output.periods) == 4

    # Periods are returned newest first: 2024, 2023, 2022, 2021
    years = [p.fy for p in output.periods]
    assert years == [2024, 2023, 2022, 2021]

    # Verify 2023 FY period standardized metrics
    fy23 = output.periods[1]
    assert fy23.fy == 2023
    assert fy23.fp == "FY"
    assert fy23.filed_date == "2023-11-03"
    assert fy23.form == "10-K"

    assert fy23.revenues == 383_285_000_000
    assert fy23.operating_income == 114_301_000_000
    assert fy23.net_income == 96_995_000_000
    assert fy23.operating_cash_flow == 110_543_000_000
    assert fy23.capex == 10_959_000_000
    assert fy23.cash_and_cash_equivalents == 29_965_000_000
    assert fy23.debt == 95_281_000_000

    # Source info with accession number, filed date, and tag
    assert "revenues" in fy23.source_info
    rev_source = fy23.source_info["revenues"]
    assert rev_source.tag == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert rev_source.accession_number == "0000320193-23-000106"
    assert rev_source.filed_date == "2023-11-03"
    assert rev_source.form == "10-K"

    # Verify derived metrics against exact hand calculations:
    # 1. Revenue YoY: (383_285 - 394_328) / 394_328 = -0.0280046... (-2.800%)
    expected_rev_yoy = (383_285_000_000 - 394_328_000_000) / 394_328_000_000
    assert fy23.revenue_yoy == pytest.approx(expected_rev_yoy, abs=1e-6)
    assert fy23["revenue_yoy"] == pytest.approx(expected_rev_yoy, abs=1e-6)

    # 2. Operating margin: 114_301 / 383_285 = 0.298214... (29.821%)
    expected_op_margin = 114_301_000_000 / 383_285_000_000
    assert fy23.operating_margin == pytest.approx(expected_op_margin, abs=1e-6)
    assert fy23["operating_margin"] == pytest.approx(expected_op_margin, abs=1e-6)

    # 3. Net margin: 96_995 / 383_285 = 0.253062... (25.306%)
    expected_net_margin = 96_995_000_000 / 383_285_000_000
    assert fy23.net_margin == pytest.approx(expected_net_margin, abs=1e-6)

    # 4. Free cash flow: 110_543 - 10_959 = 99_584_000_000 ($99.584B)
    assert fy23.free_cash_flow == 99_584_000_000
    assert fy23["free_cash_flow"] == 99_584_000_000

    # 5. Net debt: 95_281 - 29_965 = 65_316_000_000 ($65.316B)
    assert fy23.net_debt == 65_316_000_000
    assert fy23["net_debt"] == 65_316_000_000


@pytest.mark.asyncio
async def test_get_financials_as_of_date_filter() -> None:
    """Verify point-in-time filter excludes any facts filed after as_of_date."""
    import json
    from pathlib import Path

    import httpx

    from lekka_ai.data.http import create_http_client
    from lekka_ai.data.sec import SecClient
    from lekka_ai.tools import GetFinancialsInput, GetFinancialsTool

    facts_path = Path(__file__).parent / "fixtures" / "AAPL_company_facts.json"
    with open(facts_path, encoding="utf-8") as f:
        aapl_facts_data = json.load(f)

    tickers_data = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "company_tickers.json" in url_str:
            return httpx.Response(200, json=tickers_data, request=request)
        if "companyfacts" in url_str:
            return httpx.Response(200, json=aapl_facts_data, request=request)
        return httpx.Response(404, request=request)

    transport = httpx.MockTransport(handler)
    http_client = create_http_client(transport=transport)
    sec_client = SecClient(http_client=http_client)
    tool = GetFinancialsTool(sec_client=sec_client)

    # Filter as of 2022-12-31: 2023 and 2024 must NOT appear
    output = await tool.run(
        GetFinancialsInput(
            ticker="AAPL", periods=5, as_of_date="2022-12-31", period_type="annual"
        )
    )

    for p in output.periods:
        assert p.filed_date is not None
        assert p.filed_date <= "2022-12-31"

    returned_years = [p.fy for p in output.periods]
    assert 2023 not in returned_years
    assert 2024 not in returned_years
    assert 2022 in returned_years
    assert 2021 in returned_years


@pytest.mark.asyncio
async def test_get_financials_registered() -> None:
    """get_financials is registered in default_registry and exports valid schema."""
    from lekka_ai.tools import default_registry

    tool = default_registry.get("get_financials")
    assert tool is not None
    assert tool.name == "get_financials"
    spec = tool.to_spec()
    assert spec.name == "get_financials"
    props = spec.json_schema.get("properties", {})
    assert "ticker" in props
    assert "periods" in props
    assert "as_of_date" in props
    assert "period_type" in props
