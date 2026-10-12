"""Tests for analyst estimates and industry data tools with graceful degradation."""

import os
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from lekka_ai.data.estimates import (
    AnalystEstimates,
    EstimatesProviderError,
    FinnhubEstimatesProvider,
    FixtureEstimatesProvider,
    IndustryData,
    create_estimates_provider,
)
from lekka_ai.tools.base import default_registry
from lekka_ai.tools.estimates import (
    GetAnalystEstimatesInput,
    GetAnalystEstimatesOutput,
    GetAnalystEstimatesTool,
    GetIndustryDataInput,
    GetIndustryDataOutput,
    GetIndustryDataTool,
)


def test_estimates_tools_registration() -> None:
    """Tools get_analyst_estimates and get_industry_data are registered."""
    tool_est = default_registry.get("get_analyst_estimates")
    assert tool_est is not None
    assert tool_est.name == "get_analyst_estimates"
    spec_est = tool_est.to_spec()
    assert "ticker" in spec_est.json_schema["properties"]
    assert "as_of_date" in spec_est.json_schema["properties"]

    tool_ind = default_registry.get("get_industry_data")
    assert tool_ind is not None
    assert tool_ind.name == "get_industry_data"
    spec_ind = tool_ind.to_spec()
    assert "ticker" in spec_ind.json_schema["properties"]
    assert "as_of_date" in spec_ind.json_schema["properties"]


@pytest.mark.asyncio
async def test_get_analyst_estimates_unavailable_without_key() -> None:
    """get_analyst_estimates gracefully returns structured unavailable without key."""
    # Explicit provider instance with no API key
    provider = FinnhubEstimatesProvider(api_key="")
    tool = GetAnalystEstimatesTool(provider=provider)

    inp = GetAnalystEstimatesInput(ticker="AAPL", as_of_date="2024-01-15")
    result: GetAnalystEstimatesOutput = await tool.run(inp)

    assert isinstance(result, GetAnalystEstimatesOutput)
    assert result.status == "unavailable"
    assert result.available is False
    assert result.ticker == "AAPL"
    assert result.as_of_date == "2024-01-15"
    assert result.estimates is None
    assert result.message is not None
    assert "unavailable" in result.message.lower()


@pytest.mark.asyncio
async def test_get_industry_data_unavailable_without_key() -> None:
    """get_industry_data gracefully returns structured unavailable without key."""
    provider = FinnhubEstimatesProvider(api_key="")
    tool = GetIndustryDataTool(provider=provider)

    inp = GetIndustryDataInput(ticker="MSFT", as_of_date="2024-01-15")
    result: GetIndustryDataOutput = await tool.run(inp)

    assert isinstance(result, GetIndustryDataOutput)
    assert result.status == "unavailable"
    assert result.available is False
    assert result.ticker == "MSFT"
    assert result.as_of_date == "2024-01-15"
    assert result.industry_data is None
    assert result.message is not None
    assert "unavailable" in result.message.lower()


@pytest.mark.asyncio
async def test_estimates_tools_graceful_on_provider_error() -> None:
    """Tools return unavailable when provider raises an error rather than raising."""
    failing_provider = FixtureEstimatesProvider(should_fail=True)

    tool_est = GetAnalystEstimatesTool(provider=failing_provider)
    res_est = await tool_est.run(GetAnalystEstimatesInput(ticker="AAPL"))
    assert res_est.status == "unavailable"
    assert res_est.available is False
    assert res_est.estimates is None
    assert res_est.message is not None
    assert "Simulated remote provider connection error" in res_est.message

    tool_ind = GetIndustryDataTool(provider=failing_provider)
    res_ind = await tool_ind.run(GetIndustryDataInput(ticker="AAPL"))
    assert res_ind.status == "unavailable"
    assert res_ind.available is False
    assert res_ind.industry_data is None
    assert res_ind.message is not None
    assert "Simulated remote provider connection error" in res_ind.message


@pytest.mark.asyncio
async def test_get_analyst_estimates_available_path() -> None:
    """get_analyst_estimates returns structured available payload when data exists."""
    sample_est = AnalystEstimates(
        ticker="AAPL",
        target_price_mean=210.0,
        target_price_high=250.0,
        target_price_low=180.0,
        target_price_median=215.0,
        consensus_rating="Buy",
        num_analysts=35,
        eps_estimate_current_year=6.55,
        revenue_estimate_current_year=390000.0,
        source="TestFixture",
    )
    provider = FixtureEstimatesProvider(estimates_map={"AAPL": sample_est})
    tool = GetAnalystEstimatesTool(provider=provider)

    res = await tool.run(
        GetAnalystEstimatesInput(ticker="aapl", as_of_date="2024-01-15")
    )
    assert res.status == "available"
    assert res.available is True
    assert res.ticker == "AAPL"
    assert res.as_of_date == "2024-01-15"
    assert res.estimates is not None
    assert res.estimates.target_price_mean == 210.0
    assert res.estimates.target_price_high == 250.0
    assert res.estimates.consensus_rating == "Buy"
    assert res.estimates.source == "TestFixture"


@pytest.mark.asyncio
async def test_get_industry_data_available_path() -> None:
    """get_industry_data returns structured available payload when data exists."""
    sample_ind = IndustryData(
        ticker="AAPL",
        sector="Technology",
        industry="Consumer Electronics",
        peer_tickers=["MSFT", "GOOGL", "DELL", "HPQ"],
        pe_ratio_median=28.5,
        ev_ebitda_median=21.0,
        source="TestFixture",
    )
    provider = FixtureEstimatesProvider(industry_map={"AAPL": sample_ind})
    tool = GetIndustryDataTool(provider=provider)

    res = await tool.run(GetIndustryDataInput(ticker="aapl", as_of_date="2024-01-15"))
    assert res.status == "available"
    assert res.available is True
    assert res.ticker == "AAPL"
    assert res.industry_data is not None
    assert res.industry_data.sector == "Technology"
    assert res.industry_data.industry == "Consumer Electronics"
    assert "MSFT" in res.industry_data.peer_tickers
    assert res.industry_data.pe_ratio_median == 28.5


@pytest.mark.asyncio
async def test_finnhub_estimates_provider_mocked() -> None:
    """FinnhubEstimatesProvider fetches and parses price targets from Finnhub API."""
    mock_price_target = {
        "symbol": "AAPL",
        "targetHigh": 250.0,
        "targetLow": 175.0,
        "targetMean": 215.5,
        "targetMedian": 220.0,
        "lastUpdated": "2024-01-10",
    }
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=mock_price_target)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(return_value=mock_resp)

    provider = FinnhubEstimatesProvider(
        api_key="mock_key",
        http_client=mock_client,
    )
    est = await provider.get_estimates("AAPL", as_of_date=date(2024, 1, 15))

    assert est is not None
    assert est.ticker == "AAPL"
    assert est.target_price_mean == 215.5
    assert est.target_price_high == 250.0
    assert est.target_price_low == 175.0
    assert est.target_price_median == 220.0
    assert est.source == "Finnhub"
    assert est.as_of_date == "2024-01-15"


@pytest.mark.asyncio
async def test_finnhub_industry_provider_mocked() -> None:
    """FinnhubEstimatesProvider fetches peers and profile info from Finnhub API."""
    mock_peers = ["AAPL", "MSFT", "GOOGL", "HPQ"]
    mock_profile = {
        "finnhubIndustry": "Consumer Electronics",
        "name": "Apple Inc",
        "ticker": "AAPL",
    }

    mock_peers_resp = MagicMock(spec=httpx.Response)
    mock_peers_resp.status_code = 200
    mock_peers_resp.raise_for_status = MagicMock()
    mock_peers_resp.json = MagicMock(return_value=mock_peers)

    mock_profile_resp = MagicMock(spec=httpx.Response)
    mock_profile_resp.status_code = 200
    mock_profile_resp.raise_for_status = MagicMock()
    mock_profile_resp.json = MagicMock(return_value=mock_profile)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(side_effect=[mock_peers_resp, mock_profile_resp])

    provider = FinnhubEstimatesProvider(
        api_key="mock_key",
        http_client=mock_client,
    )
    ind = await provider.get_industry_data("AAPL")

    assert ind is not None
    assert ind.ticker == "AAPL"
    assert ind.sector == "Consumer Electronics"
    assert "MSFT" in ind.peer_tickers
    assert "AAPL" not in ind.peer_tickers  # Cleaned self ticker
    assert ind.source == "Finnhub"


@pytest.mark.asyncio
async def test_finnhub_provider_raises_error_on_network_failure() -> None:
    """FinnhubEstimatesProvider raises EstimatesProviderError on HTTP failure."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(side_effect=httpx.RequestError("Connection timeout"))

    provider = FinnhubEstimatesProvider(
        api_key="mock_key",
        http_client=mock_client,
    )
    with pytest.raises(EstimatesProviderError):
        await provider.get_estimates("AAPL")

    with pytest.raises(EstimatesProviderError):
        await provider.get_industry_data("AAPL")


def test_create_estimates_provider_factory() -> None:
    """Factory create_estimates_provider returns an EstimatesProvider instance."""
    prov = create_estimates_provider(api_key="test-key")
    assert isinstance(prov, FinnhubEstimatesProvider)
    assert prov.api_key == "test-key"


@pytest.mark.asyncio
async def test_live_estimates_if_key_available() -> None:
    """Optional live test if ESTIMATES_API_KEY is configured in environment."""
    api_key = os.getenv("ESTIMATES_API_KEY")
    if not api_key:
        pytest.skip("ESTIMATES_API_KEY not configured for live test")

    tool = GetAnalystEstimatesTool()
    res = await tool.run(GetAnalystEstimatesInput(ticker="AAPL"))
    assert res.ticker == "AAPL"
    # Even if Finnhub rate limits or succeeds, tool must return valid output
    assert res.status in ("available", "unavailable")
