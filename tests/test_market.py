"""Tests for market data providers (CSV fixture, yfinance, paid provider)."""

import os
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from lekka_ai.data.market import (
    PRICE_COLUMNS,
    CsvFixtureProvider,
    PaidMarketDataProvider,
    TickerNotFoundError,
    YFinanceProvider,
)


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_provider(fixtures_dir: Path) -> CsvFixtureProvider:
    return CsvFixtureProvider(fixtures_dir=fixtures_dir)


@pytest.mark.asyncio
async def test_csv_fixture_provider_aapl(fixture_provider: CsvFixtureProvider) -> None:
    """Verify loading AAPL fixture CSV returns correctly formatted DataFrame."""
    df = await fixture_provider.get_prices("AAPL")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == PRICE_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_float_dtype(df["open"])
    assert pd.api.types.is_float_dtype(df["high"])
    assert pd.api.types.is_float_dtype(df["low"])
    assert pd.api.types.is_float_dtype(df["close"])
    assert pd.api.types.is_float_dtype(df["adj_close"])
    assert pd.api.types.is_integer_dtype(df["volume"])

    # All prices and volume should be strictly positive
    assert (df["open"] > 0).all()
    assert (df["volume"] > 0).all()


@pytest.mark.asyncio
async def test_csv_fixture_provider_spy_case_insensitive(
    fixture_provider: CsvFixtureProvider,
) -> None:
    """Verify loading SPY fixture works with lowercase ticker."""
    df = await fixture_provider.get_prices("spy")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == PRICE_COLUMNS
    assert len(df) == 9  # 9 trading days in fixture


@pytest.mark.asyncio
async def test_csv_fixture_provider_date_filtering(
    fixture_provider: CsvFixtureProvider,
) -> None:
    """Verify start and end bounds filter the resulting DataFrame."""
    df = await fixture_provider.get_prices(
        "AAPL",
        start="2024-01-04",
        end="2024-01-08",
    )

    assert len(df) == 3  # 2024-01-04, 2024-01-05, 2024-01-08
    dates = [d.strftime("%Y-%m-%d") for d in df["date"]]
    assert dates == ["2024-01-04", "2024-01-05", "2024-01-08"]


@pytest.mark.asyncio
async def test_csv_fixture_provider_date_objects(
    fixture_provider: CsvFixtureProvider,
) -> None:
    """Verify start and end bounds accept datetime.date objects."""
    df = await fixture_provider.get_prices(
        "AAPL",
        start=date(2024, 1, 8),
        end=date(2024, 1, 10),
    )

    assert len(df) == 3
    dates = [d.strftime("%Y-%m-%d") for d in df["date"]]
    assert dates == ["2024-01-08", "2024-01-09", "2024-01-10"]


@pytest.mark.asyncio
async def test_csv_fixture_provider_not_found(
    fixture_provider: CsvFixtureProvider,
) -> None:
    """Verify TickerNotFoundError is raised when fixture CSV is missing."""
    with pytest.raises(TickerNotFoundError, match="Fixture not found for ticker"):
        await fixture_provider.get_prices("NONEXISTENT")


def test_paid_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """PaidMarketDataProvider must raise ValueError if no API key is supplied."""
    from lekka_ai.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "MARKET_DATA_API_KEY", None)

    with pytest.raises(ValueError, match="MARKET_DATA_API_KEY is required"):
        PaidMarketDataProvider()


@pytest.mark.asyncio
async def test_paid_provider_extension_point_not_implemented() -> None:
    """PaidMarketDataProvider raises NotImplementedError when get_prices is called."""
    provider = PaidMarketDataProvider(api_key="test_paid_key")
    with pytest.raises(NotImplementedError, match="extension point"):
        await provider.get_prices("AAPL", start="2024-01-01", end="2024-01-10")


@pytest.mark.live
@pytest.mark.asyncio
async def test_yfinance_live_get_prices() -> None:
    """Live integration test: verify yfinance fetches live market prices for AAPL."""
    if os.getenv("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE is not set to 1; skipping live yfinance test")

    provider = YFinanceProvider()
    df = await provider.get_prices(
        "AAPL",
        start="2024-01-02",
        end="2024-01-05",
        interval="1d",
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == PRICE_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert (df["open"] > 0).all()
    assert (df["high"] >= df["low"]).all()
    assert (df["volume"] > 0).all()
