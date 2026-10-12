"""Market data provider interfaces and implementations."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from lekka_ai.config import get_settings

PRICE_COLUMNS: list[str] = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]


class MarketDataError(Exception):
    """Base exception for market data operations."""


class TickerNotFoundError(MarketDataError):
    """Raised when data for a requested ticker cannot be found."""


def _standardize_market_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, ordering, and dtypes for market data.

    Expected columns:
        date (datetime64[ns] tz-naive), open (float), high (float),
        low (float), close (float), adj_close (float), volume (int64)
    """
    if df.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    # Normalize column names to lowercase snake_case
    df = df.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

    if "datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"datetime": "date"})

    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise MarketDataError(f"Missing required price columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    if hasattr(df["date"].dt, "tz") and df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    for col in ["open", "high", "low", "close", "adj_close"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype("int64")

    df = df.sort_values(by="date").reset_index(drop=True)
    return df[PRICE_COLUMNS]


class MarketDataProvider(ABC):
    """Abstract base provider for fetching historical financial market data."""

    @abstractmethod
    async def get_prices(
        self,
        ticker: str,
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical price bars for the given ticker.

        Args:
            ticker: Asset symbol (e.g. 'AAPL', 'SPY').
            start: Start date/timestamp (inclusive if data exists).
            end: End date/timestamp (inclusive if data exists).
            interval: Bar resolution (e.g. '1d', '1h', '5m').

        Returns:
            A pandas DataFrame with columns:
            [date, open, high, low, close, adj_close, volume].
        """


class CsvFixtureProvider(MarketDataProvider):
    """Market data provider reading static CSV fixture files for deterministic tests."""

    def __init__(self, fixtures_dir: Path | str | None = None) -> None:
        if fixtures_dir is not None:
            self.fixtures_dir = Path(fixtures_dir)
        else:
            # Default to tests/fixtures relative to project root
            self.fixtures_dir = (
                Path(__file__).resolve().parents[3] / "tests" / "fixtures"
            )

    async def get_prices(
        self,
        ticker: str,
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Load fixture data for ticker and filter by optional start/end bounds."""
        norm_ticker = ticker.strip().upper()
        fixture_path = self.fixtures_dir / f"{norm_ticker}.csv"
        if not fixture_path.exists():
            # Try exact case
            fixture_path = self.fixtures_dir / f"{ticker.strip()}.csv"
            if not fixture_path.exists():
                raise TickerNotFoundError(
                    f"Fixture not found for ticker '{ticker}' at {fixture_path}"
                )

        raw_df = pd.read_csv(fixture_path)
        df = _standardize_market_dataframe(raw_df)

        if start is not None:
            start_ts = pd.to_datetime(start)
            if hasattr(start_ts, "tz") and start_ts.tz is not None:
                start_ts = start_ts.tz_localize(None)
            df = df[df["date"] >= start_ts]

        if end is not None:
            end_ts = pd.to_datetime(end)
            if hasattr(end_ts, "tz") and end_ts.tz is not None:
                end_ts = end_ts.tz_localize(None)
            df = df[df["date"] <= end_ts]

        return df.reset_index(drop=True)


class YFinanceProvider(MarketDataProvider):
    """Unofficial market data provider backed by Yahoo Finance (via yfinance).

    NOTE: yfinance is an unofficial scraper source that requires no API key.
    It does not provide uptime SLAs and is susceptible to upstream Yahoo Finance changes
    and strict rate limits. For production environments, prefer a paid provider.
    """

    def __init__(self) -> None:
        pass

    def _fetch_sync(
        self,
        ticker: str,
        start: str | date | datetime | None,
        end: str | date | datetime | None,
        interval: str,
    ) -> pd.DataFrame:
        t = yf.Ticker(ticker.strip().upper())
        # yfinance history uses start/end or period
        history_kwargs: dict[str, str | date | datetime] = {"interval": interval}
        if start is not None:
            history_kwargs["start"] = start
        if end is not None:
            history_kwargs["end"] = end

        raw_df = t.history(auto_adjust=False, **history_kwargs)
        if raw_df.empty:
            return pd.DataFrame(columns=PRICE_COLUMNS)

        df = raw_df.reset_index()
        return _standardize_market_dataframe(df)

    async def get_prices(
        self,
        ticker: str,
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical price data asynchronously from Yahoo Finance."""
        return await asyncio.to_thread(self._fetch_sync, ticker, start, end, interval)


# ==============================================================================
# EXTENSION POINT: Paid Market Data Provider using MARKET_DATA_API_KEY
# ==============================================================================
class PaidMarketDataProvider(MarketDataProvider):
    """Extension point for commercial, paid market data providers.

    Requires MARKET_DATA_API_KEY configured in Settings / environment.
    Supported future vendors: Polygon.io, Financial Modeling Prep (FMP),
    Alpha Vantage, or Tiingo.
    """

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.MARKET_DATA_API_KEY
        if not self.api_key:
            raise ValueError(
                "MARKET_DATA_API_KEY is required for PaidMarketDataProvider. "
                "Set MARKET_DATA_API_KEY in your .env or environment variables."
            )

    async def get_prices(
        self,
        ticker: str,
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch prices from a commercial paid market data API.

        To implement:
            1. Use lekka_ai.data.http.AsyncHttpClient to query the vendor REST endpoint.
            2. Parse the vendor JSON payload into PRICE_COLUMNS DataFrame schema.
            3. Apply rate limiting and retry handling.
        """
        raise NotImplementedError(
            "PaidMarketDataProvider is an extension point. "
            "Implement vendor-specific API integration "
            "using MARKET_DATA_API_KEY."
        )
