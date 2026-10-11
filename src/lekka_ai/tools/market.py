"""Tool for retrieving historical market prices with point-in-time cutoff."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lekka_ai.data.market import MarketDataProvider, YFinanceProvider
from lekka_ai.tools.base import Tool, default_registry


class GetMarketDataInput(BaseModel):
    """Input payload for get_market_data tool."""

    ticker: str = Field(..., description="Asset ticker symbol, e.g. 'AAPL' or 'SPY'")
    start: str = Field(..., description="Start date formatted as YYYY-MM-DD")
    end: str = Field(..., description="End date formatted as YYYY-MM-DD")
    as_of_date: str | None = Field(
        default=None,
        description=(
            "Optional point-in-time cutoff date (YYYY-MM-DD). "
            "Drops all rows strictly after this date to avoid lookahead bias."
        ),
    )

    @field_validator("start", "end", mode="before")
    @classmethod
    def _validate_dates(cls, v: Any) -> str:
        if isinstance(v, (date, datetime)):
            return v.strftime("%Y-%m-%d")
        return str(v)

    @field_validator("as_of_date", mode="before")
    @classmethod
    def _validate_as_of(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, (date, datetime)):
            return v.strftime("%Y-%m-%d")
        return str(v)


class MarketSummary(BaseModel):
    """Summary metrics of the market data within the point-in-time window."""

    model_config = ConfigDict(extra="allow")

    latest_close: float | None = Field(
        default=None, description="Latest closing price in the window"
    )
    date: str | None = Field(
        default=None, description="Date of the latest closing price (YYYY-MM-DD)"
    )
    latest_date: str | None = Field(
        default=None, description="Alias for date of latest closing price"
    )
    total_bars: int = Field(
        default=0, description="Total number of price bars in the window"
    )
    min_close: float | None = Field(
        default=None, description="Minimum close price in window"
    )
    max_close: float | None = Field(
        default=None, description="Maximum close price in window"
    )

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class CompactBar(BaseModel):
    """Compact daily market bar representing a single trading day."""

    date: str = Field(..., description="Trading date (YYYY-MM-DD)")
    close: float = Field(..., description="Closing price")
    volume: int | None = Field(default=None, description="Trading volume")


class GetMarketDataOutput(BaseModel):
    """Output payload for get_market_data tool."""

    model_config = ConfigDict(extra="allow")

    summary: MarketSummary = Field(
        ..., description="Summary metrics including latest close and date"
    )
    series_reference: str = Field(
        ...,
        description="Compact reference identifier for the clamped price series",
    )
    compact_series: list[CompactBar] = Field(
        default_factory=list,
        description=(
            "Compact list of price bars in window (capped to prevent context explosion)"
        ),
    )

    @property
    def latest_close(self) -> float | None:
        return self.summary.latest_close

    @property
    def date(self) -> str | None:
        return self.summary.date

    @property
    def bars(self) -> list[CompactBar]:
        return self.compact_series

    @property
    def series(self) -> list[CompactBar]:
        return self.compact_series

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class GetMarketDataTool(Tool[GetMarketDataInput, GetMarketDataOutput]):
    """Tool that fetches market data with point-in-time as_of_date clamping."""

    name: str = "get_market_data"
    description: str = (
        "Retrieve historical market price data (OHLCV) for a stock or ETF ticker. "
        "Supports an optional as_of_date cutoff to enforce point-in-time analysis "
        "and prevent future data leakage. Returns a summary (latest close, date) "
        "and a compact series reference."
    )
    input_model: type[GetMarketDataInput] = GetMarketDataInput
    output_model: type[GetMarketDataOutput] = GetMarketDataOutput

    def __init__(self, provider: MarketDataProvider | None = None) -> None:
        self.provider: MarketDataProvider = provider or YFinanceProvider()

    async def run(self, input_data: GetMarketDataInput) -> GetMarketDataOutput:
        """Execute market data retrieval and clamp by as_of_date."""
        ticker = input_data.ticker.strip().upper()

        # Query provider
        df = await self.provider.get_prices(
            ticker=ticker,
            start=input_data.start,
            end=input_data.end,
        )

        # Enforce point-in-time clamp: drop all rows strictly after as_of_date
        if input_data.as_of_date is not None and not df.empty:
            as_of_str = input_data.as_of_date.strip()
            df = df[df["date"].dt.strftime("%Y-%m-%d") <= as_of_str].copy()

        if df.empty:
            summary = MarketSummary(
                latest_close=None,
                date=None,
                latest_date=None,
                total_bars=0,
                min_close=None,
                max_close=None,
            )
            series_ref = (
                f"market_series:{ticker}:{input_data.start}:"
                f"{input_data.as_of_date or input_data.end}:0_bars"
            )
            return GetMarketDataOutput(
                summary=summary,
                series_reference=series_ref,
                compact_series=[],
            )

        df = df.sort_values(by="date").reset_index(drop=True)
        last_row = df.iloc[-1]
        latest_date = pd.to_datetime(last_row["date"]).strftime("%Y-%m-%d")
        latest_close = float(last_row["close"])
        total_bars = len(df)

        summary = MarketSummary(
            latest_close=latest_close,
            date=latest_date,
            latest_date=latest_date,
            total_bars=total_bars,
            min_close=float(df["close"].min()),
            max_close=float(df["close"].max()),
        )

        series_ref = (
            f"market_series:{ticker}:{input_data.start}:{latest_date}:{total_bars}_bars"
        )

        # Compact series capped at most recent 20 bars to avoid context bloat
        compact_df = df.tail(20) if total_bars > 20 else df
        compact_bars = [
            CompactBar(
                date=pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                close=float(row["close"]),
                volume=(
                    int(row["volume"])
                    if "volume" in row and pd.notna(row["volume"])
                    else None
                ),
            )
            for _, row in compact_df.iterrows()
        ]

        return GetMarketDataOutput(
            summary=summary,
            series_reference=series_ref,
            compact_series=compact_bars,
        )


MarketDataTool = GetMarketDataTool

# Register default instance in tool registry
default_market_tool = GetMarketDataTool()
default_registry.register(default_market_tool, overwrite=True)
