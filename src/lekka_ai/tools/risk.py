"""Risk and performance quantitative statistics tool."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lekka_ai.data.market import MarketDataProvider, YFinanceProvider
from lekka_ai.quant.stats import (
    annualized_volatility,
    beta,
    correlation,
    max_drawdown,
    relative_strength,
    simple_moving_average,
    simple_returns,
)
from lekka_ai.tools.base import Tool, default_registry


class ComputeRiskStatsInput(BaseModel):
    """Input payload for compute_risk_stats tool."""

    ticker: str = Field(..., description="Stock or asset ticker symbol, e.g. 'AAPL'")
    as_of_date: str | None = Field(
        default=None,
        description="Optional point-in-time cutoff date (YYYY-MM-DD)",
    )
    benchmark: str = Field(
        default="SPY",
        description="Benchmark ticker symbol to compare against, defaults to 'SPY'",
    )
    start: str | None = Field(
        default=None,
        description="Optional start date (YYYY-MM-DD) for historical window",
    )

    @field_validator("as_of_date", "start", mode="before")
    @classmethod
    def _validate_dates(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, (date, datetime)):
            return v.strftime("%Y-%m-%d")
        return str(v)


class ComputeRiskStatsOutput(BaseModel):
    """Output payload for compute_risk_stats tool including input echoes."""

    model_config = ConfigDict(extra="allow")

    # Echo inputs used
    ticker: str = Field(..., description="Stock ticker symbol analysed")
    as_of_date: str | None = Field(
        default=None, description="Point-in-time cutoff date used"
    )
    benchmark: str = Field(..., description="Benchmark ticker used for comparison")
    start_date: str | None = Field(
        default=None, description="Actual start date of data window"
    )
    end_date: str | None = Field(
        default=None, description="Actual end date of data window"
    )
    total_bars: int = Field(default=0, description="Total trading bars in window")

    # Core quantitative risk & performance metrics
    returns: float = Field(
        ..., description="Cumulative holding period return of the stock over window"
    )
    volatility: float = Field(
        ..., description="Annualized standard deviation (volatility) of the stock"
    )
    drawdown: float = Field(..., description="Maximum peak-to-trough drawdown")
    max_drawdown: float = Field(..., description="Alias for maximum drawdown")
    drawdown_peak_date: str | None = Field(
        default=None, description="Date of drawdown peak"
    )
    drawdown_trough_date: str | None = Field(
        default=None, description="Date of drawdown trough"
    )
    drawdown_recovery_date: str | None = Field(
        default=None, description="Date of recovery, or None if unrecovered"
    )

    # Benchmark relative metrics
    beta: float = Field(
        ..., description="Systematic risk sensitivity (beta) relative to benchmark"
    )
    correlation: float = Field(
        ..., description="Pearson correlation coefficient vs benchmark"
    )
    relative_strength: float = Field(
        ..., description="Relative strength ratio return vs benchmark over window"
    )
    benchmark_returns: float = Field(
        default=0.0, description="Cumulative return of benchmark over window"
    )

    # Trend classification
    trend_label: str = Field(
        ..., description="Simple trend label based on moving averages"
    )
    moving_averages: dict[str, float] = Field(
        default_factory=dict, description="Moving average levels computed"
    )

    @property
    def trend(self) -> str:
        return self.trend_label

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


def _determine_trend_label(
    prices: pd.Series,
) -> tuple[str, dict[str, float]]:
    """Determine trend label (BULLISH, BEARISH, NEUTRAL) from moving averages."""
    n = len(prices)
    if n < 2:
        return "NEUTRAL", {}

    short_w = max(2, min(20, n // 3))
    long_w = max(3, min(50, (2 * n) // 3))

    sma_short_series = simple_moving_average(prices, short_w)
    sma_long_series = simple_moving_average(prices, long_w)

    sma_short = float(sma_short_series.dropna().iloc[-1])
    sma_long = float(sma_long_series.dropna().iloc[-1])
    latest_price = float(prices.iloc[-1])

    ma_dict = {
        f"sma_{short_w}": sma_short,
        f"sma_{long_w}": sma_long,
    }

    if latest_price > sma_short >= sma_long:
        return "BULLISH", ma_dict
    if latest_price < sma_short <= sma_long:
        return "BEARISH", ma_dict
    if latest_price > sma_short:
        return "BULLISH", ma_dict
    if latest_price < sma_short:
        return "BEARISH", ma_dict
    return "NEUTRAL", ma_dict


class ComputeRiskStatsTool(Tool[ComputeRiskStatsInput, ComputeRiskStatsOutput]):
    """Tool that computes quant risk and performance stats against a benchmark."""

    name: str = "compute_risk_stats"
    description: str = (
        "Compute quantitative risk and performance statistics for a stock ticker "
        "relative to a benchmark (e.g., SPY) as of a specific point-in-time date. "
        "Calculates returns, annualized volatility, max drawdown, beta, correlation, "
        "relative strength, and trend labels based on moving averages."
    )
    input_model: type[ComputeRiskStatsInput] = ComputeRiskStatsInput
    output_model: type[ComputeRiskStatsOutput] = ComputeRiskStatsOutput

    def __init__(self, provider: MarketDataProvider | None = None) -> None:
        self.provider: MarketDataProvider = provider or YFinanceProvider()

    async def run(self, input_data: ComputeRiskStatsInput) -> ComputeRiskStatsOutput:
        """Fetch market data and compute quantitative risk statistics."""
        ticker = input_data.ticker.strip().upper()
        benchmark = input_data.benchmark.strip().upper()

        # Query market data for stock and benchmark up to as_of_date
        df_stock = await self.provider.get_prices(
            ticker=ticker,
            start=input_data.start,
            end=input_data.as_of_date,
        )
        df_bench = await self.provider.get_prices(
            ticker=benchmark,
            start=input_data.start,
            end=input_data.as_of_date,
        )

        # Enforce point-in-time clamp if as_of_date is specified
        if input_data.as_of_date is not None:
            as_of_str = input_data.as_of_date.strip()
            if not df_stock.empty:
                df_stock = df_stock[
                    df_stock["date"].dt.strftime("%Y-%m-%d") <= as_of_str
                ].copy()
            if not df_bench.empty:
                df_bench = df_bench[
                    df_bench["date"].dt.strftime("%Y-%m-%d") <= as_of_str
                ].copy()

        if df_stock.empty:
            raise ValueError(
                f"No market price data found for ticker '{ticker}' "
                f"up to as_of_date '{input_data.as_of_date}'."
            )
        if df_bench.empty:
            raise ValueError(
                f"No benchmark price data found for '{benchmark}' "
                f"up to as_of_date '{input_data.as_of_date}'."
            )

        df_stock = df_stock.sort_values(by="date").reset_index(drop=True)
        df_bench = df_bench.sort_values(by="date").reset_index(drop=True)

        s_stock = pd.Series(df_stock["close"].values, index=df_stock["date"])
        s_bench = pd.Series(df_bench["close"].values, index=df_bench["date"])

        start_date = pd.to_datetime(s_stock.index[0]).strftime("%Y-%m-%d")
        end_date = pd.to_datetime(s_stock.index[-1]).strftime("%Y-%m-%d")
        total_bars = len(s_stock)

        # 1. Holding period return
        p_start_s = float(s_stock.iloc[0])
        p_end_s = float(s_stock.iloc[-1])
        stock_return = (p_end_s - p_start_s) / p_start_s

        p_start_b = float(s_bench.iloc[0])
        p_end_b = float(s_bench.iloc[-1])
        bench_return = (p_end_b - p_start_b) / p_start_b

        # 2. Annualized volatility
        r_stock = simple_returns(s_stock).dropna()
        stock_vol = annualized_volatility(r_stock)

        # 3. Maximum Drawdown
        mdd_res = max_drawdown(s_stock)
        drawdown_val = mdd_res.max_drawdown
        peak_d = (
            pd.to_datetime(mdd_res.peak_date).strftime("%Y-%m-%d")
            if mdd_res.peak_date is not None
            else None
        )
        trough_d = (
            pd.to_datetime(mdd_res.trough_date).strftime("%Y-%m-%d")
            if mdd_res.trough_date is not None
            else None
        )
        rec_d = (
            pd.to_datetime(mdd_res.recovery_date).strftime("%Y-%m-%d")
            if mdd_res.recovery_date is not None
            else None
        )

        # 4. Beta and correlation vs benchmark
        b_val = beta(s_stock, s_bench)
        c_val = correlation(s_stock, s_bench)

        # 5. Relative strength
        rs_val = relative_strength(s_stock, s_bench)

        # 6. Moving average trend label
        trend_label, ma_dict = _determine_trend_label(s_stock)

        return ComputeRiskStatsOutput(
            ticker=ticker,
            as_of_date=input_data.as_of_date,
            benchmark=benchmark,
            start_date=start_date,
            end_date=end_date,
            total_bars=total_bars,
            returns=float(stock_return),
            volatility=float(stock_vol),
            drawdown=float(drawdown_val),
            max_drawdown=float(drawdown_val),
            drawdown_peak_date=peak_d,
            drawdown_trough_date=trough_d,
            drawdown_recovery_date=rec_d,
            beta=float(b_val),
            correlation=float(c_val),
            relative_strength=float(rs_val),
            benchmark_returns=float(bench_return),
            trend_label=trend_label,
            moving_averages=ma_dict,
        )


RiskStatsTool = ComputeRiskStatsTool

# Register default instance in tool registry
default_risk_tool = ComputeRiskStatsTool()
default_registry.register(default_risk_tool, overwrite=True)
