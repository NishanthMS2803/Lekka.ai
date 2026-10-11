"""Quantitative finance statistical and technical analysis pure functions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Default trading days per period for non-datetime indexed series
TRADING_DAYS_PER_PERIOD: dict[str, int] = {
    "1W": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
    "2Y": 504,
    "3Y": 756,
    "5Y": 1260,
}


@dataclass(frozen=True)
class MaxDrawdownResult:
    """Maximum drawdown metric with key turning-point dates and price levels."""

    max_drawdown: float
    peak_date: Any
    trough_date: Any
    recovery_date: Any | None = None
    peak_price: float = 0.0
    trough_price: float = 0.0

    @property
    def drawdown(self) -> float:
        """Alias for max_drawdown."""
        return self.max_drawdown

    @property
    def magnitude(self) -> float:
        """Absolute magnitude of maximum drawdown."""
        return abs(self.max_drawdown)

    def __iter__(self) -> Any:
        """Allow tuple unpacking: (max_dd, peak_date, trough_date, recovery_date)."""
        return iter(
            (
                self.max_drawdown,
                self.peak_date,
                self.trough_date,
                self.recovery_date,
            )
        )


def _to_series(data: pd.Series | Sequence[float]) -> pd.Series:
    """Convert input data to a float pandas Series if not already one."""
    if isinstance(data, pd.Series):
        return data.astype(float)
    return pd.Series(data, dtype=float)


def simple_returns(prices: pd.Series | Sequence[float]) -> pd.Series:
    """Calculate simple percentage returns from a price series.

    Formula: R_t = (P_t - P_{t-1}) / P_{t-1} = P_t / P_{t-1} - 1
    First element is NaN.
    """
    s = _to_series(prices)
    if s.empty:
        return pd.Series([], dtype=float)
    return (s / s.shift(1)) - 1.0


def log_returns(prices: pd.Series | Sequence[float]) -> pd.Series:
    """Calculate continuously compounded log returns from a price series.

    Formula: r_t = ln(P_t / P_{t-1})
    First element is NaN.
    """
    s = _to_series(prices)
    if s.empty:
        return pd.Series([], dtype=float)
    if (s.dropna() <= 0).any():
        raise ValueError("Prices must be strictly positive to compute log returns.")
    return pd.Series(np.log(s / s.shift(1)), index=s.index, dtype=float)


def period_return(prices: pd.Series, period: str = "1M") -> float:
    """Calculate holding period return over a standard lookback (e.g. 1M, 6M, 1Y).

    Supports DatetimeIndex calendar offsets as well as bar-count lookbacks.
    Formula: (P_end - P_start) / P_start
    """
    if prices.empty:
        raise ValueError("Prices series cannot be empty.")
    if len(prices) == 1:
        return 0.0

    s = _to_series(prices).dropna()
    if s.empty:
        raise ValueError("Prices series contains no valid non-null prices.")

    end_price = float(s.iloc[-1])

    norm_period = period.strip().upper()

    # Case 1: DatetimeIndex (or datetime-convertible index)
    if isinstance(s.index, pd.DatetimeIndex) or (
        len(s.index) > 0 and isinstance(s.index[0], (pd.Timestamp, str))
    ):
        try:
            dt_index = (
                s.index
                if isinstance(s.index, pd.DatetimeIndex)
                else pd.to_datetime(s.index)
            )
            s_dt = pd.Series(s.values, index=dt_index).sort_index()
            end_date = s_dt.index[-1]
            end_price = float(s_dt.iloc[-1])

            offset: pd.DateOffset
            if norm_period in ("1M", "1MO"):
                offset = pd.DateOffset(months=1)
            elif norm_period in ("3M", "3MO"):
                offset = pd.DateOffset(months=3)
            elif norm_period in ("6M", "6MO"):
                offset = pd.DateOffset(months=6)
            elif norm_period in ("1Y", "12M"):
                offset = pd.DateOffset(years=1)
            elif norm_period in ("2Y", "24M"):
                offset = pd.DateOffset(years=2)
            elif norm_period in ("3Y", "36M"):
                offset = pd.DateOffset(years=3)
            elif norm_period in ("5Y", "60M"):
                offset = pd.DateOffset(years=5)
            elif norm_period == "YTD":
                target_date = pd.Timestamp(end_date.year, 1, 1)
                prior = s_dt.loc[:target_date]
                start_price = (
                    float(prior.iloc[-1]) if not prior.empty else float(s_dt.iloc[0])
                )
                if start_price == 0.0:
                    raise ZeroDivisionError("Start price cannot be zero.")
                return (end_price - start_price) / start_price
            else:
                # Default to trading day lookback
                lookback = TRADING_DAYS_PER_PERIOD.get(norm_period, 21)
                start_idx = max(0, len(s_dt) - 1 - lookback)
                start_price = float(s_dt.iloc[start_idx])
                if start_price == 0.0:
                    raise ZeroDivisionError("Start price cannot be zero.")
                return (end_price - start_price) / start_price

            target_date = end_date - offset
            prior = s_dt.loc[:target_date]
            start_price = (
                float(prior.iloc[-1]) if not prior.empty else float(s_dt.iloc[0])
            )
            if start_price == 0.0:
                raise ZeroDivisionError("Start price cannot be zero.")
            return (end_price - start_price) / start_price
        except (ValueError, TypeError):
            # Fall back to bar count if datetime conversion fails
            pass

    # Case 2: Bar count lookback
    lookback = TRADING_DAYS_PER_PERIOD.get(norm_period, 21)
    start_idx = max(0, len(s) - 1 - lookback)
    start_price = float(s.iloc[start_idx])
    if start_price == 0.0:
        raise ZeroDivisionError("Start price cannot be zero.")
    return (end_price - start_price) / start_price


def period_returns(
    prices: pd.Series,
    periods: Sequence[str] = ("1M", "6M", "1Y"),
) -> dict[str, float]:
    """Calculate holding period returns across multiple horizons."""
    return {p: period_return(prices, period=p) for p in periods}


def annualized_volatility(
    returns: pd.Series | Sequence[float],
    periods_per_year: int = 252,
    ddof: int = 1,
) -> float:
    """Calculate annualized standard deviation of returns.

    Formula: sigma_ann = sigma_daily * sqrt(periods_per_year)
    """
    s = _to_series(returns).dropna()
    if len(s) < 2:
        return 0.0
    daily_std = float(s.std(ddof=ddof))
    return float(daily_std * np.sqrt(periods_per_year))


def max_drawdown(prices: pd.Series | Sequence[float]) -> MaxDrawdownResult:
    """Calculate maximum peak-to-trough drawdown with turning point dates.

    Formula: DD_t = (P_t - max_{s <= t} P_s) / (max_{s <= t} P_s)
    Returns: MaxDrawdownResult with max_drawdown, peak_date, trough_date, recovery_date.
    """
    s = _to_series(prices).dropna()
    if s.empty:
        return MaxDrawdownResult(
            max_drawdown=0.0,
            peak_date=None,
            trough_date=None,
            recovery_date=None,
            peak_price=0.0,
            trough_price=0.0,
        )
    if len(s) == 1:
        first_idx = s.index[0]
        first_price = float(s.iloc[0])
        return MaxDrawdownResult(
            max_drawdown=0.0,
            peak_date=first_idx,
            trough_date=first_idx,
            recovery_date=first_idx,
            peak_price=first_price,
            trough_price=first_price,
        )

    running_max = s.cummax()
    drawdowns = (s - running_max) / running_max

    min_dd = float(drawdowns.min())
    if min_dd >= 0.0:
        # Never experienced a drawdown
        first_idx = s.index[0]
        first_price = float(s.iloc[0])
        return MaxDrawdownResult(
            max_drawdown=0.0,
            peak_date=first_idx,
            trough_date=first_idx,
            recovery_date=first_idx,
            peak_price=first_price,
            trough_price=first_price,
        )

    trough_idx: Any = drawdowns.idxmin()
    sub_prices = s.loc[:trough_idx]
    peak_idx: Any = sub_prices.idxmax()

    peak_price = float(s.loc[peak_idx])
    trough_price = float(s.loc[trough_idx])

    # Find recovery date (first point after trough where price >= peak_price)
    after_trough = s.loc[trough_idx:]
    recovered = after_trough[after_trough >= peak_price]
    recovery_idx = recovered.index[0] if not recovered.empty else None

    return MaxDrawdownResult(
        max_drawdown=min_dd,
        peak_date=peak_idx,
        trough_date=trough_idx,
        recovery_date=recovery_idx,
        peak_price=peak_price,
        trough_price=trough_price,
    )


def simple_moving_average(
    prices: pd.Series | Sequence[float],
    window: int,
) -> pd.Series:
    """Calculate simple moving average (SMA) over a rolling window."""
    if window <= 0:
        raise ValueError("Window size must be greater than zero.")
    s = _to_series(prices)
    return s.rolling(window=window).mean()


def exponential_moving_average(
    prices: pd.Series | Sequence[float],
    window: int,
    span: int | None = None,
) -> pd.Series:
    """Calculate exponential moving average (EMA) with span smoothing."""
    eff_span = span if span is not None else window
    if eff_span <= 0:
        raise ValueError("Span/window size must be greater than zero.")
    s = _to_series(prices)
    return s.ewm(span=eff_span, adjust=False).mean()


def moving_average(
    prices: pd.Series | Sequence[float],
    window: int,
    kind: str = "sma",
) -> pd.Series:
    """Compute moving average of specified kind ('sma' or 'ema')."""
    kind_lower = kind.strip().lower()
    if kind_lower in ("sma", "simple"):
        return simple_moving_average(prices, window=window)
    if kind_lower in ("ema", "exponential"):
        return exponential_moving_average(prices, window=window)
    raise ValueError(
        f"Unsupported moving average kind: '{kind}'. Expected 'sma' or 'ema'."
    )


# Aliases
sma = simple_moving_average
ema = exponential_moving_average


def align_series(
    series1: pd.Series | Sequence[float],
    series2: pd.Series | Sequence[float],
) -> tuple[pd.Series, pd.Series]:
    """Align two time series on common index labels and drop missing/NaN values.

    Handles date alignment, differing series lengths, and missing observations.
    """
    s1 = _to_series(series1)
    s2 = _to_series(series2)

    has_labels1 = not s1.index.equals(pd.RangeIndex(len(s1)))
    has_labels2 = not s2.index.equals(pd.RangeIndex(len(s2)))

    if has_labels1 and has_labels2:
        df = pd.concat(
            [s1.rename("s1"), s2.rename("s2")], axis=1, join="inner"
        ).dropna()
        return df["s1"], df["s2"]

    # Positional alignment for non-labeled sequence inputs
    min_len = min(len(s1), len(s2))
    sub1 = s1.iloc[-min_len:].reset_index(drop=True)
    sub2 = s2.iloc[-min_len:].reset_index(drop=True)
    df = pd.concat([sub1.rename("s1"), sub2.rename("s2")], axis=1).dropna()
    return df["s1"], df["s2"]


def _resolve_returns(
    stock: pd.Series | Sequence[float],
    benchmark: pd.Series | Sequence[float],
    is_returns: bool | None,
) -> tuple[pd.Series, pd.Series]:
    """Align series and convert to returns if inputs are price series."""
    s, b = align_series(stock, benchmark)
    if len(s) < 2:
        return pd.Series([], dtype=float), pd.Series([], dtype=float)

    if is_returns is None:
        # Auto-detect returns vs prices
        # Returns typically have negative values or small magnitudes (<= 2.0)
        is_ret = bool(
            (s < 0).any()
            or (b < 0).any()
            or (s.abs().max() <= 2.0 and b.abs().max() <= 2.0)
        )
    else:
        is_ret = is_returns

    if not is_ret:
        r_stock = simple_returns(s).dropna()
        r_bench = simple_returns(b).dropna()
        return align_series(r_stock, r_bench)

    return s, b


def beta(
    stock: pd.Series | Sequence[float],
    benchmark: pd.Series | Sequence[float],
    is_returns: bool | None = None,
) -> float:
    """Calculate the beta of a stock relative to a benchmark (e.g., SPY).

    Formula: beta = Cov(R_stock, R_benchmark) / Var(R_benchmark)
    Automatically handles date alignment and missing data.
    """
    r_stock, r_bench = _resolve_returns(stock, benchmark, is_returns)
    if len(r_stock) < 2:
        return 0.0

    var_b = float(r_bench.var(ddof=1))
    if var_b == 0.0:
        return 0.0

    cov = float(r_stock.cov(r_bench))
    return float(cov / var_b)


def correlation(
    stock: pd.Series | Sequence[float],
    benchmark: pd.Series | Sequence[float],
    is_returns: bool | None = None,
) -> float:
    """Calculate the Pearson correlation coefficient between a stock and benchmark.

    Formula: rho = Cov(R_stock, R_benchmark) / (std(R_stock) * std(R_benchmark))
    Automatically handles date alignment and missing data.
    """
    r_stock, r_bench = _resolve_returns(stock, benchmark, is_returns)
    if len(r_stock) < 2:
        return 0.0

    std_s = float(r_stock.std(ddof=1))
    std_b = float(r_bench.std(ddof=1))
    if std_s == 0.0 or std_b == 0.0:
        return 0.0

    cov = float(r_stock.cov(r_bench))
    corr = cov / (std_s * std_b)
    return float(np.clip(corr, -1.0, 1.0))


def relative_strength(
    stock: pd.Series | Sequence[float],
    benchmark: pd.Series | Sequence[float],
    window: int | None = None,
) -> float:
    """Calculate relative strength of a stock vs a benchmark over a window.

    Formula: RS = ((1 + R_stock) / (1 + R_benchmark)) - 1
    Equivalent to the percentage return of the price ratio (P_stock / P_benchmark).
    Automatically handles date alignment and missing data.
    """
    s, b = align_series(stock, benchmark)
    if len(s) < 2:
        return 0.0

    if window is not None:
        if window <= 0:
            raise ValueError("Window must be greater than zero.")
        effective_window = min(window, len(s))
        sub_s = s.iloc[-effective_window:]
        sub_b = b.iloc[-effective_window:]
    else:
        sub_s = s
        sub_b = b

    start_s, end_s = float(sub_s.iloc[0]), float(sub_s.iloc[-1])
    start_b, end_b = float(sub_b.iloc[0]), float(sub_b.iloc[-1])

    if start_s == 0.0 or start_b == 0.0:
        raise ZeroDivisionError("Start price cannot be zero.")

    r_s = (end_s - start_s) / start_s
    r_b = (end_b - start_b) / start_b

    denom = 1.0 + r_b
    if denom == 0.0:
        raise ZeroDivisionError("Benchmark return cannot be -100%.")

    return float(((1.0 + r_s) / denom) - 1.0)


def relative_strength_series(
    stock: pd.Series | Sequence[float],
    benchmark: pd.Series | Sequence[float],
    window: int | None = None,
) -> pd.Series:
    """Calculate rolling relative strength series of a stock vs a benchmark.

    Automatically handles date alignment and missing data.
    """
    s, b = align_series(stock, benchmark)
    if s.empty:
        return pd.Series([], dtype=float)

    ratio = s / b
    if window is None or window <= 1:
        first_val = float(ratio.iloc[0])
        return pd.Series((ratio / first_val) - 1.0, dtype=float)
    return pd.Series((ratio / ratio.shift(window - 1)) - 1.0, dtype=float)
