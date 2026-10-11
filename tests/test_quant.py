"""Unit tests for quantitative statistics pure functions with hand-computed answers."""

import math

import numpy as np
import pandas as pd
import pytest

from lekka_ai.quant import (
    MaxDrawdownResult,
    annualized_volatility,
    ema,
    exponential_moving_average,
    log_returns,
    max_drawdown,
    moving_average,
    period_return,
    period_returns,
    simple_moving_average,
    simple_returns,
    sma,
)


def test_simple_returns_hand_computed() -> None:
    """Verify simple returns against exact manual calculations."""
    prices = [100.0, 105.0, 94.5, 103.95]
    returns = simple_returns(prices)

    assert len(returns) == 4
    assert np.isnan(returns.iloc[0])
    assert returns.iloc[1] == pytest.approx(0.05, abs=1e-6)  # (105-100)/100 = +5%
    assert returns.iloc[2] == pytest.approx(-0.10, abs=1e-6)  # (94.5-105)/105 = -10%
    assert returns.iloc[3] == pytest.approx(0.10, abs=1e-6)  # (103.95-94.5)/94.5 = +10%


def test_log_returns_hand_computed() -> None:
    """Verify log returns match analytical ln(P_t / P_{t-1})."""
    prices = [100.0, 105.0, 94.5]
    returns = log_returns(prices)

    assert len(returns) == 3
    assert np.isnan(returns.iloc[0])
    assert returns.iloc[1] == pytest.approx(math.log(1.05), abs=1e-6)
    assert returns.iloc[2] == pytest.approx(math.log(0.90), abs=1e-6)

    # Sum of log returns must equal ln(final / initial)
    total_log_return = float(returns.iloc[1] + returns.iloc[2])
    assert total_log_return == pytest.approx(math.log(94.5 / 100.0), abs=1e-6)


def test_log_returns_zero_or_negative_raises() -> None:
    """Non-positive prices must raise ValueError for log returns."""
    with pytest.raises(ValueError, match="strictly positive"):
        log_returns([100.0, 0.0, 50.0])

    with pytest.raises(ValueError, match="strictly positive"):
        log_returns([100.0, -10.0, 50.0])


def test_period_returns_calendar_offsets() -> None:
    """Verify 1M, 6M, 1Y returns with exact calendar DatetimeIndex dates."""
    dates = pd.to_datetime(["2023-01-01", "2023-07-01", "2023-12-01", "2024-01-01"])
    prices = pd.Series([100.0, 120.0, 110.0, 132.0], index=dates)

    # 1M: 2023-12-01 to 2024-01-01 -> (132 - 110) / 110 = 22 / 110 = 0.20 (+20%)
    ret_1m = period_return(prices, "1M")
    assert ret_1m == pytest.approx(0.20, abs=1e-6)

    # 6M: 2023-07-01 to 2024-01-01 -> (132 - 120) / 120 = 12 / 120 = 0.10 (+10%)
    ret_6m = period_return(prices, "6M")
    assert ret_6m == pytest.approx(0.10, abs=1e-6)

    # 1Y: 2023-01-01 to 2024-01-01 -> (132 - 100) / 100 = 32 / 100 = 0.32 (+32%)
    ret_1y = period_return(prices, "1Y")
    assert ret_1y == pytest.approx(0.32, abs=1e-6)

    # Batch calculation
    all_rets = period_returns(prices, ("1M", "6M", "1Y"))
    assert all_rets == {
        "1M": pytest.approx(0.20, abs=1e-6),
        "6M": pytest.approx(0.10, abs=1e-6),
        "1Y": pytest.approx(0.32, abs=1e-6),
    }


def test_period_returns_trading_day_bars() -> None:
    """Verify period returns fallback to trading day lookbacks when non-dated."""
    # 253 trading days: 0..252
    values = [100.0] * 253
    values[231] = 110.0  # 21 bars before 252 (1M lookback)
    values[126] = 120.0  # 126 bars before 252 (6M lookback)
    values[0] = 100.0  # 252 bars before 252 (1Y lookback)
    values[252] = 132.0  # current end price

    prices = pd.Series(values)
    assert period_return(prices, "1M") == pytest.approx(0.20, abs=1e-6)
    assert period_return(prices, "6M") == pytest.approx(0.10, abs=1e-6)
    assert period_return(prices, "1Y") == pytest.approx(0.32, abs=1e-6)


def test_annualized_volatility_hand_computed() -> None:
    """Verify annualized volatility against analytical standard deviation."""
    # Returns with zero mean: deviations [0.01, -0.01, 0.01, -0.01]
    # Sum of squares = 4 * 0.0001 = 0.0004
    # Sample std (N-1 = 3) = sqrt(0.0004 / 3) = 0.02 / sqrt(3)
    returns = [0.01, -0.01, 0.01, -0.01]

    # If periods_per_year = 3, (0.02 / sqrt(3)) * sqrt(3) = 0.02 exactly
    vol_3 = annualized_volatility(returns, periods_per_year=3)
    assert vol_3 == pytest.approx(0.02, abs=1e-6)

    # Standard daily 252 periods: (0.02 / sqrt(3)) * sqrt(252) = 0.02 * sqrt(84)
    expected_252 = 0.02 * math.sqrt(84)
    vol_252 = annualized_volatility(returns, periods_per_year=252)
    assert vol_252 == pytest.approx(expected_252, abs=1e-6)

    # Degenerate cases
    assert annualized_volatility([]) == 0.0
    assert annualized_volatility([0.05]) == 0.0


def test_max_drawdown_with_recovery_dates() -> None:
    """Verify max drawdown identifies exact trough, peak, and recovery dates."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    prices = pd.Series([100.0, 120.0, 90.0, 110.0, 130.0], index=dates)

    result = max_drawdown(prices)

    assert isinstance(result, MaxDrawdownResult)
    # Peak is 120.0 on 2024-01-02, trough is 90.0 on 2024-01-03
    # Drawdown = (90 - 120) / 120 = -30 / 120 = -0.25 (-25%)
    assert result.max_drawdown == pytest.approx(-0.25, abs=1e-6)
    assert result.magnitude == pytest.approx(0.25, abs=1e-6)
    assert result.peak_date == "2024-01-02"
    assert result.peak_price == 120.0
    assert result.trough_date == "2024-01-03"
    assert result.trough_price == 90.0
    assert result.recovery_date == "2024-01-05"  # 130 >= 120 reached on 2024-01-05

    # Unpack support
    mdd, peak_d, trough_d, rec_d = result
    assert mdd == pytest.approx(-0.25, abs=1e-6)
    assert peak_d == "2024-01-02"
    assert trough_d == "2024-01-03"
    assert rec_d == "2024-01-05"


def test_max_drawdown_without_recovery() -> None:
    """Verify unrecovered series has recovery_date=None."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    prices = pd.Series([100.0, 80.0, 85.0], index=dates)

    result = max_drawdown(prices)
    assert result.max_drawdown == pytest.approx(-0.20, abs=1e-6)
    assert result.peak_date == "2024-01-01"
    assert result.trough_date == "2024-01-02"
    assert result.recovery_date is None


def test_max_drawdown_monotonically_increasing() -> None:
    """A series with only gains has zero drawdown."""
    prices = pd.Series([100.0, 105.0, 110.0])
    result = max_drawdown(prices)
    assert result.max_drawdown == 0.0
    assert result.magnitude == 0.0


def test_moving_averages_hand_computed() -> None:
    """Verify SMA and EMA against hand-computed values."""
    prices = [10.0, 20.0, 30.0, 40.0, 50.0]

    # SMA window 3:
    # idx 2: (10 + 20 + 30)/3 = 20.0
    # idx 3: (20 + 30 + 40)/3 = 30.0
    # idx 4: (30 + 40 + 50)/3 = 40.0
    sma_vals = simple_moving_average(prices, window=3)
    assert np.isnan(sma_vals.iloc[0])
    assert np.isnan(sma_vals.iloc[1])
    assert sma_vals.iloc[2] == pytest.approx(20.0, abs=1e-6)
    assert sma_vals.iloc[3] == pytest.approx(30.0, abs=1e-6)
    assert sma_vals.iloc[4] == pytest.approx(40.0, abs=1e-6)

    # Check alias
    assert sma(prices, window=3).equals(sma_vals)
    assert moving_average(prices, window=3, kind="sma").equals(sma_vals)

    # EMA window 3 (span=3, alpha = 2/(3+1) = 0.5):
    # idx 0: 10.0
    # idx 1: 0.5 * 20 + 0.5 * 10 = 15.0
    # idx 2: 0.5 * 30 + 0.5 * 15 = 22.5
    # idx 3: 0.5 * 40 + 0.5 * 22.5 = 31.25
    # idx 4: 0.5 * 50 + 0.5 * 31.25 = 40.625
    ema_vals = exponential_moving_average(prices, window=3)
    assert ema_vals.iloc[0] == pytest.approx(10.0, abs=1e-6)
    assert ema_vals.iloc[1] == pytest.approx(15.0, abs=1e-6)
    assert ema_vals.iloc[2] == pytest.approx(22.5, abs=1e-6)
    assert ema_vals.iloc[3] == pytest.approx(31.25, abs=1e-6)
    assert ema_vals.iloc[4] == pytest.approx(40.625, abs=1e-6)

    # Check alias
    assert ema(prices, window=3).equals(ema_vals)
    assert moving_average(prices, window=3, kind="ema").equals(ema_vals)
