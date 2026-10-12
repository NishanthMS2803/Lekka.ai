"""Unit tests for quantitative statistics pure functions with hand-computed answers."""

import math

import numpy as np
import pandas as pd
import pytest

from lekka_ai.quant import (
    DerivedFinancials,
    MaxDrawdownResult,
    align_series,
    annualized_volatility,
    beta,
    compute_derived_financials,
    correlation,
    ema,
    exponential_moving_average,
    free_cash_flow,
    log_returns,
    max_drawdown,
    moving_average,
    net_debt,
    net_margin,
    operating_margin,
    period_return,
    period_returns,
    relative_strength,
    relative_strength_series,
    simple_moving_average,
    simple_returns,
    sma,
    yoy_growth,
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


def test_beta_and_correlation_synthetic_returns_exact() -> None:
    """Stock = 2x benchmark returns gives exact beta 2.0 and correlation 1.0."""
    bench_returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    stock_returns = 2.0 * bench_returns

    b = beta(stock_returns, bench_returns)
    c = correlation(stock_returns, bench_returns)

    assert b == pytest.approx(2.0, rel=1e-5)
    assert c == pytest.approx(1.0, rel=1e-5)


def test_beta_and_correlation_inverse_synthetic() -> None:
    """Stock = -1.5x benchmark returns gives beta -1.5 and correlation -1.0."""
    bench_returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    stock_returns = -1.5 * bench_returns

    b = beta(stock_returns, bench_returns)
    c = correlation(stock_returns, bench_returns)

    assert b == pytest.approx(-1.5, rel=1e-5)
    assert c == pytest.approx(-1.0, rel=1e-5)


def test_beta_and_correlation_from_prices() -> None:
    """Beta and correlation computed directly from price series."""
    # Benchmark returns: +2%, -2%, +3%
    bench_prices = pd.Series([100.0, 102.0, 99.96, 102.9588])
    # Stock returns: +4%, -4%, +6% (exact 2x benchmark returns)
    stock_prices = pd.Series([50.0, 52.0, 49.92, 52.9152])

    b = beta(stock_prices, bench_prices)
    c = correlation(stock_prices, bench_prices)

    assert b == pytest.approx(2.0, rel=1e-5)
    assert c == pytest.approx(1.0, rel=1e-5)


def test_date_alignment_and_missing_data() -> None:
    """Date alignment handles missing dates and NaN values without crash."""
    bench_dates = pd.date_range("2024-01-01", periods=6)
    # Benchmark prices with known daily returns
    b = pd.Series([100.0, 102.0, 101.0, 104.0, 103.0, 105.0], index=bench_dates)

    # Stock has missing date 2024-01-03, and NaN on 2024-01-02
    stock_dates = [
        bench_dates[0],
        bench_dates[1],
        bench_dates[3],
        bench_dates[4],
        bench_dates[5],
    ]
    s = pd.Series([50.0, np.nan, 52.0, 51.5, 53.0], index=stock_dates)

    aligned_s, aligned_b = align_series(s, b)

    # Common dates: 2024-01-01, 2024-01-04, 2024-01-05, 2024-01-06 (4 rows, NaN dropped)
    assert len(aligned_s) == 4
    assert len(aligned_b) == 4
    assert not aligned_s.isna().any()
    assert not aligned_b.isna().any()
    assert aligned_s.index.equals(aligned_b.index)

    # Beta and correlation execute cleanly on aligned data
    b_val = beta(s, b)
    c_val = correlation(s, b)
    assert isinstance(b_val, float)
    assert isinstance(c_val, float)
    assert -1.0 <= c_val <= 1.0


def test_relative_strength_window() -> None:
    """Verify relative strength over window vs benchmark."""
    # Stock gains 20% over 3-bar window
    stock = pd.Series([100.0, 110.0, 120.0])
    # Benchmark gains 10% over 3-bar window
    bench = pd.Series([100.0, 105.0, 110.0])

    rs = relative_strength(stock, bench, window=3)
    expected_rs = (1.20 / 1.10) - 1.0  # ~0.090909 (9.09% outperformance)

    assert rs == pytest.approx(expected_rs, abs=1e-6)

    # Verify rolling series endpoint matches scalar
    rs_series = relative_strength_series(stock, bench, window=3)
    assert rs_series.iloc[-1] == pytest.approx(rs, abs=1e-6)


def test_relative_strength_flat_benchmark() -> None:
    """When benchmark is flat (0%), relative strength equals stock return."""
    stock = pd.Series([100.0, 105.0, 115.0])  # +15%
    bench = pd.Series([200.0, 200.0, 200.0])  # 0%

    rs = relative_strength(stock, bench, window=3)
    assert rs == pytest.approx(0.15, abs=1e-6)


def test_zero_variance_benchmark() -> None:
    """Constant benchmark returns variance=0, handled without error."""
    stock = pd.Series([100.0, 102.0, 104.0])
    bench = pd.Series([200.0, 200.0, 200.0])

    assert beta(stock, bench) == 0.0
    assert correlation(stock, bench) == 0.0


def test_yoy_growth_hand_computed() -> None:
    """Verify YoY growth against exact manual calculations."""
    # Positive growth: (120 - 100) / 100 = +20%
    assert yoy_growth(120, 100) == pytest.approx(0.20, abs=1e-6)

    # Negative growth: (90 - 100) / 100 = -10%
    assert yoy_growth(90, 100) == pytest.approx(-0.10, abs=1e-6)

    # Negative prior base improving: (-50 - (-100)) / 100 = +50%
    assert yoy_growth(-50, -100) == pytest.approx(0.50, abs=1e-6)

    # Edge cases
    assert yoy_growth(100, 0) is None
    assert yoy_growth(None, 100) is None
    assert yoy_growth(100, None) is None
    assert yoy_growth(None, None) is None


def test_operating_margin_hand_computed() -> None:
    """Verify operating margin: operating_income / revenues."""
    # Hand-checked: 25 / 100 = 0.25 (25%)
    assert operating_margin(25, 100) == pytest.approx(0.25, abs=1e-6)

    # Apple FY23 numbers: 114_301_000_000 / 383_285_000_000
    expected = 114_301_000_000 / 383_285_000_000  # ~0.298214
    assert operating_margin(114_301_000_000, 383_285_000_000) == pytest.approx(
        expected, abs=1e-6
    )

    # Operating loss: -15 / 100 = -0.15
    assert operating_margin(-15, 100) == pytest.approx(-0.15, abs=1e-6)

    # Edge cases
    assert operating_margin(10, 0) is None
    assert operating_margin(None, 100) is None
    assert operating_margin(10, None) is None


def test_net_margin_hand_computed() -> None:
    """Verify net margin: net_income / revenues."""
    # Hand-checked: 20 / 100 = 0.20 (20%)
    assert net_margin(20, 100) == pytest.approx(0.20, abs=1e-6)

    # Apple FY23 numbers: 96_995_000_000 / 383_285_000_000
    expected = 96_995_000_000 / 383_285_000_000  # ~0.253062
    assert net_margin(96_995_000_000, 383_285_000_000) == pytest.approx(
        expected, abs=1e-6
    )

    # Net loss: -5 / 100 = -0.05
    assert net_margin(-5, 100) == pytest.approx(-0.05, abs=1e-6)

    # Edge cases
    assert net_margin(10, 0) is None
    assert net_margin(None, 100) is None
    assert net_margin(10, None) is None


def test_free_cash_flow_hand_computed() -> None:
    """Verify FCF: CFO minus Capex."""
    # Hand-checked: 100 - 30 = 70
    assert free_cash_flow(100, 30) == 70

    # Apple FY23 numbers: 110_543_000_000 - 10_959_000_000 = 99_584_000_000
    assert free_cash_flow(110_543_000_000, 10_959_000_000) == 99_584_000_000

    # Negative capex input formatted as outflow: CFO 100, Capex -30 -> 70
    assert free_cash_flow(100, -30) == 70

    # Negative CFO (cash burn): -20 - 30 = -50
    assert free_cash_flow(-20, 30) == -50

    # Edge cases
    assert free_cash_flow(None, 30) is None
    assert free_cash_flow(100, None) is None


def test_net_debt_hand_computed() -> None:
    """Verify Net Debt: Debt minus Cash."""
    # Hand-checked: 100 - 40 = 60
    assert net_debt(100, 40) == 60

    # Net cash position (cash exceeds debt): 30 - 50 = -20
    assert net_debt(30, 50) == -20

    # Apple FY23 numbers: 95_281_000_000 - 29_965_000_000 = 65_316_000_000
    assert net_debt(95_281_000_000, 29_965_000_000) == 65_316_000_000

    # Edge cases
    assert net_debt(None, None) is None
    assert net_debt(100, None) == 100
    assert net_debt(None, 50) == -50


def test_compute_derived_financials_hand_computed() -> None:
    """Verify compute_derived_financials integrates all metrics correctly."""
    from lekka_ai.data.xbrl import PeriodFinancials

    fy22 = PeriodFinancials(
        fy=2022,
        fp="FY",
        revenues=394_328_000_000,
        net_income=99_803_000_000,
        operating_income=119_437_000_000,
        operating_cash_flow=122_151_000_000,
        capex=10_708_000_000,
        cash_and_cash_equivalents=23_646_000_000,
        debt=98_959_000_000,
    )
    fy23 = PeriodFinancials(
        fy=2023,
        fp="FY",
        revenues=383_285_000_000,
        net_income=96_995_000_000,
        operating_income=114_301_000_000,
        operating_cash_flow=110_543_000_000,
        capex=10_959_000_000,
        cash_and_cash_equivalents=29_965_000_000,
        debt=95_281_000_000,
    )

    derived = compute_derived_financials(current=fy23, prior=fy22)
    assert isinstance(derived, DerivedFinancials)

    # YoY revenue: (383_285 - 394_328) / 394_328 = -2.80%
    expected_rev_yoy = (383_285_000_000 - 394_328_000_000) / 394_328_000_000
    assert derived.revenue_yoy == pytest.approx(expected_rev_yoy, abs=1e-6)

    # YoY net income: (96_995 - 99_803) / 99_803 = -2.813%
    expected_ni_yoy = (96_995_000_000 - 99_803_000_000) / 99_803_000_000
    assert derived.net_income_yoy == pytest.approx(expected_ni_yoy, abs=1e-6)

    # Operating margin: 114_301 / 383_285 = 29.82%
    expected_op_margin = 114_301_000_000 / 383_285_000_000
    assert derived.operating_margin == pytest.approx(expected_op_margin, abs=1e-6)

    # Net margin: 96_995 / 383_285 = 25.31%
    expected_net_margin = 96_995_000_000 / 383_285_000_000
    assert derived.net_margin == pytest.approx(expected_net_margin, abs=1e-6)

    # Free cash flow: 110_543 - 10_959 = 99_584
    assert derived.free_cash_flow == 99_584_000_000

    # Net debt: 95_281 - 29_965 = 65_316
    assert derived.net_debt == 65_316_000_000
