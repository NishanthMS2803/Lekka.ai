"""Tests for scenario analysis and valuation sensitivity grid modeling."""

import pytest

from lekka_ai.quant.dcf import DCFAssumptions
from lekka_ai.quant.scenarios import (
    ScenarioAssumption,
    ScenarioTableResult,
    SensitivityGridResult,
    scenario_table,
    sensitivity_grid,
)


def test_scenario_table_bear_base_bull() -> None:
    """Test scenario_table probability-weighted valuation and upside/downside."""
    # Bear: 25% probability, 3% growth
    # Base: 50% probability, 8% growth
    # Bull: 25% probability, 15% growth
    bear = ScenarioAssumption(
        name="Bear",
        probability=0.25,
        starting_fcf=100.0,
        fcf_growth_path=[0.03] * 5,
        discount_rate=0.10,
        terminal_growth=0.015,
        shares_outstanding=50.0,
        net_debt=50.0,
    )
    base = ScenarioAssumption(
        name="Base",
        probability=0.50,
        starting_fcf=100.0,
        fcf_growth_path=[0.08] * 5,
        discount_rate=0.10,
        terminal_growth=0.02,
        shares_outstanding=50.0,
        net_debt=50.0,
    )
    bull = ScenarioAssumption(
        name="Bull",
        probability=0.25,
        starting_fcf=100.0,
        fcf_growth_path=[0.15] * 5,
        discount_rate=0.10,
        terminal_growth=0.025,
        shares_outstanding=50.0,
        net_debt=50.0,
    )

    current_price = 28.0
    res = scenario_table(
        bear=bear,
        base=base,
        bull=bull,
        current_price=current_price,
    )

    assert isinstance(res, ScenarioTableResult)
    assert len(res.scenarios) == 3

    bear_out = res.scenarios[0]
    base_out = res.scenarios[1]
    bull_out = res.scenarios[2]

    assert bear_out.name == "Bear"
    assert base_out.name == "Base"
    assert bull_out.name == "Bull"

    # Monotonicity: Bear < Base < Bull fair values
    assert (
        bear_out.fair_value_per_share
        < base_out.fair_value_per_share
        < bull_out.fair_value_per_share
    )

    # Upside / Downside calculation relative to current price
    expected_bear_upside = (
        bear_out.fair_value_per_share - current_price
    ) / current_price
    assert (
        pytest.approx(bear_out.upside_downside_pct or 0.0, rel=1e-5)
        == expected_bear_upside
    )

    # Probability-weighted fair value
    expected_weighted = (
        0.25 * bear_out.fair_value_per_share
        + 0.50 * base_out.fair_value_per_share
        + 0.25 * bull_out.fair_value_per_share
    )
    assert (
        pytest.approx(res.probability_weighted_fair_value, rel=1e-5)
        == expected_weighted
    )

    # Weighted upside
    expected_weighted_upside = (expected_weighted - current_price) / current_price
    assert (
        pytest.approx(res.probability_weighted_upside_pct or 0.0, rel=1e-5)
        == expected_weighted_upside
    )

    # Serialization
    d = res.to_dict()
    assert len(d["scenarios"]) == 3
    assert d["current_price"] == current_price


def test_scenario_table_probability_normalization() -> None:
    """Probabilities not summing to 1 are normalized automatically."""
    s1 = ScenarioAssumption(
        name="Case1",
        probability=1.0,
        starting_fcf=100.0,
        fcf_growth_path=[0.05],
        discount_rate=0.10,
        terminal_growth=0.02,
        shares_outstanding=10.0,
    )
    s2 = ScenarioAssumption(
        name="Case2",
        probability=3.0,
        starting_fcf=100.0,
        fcf_growth_path=[0.10],
        discount_rate=0.10,
        terminal_growth=0.02,
        shares_outstanding=10.0,
    )

    res = scenario_table([s1, s2])
    assert res.scenarios[0].probability == 0.25
    assert res.scenarios[1].probability == 0.75


def test_sensitivity_grid_matrix_and_lookups() -> None:
    """Test 2D sensitivity grid over discount rate and terminal growth."""
    base_assumptions = DCFAssumptions(
        starting_fcf=100.0,
        fcf_growth_path=[0.08, 0.08, 0.08],
        discount_rate=0.10,
        terminal_growth=0.02,
        shares_outstanding=50.0,
        net_debt=50.0,
    )

    discount_rates = [0.08, 0.09, 0.10, 0.11]
    terminal_growths = [0.015, 0.02, 0.025, 0.03]
    current_price = 30.0

    grid = sensitivity_grid(
        base_assumptions=base_assumptions,
        discount_rates=discount_rates,
        terminal_growths=terminal_growths,
        current_price=current_price,
    )

    assert isinstance(grid, SensitivityGridResult)
    assert len(grid.fair_values) == 4
    assert all(len(row) == 4 for row in grid.fair_values)

    # Higher discount rate implies lower fair value (for fixed terminal growth)
    val_r8 = grid.get_fair_value(0.08, 0.02)
    val_r10 = grid.get_fair_value(0.10, 0.02)
    val_r11 = grid.get_fair_value(0.11, 0.02)
    assert val_r8 is not None and val_r10 is not None and val_r11 is not None
    assert val_r8 > val_r10 > val_r11

    # Higher terminal growth implies higher fair value (for fixed discount rate)
    val_g15 = grid.get_fair_value(0.10, 0.015)
    val_g25 = grid.get_fair_value(0.10, 0.025)
    assert val_g15 is not None and val_g25 is not None
    assert val_g15 < val_r10 < val_g25

    # Check upside percentages matrix
    assert grid.upside_percentages is not None
    r10_idx = discount_rates.index(0.10)
    g20_idx = terminal_growths.index(0.02)
    expected_upside = (val_r10 - current_price) / current_price
    assert (
        pytest.approx(grid.upside_percentages[r10_idx][g20_idx] or 0.0, rel=1e-5)
        == expected_upside
    )


def test_sensitivity_grid_invalid_cells_gracefully_none() -> None:
    """Cells where terminal_growth >= discount_rate are safely set to None."""
    discount_rates = [0.03, 0.08]
    terminal_growths = [0.02, 0.035]

    grid = sensitivity_grid(
        starting_fcf=100.0,
        fcf_growth_path=[0.05],
        shares_outstanding=10.0,
        discount_rates=discount_rates,
        terminal_growths=terminal_growths,
    )

    # For r = 0.03 and g = 0.035: g >= r -> cell must be None
    assert grid.get_fair_value(0.03, 0.035) is None
    # For r = 0.08 and g = 0.02: valid -> cell must be a float
    assert grid.get_fair_value(0.08, 0.02) is not None
