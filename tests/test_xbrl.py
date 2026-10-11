"""Tests for XBRL company facts parser with tag fallbacks and real SEC fixture."""

import json
from pathlib import Path
from typing import Any

import pytest

from lekka_ai.data.xbrl import (
    DEFAULT_TAG_FALLBACKS,
    CompanyFactsParser,
    PeriodFinancials,
    parse_company_facts,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "AAPL_company_facts.json"


@pytest.fixture
def aapl_facts_fixture() -> dict[str, Any]:
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


def test_parse_company_facts_known_years_revenues(
    aapl_facts_fixture: dict[str, Any],
) -> None:
    """Verify parser returns expected revenue values for known years."""
    parsed = parse_company_facts(aapl_facts_fixture)
    assert len(parsed) > 0

    annual = parsed.annual()
    assert len(annual) >= 5

    # Known canonical annual revenues for Apple Inc.
    expected_revenues = {
        2021: (365_817_000_000, "2021-10-29"),
        2022: (394_328_000_000, "2022-10-28"),
        2023: (383_285_000_000, "2023-11-03"),
        2024: (391_035_000_000, "2024-11-01"),
    }

    for fy, (expected_rev, expected_filed) in expected_revenues.items():
        period = parsed.get((fy, "FY"))
        assert period is not None, f"Expected period ({fy}, 'FY') to exist."
        assert period.fy == fy
        assert period.fp == "FY"
        assert period.revenues == expected_rev
        assert period.revenue == expected_rev
        assert period.Revenues == expected_rev
        assert period["Revenues"] == expected_rev
        assert period["revenues"] == expected_rev
        assert period.filed == expected_filed
        assert period.filed_date == expected_filed
        assert period["filed"] == expected_filed


def test_parse_company_facts_core_financial_metrics(
    aapl_facts_fixture: dict[str, Any],
) -> None:
    """Verify extraction of all core statement metrics for 2023 FY."""
    parsed = parse_company_facts(aapl_facts_fixture)
    fy23 = parsed[(2023, "FY")]

    assert fy23.fy == 2023
    assert fy23.fp == "FY"
    assert fy23.form == "10-K"
    assert fy23.filed == "2023-11-03"
    assert fy23.end == "2023-09-30"

    # Exact audited figures from Apple 2023 10-K
    assert fy23.revenues == 383_285_000_000
    assert fy23.net_income == 96_995_000_000
    assert fy23.NetIncomeLoss == 96_995_000_000
    assert fy23["NetIncomeLoss"] == 96_995_000_000

    assert fy23.operating_income == 114_301_000_000
    assert fy23.OperatingIncomeLoss == 114_301_000_000
    assert fy23["OperatingIncomeLoss"] == 114_301_000_000

    assert fy23.operating_cash_flow == 110_543_000_000
    assert fy23.NetCashProvidedByOperatingActivities == 110_543_000_000
    assert fy23["NetCashProvidedByOperatingActivities"] == 110_543_000_000

    assert fy23.capex == 10_959_000_000
    assert fy23.PaymentsToAcquirePropertyPlantAndEquipment == 10_959_000_000
    assert fy23["PaymentsToAcquirePropertyPlantAndEquipment"] == 10_959_000_000

    assert fy23.cash_and_cash_equivalents == 29_965_000_000
    assert fy23.CashAndCashEquivalents == 29_965_000_000
    assert fy23["CashAndCashEquivalents"] == 29_965_000_000

    assert fy23.debt == 95_281_000_000
    assert fy23["debt"] == 95_281_000_000

    # Provenance tags tracking
    assert (
        fy23.tag_sources["revenues"]
        == "RevenueFromContractWithCustomerExcludingAssessedTax"
    )
    assert fy23.tag_sources["net_income"] == "NetIncomeLoss"
    assert fy23.tag_sources["operating_income"] == "OperatingIncomeLoss"
    assert (
        fy23.tag_sources["operating_cash_flow"]
        == "NetCashProvidedByUsedInOperatingActivities"
    )
    assert fy23.tag_sources["capex"] == "PaymentsToAcquirePropertyPlantAndEquipment"
    assert (
        fy23.tag_sources["cash_and_cash_equivalents"]
        == "CashAndCashEquivalentsAtCarryingValue"
    )
    assert fy23.tag_sources["debt"] == "LongTermDebtNoncurrent"


def test_tag_fallbacks_alternative_tags(
    aapl_facts_fixture: dict[str, Any],
) -> None:
    """Verify fallback mechanism extracts revenue when primary tag is absent."""
    parsed = parse_company_facts(aapl_facts_fixture)

    # In 2018, Apple used the tag 'Revenues' before adopting ASC 606 tag
    fy18 = parsed[(2018, "FY")]
    assert fy18.revenues == 265_595_000_000
    assert fy18.tag_sources["revenues"] == "Revenues"

    # Test custom fallback priority: force tag to fallback
    custom_fallbacks = {
        **DEFAULT_TAG_FALLBACKS,
        "revenues": ["NonExistentRevenueTag", "SalesRevenueNet", "Revenues"],
    }
    custom_parser = CompanyFactsParser(tag_fallbacks=custom_fallbacks)
    custom_parsed = custom_parser.parse(aapl_facts_fixture)

    # For 2018, should resolve to 'SalesRevenueNet' or 'Revenues'
    p18 = custom_parsed[(2018, "FY")]
    assert p18.revenues is not None
    assert p18.tag_sources["revenues"] in ("SalesRevenueNet", "Revenues")


def test_as_of_date_point_in_time_filter(
    aapl_facts_fixture: dict[str, Any],
) -> None:
    """Facts filed strictly after as_of_date are excluded to prevent lookahead."""
    # Filter as of 2022-12-31: 2023 10-K (filed 2023-11-03) and 2024 must NOT appear
    parsed_clamped = parse_company_facts(aapl_facts_fixture, as_of_date="2022-12-31")

    all_filed_dates = [p.filed for p in parsed_clamped if p.filed]
    for d in all_filed_dates:
        assert d <= "2022-12-31"

    # 2023 and 2024 FY must not be present
    assert parsed_clamped.get((2023, "FY")) is None
    assert parsed_clamped.get((2024, "FY")) is None

    # 2021 and 2022 FY must be present
    fy21 = parsed_clamped.get((2021, "FY"))
    fy22 = parsed_clamped.get((2022, "FY"))
    assert fy21 is not None and fy21.revenues == 365_817_000_000
    assert fy22 is not None and fy22.revenues == 394_328_000_000


def test_quarterly_period_extraction(
    aapl_facts_fixture: dict[str, Any],
) -> None:
    """Verify quarterly periods (Q1, Q2, Q3) are extracted with 3-month duration."""
    parsed = parse_company_facts(aapl_facts_fixture)
    quarters = parsed.quarterly()
    assert len(quarters) > 0

    q1_23 = parsed.get((2023, "Q1"))
    q2_23 = parsed.get((2023, "Q2"))
    q3_23 = parsed.get((2023, "Q3"))

    assert q1_23 is not None
    assert q1_23.revenues == 117_154_000_000
    assert q1_23.filed == "2023-02-03"

    assert q2_23 is not None
    assert q2_23.revenues == 94_836_000_000
    assert q2_23.filed == "2023-05-05"

    assert q3_23 is not None
    assert q3_23.revenues == 81_797_000_000
    assert q3_23.filed == "2023-08-04"


def test_company_facts_result_container_lookups(
    aapl_facts_fixture: dict[str, Any],
) -> None:
    """Verify CompanyFactsResult container lookup styles."""
    parsed = parse_company_facts(aapl_facts_fixture)

    # 1. Tuple lookup
    p1 = parsed[(2023, "FY")]
    assert p1.revenues == 383_285_000_000

    # 2. String lookup
    p2 = parsed["2023-FY"]
    assert p2.revenues == 383_285_000_000

    # 3. Year int lookup (defaults to FY)
    p3 = parsed[2023]
    assert p3.revenues == 383_285_000_000

    # 4. Standard list indexing
    assert isinstance(parsed[0], PeriodFinancials)

    # 5. Iteration
    periods_list = list(parsed)
    assert len(periods_list) == len(parsed)

    # 6. Safe get
    assert parsed.get("9999-FY") is None
    with pytest.raises(KeyError):
        _ = parsed["9999-FY"]
