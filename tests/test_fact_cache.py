"""Tests for FactCache (lite memory), TTL policies, revalidation, and profiles."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import text

from lekka_ai.agents import load_agent_profile
from lekka_ai.db import get_engine, get_session
from lekka_ai.schemas.claims import Claim, ClaimType, Evidence
from lekka_ai.tools.base import default_registry
from lekka_ai.tools.recall_facts import (
    RecallFactsInput,
    RecallFactsOutput,
    RecallFactsTool,
)
from lekka_ai.verify.facts_cache import (
    FactCacheRepository,
    compute_expires_at,
    get_ttl_for_fact_type,
)


def is_database_reachable() -> bool:
    """Check if the database is reachable."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def test_ttl_policies_durations() -> None:
    """Verify TTL policies match the specification for all 5 fact types."""
    # price: 5 min
    assert get_ttl_for_fact_type("price") == timedelta(minutes=5)
    assert get_ttl_for_fact_type("market_data") == timedelta(minutes=5)

    # news: 6h
    assert get_ttl_for_fact_type("news") == timedelta(hours=6)

    # estimates: 24h
    assert get_ttl_for_fact_type("estimates") == timedelta(hours=24)
    assert get_ttl_for_fact_type("analyst_estimates") == timedelta(hours=24)

    # quarterly financials: until next filing (~90 days)
    assert get_ttl_for_fact_type("quarterly financials") == timedelta(days=90)
    assert get_ttl_for_fact_type("quarterly_financials") == timedelta(days=90)
    assert get_ttl_for_fact_type("financials") == timedelta(days=90)

    # filings: never (None)
    assert get_ttl_for_fact_type("filings") is None
    assert get_ttl_for_fact_type("sec_filing") is None
    assert get_ttl_for_fact_type("filing") is None


def test_compute_expires_at() -> None:
    """Verify expiration timestamp computation based on base timestamp and TTL."""
    base_time = datetime(2026, 10, 12, 12, 0, 0, tzinfo=UTC)

    # price: +5 min
    exp_price = compute_expires_at("price", fetched_at=base_time)
    assert exp_price == datetime(2026, 10, 12, 12, 5, 0, tzinfo=UTC)

    # news: +6h
    exp_news = compute_expires_at("news", fetched_at=base_time)
    assert exp_news == datetime(2026, 10, 12, 18, 0, 0, tzinfo=UTC)

    # estimates: +24h
    exp_est = compute_expires_at("estimates", fetched_at=base_time)
    assert exp_est == datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)

    # quarterly financials: +90d
    exp_fin = compute_expires_at("quarterly financials", fetched_at=base_time)
    assert exp_fin == datetime(2027, 1, 10, 12, 0, 0, tzinfo=UTC)

    # filings: never -> None
    exp_filing = compute_expires_at("filings", fetched_at=base_time)
    assert exp_filing is None


def test_expired_facts_are_not_returned_as_valid() -> None:
    """Core verification invariant: expired facts must NOT be in unexpired_facts.

    They must be flagged in needs_revalidation instead.
    """
    repo = FactCacheRepository()
    repo.clear()

    # Time anchor: 12:00:00 UTC
    t0 = datetime(2026, 10, 12, 12, 0, 0, tzinfo=UTC)

    # Store price fact at 12:00:00 (TTL 5 min -> expires at 12:05:00)
    repo.store_fact(
        ticker="AAPL",
        metric_key="price",
        value=185.50,
        fact_type="price",
        fetched_at=t0,
    )

    # Store news fact at 12:00:00 (TTL 6h -> expires at 18:00:00)
    repo.store_fact(
        ticker="AAPL",
        metric_key="news_sentiment",
        value="positive",
        fact_type="news",
        fetched_at=t0,
    )

    # Store SEC filing fact at 12:00:00 (TTL None -> never expires)
    repo.store_fact(
        ticker="AAPL",
        metric_key="cik",
        value="0000320193",
        fact_type="filings",
        fetched_at=t0,
    )

    # Query 1: At 12:02:00 (before price expiry)
    t_early = datetime(2026, 10, 12, 12, 2, 0, tzinfo=UTC)
    result_early = repo.recall_facts(ticker="AAPL", as_of_time=t_early)
    unexpired_keys_early = {f.metric_key for f in result_early.unexpired_facts}
    assert "price" in unexpired_keys_early
    assert "news_sentiment" in unexpired_keys_early
    assert "cik" in unexpired_keys_early
    assert "price" not in result_early.needs_revalidation

    # Query 2: At 12:10:00 (price expired, news unexpired, filing unexpired)
    t_after_price_exp = datetime(2026, 10, 12, 12, 10, 0, tzinfo=UTC)
    result_mid = repo.recall_facts(ticker="AAPL", as_of_time=t_after_price_exp)
    unexpired_keys_mid = {f.metric_key for f in result_mid.unexpired_facts}

    # CRITICAL: Expired price is NOT in unexpired_facts!
    assert "price" not in unexpired_keys_mid
    assert "news_sentiment" in unexpired_keys_mid
    assert "cik" in unexpired_keys_mid

    # CRITICAL: Expired price IS in needs_revalidation!
    assert "price" in result_mid.needs_revalidation

    # Query 3: At 19:00:00 (price AND news expired, filing still unexpired)
    t_after_news_exp = datetime(2026, 10, 12, 19, 0, 0, tzinfo=UTC)
    result_late = repo.recall_facts(ticker="AAPL", as_of_time=t_after_news_exp)
    unexpired_keys_late = {f.metric_key for f in result_late.unexpired_facts}

    assert "price" not in unexpired_keys_late
    assert "news_sentiment" not in unexpired_keys_late
    assert "cik" in unexpired_keys_late
    assert "price" in result_late.needs_revalidation
    assert "news_sentiment" in result_late.needs_revalidation


def test_metric_filtering_and_missing_revalidation() -> None:
    """Missing or expired metrics appear in needs_revalidation."""
    repo = FactCacheRepository()
    repo.clear()

    t0 = datetime(2026, 10, 12, 12, 0, 0, tzinfo=UTC)

    # Store unexpired revenue
    repo.store_fact(
        ticker="MSFT",
        metric_key="revenue",
        value=245120000000.0,
        fact_type="quarterly financials",
        fetched_at=t0,
    )

    # Store expired price
    repo.store_fact(
        ticker="MSFT",
        metric_key="price",
        value=420.0,
        fact_type="price",
        fetched_at=t0 - timedelta(minutes=10),  # expired
    )

    # Query for revenue, price, and non-cached pe_ratio
    res = repo.recall_facts(
        ticker="MSFT",
        metrics=["revenue", "price", "pe_ratio"],
        as_of_time=t0,
    )

    unexpired_metrics = {f.metric_key for f in res.unexpired_facts}
    assert "revenue" in unexpired_metrics
    assert "price" not in unexpired_metrics
    assert "pe_ratio" not in unexpired_metrics

    # Both expired price and missing pe_ratio must be in needs_revalidation
    assert "price" in res.needs_revalidation
    assert "pe_ratio" in res.needs_revalidation
    assert "revenue" not in res.needs_revalidation


@pytest.mark.asyncio
async def test_recall_facts_tool_invocation() -> None:
    """Verify RecallFactsTool executes properly and matches tool registry contract."""
    repo = FactCacheRepository()
    repo.clear()

    now = datetime.now(UTC)
    repo.store_fact(
        ticker="NVDA",
        metric_key="revenue",
        value=30040000000.0,
        fact_type="quarterly financials",
        source="10-Q",
        as_of_date="2024-07-28",
        fetched_at=now,
    )

    tool = RecallFactsTool(repository=repo)
    assert tool.name == "recall_facts"

    # Invoke tool
    output = await tool.run(RecallFactsInput(ticker="NVDA", metrics=["revenue"]))
    assert isinstance(output, RecallFactsOutput)
    assert output.ticker == "NVDA"
    assert len(output.unexpired_facts) == 1
    assert output.unexpired_facts[0].metric_key == "revenue"
    assert output.unexpired_facts[0].value == 30040000000.0
    assert output.unexpired_facts[0].source == "10-Q"
    assert output.needs_revalidation == []


def test_store_verified_claims_in_cache_ledger_integration() -> None:
    """Verify storing verified claims applies fact_type TTLs."""
    repo = FactCacheRepository()
    repo.clear()

    now = datetime.now(UTC)
    claim_verified = Claim(
        claim_id="claim_rev_001",
        text="Apple Inc. reported quarterly revenue of $85.8B for Q3 2024.",
        type=ClaimType.FACT,
        confidence=1.0,
        produced_by="fundamental",
        evidence=[
            Evidence(
                source_type="quarterly_financials",
                document_id="AAPL",
                tool_call_id="tc_rev_001",
                retrieved_at=now.isoformat(),
                as_of_date="2024-06-29",
                raw_value={"revenue": 85780000000.0},
            )
        ],
    )

    claim_contradicted = Claim(
        claim_id="claim_rev_002",
        text="Apple reported revenue of $95B for Q3 2024.",
        type=ClaimType.FACT,
        confidence=0.5,
        contradicted_by=["claim_rev_001"],
        evidence=[
            Evidence(
                source_type="quarterly_financials",
                document_id="AAPL",
                tool_call_id="tc_rev_002",
                retrieved_at=now.isoformat(),
                as_of_date="2024-06-29",
            )
        ],
    )

    stored = repo.store_verified_claims(
        [claim_verified, claim_contradicted],
        ticker="AAPL",
    )

    # Only verified (uncontradicted) claim stored
    assert len(stored) == 1
    rec = stored[0]
    assert rec["ticker"] == "AAPL"
    assert rec["metric_key"] == "revenue"
    assert rec["fact_type"] == "quarterly financials"
    assert rec["expires_at"] is not None
    assert rec["claim_id"] == "claim_rev_001"

    # Recall
    res = repo.recall_facts("AAPL", as_of_time=now)
    assert len(res.unexpired_facts) == 1
    assert res.unexpired_facts[0].metric_key == "revenue"
    assert res.unexpired_facts[0].claim_id == "claim_rev_001"


def test_profile_tool_permissions() -> None:
    """Verify recall_facts is allowed in 4 profiles and NEVER risk."""
    # 1. Market Profile
    market = load_agent_profile("market")
    assert market.is_tool_allowed("recall_facts")

    # 2. Fundamental Profile
    fundamental = load_agent_profile("fundamental")
    assert fundamental.is_tool_allowed("recall_facts")

    # 3. Valuation Profile
    valuation = load_agent_profile("valuation")
    assert valuation.is_tool_allowed("recall_facts")

    # 4. News Profile
    news = load_agent_profile("news")
    assert news.is_tool_allowed("recall_facts")

    # 5. Risk Profile: MUST NEVER BE ALLOWED
    risk = load_agent_profile("risk")
    assert not risk.is_tool_allowed("recall_facts")

    # Verify tool is in default_registry
    assert "recall_facts" in default_registry


@pytest.mark.db
def test_fact_cache_db_persistence() -> None:
    """Verify database persistence in facts_cache table when PostgreSQL is available."""
    if not is_database_reachable():
        pytest.skip("Database is unreachable")

    session_generator = get_session()
    session = next(session_generator)
    try:
        repo = FactCacheRepository(session=session)
        repo.clear(ticker="TESTDB", session=session)

        now = datetime.now(UTC)
        fact_dict = repo.store_fact(
            ticker="TESTDB",
            metric_key="operating_income",
            value=35000000000.0,
            fact_type="quarterly financials",
            source="Test SEC 10-Q",
            as_of_date=date(2024, 6, 30),
            fetched_at=now,
            session=session,
        )
        assert fact_dict["ticker"] == "TESTDB"

        # Verify querying directly from DB
        res = repo.recall_facts("TESTDB", session=session, as_of_time=now)
        assert len(res.unexpired_facts) == 1
        assert res.unexpired_facts[0].metric_key == "operating_income"
        assert res.unexpired_facts[0].value == 35000000000.0

        # Cleanup
        repo.clear(ticker="TESTDB", session=session)
    finally:
        try:
            next(session_generator)
        except StopIteration:
            pass
