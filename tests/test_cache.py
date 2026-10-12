"""Tests for Tool cache with fact-based TTL policies and tool decorator."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import BaseModel, Field
from sqlalchemy import text

from lekka_ai.db import get_engine, get_session
from lekka_ai.tools.base import Tool
from lekka_ai.tools.cache import (
    TTL_POLICIES,
    Cache,
    build_cache_key,
    cached_tool,
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


class PriceInput(BaseModel):
    """Input payload for PriceTool."""

    ticker: str = Field(..., description="Stock ticker symbol")


class PriceOutput(BaseModel):
    """Output payload for PriceTool."""

    ticker: str
    price: float


class CountingPriceTool(Tool[PriceInput, PriceOutput]):
    """Tool that tracks execution count to verify caching behavior."""

    name: str = "get_stock_price"
    description: str = "Returns stock price with execution counting."
    input_model: type[PriceInput] = PriceInput
    output_model: type[PriceOutput] = PriceOutput

    def __init__(self, price: float = 150.0) -> None:
        self.call_count = 0
        self.price = price

    async def run(self, input_data: PriceInput) -> PriceOutput:
        """Increment count and return price."""
        self.call_count += 1
        return PriceOutput(ticker=input_data.ticker.upper(), price=self.price)


@pytest.mark.db
def test_cache_miss_and_hit() -> None:
    """Cache returns None on miss, and returns stored value on hit."""
    if not is_database_reachable():
        pytest.skip("Database is unavailable")

    session_generator = get_session()
    session = next(session_generator)
    try:
        cache = Cache(session=session)
        test_key = "test_tool:ticker=AAPL"
        cache.delete(test_key)

        # 1. Cache Miss
        assert cache.get(test_key) is None

        # 2. Cache Set
        record = cache.set(
            key=test_key,
            value={"price": 185.50, "ticker": "AAPL"},
            fact_type="price",
        )
        assert record.key == test_key
        assert record.fact_type == "price"
        assert record.expires_at is not None

        # 3. Cache Hit
        cached_value = cache.get(test_key)
        assert cached_value == {"price": 185.50, "ticker": "AAPL"}

        # Cleanup
        cache.delete(test_key)
    finally:
        try:
            next(session_generator)
        except StopIteration:
            pass


@pytest.mark.db
def test_cache_ttl_policies_and_expiration() -> None:
    """Cache policies enforce specific TTL durations, and expired items return None."""
    if not is_database_reachable():
        pytest.skip("Database is unavailable")

    session_generator = get_session()
    session = next(session_generator)
    try:
        cache = Cache(session=session)
        now = datetime.now(UTC)

        # Verify TTL policies mapping
        assert TTL_POLICIES["price"] == timedelta(minutes=5)
        assert TTL_POLICIES["news"] == timedelta(hours=6)
        assert TTL_POLICIES["estimates"] == timedelta(hours=24)
        assert TTL_POLICIES["financials"] == timedelta(days=7)
        assert TTL_POLICIES["filing_doc"] is None

        # 1. filing_doc never expires
        filing_rec = cache.set(
            "doc:10k", {"content": "annual report"}, fact_type="filing_doc"
        )
        assert filing_rec.expires_at is None
        assert cache.get("doc:10k") == {"content": "annual report"}
        cache.delete("doc:10k")

        # 2. price expires in 5 minutes
        price_rec = cache.set(
            "price:msft", {"price": 400.0}, fact_type="price", fetched_at=now
        )
        assert price_rec.expires_at is not None
        diff = price_rec.expires_at - now
        assert abs(diff.total_seconds() - 300) < 2
        cache.delete("price:msft")

        # 3. Expired record returns None (Cache Miss on Expiry)
        expired_key = "expired:quote"
        past_time = now - timedelta(minutes=10)
        cache.set(
            key=expired_key,
            value={"price": 99.0},
            fact_type="price",
            fetched_at=past_time - timedelta(minutes=5),
            expires_at=past_time,
        )
        assert cache.get(expired_key) is None

        # 4. Unknown fact_type raises ValueError
        with pytest.raises(ValueError, match="Unknown fact_type"):
            cache.set("bad:type", {"x": 1}, fact_type="unknown_type")

        cache.delete(expired_key)
    finally:
        try:
            next(session_generator)
        except StopIteration:
            pass


@pytest.mark.db
@pytest.mark.asyncio
async def test_cached_tool_decorator_hit_miss_and_expiry() -> None:
    """cached_tool decorator serves hits from cache and respects expiry."""
    if not is_database_reachable():
        pytest.skip("Database is unavailable")

    session_generator = get_session()
    session = next(session_generator)
    try:
        cache = Cache(session=session)
        raw_tool = CountingPriceTool(price=220.0)

        # Apply decorator
        decorated_tool = cached_tool(cache, fact_type="price")(raw_tool)

        key_aapl = build_cache_key("get_stock_price", {"ticker": "AAPL"})
        key_goog = build_cache_key("get_stock_price", {"ticker": "GOOG"})
        cache.delete(key_aapl)
        cache.delete(key_goog)

        # Initial call: Cache MISS (executes run, increments counter to 1)
        res1 = await decorated_tool.execute({"ticker": "AAPL"})
        assert isinstance(res1, PriceOutput)
        assert res1.ticker == "AAPL"
        assert res1.price == 220.0
        assert raw_tool.call_count == 1

        # Second call with identical input: Cache HIT (counter stays 1)
        res2 = await decorated_tool.execute({"ticker": "AAPL"})
        assert isinstance(res2, PriceOutput)
        assert res2.price == 220.0
        assert raw_tool.call_count == 1

        # Third call with different input: Cache MISS (increments counter to 2)
        res3 = await decorated_tool.execute({"ticker": "GOOG"})
        assert res3.ticker == "GOOG"
        assert raw_tool.call_count == 2

        # Expire AAPL cache entry manually
        past_expiration = datetime.now(UTC) - timedelta(minutes=1)
        cache.set(
            key=key_aapl,
            value={"ticker": "AAPL", "price": 220.0},
            fact_type="price",
            expires_at=past_expiration,
        )

        # Call again after expiration: Cache MISS on expiry (increments counter to 3)
        res4 = await decorated_tool.execute({"ticker": "AAPL"})
        assert res4.ticker == "AAPL"
        assert raw_tool.call_count == 3

        # Cleanup
        cache.delete(key_aapl)
        cache.delete(key_goog)
    finally:
        try:
            next(session_generator)
        except StopIteration:
            pass
