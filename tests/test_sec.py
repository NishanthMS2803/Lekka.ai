"""Tests for SEC EDGAR client (SecClient) including mocked and live tests."""

import json
import os
from typing import Any

import httpx
import pytest

from lekka_ai.config import get_settings
from lekka_ai.data.http import create_http_client
from lekka_ai.data.sec import (
    CIKNotFoundError,
    SecClient,
)

SAMPLE_SEC_TICKERS_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    "2": {"cik_str": 1018724, "ticker": "AMZN", "title": "AMAZON COM INC"},
}


def create_mock_transport(
    status_code: int = 200, json_data: Any = None
) -> httpx.MockTransport:
    """Helper creating a MockTransport returning standard JSON responses."""

    def handler(request: httpx.Request) -> httpx.Response:
        content = (
            json.dumps(json_data).encode("utf-8") if json_data is not None else b"{}"
        )
        return httpx.Response(
            status_code=status_code,
            headers={"Content-Type": "application/json"},
            content=content,
            request=request,
        )

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_sec_client_user_agent_header_and_lookup() -> None:
    """SecClient sends configured User-Agent and resolves zero-padded 10-digit CIK."""
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=json.dumps(SAMPLE_SEC_TICKERS_PAYLOAD).encode("utf-8"),
            request=request,
        )

    transport = httpx.MockTransport(handler)
    http_client = create_http_client(transport=transport)

    settings = get_settings()
    sec_client = SecClient(
        user_agent=settings.SEC_USER_AGENT,
        http_client=http_client,
    )

    cik_aapl = await sec_client.get_cik("AAPL")
    assert cik_aapl == "0000320193"
    assert len(cik_aapl) == 10

    # Verify request headers
    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req.headers["User-Agent"] == settings.SEC_USER_AGENT
    assert "company_tickers.json" in str(req.url)

    # Verify case-insensitivity
    cik_msft = await sec_client.get_cik("msft")
    assert cik_msft == "0000789019"
    assert len(cik_msft) == 10

    # Second lookup should hit cache and NOT send a second network request
    assert len(captured_requests) == 1


@pytest.mark.asyncio
async def test_sec_client_numeric_cik() -> None:
    """Direct numeric CIK strings are zero-padded without network call."""
    captured: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)
    http_client = create_http_client(transport=transport)

    sec_client = SecClient(http_client=http_client)
    cik = await sec_client.get_cik("320193")
    assert cik == "0000320193"
    assert len(captured) == 0


@pytest.mark.asyncio
async def test_sec_client_not_found() -> None:
    """Unknown ticker symbol raises CIKNotFoundError."""
    transport = create_mock_transport(200, SAMPLE_SEC_TICKERS_PAYLOAD)
    http_client = create_http_client(transport=transport)

    sec_client = SecClient(http_client=http_client)
    with pytest.raises(CIKNotFoundError, match="not found in SEC"):
        await sec_client.get_cik("UNKNOWN_TICKER")


def test_sec_client_rate_limit_validation() -> None:
    """Configuring rate limit >= 10 req/s raises ValueError per SEC policy."""
    with pytest.raises(ValueError, match="strictly under 10 req/s"):
        SecClient(rate_limit=10.0)

    with pytest.raises(ValueError, match="strictly under 10 req/s"):
        SecClient(rate_limit=15.0)


@pytest.mark.asyncio
async def test_sec_client_get_company_facts() -> None:
    """SecClient.get_company_facts requests zero-padded CIK URL and returns JSON."""
    captured: list[httpx.Request] = []
    fake_facts = {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {"us-gaap": {}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=json.dumps(fake_facts).encode("utf-8"),
            request=request,
        )

    transport = httpx.MockTransport(handler)
    http_client = create_http_client(transport=transport)
    sec_client = SecClient(http_client=http_client)

    result = await sec_client.get_company_facts("320193")
    assert result["cik"] == 320193
    assert result["entityName"] == "Apple Inc."

    assert len(captured) == 1
    req = captured[0]
    assert (
        str(req.url) == "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
    )


@pytest.mark.live
@pytest.mark.asyncio
async def test_sec_client_live_aapl_cik() -> None:
    """Live integration test: query live SEC EDGAR API for Apple's CIK."""
    if os.getenv("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE is not set to 1; skipping live SEC EDGAR test")

    async with SecClient() as client:
        cik = await client.get_cik("AAPL")
        assert cik == "0000320193"
        name = await client.get_company_name("AAPL")
        assert name is not None
        assert "Apple" in name

        facts = await client.get_company_facts(cik)
        assert facts.get("entityName") is not None
        assert "Apple" in facts["entityName"]
        assert "us-gaap" in facts.get("facts", {})
