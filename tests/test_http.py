"""Tests for shared HTTP client, rate-limiting, and tenacity retries."""

import time

import httpx
import pytest

from lekka_ai.data.http import (
    HostRateLimiter,
    RetryableHTTPStatusError,
    create_http_client,
)


@pytest.mark.asyncio
async def test_http_client_retries_on_500() -> None:
    """HTTP client retries on transient 500 Internal Server Error and succeeds."""
    attempts = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(500, text="Internal Server Error")
        return httpx.Response(200, json={"status": "recovered"})

    transport = httpx.MockTransport(mock_handler)
    async with create_http_client(
        transport=transport,
        max_retries=4,
        backoff_multiplier=0.01,
        min_backoff=0.01,
        max_backoff=0.05,
    ) as client:
        response = await client.get("http://api.example.com/test-endpoint")
        assert response.status_code == 200
        assert response.json() == {"status": "recovered"}
        assert attempts == 3


@pytest.mark.asyncio
async def test_http_client_retries_exhausted_raises_error() -> None:
    """When 500 errors persist past max_retries, RetryableHTTPStatusError is raised."""
    attempts = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, text="Persistent 500 Error")

    transport = httpx.MockTransport(mock_handler)
    async with create_http_client(
        transport=transport,
        max_retries=3,
        backoff_multiplier=0.01,
        min_backoff=0.01,
        max_backoff=0.05,
    ) as client:
        with pytest.raises(RetryableHTTPStatusError) as exc_info:
            await client.get("http://api.example.com/fails-persistently")

        assert exc_info.value.response.status_code == 500
        assert attempts == 3


@pytest.mark.asyncio
async def test_http_client_retries_on_429_too_many_requests() -> None:
    """HTTP client retries on 429 Too Many Requests."""
    attempts = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, text="Rate Limited")
        return httpx.Response(200, json={"data": "success"})

    transport = httpx.MockTransport(mock_handler)
    async with create_http_client(
        transport=transport,
        max_retries=3,
        backoff_multiplier=0.01,
        min_backoff=0.01,
        max_backoff=0.05,
    ) as client:
        response = await client.get("http://api.example.com/data")
        assert response.status_code == 200
        assert response.json() == {"data": "success"}
        assert attempts == 2


@pytest.mark.asyncio
async def test_http_client_raises_on_404_without_retrying() -> None:
    """404 Not Found raises httpx.HTTPStatusError immediately without retrying."""
    attempts = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404, text="Resource Not Found")

    transport = httpx.MockTransport(mock_handler)
    async with create_http_client(
        transport=transport,
        max_retries=4,
    ) as client:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.get("http://api.example.com/missing-resource")

        assert exc_info.value.response.status_code == 404
        # 404 should never be retried
        assert attempts == 1


@pytest.mark.asyncio
async def test_http_client_respects_rate_limit() -> None:
    """HostRateLimiter enforces token-bucket rate limits per configured host."""
    limiter = HostRateLimiter(default_rate=50.0)
    # Configure 5 requests/second (0.2s interval) with capacity 1 for rate.limited.com
    limiter.set_limit("rate.limited.com", rate=5.0, capacity=1.0)

    transport = httpx.MockTransport(lambda req: httpx.Response(200, text="ok"))

    async with create_http_client(rate_limiter=limiter, transport=transport) as client:
        start_time = time.monotonic()
        for _ in range(3):
            res = await client.get("http://rate.limited.com/items")
            assert res.status_code == 200
        elapsed = time.monotonic() - start_time

        # 3 requests with capacity 1 at 5 req/s must take at least ~0.35s
        assert elapsed >= 0.35, f"Elapsed time {elapsed:.3f}s was less than rate limit"

        # Request to a different host with high limit should run without delay
        unlimited_start = time.monotonic()
        for _ in range(3):
            res = await client.get("http://other.unlimited.com/items")
            assert res.status_code == 200
        unlimited_elapsed = time.monotonic() - unlimited_start
        assert unlimited_elapsed < 0.15


@pytest.mark.asyncio
async def test_http_client_custom_timeout_and_raise_for_status_flag() -> None:
    """Client allows disabling raise_for_status to inspect 4xx responses."""
    transport = httpx.MockTransport(lambda req: httpx.Response(404, text="Not Found"))

    async with create_http_client(
        transport=transport,
        raise_for_status=False,
        timeout=25.0,
    ) as client:
        response = await client.get("http://api.example.com/check")
        assert response.status_code == 404
        assert response.text == "Not Found"
        assert client.timeout.read == 25.0
