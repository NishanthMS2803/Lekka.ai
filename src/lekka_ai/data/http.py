"""Shared asynchronous HTTP client with rate-limiting and tenacity retries."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from lekka_ai.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=15.0,
    write=5.0,
    pool=5.0,
)


class RetryableHTTPStatusError(httpx.HTTPStatusError):
    """Raised when an HTTP response status is transient and eligible for retry."""


def is_retryable_exception(exc: BaseException) -> bool:
    """Determine whether an exception or status code should trigger a retry."""
    if isinstance(exc, RetryableHTTPStatusError):
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    return False


class TokenBucket:
    """Thread-safe and asyncio-safe token-bucket rate limiter."""

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        if rate <= 0:
            raise ValueError("Rate must be greater than zero.")
        self.rate = float(rate)
        self.capacity = float(capacity) if capacity is not None else float(rate)
        self.tokens = self.capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, amount: float = 1.0) -> None:
        """Wait until the requested number of tokens is available, then consume."""
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.last_update = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

                if self.tokens >= amount:
                    self.tokens -= amount
                    return

                needed = amount - self.tokens
                wait_seconds = needed / self.rate
                await asyncio.sleep(wait_seconds)


class HostRateLimiter:
    """Per-host token-bucket rate limiter registry."""

    def __init__(
        self,
        default_rate: float = 10.0,
        default_capacity: float | None = None,
        host_limits: dict[str, tuple[float, float] | float] | None = None,
    ) -> None:
        self.default_rate = float(default_rate)
        if default_capacity is not None:
            self.default_capacity = float(default_capacity)
        else:
            self.default_capacity = float(default_rate)
        self.host_limits: dict[str, tuple[float, float] | float] = host_limits or {}
        self._limiters: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    def set_limit(self, host: str, rate: float, capacity: float | None = None) -> None:
        """Configure or update the token bucket limit for a specific host."""
        cap = capacity if capacity is not None else rate
        self.host_limits[host] = (rate, cap)
        self._limiters[host] = TokenBucket(rate=rate, capacity=cap)

    def get_limiter(self, host: str) -> TokenBucket:
        """Retrieve or initialize the TokenBucket for a given host."""
        if host not in self._limiters:
            config = self.host_limits.get(host)
            if config is None:
                rate = self.default_rate
                capacity = self.default_capacity
            elif isinstance(config, (int, float)):
                rate = float(config)
                capacity = float(config)
            else:
                rate, capacity = config
            self._limiters[host] = TokenBucket(rate=rate, capacity=capacity)
        return self._limiters[host]

    async def acquire(self, host: str, amount: float = 1.0) -> None:
        """Acquire a token for the specified host."""
        limiter = self.get_limiter(host)
        await limiter.acquire(amount)


class AsyncHttpClient(httpx.AsyncClient):
    """Async HTTP client with automatic rate limiting and exponential retries."""

    def __init__(
        self,
        rate_limiter: HostRateLimiter | None = None,
        max_retries: int = 3,
        backoff_multiplier: float = 0.5,
        min_backoff: float = 0.1,
        max_backoff: float = 2.0,
        raise_for_status: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.rate_limiter = rate_limiter or HostRateLimiter()
        self.max_retries = max_retries
        self.backoff_multiplier = backoff_multiplier
        self.min_backoff = min_backoff
        self.max_backoff = max_backoff
        self.raise_for_status_default = raise_for_status

    async def send(
        self,
        request: httpx.Request,
        *,
        stream: bool = False,
        auth: Any = None,
        follow_redirects: Any = None,
    ) -> httpx.Response:
        """Send an HTTP request with per-host rate limiting and retries on 429/5xx."""
        host = request.url.host
        if self.rate_limiter and host:
            await self.rate_limiter.acquire(host)

        retryer = AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=self.backoff_multiplier,
                min=self.min_backoff,
                max=self.max_backoff,
            ),
            retry=retry_if_exception(is_retryable_exception),
            reraise=True,
        )

        async for attempt in retryer:
            with attempt:
                if attempt.retry_state.attempt_number > 1:
                    logger.warning(
                        "Retrying HTTP request",
                        url=str(request.url),
                        attempt=attempt.retry_state.attempt_number,
                    )
                    if self.rate_limiter and host:
                        await self.rate_limiter.acquire(host)

                response = await super().send(
                    request,
                    stream=stream,
                    auth=auth,
                    follow_redirects=follow_redirects,
                )

                if response.status_code in (429, 500, 502, 503, 504):
                    raise RetryableHTTPStatusError(
                        f"Retryable status code {response.status_code}",
                        request=request,
                        response=response,
                    )

                if self.raise_for_status_default:
                    response.raise_for_status()

                return response

        raise RuntimeError("Request ended unexpectedly without response.")


def create_http_client(
    rate_limiter: HostRateLimiter | None = None,
    timeout: float | httpx.Timeout | None = None,
    max_retries: int = 3,
    backoff_multiplier: float = 0.5,
    min_backoff: float = 0.1,
    max_backoff: float = 2.0,
    raise_for_status: bool = True,
    transport: httpx.AsyncBaseTransport | None = None,
    headers: dict[str, str] | None = None,
    **kwargs: Any,
) -> AsyncHttpClient:
    """Factory creating an AsyncHttpClient with timeouts, retries, and rate limiting."""
    target_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
    return AsyncHttpClient(
        rate_limiter=rate_limiter,
        max_retries=max_retries,
        backoff_multiplier=backoff_multiplier,
        min_backoff=min_backoff,
        max_backoff=max_backoff,
        raise_for_status=raise_for_status,
        timeout=target_timeout,
        transport=transport,
        headers=headers,
        **kwargs,
    )
