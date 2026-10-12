"""Rate limiting utilities for API endpoints."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

from fastapi import HTTPException, Request, status

from lekka_ai.config import get_settings


@dataclass
class RateLimiter:
    """Sliding window request rate limiter per client IP address."""

    requests: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def check(
        self,
        key: str,
        max_requests: int | None = None,
        window_seconds: float = 60.0,
    ) -> None:
        """Check whether request is allowed under current rate limit.

        Raises:
            HTTPException: 429 Too Many Requests if rate limit is exceeded.
        """
        settings = get_settings()
        if not settings.RATE_LIMIT_ENABLED:
            return

        limit = (
            max_requests
            if max_requests is not None
            else settings.RATE_LIMIT_RUNS_PER_MINUTE
        )
        now = time.monotonic()
        cutoff = now - window_seconds

        # Prune old request timestamps
        timestamps = [ts for ts in self.requests[key] if ts > cutoff]
        if len(timestamps) >= limit:
            retry_after = int(max(1.0, window_seconds - (now - timestamps[0])))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded: maximum {limit} requests allowed per "
                    f"{int(window_seconds)}s. Please try again in {retry_after}s."
                ),
                headers={"Retry-After": str(retry_after)},
            )

        timestamps.append(now)
        self.requests[key] = timestamps

    def reset(self) -> None:
        """Reset internal rate limiting counters (for testing)."""
        self.requests.clear()


default_rate_limiter = RateLimiter()


def get_client_ip(request: Request) -> str:
    """Extract client IP address from proxy headers or connection host."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"
