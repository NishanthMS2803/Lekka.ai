"""SEC EDGAR client with rate limiting, User-Agent enforcement, and CIK mapping."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from lekka_ai.config import get_settings
from lekka_ai.data.http import AsyncHttpClient, HostRateLimiter, create_http_client
from lekka_ai.logging import get_logger

logger = get_logger(__name__)

SEC_COMPANY_TICKERS_URL: str = "https://www.sec.gov/files/company_tickers.json"
DEFAULT_SEC_RATE_LIMIT: float = 8.0  # SEC mandates under 10 requests/second


class SecClientError(Exception):
    """Base exception for SEC EDGAR client operations."""


class CIKNotFoundError(SecClientError):
    """Raised when a ticker cannot be resolved to an SEC CIK."""


class SecClient:
    """Async SEC EDGAR client with compliant rate-limiting and CIK lookup.

    Complies with SEC Fair Access policy:
    - User-Agent header in format: Sample Company Name AdminContact@domain.com
    - Strict rate limiting strictly under 10 requests per second.
    """

    def __init__(
        self,
        user_agent: str | None = None,
        http_client: AsyncHttpClient | None = None,
        rate_limit: float = DEFAULT_SEC_RATE_LIMIT,
    ) -> None:
        if rate_limit >= 10.0:
            raise ValueError(
                f"SEC rate limit must be strictly under 10 req/s, got {rate_limit}."
            )

        settings = get_settings()
        self.user_agent = user_agent or settings.SEC_USER_AGENT
        if not self.user_agent:
            raise ValueError("SEC_USER_AGENT must be configured in settings or passed.")

        self.rate_limit = rate_limit
        self._owns_client = False

        if http_client is not None:
            self._http_client = http_client
        else:
            limiter = HostRateLimiter(
                default_rate=self.rate_limit,
                default_capacity=self.rate_limit,
                host_limits={
                    "www.sec.gov": (self.rate_limit, self.rate_limit),
                    "data.sec.gov": (self.rate_limit, self.rate_limit),
                    "sec.gov": (self.rate_limit, self.rate_limit),
                },
            )
            self._http_client = create_http_client(
                rate_limiter=limiter,
                headers={"User-Agent": self.user_agent},
            )
            self._owns_client = True

        # In-memory caches for ticker mappings
        self._ticker_to_cik: dict[str, str] = {}
        self._ticker_to_title: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Execute rate-limited GET request to SEC with compliant headers."""
        headers = kwargs.pop("headers", {})
        merged_headers = {"User-Agent": self.user_agent, **headers}
        return await self._http_client.get(url, headers=merged_headers, **kwargs)

    async def get_company_tickers(self, force_refresh: bool = False) -> dict[str, str]:
        """Fetch and cache ticker-to-CIK mapping from SEC EDGAR.

        Returns:
            Dictionary mapping uppercase ticker symbol to zero-padded 10-digit CIK.
        """
        async with self._lock:
            if self._ticker_to_cik and not force_refresh:
                return dict(self._ticker_to_cik)

            logger.info("Fetching SEC company tickers", url=SEC_COMPANY_TICKERS_URL)
            response = await self.get(SEC_COMPANY_TICKERS_URL)
            response.raise_for_status()
            data: dict[str, dict[str, Any]] = response.json()

            mapping: dict[str, str] = {}
            titles: dict[str, str] = {}
            for item in data.values():
                ticker_raw = item.get("ticker")
                cik_raw = item.get("cik_str")
                title_raw = item.get("title")

                if ticker_raw and cik_raw is not None:
                    ticker = str(ticker_raw).strip().upper()
                    # Zero-pad CIK to 10 digits as required by SEC EDGAR APIs
                    padded_cik = str(cik_raw).zfill(10)
                    mapping[ticker] = padded_cik
                    if title_raw:
                        titles[ticker] = str(title_raw).strip()

            self._ticker_to_cik = mapping
            self._ticker_to_title = titles
            return dict(self._ticker_to_cik)

    async def get_cik(self, ticker: str) -> str:
        """Resolve a ticker symbol to a zero-padded 10-digit CIK.

        Args:
            ticker: Stock ticker symbol (e.g. 'AAPL') or numeric CIK.

        Returns:
            10-digit string CIK (e.g. '0000320193').
        """
        norm = ticker.strip().upper()

        # If already a numeric CIK
        if norm.isdigit():
            return norm.zfill(10)

        # Fast path from cache
        if norm in self._ticker_to_cik:
            return self._ticker_to_cik[norm]

        # Fetch and refresh cache
        tickers = await self.get_company_tickers()
        if norm in tickers:
            return tickers[norm]

        raise CIKNotFoundError(
            f"Ticker '{ticker}' not found in SEC company tickers database."
        )

    async def get_submissions(self, cik: str) -> dict[str, Any]:
        """Fetch company submissions metadata for a given CIK from SEC EDGAR.

        Endpoint: https://data.sec.gov/submissions/CIK##########.json
        """
        padded_cik = str(cik).strip().zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
        response = await self.get(url)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    @staticmethod
    def build_document_url(
        cik: str | int,
        accession_number: str,
        primary_document: str,
    ) -> str:
        """Construct canonical SEC EDGAR URL for a primary filing document."""
        clean_cik = str(int(str(cik).strip()))
        clean_acc = accession_number.strip().replace("-", "")
        clean_doc = primary_document.strip()
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{clean_cik}/{clean_acc}/{clean_doc}"
        )

    async def get_company_name(self, ticker: str) -> str | None:
        """Retrieve company title/name for ticker if available."""
        norm = ticker.strip().upper()
        if not self._ticker_to_title:
            await self.get_company_tickers()
        return self._ticker_to_title.get(norm)

    async def close(self) -> None:
        """Close underlying HTTP client if owned by this instance."""
        if self._owns_client:
            await self._http_client.aclose()

    async def __aenter__(self) -> SecClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
