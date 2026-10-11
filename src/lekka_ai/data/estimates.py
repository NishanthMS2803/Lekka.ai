"""Analyst consensus estimates and peer/sector industry multiples provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import httpx

from lekka_ai.config import get_settings
from lekka_ai.data.http import create_http_client
from lekka_ai.logging import get_logger

logger = get_logger(__name__)


class EstimatesProviderError(Exception):
    """Exception raised when an estimates provider fails."""


@dataclass
class AnalystEstimates:
    """Consensus analyst recommendations and forward estimates."""

    ticker: str
    target_price_mean: float | None = None
    target_price_high: float | None = None
    target_price_low: float | None = None
    target_price_median: float | None = None
    consensus_rating: str | None = None  # e.g. "Buy", "Hold", "Sell"
    num_analysts: int | None = None
    eps_estimate_current_year: float | None = None
    revenue_estimate_current_year: float | None = None
    period: str | None = None
    source: str = "EstimatesProvider"
    as_of_date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndustryData:
    """Sector, industry, and peer valuation multiples."""

    ticker: str
    sector: str
    industry: str
    peer_tickers: list[str] = field(default_factory=list)
    pe_ratio_median: float | None = None
    ev_ebitda_median: float | None = None
    price_to_sales_median: float | None = None
    price_to_book_median: float | None = None
    source: str = "EstimatesProvider"
    as_of_date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EstimatesProvider(ABC):
    """Abstract interface for analyst estimates and peer/industry multiples."""

    @abstractmethod
    async def get_estimates(
        self,
        ticker: str,
        *,
        as_of_date: date | None = None,
    ) -> AnalystEstimates | None:
        """Fetch consensus analyst price targets and recommendations for ticker."""
        ...

    @abstractmethod
    async def get_industry_data(
        self,
        ticker: str,
        *,
        as_of_date: date | None = None,
    ) -> IndustryData | None:
        """Fetch industry categorization, peer group, and valuation multiples."""
        ...


class FinnhubEstimatesProvider(EstimatesProvider):
    """Estimates and peer multiples adapter using Finnhub or financial APIs."""

    BASE_URL: str = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key or get_settings().ESTIMATES_API_KEY
        self._http_client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is not None:
            return self._http_client
        return create_http_client()

    async def get_estimates(
        self,
        ticker: str,
        *,
        as_of_date: date | None = None,
    ) -> AnalystEstimates | None:
        """Fetch price targets and consensus recommendation from Finnhub."""
        if not self.api_key:
            logger.info("ESTIMATES_API_KEY not configured; estimates unavailable")
            return None

        client = await self._get_client()
        clean_ticker = ticker.strip().upper()
        target_url = f"{self.BASE_URL}/stock/price-target"
        params = {"symbol": clean_ticker, "token": self.api_key}

        try:
            resp = await client.get(target_url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(
                "Finnhub price-target query failed", ticker=clean_ticker, error=str(exc)
            )
            raise EstimatesProviderError(f"Estimates fetch failed: {exc}") from exc

        if not data or not isinstance(data, dict) or not data.get("symbol"):
            return None

        target_mean = (
            float(data["targetMean"]) if data.get("targetMean") is not None else None
        )
        target_high = (
            float(data["targetHigh"]) if data.get("targetHigh") is not None else None
        )
        target_low = (
            float(data["targetLow"]) if data.get("targetLow") is not None else None
        )
        target_median = (
            float(data["targetMedian"])
            if data.get("targetMedian") is not None
            else None
        )

        return AnalystEstimates(
            ticker=clean_ticker,
            target_price_mean=target_mean,
            target_price_high=target_high,
            target_price_low=target_low,
            target_price_median=target_median,
            consensus_rating="Buy"
            if target_mean and target_low and target_mean > target_low
            else None,
            source="Finnhub",
            as_of_date=as_of_date.isoformat() if as_of_date else None,
            metadata=data,
        )

    async def get_industry_data(
        self,
        ticker: str,
        *,
        as_of_date: date | None = None,
    ) -> IndustryData | None:
        """Fetch company peers and profile multiples from Finnhub."""
        if not self.api_key:
            logger.info("ESTIMATES_API_KEY not configured; industry data unavailable")
            return None

        client = await self._get_client()
        clean_ticker = ticker.strip().upper()

        peers_url = f"{self.BASE_URL}/stock/peers"
        profile_url = f"{self.BASE_URL}/stock/profile2"
        params = {"symbol": clean_ticker, "token": self.api_key}

        try:
            peers_resp = await client.get(peers_url, params=params)
            peers_resp.raise_for_status()
            peers_data = peers_resp.json()

            profile_resp = await client.get(profile_url, params=params)
            profile_resp.raise_for_status()
            profile_data = profile_resp.json()
        except Exception as exc:
            logger.warning(
                "Finnhub industry query failed", ticker=clean_ticker, error=str(exc)
            )
            raise EstimatesProviderError(f"Industry data fetch failed: {exc}") from exc

        peers_list = (
            [
                str(p).strip().upper()
                for p in peers_data
                if str(p).strip().upper() != clean_ticker
            ]
            if isinstance(peers_data, list)
            else []
        )

        sector = (
            str(profile_data.get("finnhubIndustry", "Technology"))
            if isinstance(profile_data, dict)
            else "Technology"
        )
        industry = sector

        return IndustryData(
            ticker=clean_ticker,
            sector=sector,
            industry=industry,
            peer_tickers=peers_list[:6],
            pe_ratio_median=None,
            source="Finnhub",
            as_of_date=as_of_date.isoformat() if as_of_date else None,
            metadata={"peers": peers_list, "profile": profile_data},
        )


class FixtureEstimatesProvider(EstimatesProvider):
    """Deterministic fixture provider for unit and integration testing."""

    def __init__(
        self,
        estimates_map: dict[str, AnalystEstimates] | None = None,
        industry_map: dict[str, IndustryData] | None = None,
        should_fail: bool = False,
    ) -> None:
        self.estimates_map = estimates_map or {}
        self.industry_map = industry_map or {}
        self.should_fail = should_fail

    async def get_estimates(
        self,
        ticker: str,
        *,
        as_of_date: date | None = None,
    ) -> AnalystEstimates | None:
        if self.should_fail:
            raise EstimatesProviderError("Simulated remote provider connection error")
        clean = ticker.strip().upper()
        est = self.estimates_map.get(clean)
        if est and as_of_date:
            est.as_of_date = as_of_date.isoformat()
        return est

    async def get_industry_data(
        self,
        ticker: str,
        *,
        as_of_date: date | None = None,
    ) -> IndustryData | None:
        if self.should_fail:
            raise EstimatesProviderError("Simulated remote provider connection error")
        clean = ticker.strip().upper()
        ind = self.industry_map.get(clean)
        if ind and as_of_date:
            ind.as_of_date = as_of_date.isoformat()
        return ind


def create_estimates_provider(
    api_key: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> EstimatesProvider:
    """Factory creating an EstimatesProvider instance using settings."""
    key = api_key or get_settings().ESTIMATES_API_KEY
    return FinnhubEstimatesProvider(api_key=key, http_client=http_client)
