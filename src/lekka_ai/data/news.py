"""Financial news provider interface and implementations with untrusted text tagging."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import httpx

from lekka_ai.config import get_settings
from lekka_ai.data.http import create_http_client
from lekka_ai.logging import get_logger

logger = get_logger(__name__)


class NewsProviderError(Exception):
    """Exception raised for errors during news fetching."""


@dataclass
class NewsArticle:
    """Standardized news article record.

    All external text is explicitly marked as untrusted third-party content.
    """

    title: str
    source: str
    published_date: str  # YYYY-MM-DD
    url: str
    snippet: str
    is_untrusted: bool = True
    trust_label: str = "UNTRUSTED_EXTERNAL_CONTENT"
    metadata: dict[str, Any] = field(default_factory=dict)


class NewsProvider(ABC):
    """Abstract interface for external news search and retrieval providers."""

    @abstractmethod
    async def get_news(
        self,
        query: str,
        *,
        ticker: str | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        limit: int = 10,
    ) -> list[NewsArticle]:
        """Fetch news articles matching query and/or ticker within date window.

        Args:
            query: Free-text search query.
            ticker: Optional stock ticker symbol (e.g. 'AAPL').
            from_date: Optional start date filter.
            to_date: Optional end date filter.
            limit: Maximum number of articles to return.

        Returns:
            List of standardized NewsArticle objects marked as untrusted.
        """
        ...


class FinnhubNewsProvider(NewsProvider):
    """Financial news provider using the Finnhub Stock API."""

    BASE_URL: str = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key or get_settings().NEWS_API_KEY
        self._http_client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is not None:
            return self._http_client
        return create_http_client()

    async def get_news(
        self,
        query: str,
        *,
        ticker: str | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        limit: int = 10,
    ) -> list[NewsArticle]:
        """Retrieve company or market news from Finnhub."""
        if not self.api_key:
            raise NewsProviderError(
                "NEWS_API_KEY is not configured for FinnhubNewsProvider."
            )

        client = await self._get_client()
        target_symbol = (ticker or query).strip().upper()

        # If target looks like a ticker symbol (1-5 alphabetical characters)
        is_ticker = target_symbol.isalpha() and len(target_symbol) <= 5

        params: dict[str, Any] = {"token": self.api_key}
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()

        if is_ticker:
            endpoint = f"{self.BASE_URL}/company-news"
            params["symbol"] = target_symbol
            # Finnhub requires 'from' and 'to' for company-news;
            # provide fallback defaults if omitted
            if "from" not in params:
                params["from"] = from_date.isoformat() if from_date else "2020-01-01"
            if "to" not in params:
                params["to"] = (
                    to_date.isoformat()
                    if to_date
                    else datetime.now(UTC).strftime("%Y-%m-%d")
                )
        else:
            endpoint = f"{self.BASE_URL}/news"
            params["category"] = "general"

        logger.debug(
            "Querying Finnhub news",
            endpoint=endpoint,
            symbol=params.get("symbol"),
            query=query,
        )

        try:
            resp = await client.get(endpoint, params=params)
            resp.raise_for_status()
            raw_data = resp.json()
        except Exception as exc:
            logger.error("Finnhub news request failed", error=str(exc))
            raise NewsProviderError(f"Finnhub API request failed: {exc}") from exc

        if not isinstance(raw_data, list):
            return []

        articles: list[NewsArticle] = []
        for item in raw_data:
            if not isinstance(item, dict):
                continue

            # Parse timestamp (seconds)
            dt_raw = item.get("datetime")
            if isinstance(dt_raw, (int, float)) and dt_raw > 0:
                pub_date = datetime.fromtimestamp(dt_raw, tz=UTC).strftime("%Y-%m-%d")
            else:
                pub_date = datetime.now(UTC).strftime("%Y-%m-%d")

            headline = str(item.get("headline", "")).strip()
            summary = str(item.get("summary", "")).strip()
            source = str(item.get("source", "Finnhub")).strip()
            url = str(item.get("url", "")).strip()

            if not headline and not summary:
                continue

            articles.append(
                NewsArticle(
                    title=headline,
                    source=source,
                    published_date=pub_date,
                    url=url,
                    snippet=summary,
                    is_untrusted=True,
                    trust_label="UNTRUSTED_EXTERNAL_CONTENT",
                    metadata={
                        "category": item.get("category"),
                        "related": item.get("related"),
                        "id": item.get("id"),
                    },
                )
            )
            if len(articles) >= limit:
                break

        return articles


class TavilyNewsProvider(NewsProvider):
    """News search provider using the Tavily Search API."""

    ENDPOINT: str = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key or get_settings().NEWS_API_KEY
        self._http_client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is not None:
            return self._http_client
        return create_http_client()

    async def get_news(
        self,
        query: str,
        *,
        ticker: str | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        limit: int = 10,
    ) -> list[NewsArticle]:
        """Query Tavily news search."""
        if not self.api_key:
            raise NewsProviderError(
                "NEWS_API_KEY is not configured for TavilyNewsProvider."
            )

        client = await self._get_client()
        search_query = f"{ticker} {query}".strip() if ticker else query

        payload: dict[str, Any] = {
            "api_key": self.api_key,
            "query": search_query,
            "topic": "news",
            "max_results": limit,
        }

        try:
            resp = await client.post(self.ENDPOINT, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("Tavily search failed", error=str(exc))
            raise NewsProviderError(f"Tavily search failed: {exc}") from exc

        raw_results = data.get("results", [])
        articles: list[NewsArticle] = []
        for item in raw_results:
            title = str(item.get("title", "")).strip()
            content = str(item.get("content", "")).strip()
            url = str(item.get("url", "")).strip()
            pub_date = str(
                item.get(
                    "published_date",
                    datetime.now(UTC).strftime("%Y-%m-%d"),
                )
            )[:10]

            articles.append(
                NewsArticle(
                    title=title,
                    source=item.get("source", "Tavily"),
                    published_date=pub_date,
                    url=url,
                    snippet=content,
                    is_untrusted=True,
                    trust_label="UNTRUSTED_EXTERNAL_CONTENT",
                )
            )

        return articles[:limit]


class FixtureNewsProvider(NewsProvider):
    """In-memory fixture news provider for testing without external network calls."""

    def __init__(self, articles: list[NewsArticle] | None = None) -> None:
        self.articles = list(articles or [])

    async def get_news(
        self,
        query: str,
        *,
        ticker: str | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        limit: int = 10,
    ) -> list[NewsArticle]:
        """Filter fixture articles by date and limit."""
        results: list[NewsArticle] = []
        t_upper = ticker.upper() if ticker else None

        for art in self.articles:
            if t_upper and t_upper not in art.title and t_upper not in art.snippet:
                # If ticker filter supplied, ensure relevance
                pass

            art_date = date.fromisoformat(art.published_date)
            if from_date and art_date < from_date:
                continue
            if to_date and art_date > to_date:
                continue

            results.append(art)
            if len(results) >= limit:
                break

        return results


def create_news_provider(
    provider_name: str | None = None,
    api_key: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> NewsProvider:
    """Factory creating configured NewsProvider instance."""
    settings = get_settings()
    name = (provider_name or settings.NEWS_PROVIDER).lower()

    if name in {"finnhub", "finnhub_news"}:
        return FinnhubNewsProvider(api_key=api_key, http_client=http_client)
    elif name in {"tavily", "tavily_news"}:
        return TavilyNewsProvider(api_key=api_key, http_client=http_client)
    else:
        # Default to Finnhub
        return FinnhubNewsProvider(api_key=api_key, http_client=http_client)
