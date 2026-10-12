"""Tests for news providers and get_news tool with untrusted data tagging."""

import os
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from lekka_ai.data.news import (
    FinnhubNewsProvider,
    FixtureNewsProvider,
    NewsArticle,
    NewsProviderError,
    TavilyNewsProvider,
    create_news_provider,
)
from lekka_ai.tools.base import default_registry
from lekka_ai.tools.news import (
    GetNewsInput,
    GetNewsTool,
)


def test_news_article_untrusted_flag() -> None:
    """NewsArticle records are automatically labeled as untrusted third-party data."""
    art = NewsArticle(
        title="Apple Reports Record Results",
        source="Reuters",
        published_date="2024-01-15",
        url="https://example.com/article1",
        snippet="Apple shares surged following record quarterly revenues.",
    )
    assert art.is_untrusted is True
    assert art.trust_label == "UNTRUSTED_EXTERNAL_CONTENT"


def test_get_news_tool_registration() -> None:
    """get_news tool is registered in the tool registry with correct schema."""
    tool = default_registry.get("get_news")
    assert tool is not None
    assert tool.name == "get_news"
    spec = tool.to_spec()
    assert spec.name == "get_news"
    props = spec.json_schema["properties"]
    assert "query" in props
    assert "ticker" in props
    assert "as_of_date" in props


@pytest.mark.asyncio
async def test_finnhub_news_provider_mocked() -> None:
    """FinnhubNewsProvider formats company news and marks content as untrusted."""
    mock_payload = [
        {
            "category": "company",
            "datetime": 1705320000,  # 2024-01-15T12:00:00Z
            "headline": "Apple Unveils Next Gen Silicon",
            "id": 101,
            "image": "https://example.com/img.png",
            "related": "AAPL",
            "source": "Bloomberg",
            "summary": "Apple introduced new M3 chips with advanced neural processing.",
            "url": "https://bloomberg.com/news/apple-m3",
        },
        {
            "category": "company",
            "datetime": 1705233600,  # 2024-01-14T12:00:00Z
            "headline": "Analysts Raise Apple Price Targets",
            "id": 102,
            "related": "AAPL",
            "source": "CNBC",
            "summary": "Analysts upgraded Apple following strong iPhone demand.",
            "url": "https://cnbc.com/apple-upgrade",
        },
    ]

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=mock_payload)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(return_value=mock_resp)

    provider = FinnhubNewsProvider(api_key="test-api-key", http_client=mock_client)
    articles = await provider.get_news(
        query="AAPL",
        ticker="AAPL",
        from_date=date(2024, 1, 1),
        to_date=date(2024, 1, 31),
        limit=5,
    )

    assert len(articles) == 2
    art1 = articles[0]
    assert art1.title == "Apple Unveils Next Gen Silicon"
    assert art1.source == "Bloomberg"
    assert art1.published_date == "2024-01-15"
    assert art1.url == "https://bloomberg.com/news/apple-m3"
    assert "M3 chips" in art1.snippet
    assert art1.is_untrusted is True

    # Verify query params passed to Finnhub
    call_args = mock_client.get.call_args
    assert call_args[0][0] == "https://finnhub.io/api/v1/company-news"
    params = call_args[1]["params"]
    assert params["symbol"] == "AAPL"
    assert params["from"] == "2024-01-01"
    assert params["to"] == "2024-01-31"
    assert params["token"] == "test-api-key"


@pytest.mark.asyncio
async def test_finnhub_missing_api_key_raises_error() -> None:
    """Missing API key raises NewsProviderError."""
    provider = FinnhubNewsProvider(api_key=None)
    with pytest.raises(NewsProviderError) as exc_info:
        await provider.get_news(query="AAPL")
    assert "NEWS_API_KEY is not configured" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tavily_news_provider_mocked() -> None:
    """TavilyNewsProvider formats query results and labels untrusted data."""
    mock_payload = {
        "results": [
            {
                "title": "Microsoft and OpenAI Partner on Supercomputing",
                "content": "Microsoft is planning a $100B AI datacenter.",
                "url": "https://example.com/msft-ai",
                "published_date": "2024-03-29",
                "source": "The Information",
            }
        ]
    }

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=mock_payload)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(return_value=mock_resp)

    provider = TavilyNewsProvider(api_key="test-tavily-key", http_client=mock_client)
    articles = await provider.get_news(query="Microsoft AI datacenter", limit=2)

    assert len(articles) == 1
    art = articles[0]
    assert "Microsoft and OpenAI" in art.title
    assert "supercomputing" in art.snippet.lower() or "$100B" in art.snippet
    assert art.is_untrusted is True


@pytest.mark.asyncio
async def test_get_news_tool_point_in_time_enforcement() -> None:
    """get_news strictly excludes articles published after as_of_date."""
    # Synthetic articles across January - March 2024
    test_articles = [
        NewsArticle(
            title="Old Article Jan 10",
            source="NewsA",
            published_date="2024-01-10",
            url="https://example.com/1",
            snippet="Old news from early January.",
        ),
        NewsArticle(
            title="Target Date Article Jan 15",
            source="NewsB",
            published_date="2024-01-15",
            url="https://example.com/2",
            snippet="News on the exact cutoff date.",
        ),
        NewsArticle(
            title="Future Article Jan 20 (LEAK)",
            source="NewsC",
            published_date="2024-01-20",
            url="https://example.com/3",
            snippet="Future leak that must NOT be returned.",
        ),
        NewsArticle(
            title="Far Future Article Feb 10 (LEAK)",
            source="NewsD",
            published_date="2024-02-10",
            url="https://example.com/4",
            snippet="Far future leak.",
        ),
    ]

    fixture_provider = FixtureNewsProvider(articles=test_articles)
    tool = GetNewsTool(provider=fixture_provider)

    # Cutoff date: 2024-01-15
    output = await tool.run(
        GetNewsInput(
            ticker="AAPL",
            as_of_date="2024-01-15",
            limit=10,
        )
    )

    assert output.as_of_date == "2024-01-15"
    assert len(output.articles) == 2

    # None of the returned articles can be after 2024-01-15
    for item in output.articles:
        art_date = date.fromisoformat(item.published_date)
        assert art_date <= date(2024, 1, 15)
        # All items must be marked untrusted
        assert item.is_untrusted is True
        assert item.trust_label == "UNTRUSTED_DATA"

    titles = [a.title for a in output.articles]
    assert "Old Article Jan 10" in titles
    assert "Target Date Article Jan 15" in titles
    assert "Future Article Jan 20 (LEAK)" not in titles
    assert "Far Future Article Feb 10 (LEAK)" not in titles

    # Verify safety warning in output payload
    assert "untrusted" in output.trust_warning.lower()


@pytest.mark.asyncio
async def test_get_news_tool_input_validation() -> None:
    """Validates missing query/ticker and limit clamping."""
    tool = GetNewsTool(provider=FixtureNewsProvider([]))

    # Error when both ticker and query are omitted
    with pytest.raises(ValueError):
        await tool.run(GetNewsInput())

    # Query with limit clamping
    out = await tool.run(GetNewsInput(query="semiconductors", limit=100))
    assert out.query == "semiconductors"


def test_create_news_provider_factory() -> None:
    """Factory instantiates corresponding provider based on configuration."""
    finnhub = create_news_provider(provider_name="finnhub", api_key="dummy")
    assert isinstance(finnhub, FinnhubNewsProvider)

    tavily = create_news_provider(provider_name="tavily", api_key="dummy")
    assert isinstance(tavily, TavilyNewsProvider)


@pytest.mark.live
@pytest.mark.asyncio
async def test_news_provider_live_finnhub() -> None:
    """Live integration test: query live Finnhub API for AAPL news if configured."""
    api_key = os.getenv("NEWS_API_KEY")
    if os.getenv("RUN_LIVE") != "1" or not api_key:
        pytest.skip(
            "RUN_LIVE != '1' or NEWS_API_KEY not configured; skipping live test"
        )

    provider = FinnhubNewsProvider(api_key=api_key)
    articles = await provider.get_news(
        query="AAPL",
        ticker="AAPL",
        from_date=date(2024, 1, 1),
        to_date=date(2024, 1, 15),
        limit=5,
    )

    assert len(articles) > 0
    top = articles[0]
    assert top.title is not None
    assert top.url.startswith("http")
    assert top.is_untrusted is True
    assert top.published_date >= "2024-01-01"
