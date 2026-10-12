"""Shared fixtures and test registries for the test suite."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from lekka_ai.data.http import create_http_client
from lekka_ai.data.market import CsvFixtureProvider
from lekka_ai.data.news import FixtureNewsProvider, NewsArticle
from lekka_ai.data.sec import SecClient
from lekka_ai.tools.base import ToolRegistry
from lekka_ai.tools.echo import EchoTool
from lekka_ai.tools.estimates import (
    GetAnalystEstimatesTool,
    GetIndustryDataTool,
)
from lekka_ai.tools.filings import ListFilingsTool
from lekka_ai.tools.financials import GetFinancialsTool
from lekka_ai.tools.market import GetMarketDataTool
from lekka_ai.tools.news import GetNewsTool
from lekka_ai.tools.python_sandbox import RunPythonTool
from lekka_ai.tools.quant import (
    ComputeMultiplesTool,
    RunDCFTool,
    RunReverseDCFTool,
    ScenarioTableTool,
)
from lekka_ai.tools.recall_facts import RecallFactsTool
from lekka_ai.tools.risk import ComputeRiskStatsTool
from lekka_ai.tools.search_filings import (
    Citation,
    Passage,
    SearchFilingsInput,
    SearchFilingsOutput,
    SearchFilingsTool,
)
from lekka_ai.tools.submit import SubmitAnswerTool


class FixtureSearchFilingsTool(SearchFilingsTool):
    """Fixture-backed SearchFilingsTool returning deterministic passages."""

    async def run(self, input_data: SearchFilingsInput) -> SearchFilingsOutput:
        return SearchFilingsOutput(
            ticker=input_data.ticker,
            query=input_data.query,
            top_k=input_data.top_k,
            total_results=1,
            as_of_date=input_data.as_of_date,
            passages=[
                Passage(
                    text=(
                        "Apple Inc. announced key executive transitions and"
                        " operations in Item 5.02."
                    ),
                    score=0.92,
                    citation=Citation(
                        accession="0000320193-24-000106",
                        section="Item 5.02",
                        filing_date="2024-08-26",
                        chunk_id="chunk-fix-001",
                    ),
                )
            ],
        )


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to recorded fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_news_articles() -> list[NewsArticle]:
    """Return recorded news fixture articles for isolation testing."""
    return [
        NewsArticle(
            title="Apple Announces Luca Maestri to Transition from CFO Role",
            source="Apple Newsroom",
            published_date="2024-08-26",
            url="https://www.apple.com/newsroom/2024/08/luca-maestri-cfo",
            snippet=(
                "Apple today announced that Luca Maestri will transition from"
                " the CFO role on January 1, 2025. Kevan Parekh will become CFO."
            ),
        ),
        NewsArticle(
            title="Apple Debuts iPhone 16 Lineup with Apple Intelligence",
            source="Reuters",
            published_date="2024-09-09",
            url="https://www.reuters.com/technology/apple-glowtime-2024-09-09",
            snippet=(
                "Apple unveiled its iPhone 16 family powered by the A18 chip and"
                " integrated Apple Intelligence features."
            ),
        ),
        NewsArticle(
            title="Apple Board Declares Cash Dividend of $0.25 Per Share",
            source="PR Newswire",
            published_date="2024-08-01",
            url="https://www.apple.com/newsroom/2024/08/dividend-declaration",
            snippet=(
                "Apple's board of directors declared a cash dividend of $0.25"
                " per share of the Company's common stock."
            ),
        ),
    ]


@pytest.fixture
def mock_sec_client(fixtures_dir: Path) -> SecClient:
    """Return SecClient backed by MockTransport serving local recorded fixtures."""
    facts_path = fixtures_dir / "AAPL_company_facts.json"
    with open(facts_path, encoding="utf-8") as f:
        facts_data = json.load(f)

    subs_path = fixtures_dir / "AAPL_submissions.json"
    with open(subs_path, encoding="utf-8") as f:
        subs_data = json.load(f)

    tickers_data: dict[str, Any] = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "company_tickers.json" in url_str:
            return httpx.Response(200, json=tickers_data, request=request)
        if "companyfacts" in url_str:
            return httpx.Response(200, json=facts_data, request=request)
        if "submissions" in url_str:
            return httpx.Response(200, json=subs_data, request=request)
        return httpx.Response(404, request=request)

    transport = httpx.MockTransport(handler)
    http_client = create_http_client(transport=transport)
    return SecClient(http_client=http_client)


@pytest.fixture
def fixture_tool_registry(
    fixtures_dir: Path,
    mock_sec_client: SecClient,
    fixture_news_articles: list[NewsArticle],
) -> ToolRegistry:
    """Build a ToolRegistry containing fixture-backed versions of all tools."""
    registry = ToolRegistry()

    # Market tools
    csv_provider = CsvFixtureProvider(fixtures_dir=fixtures_dir)
    registry.register(GetMarketDataTool(provider=csv_provider))
    registry.register(ComputeRiskStatsTool(provider=csv_provider))

    # SEC tools
    registry.register(GetFinancialsTool(sec_client=mock_sec_client))
    registry.register(ListFilingsTool(sec_client=mock_sec_client))
    registry.register(FixtureSearchFilingsTool())

    # News tool
    news_provider = FixtureNewsProvider(articles=fixture_news_articles)
    registry.register(GetNewsTool(provider=news_provider))

    # Fundamental & Estimates tools
    registry.register(GetAnalystEstimatesTool())
    registry.register(GetIndustryDataTool())

    # Quantitative modeling tools
    registry.register(RunDCFTool())
    registry.register(RunReverseDCFTool())
    registry.register(ComputeMultiplesTool())
    registry.register(ScenarioTableTool())

    # Sandbox & utility tools
    registry.register(RunPythonTool())
    registry.register(EchoTool())
    registry.register(SubmitAnswerTool())
    registry.register(RecallFactsTool())

    return registry
