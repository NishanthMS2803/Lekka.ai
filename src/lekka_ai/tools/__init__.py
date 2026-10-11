"""Tools framework for specialist subagents."""

from lekka_ai.tools.base import Tool, ToolRegistry, default_registry
from lekka_ai.tools.cache import TTL_POLICIES, Cache, build_cache_key, cached_tool
from lekka_ai.tools.echo import EchoInput, EchoOutput, EchoTool
from lekka_ai.tools.filings import (
    FilingItem,
    FilingsTool,
    ListFilingsInput,
    ListFilingsOutput,
    ListFilingsTool,
)
from lekka_ai.tools.financials import (
    FinancialPeriodSummary,
    FinancialsTool,
    GetFinancialsInput,
    GetFinancialsOutput,
    GetFinancialsTool,
)
from lekka_ai.tools.market import (
    CompactBar,
    GetMarketDataInput,
    GetMarketDataOutput,
    GetMarketDataTool,
    MarketDataTool,
    MarketSummary,
)
from lekka_ai.tools.news import (
    GetNewsInput,
    GetNewsOutput,
    GetNewsTool,
    NewsItem,
    NewsTool,
)
from lekka_ai.tools.risk import (
    ComputeRiskStatsInput,
    ComputeRiskStatsOutput,
    ComputeRiskStatsTool,
    RiskStatsTool,
)
from lekka_ai.tools.search_filings import (
    Citation,
    Passage,
    SearchFilingsInput,
    SearchFilingsOutput,
    SearchFilingsTool,
)

__all__ = [
    "Cache",
    "Citation",
    "CompactBar",
    "ComputeRiskStatsInput",
    "ComputeRiskStatsOutput",
    "ComputeRiskStatsTool",
    "EchoInput",
    "EchoOutput",
    "EchoTool",
    "FilingItem",
    "FilingsTool",
    "FinancialPeriodSummary",
    "FinancialsTool",
    "GetFinancialsInput",
    "GetFinancialsOutput",
    "GetFinancialsTool",
    "GetMarketDataInput",
    "GetMarketDataOutput",
    "GetMarketDataTool",
    "GetNewsInput",
    "GetNewsOutput",
    "GetNewsTool",
    "ListFilingsInput",
    "ListFilingsOutput",
    "ListFilingsTool",
    "MarketDataTool",
    "MarketSummary",
    "NewsItem",
    "NewsTool",
    "Passage",
    "RiskStatsTool",
    "SearchFilingsInput",
    "SearchFilingsOutput",
    "SearchFilingsTool",
    "TTL_POLICIES",
    "Tool",
    "ToolRegistry",
    "build_cache_key",
    "cached_tool",
    "default_registry",
]
