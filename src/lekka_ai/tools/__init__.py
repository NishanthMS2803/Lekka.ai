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
from lekka_ai.tools.market import (
    CompactBar,
    GetMarketDataInput,
    GetMarketDataOutput,
    GetMarketDataTool,
    MarketDataTool,
    MarketSummary,
)
from lekka_ai.tools.risk import (
    ComputeRiskStatsInput,
    ComputeRiskStatsOutput,
    ComputeRiskStatsTool,
    RiskStatsTool,
)

__all__ = [
    "Cache",
    "CompactBar",
    "ComputeRiskStatsInput",
    "ComputeRiskStatsOutput",
    "ComputeRiskStatsTool",
    "EchoInput",
    "EchoOutput",
    "EchoTool",
    "FilingItem",
    "FilingsTool",
    "GetMarketDataInput",
    "GetMarketDataOutput",
    "GetMarketDataTool",
    "ListFilingsInput",
    "ListFilingsOutput",
    "ListFilingsTool",
    "MarketDataTool",
    "MarketSummary",
    "RiskStatsTool",
    "TTL_POLICIES",
    "Tool",
    "ToolRegistry",
    "build_cache_key",
    "cached_tool",
    "default_registry",
]
