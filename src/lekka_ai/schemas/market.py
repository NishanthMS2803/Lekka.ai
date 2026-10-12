"""Typed answer extension schema for market specialist agent."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from lekka_ai.schemas.submit import AnswerExtension, register_answer_extension


class MarketAnswerExtension(AnswerExtension):
    """Answer extension containing market trend and key quantitative statistics."""

    model_config = ConfigDict(extra="allow")

    trend_label: str = Field(
        ...,
        description="Market trend label (e.g. 'bullish', 'bearish', 'neutral')",
    )
    key_stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Key statistics (volatility, drawdown, beta, relative strength)",
    )


# Register market answer extension globally
register_answer_extension("market", MarketAnswerExtension)
