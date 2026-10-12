"""Typed answer extension schema for news and catalyst specialist agent."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from lekka_ai.schemas.submit import AnswerExtension, register_answer_extension


class CatalystImpact(StrEnum):
    """Directional impact of a corporate catalyst event."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"


class CatalystNovelty(StrEnum):
    """Novelty state of a corporate development."""

    NEW = "new"
    ALREADY_KNOWN = "already_known"


class CatalystItem(BaseModel):
    """Structured corporate catalyst or news event item."""

    model_config = ConfigDict(extra="allow")

    headline: str = Field(
        ..., description="Summary of the corporate event or news development"
    )
    date: str = Field(
        ..., description="Date of the development or announcement (YYYY-MM-DD)"
    )
    impact: str = Field(
        ...,
        description="Impact: 'positive', 'negative', 'neutral', or 'uncertain'",
    )
    novelty: str = Field(
        ...,
        description="Novelty: 'new' vs 'already_known'",
    )
    source: str | None = Field(
        default=None,
        description="Source identifier (e.g. 'SEC 8-K', 'Yahoo News')",
    )


class NewsAnswerExtension(AnswerExtension):
    """Answer extension containing a list of analyzed corporate catalysts."""

    model_config = ConfigDict(extra="allow")

    catalysts: list[CatalystItem] = Field(
        default_factory=list,
        description="List of structured corporate catalysts",
    )
    summary: str | None = Field(
        default=None,
        description="Synthesis of overall catalyst landscape",
    )


# Register news answer extension globally
register_answer_extension("news", NewsAnswerExtension)
