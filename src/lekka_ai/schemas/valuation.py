"""Typed answer extension schema for valuation specialist agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lekka_ai.schemas.submit import AnswerExtension, register_answer_extension


class ScenarioValues(BaseModel):
    """Scenario intrinsic values per share for bear, base, and bull cases."""

    model_config = ConfigDict(extra="allow")

    bear: float = Field(..., description="Bear case fair value per share")
    base: float = Field(..., description="Base case fair value per share")
    bull: float = Field(..., description="Bull case fair value per share")
    expected_value: float | None = Field(
        default=None, description="Weighted expected fair value"
    )


class ValuationAnswerExtension(AnswerExtension):
    """Answer extension containing valuation scenario values and implied growth."""

    model_config = ConfigDict(extra="allow")

    scenario_values: ScenarioValues | dict[str, Any] = Field(
        ...,
        description="Bear, base, and bull intrinsic scenario values",
    )
    implied_growth: float | None = Field(
        default=None,
        description="Market-implied growth rate solved from reverse DCF",
    )
    current_price: float | None = Field(
        default=None,
        description="Current market price per share",
    )
    multiples: dict[str, Any] = Field(
        default_factory=dict,
        description="Valuation multiples (P/E, EV/EBITDA, etc.)",
    )


# Register valuation answer extension globally
register_answer_extension("valuation", ValuationAnswerExtension)
