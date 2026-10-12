"""Pydantic schemas for specialized evaluation datasets: risk recall and adversarial."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

AdversarialExpectedBehavior = Literal["ask", "abstain", "answer_with_caveats"]


class RiskRecallExample(BaseModel):
    """Benchmark item for evaluating recall of emerging risks prior to severe drawdowns.

    Features companies that subsequently fell sharply, parameterized with an as_of_date
    prior to the materialization of the crisis and labeled with expected risk themes.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(
        ...,
        description="Unique identifier for the risk recall example (e.g. 'risk_01')",
    )
    ticker: str = Field(
        ...,
        description="Target equity ticker symbol (e.g. 'SIVB', 'CS')",
    )
    company_name: str = Field(
        ...,
        description="Full corporate legal name of the target entity",
    )
    question: str = Field(
        ...,
        description="Research query prompting evaluation of target financial condition",
    )
    as_of_date: str = Field(
        ...,
        description="Point-in-time date strictly before the drawdown (YYYY-MM-DD)",
    )
    fall_date: str | None = Field(
        default=None,
        description="Approximate date when the severe crash materialized (YYYY-MM-DD)",
    )
    drawdown_pct: float | None = Field(
        default=None,
        description="Percentage decline in valuation or equity price (negative number)",
    )
    event_description: str | None = Field(
        default=None,
        description="Contextual summary describing the historical drawdown or failure",
    )
    risk_themes: list[str] = Field(
        ...,
        description="List of risk themes labeled as 'should have been flagged'",
    )
    type: str = Field(
        default="risk_recall",
        description="Dataset example classification type identifier",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Descriptive tags categorizing the industry, risk, or asset class",
    )

    @field_validator("as_of_date", "fall_date")
    @classmethod
    def validate_iso_date(cls, v: str | None) -> str | None:
        """Validate date string conforms to standard ISO-8601 YYYY-MM-DD format."""
        if v is None:
            return None
        try:
            date.fromisoformat(v)
        except ValueError as err:
            raise ValueError(f"Date '{v}' must match YYYY-MM-DD format: {err}") from err
        return v

    @field_validator("risk_themes")
    @classmethod
    def validate_non_empty_risk_themes(cls, v: list[str]) -> list[str]:
        """Ensure risk themes contains at least one labeled risk topic."""
        if not v:
            raise ValueError("risk_themes must contain at least one risk topic")
        clean = [t.strip() for t in v if t.strip()]
        if not clean:
            raise ValueError("risk_themes must contain non-whitespace strings")
        return clean


class AdversarialExample(BaseModel):
    """Benchmark item for evaluating behavior under adversarial or ambiguous conditions.

    Evaluates whether the system appropriately asks for clarification, abstains,
    or answers with explicit objective caveats rather than hallucinating.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(
        ...,
        description="Unique identifier for the adversarial example (e.g. 'adv_01')",
    )
    question: str = Field(
        ...,
        description="Adversarial, leading, ambiguous, or missing-data prompt",
    )
    ticker: str | None = Field(
        default=None,
        description="Primary ticker symbol if extractable from prompt, else None",
    )
    as_of_date: str | None = Field(
        default=None,
        description="Point-in-time cutoff date (YYYY-MM-DD) if applicable",
    )
    category: str = Field(
        default="ambiguous",
        description="Adversarial category (e.g. ambiguous, leading, missing_data)",
    )
    expected_behavior: AdversarialExpectedBehavior = Field(
        ...,
        description="Behavior: 'ask', 'abstain', or 'answer_with_caveats'",
    )
    behavior_rationale: str | None = Field(
        default=None,
        description="Detailed explanation of why this specific behavior is required",
    )
    expected_abstain: bool = Field(
        default=False,
        description="Flag indicating if the system must execute the abstention path",
    )
    expected: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured expectations dictionary",
    )
    type: str = Field(
        default="adversarial",
        description="Dataset example classification type identifier",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Descriptive tags categorizing the scenario and security domain",
    )

    @field_validator("as_of_date")
    @classmethod
    def validate_as_of_date(cls, v: str | None) -> str | None:
        """Validate optional as_of_date string conforms to YYYY-MM-DD format."""
        if v is None:
            return None
        try:
            date.fromisoformat(v)
        except ValueError as err:
            raise ValueError(f"Date '{v}' must match YYYY-MM-DD format: {err}") from err
        return v
