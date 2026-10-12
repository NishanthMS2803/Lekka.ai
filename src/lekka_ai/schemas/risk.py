"""Typed answer extension schema for risk specialist agent."""

from __future__ import annotations

from enum import StrEnum

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from lekka_ai.schemas.submit import AnswerExtension, register_answer_extension


class RiskSeverity(StrEnum):
    """Categorical severity classification for risk factors."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RankedRisk(BaseModel):
    """Ranked financial, operational, or market risk factor."""

    model_config = ConfigDict(extra="allow")

    rank: int = Field(
        ...,
        ge=1,
        description="Priority rank of the risk factor (1 is highest priority/severity)",
    )
    risk_factor: str = Field(
        ...,
        validation_alias=AliasChoices("risk_factor", "title", "name"),
        description="Concise description or title of the risk factor",
    )
    severity: RiskSeverity | str = Field(
        ...,
        description="Severity level: 'critical', 'high', 'medium', or 'low'",
    )
    failure_mode: str | None = Field(
        default=None,
        description="Mechanism, breakdown, or vulnerability triggering the downside",
    )
    potential_drawdown: str | float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "potential_drawdown", "potential_drawdown_pct", "drawdown_impact"
        ),
        description="Estimated drawdown or valuation impact (e.g. '30-50%' or 0.35)",
    )
    evidence_claim_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "evidence_claim_ids", "claim_ids", "evidence_claims"
        ),
        description="List of Claim IDs supporting or grounding this risk factor",
    )
    description: str | None = Field(
        default=None,
        description="In-depth analysis of the risk dynamic and transmission channel",
    )


class RiskAnswerExtension(AnswerExtension):
    """Answer extension containing ranked risks with severity and evidence citations."""

    model_config = ConfigDict(extra="allow")

    ranked_risks: list[RankedRisk] = Field(
        default_factory=list,
        validation_alias=AliasChoices("ranked_risks", "risks"),
        description="Prioritized list of ranked risk factors ordered by severity",
    )
    drawdown_catalysts: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "drawdown_catalysts", "drawdown_scenarios", "catalysts"
        ),
        description="Specific events capable of causing a 30-50% drawdown",
    )
    counter_thesis_summary: str | None = Field(
        default=None,
        validation_alias=AliasChoices("counter_thesis_summary", "counter_thesis"),
        description="Rebuttal explaining why optimistic wave-1 conclusions may fail",
    )


# Register risk answer extension globally
register_answer_extension("risk", RiskAnswerExtension)
