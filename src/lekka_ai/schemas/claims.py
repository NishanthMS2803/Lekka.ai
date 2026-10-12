"""Claim and Evidence data contracts for grounded financial analysis."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ClaimType(StrEnum):
    """Categorical classification of financial claims."""

    FACT = "fact"
    CALCULATION = "calculation"
    INTERPRETATION = "interpretation"
    BACKGROUND = "background"


class Evidence(BaseModel):
    """Verifiable supporting evidence grounded in tools or source documents."""

    model_config = ConfigDict(extra="allow", frozen=True)

    source_type: str = Field(
        ...,
        description=(
            "Type of evidence source, e.g. 'sec_filing', 'market_data', "
            "'xbrl_facts', 'news', 'analyst_estimates'"
        ),
    )
    document_id: str | None = Field(
        default=None,
        description="Accession number, ticker, or unique document identifier",
    )
    section: str | None = Field(
        default=None,
        description="Filing section heading (e.g. 'Item 1A', 'Item 7') if applicable",
    )
    tool_call_id: str | None = Field(
        default=None,
        description="Audited execution ID of the tool call producing this evidence",
    )
    retrieved_at: str | None = Field(
        default=None,
        description="ISO timestamp when the evidence was retrieved",
    )
    as_of_date: str | None = Field(
        default=None,
        description="Point-in-time cutoff date (YYYY-MM-DD)",
    )
    raw_value: Any = Field(
        default=None,
        description="Underlying raw data, table snippet, or metric value",
    )

    @field_validator("retrieved_at", mode="before")
    @classmethod
    def _validate_retrieved_at(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, (datetime, date)):
            return v.isoformat()
        return str(v).strip() or None

    @field_validator("as_of_date", mode="before")
    @classmethod
    def _validate_as_of_date(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, (date, datetime)):
            return v.strftime("%Y-%m-%d")
        return str(v).strip() or None


class Claim(BaseModel):
    """A financial claim statement with ground-truth evidence and lineage."""

    model_config = ConfigDict(extra="allow")

    claim_id: str = Field(
        default_factory=lambda: f"claim_{uuid4().hex[:10]}",
        description="Unique identifier for the claim",
    )
    text: str = Field(
        ...,
        description="Clear, declarative text statement of the claim",
    )
    type: ClaimType | str = Field(
        ...,
        description="Epistemic type: fact, calculation, interpretation, or background",
    )
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="List of verifiable supporting Evidence objects",
    )
    derived_by: str | None = Field(
        default=None,
        description="Formula, quantitative method, or tool that derived this value",
    )
    produced_by: str | None = Field(
        default=None,
        description="Agent role or tool name that generated this claim",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated confidence score between 0.0 and 1.0",
    )
    contradicted_by: list[str] = Field(
        default_factory=list,
        description="List of claim or evidence IDs that contradict this claim",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of prerequisite claim IDs for interpretation claims",
    )

    @field_validator("type", mode="before")
    @classmethod
    def _validate_type(cls, v: Any) -> ClaimType:
        if isinstance(v, ClaimType):
            return v
        if isinstance(v, str):
            clean = v.strip().lower()
            try:
                return ClaimType(clean)
            except ValueError:
                pass
        raise ValueError(
            f"Invalid claim type '{v}'. Must be one of: {[t.value for t in ClaimType]}"
        )

    @model_validator(mode="after")
    def _validate_claim_rules(self) -> Self:
        """Validate integrity rules for facts, calculations, and interpretations."""
        # 1. Fact claims must have at least one Evidence object with tool_call_id
        if self.type == ClaimType.FACT:
            if not self.evidence:
                raise ValueError("Fact claims require at least one Evidence object.")
            has_tool_call = any(
                e.tool_call_id is not None and bool(e.tool_call_id.strip())
                for e in self.evidence
            )
            if not has_tool_call:
                raise ValueError(
                    "Fact claims require at least one Evidence with tool_call_id."
                )

        # 2. Calculation claims require derived_by or evidence
        elif self.type == ClaimType.CALCULATION:
            has_derivation = bool(self.derived_by and self.derived_by.strip())
            has_evidence = len(self.evidence) > 0
            if not has_derivation and not has_evidence:
                raise ValueError(
                    "Calculation claims require either 'derived_by' (formula/method) "
                    "or supporting 'evidence'."
                )

        # 3. Interpretation claims require depends_on claim IDs
        elif self.type == ClaimType.INTERPRETATION:
            if not self.depends_on:
                raise ValueError(
                    "Interpretation claims require at least one prerequisite "
                    "claim ID in 'depends_on'."
                )

        return self
