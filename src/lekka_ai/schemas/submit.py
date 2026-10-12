"""SubmitAnswer schema and typed agent answer extension hook."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lekka_ai.schemas.claims import Claim


class AnswerStatus(StrEnum):
    """Execution outcome status for an agent sub-task."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class AnswerExtension(BaseModel):
    """Base class for per-agent-type typed answer extension payloads."""

    model_config = ConfigDict(extra="allow")


# Global registry mapping agent_type string to typed AnswerExtension models
_EXTENSION_REGISTRY: dict[str, type[BaseModel]] = {}


def register_answer_extension(
    agent_type: str,
    extension_model: type[BaseModel],
) -> None:
    """Register a typed answer extension model for a specific agent type."""
    clean_type = agent_type.strip().lower()
    _EXTENSION_REGISTRY[clean_type] = extension_model


def get_answer_extension(agent_type: str) -> type[BaseModel] | None:
    """Retrieve registered extension model for an agent type, if registered."""
    clean_type = agent_type.strip().lower()
    return _EXTENSION_REGISTRY.get(clean_type)


def clear_answer_extensions() -> None:
    """Clear registered answer extensions (useful for unit tests)."""
    _EXTENSION_REGISTRY.clear()


class SubmitAnswer(BaseModel):
    """Audited submission schema returned by specialist subagents."""

    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        ...,
        description="Unique identifier of the task being answered",
    )
    agent_type: str = Field(
        ...,
        description="Type/role of the submitting specialist agent",
    )
    status: AnswerStatus | str = Field(
        ...,
        description="Resolution status: completed, partial, or insufficient_evidence",
    )
    claims: list[Claim] = Field(
        default_factory=list,
        description="Grounded Claim objects supporting the answer",
    )
    assumptions: dict[str, Any] = Field(
        default_factory=dict,
        description="Explicit assumptions made by the agent during analysis",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Unresolved questions, data gaps, or ambiguities identified",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the answer between 0.0 and 1.0",
    )
    confidence_rationale: str = Field(
        ...,
        description="Detailed explanation and justification for the confidence rating",
    )
    limits_hit: list[str] = Field(
        default_factory=list,
        description="Resource limits hit (e.g. rate_limit, max_iterations)",
    )
    extension: Any | None = Field(
        default=None,
        description="Per-agent typed extension payload (e.g. DCF scenarios, multiples)",
    )

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, v: Any) -> AnswerStatus:
        if isinstance(v, AnswerStatus):
            return v
        if isinstance(v, str):
            clean = v.strip().lower()
            try:
                return AnswerStatus(clean)
            except ValueError:
                pass
        raise ValueError(
            f"Invalid status '{v}'. Must be one of: {[s.value for s in AnswerStatus]}"
        )

    @field_validator("assumptions", mode="before")
    @classmethod
    def _validate_assumptions(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, (list, tuple)):
            return {f"assumption_{idx + 1}": item for idx, item in enumerate(v)}
        return {"general": str(v)}

    @model_validator(mode="after")
    def _validate_answer_rules(self) -> Self:
        """Validate status consistency and typed extensions."""
        # Completed answers require at least one grounded claim
        if self.status == AnswerStatus.COMPLETED and len(self.claims) == 0:
            raise ValueError(
                "Completed answers require at least one grounded Claim. "
                "If insufficient data was found, use status='insufficient_evidence'."
            )

        # Validate typed extension hook if registered for agent_type
        clean_type = self.agent_type.strip().lower()
        ext_model = _EXTENSION_REGISTRY.get(clean_type)
        if ext_model is not None and self.extension is not None:
            if isinstance(self.extension, ext_model):
                pass
            elif isinstance(self.extension, dict):
                # Validate and parse dictionary into registered extension model
                object.__setattr__(
                    self, "extension", ext_model.model_validate(self.extension)
                )
            elif isinstance(self.extension, BaseModel):
                # Validate extension instance
                object.__setattr__(
                    self,
                    "extension",
                    ext_model.model_validate(self.extension.model_dump()),
                )

        return self


def export_submit_answer_schema(agent_type: str | None = None) -> dict[str, Any]:
    """Export JSON Schema for SubmitAnswer, optionally tailored with agent extension."""
    schema = SubmitAnswer.model_json_schema()
    if agent_type is not None:
        ext_model = get_answer_extension(agent_type)
        if ext_model is not None:
            schema["$defs"] = schema.get("$defs", {})
            schema["$defs"][ext_model.__name__] = ext_model.model_json_schema()
    return schema
