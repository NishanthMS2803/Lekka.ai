"""Pydantic request and response schemas for the Run and Claim API."""

from __future__ import annotations

from datetime import date
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from lekka_ai.config import get_settings
from lekka_ai.db.models import ClaimModel
from lekka_ai.schemas.claims import Claim


class CreateRunRequest(BaseModel):
    """Payload to initiate an orchestrated multi-agent research run."""

    question: str = Field(
        ...,
        description="Natural language equity research prompt or question",
    )
    ticker: str | None = Field(
        default=None,
        description="Optional target equity ticker symbol (e.g. 'AAPL')",
    )
    as_of_date: str | date | None = Field(
        default=None,
        description="Cutoff date for research evidence in YYYY-MM-DD format",
    )
    budget: dict[str, Any] | None = Field(
        default=None,
        description="Optional budget limits (max_tokens, max_cost_usd, etc.)",
    )
    config: dict[str, Any] | None = Field(
        default=None,
        description="Optional run-level configuration (e.g. fault_plan)",
    )
    fault_plan: dict[str, Any] | None = Field(
        default=None,
        description="Optional chaos fault injection plan (disabled in production)",
    )

    @field_validator("question")
    @classmethod
    def validate_question_length(cls, v: str) -> str:
        """Validate question character length against configured maximum limit."""
        settings = get_settings()
        max_len = settings.MAX_QUESTION_LENGTH
        if len(v.strip()) > max_len:
            raise ValueError(
                f"Question exceeds maximum allowed length of {max_len} characters."
            )
        return v

    @model_validator(mode="after")
    def validate_security_and_budget(self) -> Self:
        """Validate ticker against allowlist and enforce hard per-run cost cap."""
        settings = get_settings()

        # 1. Ticker allowlist validation
        target_ticker = self.ticker or (self.config or {}).get("ticker")
        if target_ticker:
            allowlist = settings.TICKER_ALLOWLIST
            is_enforced = settings.TICKER_ALLOWLIST_ENABLED or allowlist is not None
            if is_enforced and allowlist:
                allowed_set = {t.strip().upper() for t in allowlist}
                if str(target_ticker).strip().upper() not in allowed_set:
                    raise ValueError(
                        f"Ticker '{target_ticker}' is not in the allowed universe."
                    )

        # 2. Hard per-run cost cap enforcement
        if self.budget and isinstance(self.budget, dict):
            cap = settings.HARD_PER_RUN_COST_CAP_USD
            for cost_key in ("cost", "max_usd", "max_cost_usd"):
                if cost_key in self.budget and self.budget[cost_key] is not None:
                    try:
                        self.budget[cost_key] = min(float(self.budget[cost_key]), cap)
                    except (ValueError, TypeError):
                        pass

        return self


class RunCreateResponse(BaseModel):
    """Response returned upon successfully launching a run."""

    run_id: str = Field(..., description="Unique ID of the initiated run")


class RunStatusResponse(BaseModel):
    """Execution status and terminal outcome of a research run."""

    run_id: str = Field(..., description="Unique ID of the run")
    status: str = Field(
        ...,
        description=(
            "Run status: 'pending', 'running', 'completed', 'failed', 'abstained'"
        ),
    )
    stop_reason: str | None = Field(
        default=None,
        description="Categorical reason execution stopped (e.g. 'done', 'abstain')",
    )
    rounds: int = Field(default=0, description="Orchestration rounds executed")
    outcome: dict[str, Any] | None = Field(
        default=None,
        description="Complete execution outcome and state metadata",
    )
    error: str | None = Field(default=None, description="Error message if run failed")


class RunReportResponse(BaseModel):
    """Synthesized research report or abstention notice for a completed run."""

    run_id: str = Field(..., description="Unique ID of the run")
    markdown: str = Field(..., description="Full report content in Markdown")
    status: str = Field(default="completed", description="Terminal run status")
    validation_passed: bool = Field(
        default=True,
        description="Whether report citations and numbers passed verification",
    )
    confidence_score: float | None = Field(
        default=None,
        description="Calculated deterministic confidence score (0.0 to 1.0)",
    )
    is_abstention: bool = Field(
        default=False,
        description="Whether the system abstained due to insufficient evidence",
    )
    sentence_to_claims: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Map from report sentence to referenced claim IDs",
    )
    claims_used: list[str] = Field(
        default_factory=list,
        description="All claim IDs grounded into the report",
    )
    report: str | None = Field(
        default=None,
        description="Alias for markdown field",
    )
    disclaimer: str = Field(
        default="Lekka AI is a research tool, not financial advice.",
        description="Mandatory regulatory disclaimer",
    )

    @model_validator(mode="after")
    def set_report_alias(self) -> Self:
        """Mirror markdown into report alias and ensure disclaimer is present."""
        if "research tool, not financial advice" not in self.markdown.lower():
            self.markdown = (
                f"{self.markdown.rstrip()}\n\n---\n"
                "> **Disclaimer:** Lekka AI is a research tool, not financial advice."
            )
        if self.report is None:
            self.report = self.markdown
        return self


class EvidenceResponse(BaseModel):
    """Verifiable supporting evidence item."""

    source_type: str = Field(..., description="Source classification type")
    document_id: str | None = Field(default=None, description="Document ID or ticker")
    section: str | None = Field(default=None, description="Section heading")
    tool_call_id: str | None = Field(default=None, description="Audited tool call ID")
    retrieved_at: str | None = Field(default=None, description="ISO timestamp")
    as_of_date: str | None = Field(default=None, description="Cutoff date")
    raw_value: Any = Field(default=None, description="Underlying raw data or metrics")


class ClaimResponse(BaseModel):
    """Declarative financial claim statement with ground-truth evidence."""

    claim_id: str = Field(..., description="Unique identifier for the claim")
    text: str = Field(..., description="Declarative claim statement")
    type: str = Field(..., description="Epistemic type: fact, calculation, etc.")
    confidence: float = Field(..., description="Confidence score")
    derived_by: str | None = Field(default=None, description="Tool deriving claim")
    produced_by: str | None = Field(default=None, description="Agent producing claim")
    contradicted_by: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    evidence: list[EvidenceResponse] = Field(default_factory=list)


def claim_to_response(claim: Claim | ClaimModel | dict[str, Any]) -> ClaimResponse:
    """Normalize any claim representation into a ClaimResponse model."""
    schema: Claim
    if isinstance(claim, ClaimModel):
        schema = claim.to_schema()
    elif isinstance(claim, dict):
        schema = Claim.model_validate(claim)
    elif isinstance(claim, Claim):
        schema = claim
    else:
        msg = f"Unsupported claim object type: {type(claim)}"
        raise TypeError(msg)

    evidence_items = [
        EvidenceResponse(
            source_type=ev.source_type,
            document_id=ev.document_id,
            section=ev.section,
            tool_call_id=str(ev.tool_call_id) if ev.tool_call_id else None,
            retrieved_at=ev.retrieved_at,
            as_of_date=ev.as_of_date,
            raw_value=ev.raw_value,
        )
        for ev in schema.evidence
    ]

    return ClaimResponse(
        claim_id=schema.claim_id,
        text=schema.text,
        type=str(schema.type),
        confidence=float(schema.confidence),
        derived_by=schema.derived_by,
        produced_by=schema.produced_by,
        contradicted_by=list(schema.contradicted_by or []),
        depends_on=list(schema.depends_on or []),
        evidence=evidence_items,
    )


class SpanResponse(BaseModel):
    """Instrumentation span record from execution tracer."""

    name: str = Field(..., description="Span name")
    id: str = Field(..., description="Unique span UUID")
    parent_id: str | None = Field(default=None, description="Parent span UUID")
    start_time: str = Field(..., description="ISO start timestamp")
    end_time: str | None = Field(default=None, description="ISO end timestamp")
    latency_ms: int | None = Field(default=None, description="Latency in milliseconds")
    status: str = Field(default="ok", description="Span status (ok, error, running)")
    error: str | None = Field(default=None, description="Error message if failed")
    span_type: str = Field(default="generic", description="Classification of span")
    tool_name: str | None = Field(default=None, description="Tool name if tool span")
    input_json: Any | None = Field(
        default=None, description="Input arguments to tool or span"
    )
    output_json: Any | None = Field(
        default=None, description="Output payload or return value"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class CostTotalsResponse(BaseModel):
    """Cumulative resource and cost metrics for a run."""

    total_tokens: int = Field(default=0, description="Total tokens used")
    tokens_in: int = Field(default=0, description="Input tokens used")
    tokens_out: int = Field(default=0, description="Output tokens used")
    tool_calls: int = Field(default=0, description="Number of tool calls executed")
    elapsed_seconds: float = Field(default=0.0, description="Total elapsed seconds")
    estimated_cost_usd: float = Field(
        default=0.0, description="Estimated total cost in USD"
    )


class RunTraceResponse(BaseModel):
    """Execution trace report containing spans and resource cost totals."""

    run_id: str = Field(..., description="Unique ID of the run")
    spans: list[SpanResponse] = Field(
        default_factory=list, description="All execution spans recorded"
    )
    cost_totals: CostTotalsResponse = Field(
        default_factory=CostTotalsResponse,
        description="Cumulative tokens, tool calls, and USD cost totals",
    )


class VariantMetricsResponse(BaseModel):
    """Aggregate evaluation metrics for a single pipeline variant."""

    variant: str = Field(..., description="Variant name, e.g. v1_direct, v5_dynamic")
    accuracy: float = Field(default=0.0, description="Task factual accuracy rate")
    citation_precision: float = Field(
        default=0.0, description="Citation precision rate"
    )
    unsupported_facts: float = Field(default=0.0, description="Unsupported fact rate")
    risk_recall: float = Field(default=0.0, description="Theme risk recall rate")
    calibration: float = Field(default=0.0, description="Calibration Brier score")
    abstention_quality: float = Field(
        default=0.0, description="Abstention quality score"
    )
    success_rate: float = Field(default=0.0, description="Pipeline run success rate")
    avg_cost: float = Field(default=0.0, description="Average cost in USD per query")
    total_cost: float = Field(default=0.0, description="Total cost in USD")
    avg_tokens: float = Field(default=0.0, description="Average tokens per query")
    avg_tool_calls: float = Field(
        default=0.0, description="Average tool calls per query"
    )
    unnecessary_calls: float = Field(
        default=0.0, description="Average unnecessary calls"
    )
    avg_latency_ms: float = Field(default=0.0, description="Average latency in ms")
    min_latency_ms: float = Field(default=0.0, description="Minimum latency in ms")
    max_latency_ms: float = Field(default=0.0, description="Maximum latency in ms")


class EvalResultsResponse(BaseModel):
    """Evaluation benchmark results across all pipeline variants and ablations."""

    eval_run_id: str = Field(..., description="Unique evaluation run identifier")
    timestamp: str = Field(..., description="ISO execution timestamp")
    budget_cap_usd: float = Field(
        default=1.0, description="Allocated budget cap in USD"
    )
    total_spend_usd: float = Field(default=0.0, description="Total spend in USD")
    variants: list[VariantMetricsResponse] = Field(
        default_factory=list, description="Performance metrics by variant"
    )
    report_markdown: str = Field(default="", description="Complete REPORT.md contents")
    available_plots: list[str] = Field(
        default_factory=list, description="List of generated plot filenames"
    )
