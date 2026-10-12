"""Dispatch glue connecting planned tasks, context resolution, runtime, and budget."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from lekka_ai.db.ledger import LedgerRepository
from lekka_ai.harness.models import AgentTask
from lekka_ai.harness.permissions import ToolView
from lekka_ai.harness.profiles import load_agent_profile
from lekka_ai.harness.runtime import AgentRunResult, AgentRuntime
from lekka_ai.llm.base import LLMClient
from lekka_ai.logging import get_logger
from lekka_ai.orchestrator.budget import RunBudget
from lekka_ai.orchestrator.state import PlannedTask, RunState, to_uuid
from lekka_ai.tools.base import ToolRegistry, default_registry
from lekka_ai.tracing import Tracer

logger = get_logger(__name__)


class TaskResult(BaseModel):
    """Structured result returned from executing a planned specialist task."""

    model_config = ConfigDict(extra="allow")

    task_id: str = Field(..., description="Planned task identifier")
    agent_type: str = Field(..., description="Executing agent profile type")
    status: str = Field(..., description="Terminal task status ('completed', etc.)")
    claims: list[Any] = Field(
        default_factory=list,
        description="List of verified grounded claims submitted by agent",
    )
    assumptions: dict[str, Any] = Field(
        default_factory=dict,
        description="Assumptions dictionary submitted by agent",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Unresolved questions identified during research",
    )
    usage: dict[str, Any] = Field(
        default_factory=dict,
        description="Resource usage statistics (tokens, tool calls, latency, USD)",
    )
    confidence: float | None = Field(
        default=None,
        description="Confidence score between 0.0 and 1.0",
    )
    confidence_rationale: str | None = Field(
        default=None,
        description="Justification for confidence assessment",
    )
    extension: dict[str, Any] | None = Field(
        default=None,
        description="Typed per-agent extension payload if available",
    )
    error: str | None = Field(
        default=None,
        description="Error description if task execution failed",
    )
    agent_run_id: str | None = Field(
        default=None,
        description="Unique identifier for the persisted agent run",
    )


def get_relevant_context_facts(
    planned_task: PlannedTask,
    state: RunState,
    ledger: LedgerRepository | None = None,
) -> list[dict[str, Any]]:
    """Retrieve grounded claims from ledger created by upstream dependency tasks.

    Extracts claims from the ledger matching the task_ids in planned_task.depends_on.
    If the ledger is unavailable, falls back to state.results from completed tasks.
    """
    context_facts: list[dict[str, Any]] = []

    # 1. Primary path: query database ledger
    if ledger is not None:
        try:
            run_uuid = to_uuid(state.run_id)
            run_claims = ledger.get_claims(run_id=run_uuid)
            dep_uuids = {to_uuid(dep) for dep in planned_task.depends_on}
            dep_slugs = set(planned_task.depends_on)

            for c in run_claims:
                has_task_dep = c.task_id and (
                    c.task_id in dep_uuids or str(c.task_id) in dep_slugs
                )
                if has_task_dep:
                    is_dep_claim = True
                elif c.produced_by and c.produced_by in dep_slugs:
                    is_dep_claim = True

                if is_dep_claim:
                    context_facts.append(
                        {
                            "claim_id": c.claim_id,
                            "text": c.text,
                            "type": c.type,
                            "confidence": c.confidence,
                            "produced_by": c.produced_by,
                            "evidence": [
                                {
                                    "source_type": ev.source_type,
                                    "raw_value": ev.raw_value,
                                    "as_of_date": ev.as_of_date,
                                    "document_id": ev.document_id,
                                }
                                for ev in c.evidence
                            ],
                        }
                    )
            if context_facts:
                return context_facts
        except Exception as exc:
            logger.warning(
                "Failed to retrieve context facts from ledger",
                task_id=planned_task.task_id,
                error=str(exc),
            )

    # 2. Fallback path: extract from completed state.results
    for dep_id in planned_task.depends_on:
        dep_result = state.results.get(dep_id)
        if isinstance(dep_result, dict):
            raw_claims = dep_result.get("claims", [])
            for rc in raw_claims:
                if isinstance(rc, dict):
                    context_facts.append(rc)
                elif hasattr(rc, "model_dump"):
                    context_facts.append(rc.model_dump())

    return context_facts


async def run_task(
    planned_task: PlannedTask,
    state: RunState,
    budget: RunBudget | None = None,
    *,
    llm: LLMClient | None = None,
    tool_registry: ToolRegistry | None = None,
    ledger: LedgerRepository | None = None,
    session: Session | None = None,
    tracer: Tracer | None = None,
    max_repair_attempts: int = 3,
) -> TaskResult:
    """Execute a single planned task through the runtime, budget, and ledger glue.

    Args:
        planned_task: The PlannedTask to execute.
        state: Shared RunState tracking overall run progress.
        budget: RunBudget manager for limits allocation and reconciliation.
        llm: LLMClient instance (defaults to GeminiClient).
        tool_registry: ToolRegistry for specialist tool authorization.
        ledger: LedgerRepository for claim and evidence persistence.
        session: Database session for task and run persistence.
        tracer: Optional Tracer instance for span instrumentation.
        max_repair_attempts: Repair iterations for submit_answer schema validation.

    Returns:
        Structured TaskResult with claims, assumptions, usage, and status.
    """
    effective_ledger = ledger or (LedgerRepository(session) if session else None)
    effective_budget = budget or RunBudget.from_settings(tracer=tracer)

    # 1. Build AgentTask with only relevant context facts from the ledger
    context_facts = get_relevant_context_facts(
        planned_task=planned_task,
        state=state,
        ledger=effective_ledger,
    )

    as_of_date: str | None = None
    if state.parsed_query and isinstance(state.parsed_query, dict):
        as_of_date = state.parsed_query.get("as_of_date")
    if not as_of_date and "as_of_date" in planned_task.metadata:
        as_of_date = str(planned_task.metadata["as_of_date"])

    task_uuid_str = str(to_uuid(planned_task.task_id))
    agent_task = AgentTask(
        task_id=task_uuid_str,
        agent_type=planned_task.agent_type,
        instruction=planned_task.instruction,
        context_facts=context_facts,
        as_of_date=as_of_date,
        run_id=str(to_uuid(state.run_id)),
    )

    # Update state to running
    state.update_task_status(
        task_id=planned_task.task_id,
        status="running",
        session=session,
    )

    # 2. Pick the agent profile
    profile = load_agent_profile(planned_task.agent_type)

    # 3. Create ToolView
    reg = tool_registry or default_registry
    tool_view = ToolView(registry=reg, profile=profile, tracer=tracer)

    # 4. Allocate Limits from budget
    limits = effective_budget.allocate(planned_task)

    # 5. Call AgentRuntime
    t0 = time.perf_counter()
    run_result: AgentRunResult
    try:
        run_result = await AgentRuntime.run(
            task=agent_task,
            profile=profile,
            llm=llm,  # type: ignore[arg-type]
            tool_view=tool_view,
            limits=limits,
            session=session,
            ledger=effective_ledger,
            max_repair_attempts=max_repair_attempts,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        err_msg = f"Task execution failed: {exc}"
        logger.error(err_msg, task_id=planned_task.task_id, exc_info=True)
        task_result = TaskResult(
            task_id=planned_task.task_id,
            agent_type=planned_task.agent_type,
            status="failed",
            usage={"elapsed_seconds": round(elapsed, 4)},
            error=err_msg,
        )
        state.update_task_status(
            task_id=planned_task.task_id,
            status="failed",
            session=session,
            result=task_result.model_dump(),
            error=err_msg,
        )
        return task_result

    elapsed = time.perf_counter() - t0

    # 6. Reconcile budget usage
    effective_budget.reconcile(run_result, elapsed_seconds=elapsed)

    # 7. Extract structured submission components
    claims: list[Any] = []
    assumptions: dict[str, Any] = {}
    open_questions: list[str] = []
    confidence: float | None = None
    confidence_rationale: str | None = None
    extension: dict[str, Any] | None = None

    if run_result.submission is not None:
        raw_claims = run_result.submission.claims
        claims = [c.model_dump() if hasattr(c, "model_dump") else c for c in raw_claims]
        assumptions = dict(getattr(run_result.submission, "assumptions", {}) or {})
        open_questions = list(
            getattr(run_result.submission, "open_questions", []) or []
        )
        confidence = run_result.submission.confidence
        confidence_rationale = run_result.submission.confidence_rationale

        if (
            hasattr(run_result.submission, "extension")
            and run_result.submission.extension is not None
        ):
            ext = run_result.submission.extension
            if hasattr(ext, "model_dump"):
                extension = ext.model_dump()
            elif isinstance(ext, dict):
                extension = dict(ext)
    elif run_result.stored_claims:
        claims = [
            c.model_dump() if hasattr(c, "model_dump") else c
            for c in run_result.stored_claims
        ]

    usage = {
        "tokens_in": run_result.tokens_in,
        "tokens_out": run_result.tokens_out,
        "total_tokens": run_result.tokens_in + run_result.tokens_out,
        "tool_calls": run_result.tool_calls_count,
        "iterations": run_result.iterations,
        "elapsed_seconds": round(elapsed, 4),
        "estimated_cost_usd": effective_budget.estimate_usd(
            run_result.tokens_in, run_result.tokens_out
        ),
    }

    # 8. Build TaskResult
    task_result = TaskResult(
        task_id=planned_task.task_id,
        agent_type=planned_task.agent_type,
        status=run_result.status,
        claims=claims,
        assumptions=assumptions,
        open_questions=open_questions,
        usage=usage,
        confidence=confidence,
        confidence_rationale=confidence_rationale,
        extension=extension,
        agent_run_id=run_result.agent_run_id,
    )

    # 9. Update RunState
    state.update_task_status(
        task_id=planned_task.task_id,
        status=task_result.status,
        session=session,
        result=task_result.model_dump(),
    )
    if task_result.open_questions:
        state.open_questions.extend(task_result.open_questions)

    return task_result
