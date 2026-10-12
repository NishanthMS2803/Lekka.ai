"""FastAPI routes for research runs management and reports."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from lekka_ai.api.dependencies import (
    get_event_bus,
    get_llm,
    get_optional_db_session,
    get_run_manager,
    get_task_runner,
    get_tool_registry,
)
from lekka_ai.api.models import (
    ClaimResponse,
    CostTotalsResponse,
    CreateRunRequest,
    RunCreateResponse,
    RunReportResponse,
    RunStatusResponse,
    RunTraceResponse,
    SpanResponse,
    claim_to_response,
)
from lekka_ai.api.services import RunManager
from lekka_ai.db.ledger import LedgerRepository
from lekka_ai.db.repository import get_run as get_db_run
from lekka_ai.events import EventBus
from lekka_ai.llm.base import LLMClient
from lekka_ai.logging import get_logger
from lekka_ai.orchestrator.dispatch import TaskResult
from lekka_ai.orchestrator.engine import RunOutcome, TaskRunnerCallable
from lekka_ai.tools.base import ToolRegistry

logger = get_logger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post(
    "",
    response_model=RunCreateResponse,
    status_code=200,
    summary="Start research run in background task",
)
@router.post(
    "/",
    response_model=RunCreateResponse,
    status_code=200,
    include_in_schema=False,
)
async def create_run(
    request: CreateRunRequest,
    background_tasks: BackgroundTasks,
    llm: Annotated[LLMClient, Depends(get_llm)],
    run_manager: Annotated[RunManager, Depends(get_run_manager)],
    task_runner: Annotated[TaskRunnerCallable | None, Depends(get_task_runner)] = None,
    tool_registry: Annotated[ToolRegistry | None, Depends(get_tool_registry)] = None,
    session: Annotated[Session | None, Depends(get_optional_db_session)] = None,
) -> RunCreateResponse:
    """Launch orchestrator engine in background task and return run_id."""
    run_id = str(uuid.uuid4())
    run_manager.create_run(
        run_id=run_id,
        question=request.question,
        as_of_date=request.as_of_date,
        budget=request.budget,
        config=request.config,
        fault_plan=request.fault_plan,
    )

    background_tasks.add_task(
        run_manager.execute_run,
        run_id=run_id,
        llm=llm,
        task_runner=task_runner,
        tool_registry=tool_registry,
        session=session,
    )

    return RunCreateResponse(run_id=run_id)


@router.get(
    "/{run_id}",
    response_model=RunStatusResponse,
    summary="Get run status and outcome",
)
async def get_run_status(
    run_id: str,
    run_manager: Annotated[RunManager, Depends(get_run_manager)],
    wait: bool = Query(
        default=False,
        description="Wait for run completion before returning if active",
    ),
    timeout: float = Query(
        default=10.0,
        description="Maximum seconds to wait if wait=True",
    ),
    session: Annotated[Session | None, Depends(get_optional_db_session)] = None,
) -> RunStatusResponse:
    """Retrieve execution status and outcome for a given run ID."""
    record = run_manager.get_run(run_id)
    if record is None:
        if session is not None:
            try:
                db_run = get_db_run(session, uuid.UUID(run_id))
                if db_run is not None:
                    return RunStatusResponse(
                        run_id=str(db_run.id),
                        status=db_run.status,
                    )
            except Exception:
                pass
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    if wait and record.status in ("pending", "running"):
        try:
            import asyncio

            await asyncio.wait_for(record.completed_event.wait(), timeout=timeout)
        except TimeoutError:
            pass

    outcome_dict: dict[str, Any] | None = None
    if record.outcome is not None:
        if hasattr(record.outcome, "model_dump"):
            try:
                outcome_dict = record.outcome.model_dump(mode="json")
            except Exception:
                outcome_dict = {
                    "status": record.status,
                    "stop_reason": record.stop_reason,
                    "rounds": record.rounds,
                }
        elif isinstance(record.outcome, dict):
            outcome_dict = record.outcome
        else:
            outcome_dict = {
                "status": record.status,
                "stop_reason": record.stop_reason,
            }

    return RunStatusResponse(
        run_id=record.run_id,
        status=record.status,
        stop_reason=record.stop_reason,
        rounds=record.rounds,
        outcome=outcome_dict,
        error=record.error,
    )


@router.get(
    "/{run_id}/report",
    response_model=RunReportResponse,
    summary="Get final synthesized research report",
)
async def get_run_report(
    run_id: str,
    run_manager: Annotated[RunManager, Depends(get_run_manager)],
    wait: bool = Query(
        default=False,
        description="Wait for run completion before returning report",
    ),
    timeout: float = Query(
        default=10.0,
        description="Maximum seconds to wait if wait=True",
    ),
) -> RunReportResponse:
    """Retrieve synthesized research report or abstention report for run."""
    record = run_manager.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    if wait and record.status in ("pending", "running"):
        try:
            import asyncio

            await asyncio.wait_for(record.completed_event.wait(), timeout=timeout)
        except TimeoutError:
            pass

    if record.status in ("pending", "running") and record.report is None:
        raise HTTPException(
            status_code=404,
            detail=f"Report not ready yet for run '{run_id}' (status: {record.status})",
        )

    if record.report is None:
        raise HTTPException(
            status_code=404,
            detail=f"No report available for run '{run_id}' (status: {record.status})",
        )

    conf_score = (
        record.confidence_breakdown.confidence_score
        if record.confidence_breakdown is not None
        else None
    )
    is_abstention = (
        record.confidence_breakdown.is_abstention
        if record.confidence_breakdown is not None
        else False
    )

    return RunReportResponse(
        run_id=record.run_id,
        markdown=record.report.markdown,
        status=record.status,
        validation_passed=record.report.validation_passed,
        confidence_score=conf_score,
        is_abstention=is_abstention,
        sentence_to_claims=record.report.sentence_to_claims,
        claims_used=record.report.claims_used,
    )


@router.get(
    "/{run_id}/claims",
    response_model=list[ClaimResponse],
    summary="Get all claims and evidence produced by run",
)
async def get_run_claims(
    run_id: str,
    run_manager: Annotated[RunManager, Depends(get_run_manager)],
    session: Annotated[Session | None, Depends(get_optional_db_session)] = None,
) -> list[ClaimResponse]:
    """Retrieve all claims associated with a given research run."""
    record = run_manager.get_run(run_id)
    if record is None:
        if session is not None:
            try:
                db_run = get_db_run(session, uuid.UUID(run_id))
                if db_run is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Run '{run_id}' not found",
                    )
            except ValueError:
                raise HTTPException(
                    status_code=404,
                    detail=f"Run '{run_id}' not found",
                ) from None
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Run '{run_id}' not found",
            )

    claims: list[Any] = []
    if session is not None:
        try:
            ledger = LedgerRepository(session)
            claims = ledger.get_claims(run_id=run_id)
        except Exception:
            pass

    if not claims and record is not None:
        claims = record.claims

    return [claim_to_response(c) for c in claims]


@router.get(
    "/{run_id}/events",
    summary="Stream research run events as Server-Sent Events (SSE)",
)
async def get_run_events(
    run_id: str,
    run_manager: Annotated[RunManager, Depends(get_run_manager)],
    event_bus: Annotated[EventBus, Depends(get_event_bus)],
) -> StreamingResponse:
    """Stream real-time and historical events for run via SSE protocol."""
    record = run_manager.get_run(run_id)
    if record is None and not event_bus.get_history(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    async def event_generator() -> AsyncIterator[str]:
        async for event in event_bus.subscribe(run_id):
            yield event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/{run_id}/trace",
    response_model=RunTraceResponse,
    summary="Get execution spans and cumulative resource cost totals",
)
async def get_run_trace(
    run_id: str,
    run_manager: Annotated[RunManager, Depends(get_run_manager)],
    session: Annotated[Session | None, Depends(get_optional_db_session)] = None,
) -> RunTraceResponse:
    """Retrieve execution trace spans and resource consumption totals."""
    record = run_manager.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    spans_data: list[SpanResponse] = []
    if record.tracer and record.tracer.spans:
        for s in record.tracer.spans:
            input_val = (
                s.input_json
                if s.input_json is not None
                else s.metadata.get("input_json")
            )
            output_val = (
                s.output_json
                if s.output_json is not None
                else s.metadata.get("output_json")
            )
            spans_data.append(
                SpanResponse(
                    name=s.name,
                    id=str(s.id),
                    parent_id=str(s.parent_id) if s.parent_id else None,
                    start_time=s.start_time.isoformat(),
                    end_time=s.end_time.isoformat() if s.end_time else None,
                    latency_ms=s.latency_ms,
                    status=s.status,
                    error=s.error,
                    span_type=s.span_type,
                    tool_name=s.tool_name,
                    input_json=input_val,
                    output_json=output_val,
                    metadata=dict(s.metadata or {}),
                )
            )

    # Reconcile or load tool calls from DB if available
    if session is not None:
        try:
            run_uuid = uuid.UUID(run_id)
            db_run = get_db_run(session, run_uuid)
            if db_run and db_run.tasks:
                # If spans_data is empty (e.g. loaded from DB), populate
                # from DB agent runs and tool calls
                if not spans_data:
                    for task in db_run.tasks:
                        for ar in task.agent_runs:
                            agent_span = SpanResponse(
                                name=f"agent:{task.agent_type}",
                                id=str(ar.id),
                                parent_id=None,
                                start_time=(
                                    ar.created_at or db_run.created_at
                                ).isoformat(),
                                end_time=(ar.finished_at).isoformat()
                                if ar.finished_at
                                else None,
                                latency_ms=None,
                                status=ar.status,
                                error=None,
                                span_type="agent",
                                input_json=None,
                                output_json=None,
                                metadata={
                                    "tokens_in": ar.tokens_in,
                                    "tokens_out": ar.tokens_out,
                                },
                            )
                            spans_data.append(agent_span)
                            for tc in ar.tool_calls:
                                tc_span = SpanResponse(
                                    name=tc.tool_name,
                                    id=str(tc.id),
                                    parent_id=str(ar.id),
                                    start_time=tc.created_at.isoformat(),
                                    end_time=None,
                                    latency_ms=tc.latency_ms,
                                    status="error" if tc.error else "ok",
                                    error=tc.error,
                                    span_type="tool",
                                    tool_name=tc.tool_name,
                                    input_json=tc.input_json,
                                    output_json=tc.output_json,
                                    metadata={"cache_hit": tc.cache_hit},
                                )
                                spans_data.append(tc_span)
        except Exception as e:
            logger.warning("Trace DB enrichment failed", error=str(e))

    budget = record.budget_manager
    total_tokens = budget.used_tokens if budget else 0
    tokens_in = total_tokens // 2
    tokens_out = total_tokens - tokens_in
    tool_calls = budget.used_tool_calls if budget else 0

    if (
        isinstance(record.outcome, RunOutcome)
        and record.outcome.state
        and record.outcome.state.results
    ):
        ti = 0
        to = 0
        tc = 0
        for res in record.outcome.state.results.values():
            if isinstance(res, TaskResult) and res.usage:
                ti += int(res.usage.get("tokens_in", 0))
                to += int(res.usage.get("tokens_out", 0))
                tc += int(res.usage.get("tool_calls", 0))
        if ti > 0 or to > 0:
            tokens_in = ti
            tokens_out = to
            total_tokens = ti + to
        if tc > tool_calls:
            tool_calls = tc

    if record.tracer:
        span_tool_count = sum(
            1 for s in record.tracer.spans if s.span_type == "tool" or s.tool_name
        )
        if span_tool_count > tool_calls:
            tool_calls = span_tool_count

    cost_totals = CostTotalsResponse(
        total_tokens=total_tokens,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tool_calls=tool_calls,
        elapsed_seconds=(budget.used_seconds if budget else 0.0),
        estimated_cost_usd=(budget.used_cost if budget else 0.0),
    )

    return RunTraceResponse(
        run_id=record.run_id,
        spans=spans_data,
        cost_totals=cost_totals,
    )
