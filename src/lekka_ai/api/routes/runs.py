"""FastAPI routes for research runs management and reports."""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from lekka_ai.api.dependencies import (
    get_llm,
    get_optional_db_session,
    get_run_manager,
    get_task_runner,
    get_tool_registry,
)
from lekka_ai.api.models import (
    ClaimResponse,
    CreateRunRequest,
    RunCreateResponse,
    RunReportResponse,
    RunStatusResponse,
    claim_to_response,
)
from lekka_ai.api.services import RunManager
from lekka_ai.db.ledger import LedgerRepository
from lekka_ai.db.repository import get_run as get_db_run
from lekka_ai.llm.base import LLMClient
from lekka_ai.orchestrator.engine import TaskRunnerCallable
from lekka_ai.tools.base import ToolRegistry

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
            outcome_dict = record.outcome.model_dump()
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
