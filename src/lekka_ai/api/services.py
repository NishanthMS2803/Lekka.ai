"""Service layer managing research run lifecycle, storage, and execution."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy.orm import Session

from lekka_ai.db.ledger import LedgerRepository
from lekka_ai.db.repository import create_run, get_run
from lekka_ai.events import EventBus, default_event_bus
from lekka_ai.llm.base import LLMClient
from lekka_ai.logging import get_logger
from lekka_ai.orchestrator.budget import RunBudget
from lekka_ai.orchestrator.dispatch import TaskResult
from lekka_ai.orchestrator.engine import (
    RunOutcome,
    TaskRunnerCallable,
    run_engine,
)
from lekka_ai.report.confidence import (
    ConfidenceBreakdown,
    synthesize_report_with_confidence,
)
from lekka_ai.report.synthesize import SynthesizedReport
from lekka_ai.schemas.claims import Claim
from lekka_ai.tools.base import ToolRegistry
from lekka_ai.tracing import Tracer

logger = get_logger(__name__)


@dataclass
class RunRecord:
    """In-memory tracking record for a research run."""

    run_id: str
    question: str
    as_of_date: str | None
    budget: dict[str, Any] | None
    status: str = "pending"
    stop_reason: str | None = None
    rounds: int = 0
    outcome: RunOutcome | dict[str, Any] | None = None
    report: SynthesizedReport | None = None
    confidence_breakdown: ConfidenceBreakdown | None = None
    claims: list[Claim] = field(default_factory=list)
    tracer: Tracer | None = None
    budget_manager: RunBudget | None = None
    error: str | None = None
    completed_event: asyncio.Event = field(default_factory=asyncio.Event)


class RunManager:
    """Manager coordinating background runs, outcomes, reports, and claims."""

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._claims_by_id: dict[str, Claim] = {}
        self.event_bus = event_bus or default_event_bus

    def create_run(
        self,
        *,
        run_id: str,
        question: str,
        as_of_date: str | date | None = None,
        budget: dict[str, Any] | None = None,
    ) -> RunRecord:
        """Create and register a new pending run record."""
        as_of_str = str(as_of_date) if as_of_date is not None else None
        record = RunRecord(
            run_id=run_id,
            question=question,
            as_of_date=as_of_str,
            budget=budget,
        )
        self._runs[run_id] = record
        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        """Retrieve run record by ID."""
        return self._runs.get(run_id)

    def store_claims(self, run_id: str, claims: list[Claim]) -> None:
        """Associate claims with a run and index by claim ID."""
        record = self._runs.get(run_id)
        if record is not None:
            record.claims.extend(claims)
        for c in claims:
            self._claims_by_id[c.claim_id] = c

    def get_claim(self, claim_id: str) -> Claim | None:
        """Retrieve a cached claim by claim ID."""
        return self._claims_by_id.get(claim_id)

    def get_claims(self, run_id: str) -> list[Claim]:
        """Retrieve claims for a run."""
        record = self._runs.get(run_id)
        if record is not None:
            return record.claims
        return []

    async def execute_run(
        self,
        run_id: str,
        *,
        llm: LLMClient | None = None,
        task_runner: TaskRunnerCallable | None = None,
        tool_registry: ToolRegistry | None = None,
        session: Session | None = None,
        available_profiles: Any = None,
    ) -> None:
        """Execute the research engine loop and synthesize report."""
        record = self.get_run(run_id)
        if record is None:
            return

        record.status = "running"
        logger.info("Executing background run", run_id=run_id, question=record.question)

        # Sync to DB if session is available
        if session is not None:
            try:
                run_uuid = uuid.UUID(run_id)
                db_run = get_run(session, run_uuid)
                if db_run is None:
                    parsed_date = (
                        date.fromisoformat(record.as_of_date)
                        if record.as_of_date
                        else None
                    )
                    create_run(
                        session,
                        question=record.question,
                        status="running",
                        as_of_date=parsed_date,
                        budget_json=record.budget or {},
                    )
                else:
                    db_run.status = "running"
                    session.commit()
            except Exception as e:
                logger.warning("DB run sync initial update failed", error=str(e))

        # Setup tracer
        tracer = Tracer(session=session, run_id=run_id, event_bus=self.event_bus)
        record.tracer = tracer

        # Build budget
        budget: RunBudget
        if record.budget:
            budget = RunBudget(tracer=tracer, **record.budget)
        else:
            budget = RunBudget.from_settings(tracer=tracer)
        record.budget_manager = budget

        # Emit run_started event
        await self.event_bus.emit(
            run_id,
            "run_started",
            {
                "question": record.question,
                "as_of_date": record.as_of_date,
            },
        )

        ledger = LedgerRepository(session) if session is not None else None
        try:
            outcome = await run_engine(
                question=record.question,
                budget=budget,
                llm=llm,
                task_runner=task_runner,
                tool_registry=tool_registry,
                ledger=ledger,
                session=session,
                tracer=tracer,
                available_profiles=available_profiles,
                event_bus=self.event_bus,
                run_id=run_id,
            )

            # Collect claims from all completed tasks
            all_claims: list[Claim] = []
            if outcome.state and outcome.state.results:
                for res in outcome.state.results.values():
                    claims_raw: list[Any] = []
                    if isinstance(res, TaskResult):
                        claims_raw = res.claims
                    elif isinstance(res, dict):
                        claims_raw = res.get("claims", [])
                    elif hasattr(res, "claims"):
                        claims_raw = getattr(res, "claims", [])

                    for c in claims_raw:
                        if isinstance(c, Claim):
                            all_claims.append(c)
                        elif isinstance(c, dict):
                            try:
                                all_claims.append(Claim.model_validate(c))
                            except Exception:
                                pass

            # Also query ledger for any stored claims
            if ledger is not None:
                try:
                    db_claims = ledger.get_claims(run_id=run_id)
                    existing_ids = {c.claim_id for c in all_claims}
                    for dbc in db_claims:
                        if dbc.claim_id not in existing_ids:
                            all_claims.append(dbc.to_schema())
                except Exception as e:
                    logger.warning("Ledger fetch claims failed", error=str(e))

            self.store_claims(run_id, all_claims)

            if outcome.status == "failed":
                record.status = "failed"
                record.error = outcome.error or "Run execution failed"
                record.outcome = outcome
                if session is not None:
                    try:
                        db_run = get_run(session, uuid.UUID(run_id))
                        if db_run is not None:
                            db_run.status = "failed"
                            db_run.finished_at = datetime.now(UTC)
                            session.commit()
                    except Exception as e:
                        logger.warning("DB run update failed", error=str(e))
                await self.event_bus.emit(
                    run_id,
                    "run_failed",
                    {"error": record.error},
                )
                return

            # Extract ticker if available
            ticker: str | None = None
            if outcome.state and outcome.state.parsed_query:
                tickers = outcome.state.parsed_query.get("tickers")
                if tickers and isinstance(tickers, list) and len(tickers) > 0:
                    ticker = str(tickers[0])

            # Synthesize final report with confidence & abstention checks
            report, breakdown = await synthesize_report_with_confidence(
                claims=all_claims,
                ticker=ticker,
                question=record.question,
                llm=llm,
            )

            record.status = outcome.status
            record.stop_reason = outcome.stop_reason
            record.rounds = outcome.rounds
            record.outcome = outcome
            record.report = report
            record.confidence_breakdown = breakdown
            record.error = outcome.error

            if session is not None:
                try:
                    db_run = get_run(session, uuid.UUID(run_id))
                    if db_run is not None:
                        db_run.status = outcome.status
                        db_run.finished_at = datetime.now(UTC)
                        session.commit()
                except Exception as e:
                    logger.warning("DB run update final status failed", error=str(e))

            # Emit verification_done and report_ready events
            await self.event_bus.emit(
                run_id,
                "verification_done",
                {
                    "confidence_score": breakdown.confidence_score,
                    "is_abstention": breakdown.is_abstention,
                    "missing_evidence": breakdown.missing_evidence,
                },
            )
            await self.event_bus.emit(
                run_id,
                "report_ready",
                {
                    "status": record.status,
                    "validation_passed": report.validation_passed,
                    "confidence_score": breakdown.confidence_score,
                    "is_abstention": breakdown.is_abstention,
                },
            )

            logger.info(
                "Run execution finished",
                run_id=run_id,
                status=record.status,
                stop_reason=record.stop_reason,
            )

        except Exception as exc:
            err_msg = str(exc)
            logger.error("Run execution failed", run_id=run_id, error=err_msg)
            record.status = "failed"
            record.error = err_msg
            await self.event_bus.emit(
                run_id,
                "run_failed",
                {"error": err_msg},
            )
        finally:
            record.completed_event.set()


# Default singleton manager instance
default_run_manager = RunManager()
