"""Main orchestration engine executing the re-plan loop with stopping rules."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from lekka_ai.db.ledger import LedgerRepository
from lekka_ai.harness.permissions import AgentProfile
from lekka_ai.llm.base import LLMClient
from lekka_ai.logging import get_logger
from lekka_ai.orchestrator.budget import RunBudget
from lekka_ai.orchestrator.dispatch import TaskResult, run_task
from lekka_ai.orchestrator.gapcheck import Decision, GapCheckAction, gap_check
from lekka_ai.orchestrator.planner import plan_query
from lekka_ai.orchestrator.state import PlannedTask, RunState, ready_tasks
from lekka_ai.orchestrator.understand import ParsedQuery, understand_query
from lekka_ai.tools.base import ToolRegistry
from lekka_ai.tracing import Tracer

logger = get_logger(__name__)

TaskRunnerCallable = Callable[[PlannedTask, Path], Awaitable[Any]]


class RunOutcome(BaseModel):
    """Final outcome and metadata of an orchestrated multi-agent research run."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    status: str = Field(
        ...,
        description=(
            "Terminal execution status ('completed', 'budget_exhausted', "
            "'max_rounds_reached', 'abstained', 'needs_clarification', 'failed')"
        ),
    )
    stop_reason: str = Field(
        ...,
        description=(
            "Categorical stopping condition: 'done', 'max_rounds', "
            "'budget_exhausted', 'no_new_evidence', 'abstain', 'needs_clarification'"
        ),
    )
    state: RunState = Field(
        ...,
        description=(
            "Final RunState containing execution plan, task results, and metadata"
        ),
    )
    rounds: int = Field(
        default=1,
        description="Total orchestration wave rounds executed",
    )
    decision: Decision | None = Field(
        default=None,
        description="Final parent gap-check decision if evaluated",
    )
    budget: RunBudget | None = Field(
        default=None,
        description="RunBudget tracking token, tool call, time, and USD consumption",
    )
    error: str | None = Field(
        default=None,
        description="Error message if run terminated due to exception",
    )

    @property
    def is_successful(self) -> bool:
        """Return True if run completed normally with answers or stable evidence."""
        return self.status == "completed" and self.stop_reason in (
            "done",
            "no_new_evidence",
        )


def _extract_claim_signatures(state: RunState) -> set[str]:
    """Extract normalized claim text signatures across all completed task results."""
    signatures: set[str] = set()
    for res in state.results.values():
        claims: list[Any] = []
        if isinstance(res, TaskResult):
            claims = res.claims
        elif isinstance(res, dict):
            claims = res.get("claims", [])
        elif hasattr(res, "claims"):
            claims = getattr(res, "claims", [])

        for c in claims:
            if isinstance(c, dict):
                text = c.get("text", "")
            elif hasattr(c, "text"):
                text = getattr(c, "text", "")
            else:
                text = str(c)
            clean = text.strip().lower()
            if clean:
                signatures.add(clean)
    return signatures


class OrchestrationEngine:
    """Multi-wave orchestration engine coordinating the iterative research loop.

    Workflow:
    1. Understand: Parse user query into structured intent, tickers, and constraints.
    2. Plan: Decompose query into an initial dependency DAG of specialist tasks.
    3. Run Wave: Dispatch ready tasks concurrently across isolated workspaces.
    4. Gap Check: Evaluate findings for factual conflicts, unsupported assumptions,
       missing evidence, and low confidence.
    5. Re-plan or Terminate:
       - 'done' -> Stop run.
       - 'abstain' -> Stop run.
       - 'follow_up' -> Schedule follow-up tasks into the plan and run next wave.

    Hard Stopping Rules:
    - max_rounds (default 3): Halts when round cap is reached.
    - budget exhaustion: Halts when token, tool call, or cost limits are exhausted.
    - 'no new evidence': Halts when follow-up wave yields no novel grounded claims.
    """

    def __init__(
        self,
        *,
        max_rounds: int = 3,
        max_concurrency: int = 5,
        default_timeout: float | None = 60.0,
        budget: RunBudget | None = None,
        llm: LLMClient | None = None,
        understand_llm: LLMClient | None = None,
        planner_llm: LLMClient | None = None,
        agent_llm: LLMClient | None = None,
        gapcheck_llm: LLMClient | None = None,
        task_runner: TaskRunnerCallable | None = None,
        tool_registry: ToolRegistry | None = None,
        ledger: LedgerRepository | None = None,
        session: Session | None = None,
        tracer: Tracer | None = None,
        event_bus: Any | None = None,
        run_id: str | None = None,
        fault_plan: Any | None = None,
    ) -> None:
        """Initialize OrchestrationEngine with configuration and dependencies."""
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")

        self.max_rounds = max_rounds
        self.max_concurrency = max_concurrency
        self.default_timeout = default_timeout
        self.budget = budget
        self.llm = llm
        self.understand_llm = understand_llm or llm
        self.planner_llm = planner_llm or llm
        self.agent_llm = agent_llm or llm
        self.gapcheck_llm = gapcheck_llm or llm
        self.task_runner = task_runner
        self.tool_registry = tool_registry
        self.ledger = ledger
        self.session = session
        self.tracer = tracer
        self.event_bus = event_bus
        self.run_id = run_id
        self.fault_plan = fault_plan

    async def run(
        self,
        question: str,
        *,
        initial_state: RunState | None = None,
        available_profiles: dict[str, AgentProfile] | Iterable[str] | None = None,
    ) -> RunOutcome:
        """Execute the re-plan orchestration loop for a research question.

        Args:
            question: User research question or investment prompt.
            initial_state: Optional pre-configured RunState (bypasses understand/plan).
            available_profiles: Permitted specialist profiles for planning.

        Returns:
            RunOutcome containing terminal status, stop reason, final state, and budget.
        """
        # Resolve shared runtime budget
        effective_budget = self.budget or RunBudget.from_settings(tracer=self.tracer)

        # -------------------------------------------------------------
        # Step 1 & 2: Query Understanding and Initial Planning
        # -------------------------------------------------------------
        state: RunState
        if initial_state is not None:
            state = initial_state
        else:
            logger.info("Executing query understanding", question=question)
            parsed_query: ParsedQuery
            try:
                parsed_query = await understand_query(
                    question=question,
                    llm=self.understand_llm,
                )
            except Exception as exc:
                err_msg = f"Query understanding failed: {exc}"
                logger.error(err_msg, exc_info=True)
                return RunOutcome(
                    status="failed",
                    stop_reason="error",
                    state=RunState(question=question),
                    rounds=0,
                    budget=effective_budget,
                    error=err_msg,
                )

            if parsed_query.needs_clarification:
                logger.info(
                    "Query requires user clarification; stopping before execution",
                    ambiguities=parsed_query.ambiguities,
                )
                return RunOutcome(
                    status="needs_clarification",
                    stop_reason="needs_clarification",
                    state=RunState(
                        question=question,
                        parsed_query=parsed_query.model_dump(),
                    ),
                    rounds=0,
                    budget=effective_budget,
                )

            logger.info("Generating initial research plan", intent=parsed_query.intent)
            try:
                plan = await plan_query(
                    query=parsed_query,
                    available_profiles=available_profiles,
                    llm=self.planner_llm,
                )
            except Exception as exc:
                err_msg = f"Planning failed: {exc}"
                logger.error(err_msg, exc_info=True)
                return RunOutcome(
                    status="failed",
                    stop_reason="error",
                    state=RunState(
                        question=question,
                        parsed_query=parsed_query.model_dump(),
                    ),
                    rounds=0,
                    budget=effective_budget,
                    error=err_msg,
                )

            state = RunState(
                run_id=self.run_id or str(uuid.uuid4()),
                question=question,
                parsed_query=parsed_query.model_dump(),
                plan=plan,
            )

        if self.run_id and state.run_id != self.run_id:
            state.run_id = self.run_id

        if self.event_bus and state.run_id:
            await self.event_bus.emit(
                state.run_id,
                "plan_created",
                {
                    "task_count": len(state.plan.tasks),
                    "tasks": [t.model_dump() for t in state.plan.tasks],
                    "intent": state.parsed_query.get("intent")
                    if state.parsed_query
                    else None,
                    "tickers": state.parsed_query.get("tickers", [])
                    if state.parsed_query
                    else [],
                },
            )

        # -------------------------------------------------------------
        # Step 3, 4, 5: Iterative Re-plan Execution Loop
        # -------------------------------------------------------------
        round_number = 1
        last_decision: Decision | None = None
        claims_seen: set[str] = _extract_claim_signatures(state)

        # Build task execution runner
        async def _default_dispatch_runner(
            task: PlannedTask,
            scratch_dir: Path,
        ) -> TaskResult:
            return await run_task(
                planned_task=task,
                state=state,
                budget=effective_budget,
                llm=self.agent_llm,
                tool_registry=self.tool_registry,
                ledger=self.ledger,
                session=self.session,
                tracer=self.tracer,
                fault_plan=self.fault_plan,
            )

        raw_runner = self.task_runner or _default_dispatch_runner

        async def _event_emitting_runner(
            task: PlannedTask,
            scratch_dir: Path,
        ) -> TaskResult:
            if self.event_bus and state.run_id:
                await self.event_bus.emit(
                    state.run_id,
                    "task_started",
                    {
                        "task_id": task.task_id,
                        "agent_type": task.agent_type,
                        "instruction": task.instruction,
                        "priority": task.priority,
                    },
                )

            res = await raw_runner(task, scratch_dir)

            if self.event_bus and state.run_id:
                tool_calls_count = res.usage.get("tool_calls", 0) if res.usage else 0
                if tool_calls_count > 0:
                    await self.event_bus.emit(
                        state.run_id,
                        "tool_called",
                        {
                            "task_id": task.task_id,
                            "agent_type": task.agent_type,
                            "count": tool_calls_count,
                        },
                    )

                await self.event_bus.emit(
                    state.run_id,
                    "task_completed",
                    {
                        "task_id": task.task_id,
                        "agent_type": task.agent_type,
                        "status": res.status,
                        "claims_count": len(res.claims),
                    },
                )
            return res

        effective_runner = _event_emitting_runner
        from lekka_ai.harness.scheduler import TaskScheduler

        scheduler = TaskScheduler(
            max_concurrency=self.max_concurrency,
            default_timeout=self.default_timeout,
            task_runner=effective_runner,
            session=self.session,
        )

        while True:
            state.round_number = round_number
            logger.info(
                "Starting orchestration round",
                round=round_number,
                max_rounds=self.max_rounds,
            )

            # Stopping Rule Check A: Hard round limit reached before launch
            if round_number > self.max_rounds:
                logger.warning(
                    "Maximum round limit reached; stopping re-plan loop",
                    max_rounds=self.max_rounds,
                )
                return RunOutcome(
                    status="max_rounds_reached",
                    stop_reason="max_rounds",
                    state=state,
                    rounds=round_number - 1,
                    decision=last_decision,
                    budget=effective_budget,
                )

            # Stopping Rule Check B: Budget exhaustion before launch
            if effective_budget.stop_launching():
                logger.warning(
                    "Budget exhausted; stopping task launches",
                    remaining=effective_budget.remaining().model_dump(),
                )
                return RunOutcome(
                    status="budget_exhausted",
                    stop_reason="budget_exhausted",
                    state=state,
                    rounds=round_number - 1 if round_number > 1 else 1,
                    decision=last_decision,
                    budget=effective_budget,
                )

            # ---------------------------------------------------------
            # Run Wave: Execute ready tasks until plan wave settles
            # ---------------------------------------------------------
            wave_executed_any = False
            while True:
                ready = ready_tasks(state.plan, state.task_statuses)
                if not ready:
                    break

                if effective_budget.stop_launching():
                    logger.warning("Budget exhausted mid-wave; halting ready tasks")
                    return RunOutcome(
                        status="budget_exhausted",
                        stop_reason="budget_exhausted",
                        state=state,
                        rounds=round_number,
                        decision=last_decision,
                        budget=effective_budget,
                    )

                logger.info(
                    "Executing parallel task wave",
                    round=round_number,
                    ready_count=len(ready),
                    task_ids=[t.task_id for t in ready],
                )
                await scheduler.run_tasks(ready, state=state)
                wave_executed_any = True

            # Stopping Rule Check C: Budget exhausted immediately after wave
            if effective_budget.stop_launching():
                logger.warning("Budget exhausted after wave execution")
                return RunOutcome(
                    status="budget_exhausted",
                    stop_reason="budget_exhausted",
                    state=state,
                    rounds=round_number,
                    decision=last_decision,
                    budget=effective_budget,
                )

            # Stopping Rule Check D: 'No new evidence' detection on follow-up rounds
            current_claims = _extract_claim_signatures(state)
            new_claims = current_claims - claims_seen
            if round_number > 1 and wave_executed_any and len(new_claims) == 0:
                logger.info(
                    "Follow-up wave produced no novel evidence; stopping re-plan loop",
                    round=round_number,
                )
                return RunOutcome(
                    status="completed",
                    stop_reason="no_new_evidence",
                    state=state,
                    rounds=round_number,
                    decision=last_decision,
                    budget=effective_budget,
                )

            # Update tracked claims
            claims_seen = current_claims

            # ---------------------------------------------------------
            # Gap Check: Audit wave findings
            # ---------------------------------------------------------
            logger.info("Executing parent gap check audit", round=round_number)
            decision: Decision
            try:
                decision = await gap_check(
                    state=state,
                    question=state.question,
                    llm=self.gapcheck_llm,
                )
            except Exception as exc:
                err_msg = f"Gap check evaluation failed: {exc}"
                logger.error(err_msg, exc_info=True)
                return RunOutcome(
                    status="failed",
                    stop_reason="error",
                    state=state,
                    rounds=round_number,
                    decision=last_decision,
                    budget=effective_budget,
                    error=err_msg,
                )

            last_decision = decision
            action = decision.action

            if self.event_bus and state.run_id:
                await self.event_bus.emit(
                    state.run_id,
                    "gap_check",
                    {
                        "round": round_number,
                        "action": decision.action.value,
                        "reasons": decision.reasons,
                        "follow_up_tasks": [
                            t.model_dump() for t in decision.follow_up_tasks
                        ],
                    },
                )

            # Decision Action 1: DONE
            if action == GapCheckAction.DONE:
                logger.info("Gap check audit concluded: DONE", round=round_number)
                return RunOutcome(
                    status="completed",
                    stop_reason="done",
                    state=state,
                    rounds=round_number,
                    decision=decision,
                    budget=effective_budget,
                )

            # Decision Action 2: ABSTAIN
            if action == GapCheckAction.ABSTAIN:
                logger.info("Gap check audit concluded: ABSTAIN", round=round_number)
                return RunOutcome(
                    status="abstained",
                    stop_reason="abstain",
                    state=state,
                    rounds=round_number,
                    decision=decision,
                    budget=effective_budget,
                )

            # Decision Action 3: FOLLOW_UP
            logger.info(
                "Gap check requested follow-up wave",
                round=round_number,
                follow_up_count=len(decision.follow_up_tasks),
            )

            # Stopping Rule Check E: Max rounds reached at follow-up decision
            if round_number >= self.max_rounds:
                logger.warning(
                    "Max rounds reached upon follow-up decision; terminating",
                    max_rounds=self.max_rounds,
                )
                return RunOutcome(
                    status="max_rounds_reached",
                    stop_reason="max_rounds",
                    state=state,
                    rounds=round_number,
                    decision=decision,
                    budget=effective_budget,
                )

            # Stopping Rule Check F: Budget exhausted upon follow-up decision
            if effective_budget.stop_launching():
                logger.warning("Budget exhausted upon follow-up decision; terminating")
                return RunOutcome(
                    status="budget_exhausted",
                    stop_reason="budget_exhausted",
                    state=state,
                    rounds=round_number,
                    decision=decision,
                    budget=effective_budget,
                )

            if not decision.follow_up_tasks:
                logger.info("Follow-up decision specified no tasks; ending loop")
                return RunOutcome(
                    status="completed",
                    stop_reason="done",
                    state=state,
                    rounds=round_number,
                    decision=decision,
                    budget=effective_budget,
                )

            # Add follow-up tasks to the plan for the next round
            for follow_up_spec in decision.to_planned_tasks():
                state.plan.add_task(follow_up_spec)

            round_number += 1


async def run_engine(
    question: str,
    *,
    initial_state: RunState | None = None,
    budget: RunBudget | None = None,
    max_rounds: int = 3,
    max_concurrency: int = 5,
    default_timeout: float | None = 60.0,
    llm: LLMClient | None = None,
    understand_llm: LLMClient | None = None,
    planner_llm: LLMClient | None = None,
    agent_llm: LLMClient | None = None,
    gapcheck_llm: LLMClient | None = None,
    task_runner: TaskRunnerCallable | None = None,
    tool_registry: ToolRegistry | None = None,
    ledger: LedgerRepository | None = None,
    session: Session | None = None,
    tracer: Tracer | None = None,
    available_profiles: dict[str, AgentProfile] | Iterable[str] | None = None,
    event_bus: Any | None = None,
    run_id: str | None = None,
    fault_plan: Any | None = None,
) -> RunOutcome:
    """Run the complete multi-wave orchestration engine loop.

    Convenience wrapper around OrchestrationEngine.run().
    """
    engine = OrchestrationEngine(
        max_rounds=max_rounds,
        max_concurrency=max_concurrency,
        default_timeout=default_timeout,
        budget=budget,
        llm=llm,
        understand_llm=understand_llm,
        planner_llm=planner_llm,
        agent_llm=agent_llm,
        gapcheck_llm=gapcheck_llm,
        task_runner=task_runner,
        tool_registry=tool_registry,
        ledger=ledger,
        session=session,
        tracer=tracer,
        event_bus=event_bus,
        run_id=run_id,
        fault_plan=fault_plan,
    )
    return await engine.run(
        question=question,
        initial_state=initial_state,
        available_profiles=available_profiles,
    )


run_orchestrator = run_engine
