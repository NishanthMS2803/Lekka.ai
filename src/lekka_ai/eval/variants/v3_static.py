"""V3 Static Variant: all five specialist agents always run, one round, no gap check."""

from __future__ import annotations

import time
import uuid
from typing import Any

from lekka_ai.eval.dataset import EvalExample
from lekka_ai.eval.variant import RunResult
from lekka_ai.eval.variants.base import (
    BaseEngineVariant,
    VariantConfig,
)
from lekka_ai.llm.base import LLMClient, LLMResponse
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.orchestrator.budget import RunBudget
from lekka_ai.orchestrator.engine import OrchestrationEngine, RunOutcome
from lekka_ai.orchestrator.state import Plan, PlannedTask, RunState
from lekka_ai.tools.base import ToolRegistry


class V3StaticVariant(BaseEngineVariant):
    """V3 Static: executes all 5 specialist agents in 1 round without gap check."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        config: VariantConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        name: str = "v3_static",
        task_runner: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            llm=llm,
            config=config,
            tool_registry=tool_registry,
            description=(
                "V3 Static variant: all specialist agents always execute in one "
                "fixed round without gap check re-planning."
            ),
        )
        self.task_runner = task_runner

    def _build_static_plan(self, question: str, run_id: str) -> Plan:
        """Construct the static plan containing all specialist agents."""
        plan = Plan()

        # Phase 1: Data gathering agents
        plan.add_task(
            PlannedTask(
                task_id=f"{run_id}_fund",
                agent_type="fundamental",
                instruction=(
                    f"Extract fundamental metrics and 10-K disclosures: {question}"
                ),
                priority=1,
            )
        )
        plan.add_task(
            PlannedTask(
                task_id=f"{run_id}_mkt",
                agent_type="market",
                instruction=(
                    f"Gather price history, volatility, and trading volume: {question}"
                ),
                priority=1,
            )
        )
        plan.add_task(
            PlannedTask(
                task_id=f"{run_id}_news",
                agent_type="news",
                instruction=(
                    "Retrieve news articles, press releases, and commentary: "
                    f"{question}"
                ),
                priority=1,
            )
        )

        # Phase 2: Synthesis & analysis agents
        plan.add_task(
            PlannedTask(
                task_id=f"{run_id}_val",
                agent_type="valuation",
                instruction=(
                    f"Compute valuation multiples and DCF intrinsic value: {question}"
                ),
                priority=2,
                depends_on=[f"{run_id}_fund", f"{run_id}_mkt"],
            )
        )

        if self.config.risk_agent:
            plan.add_task(
                PlannedTask(
                    task_id=f"{run_id}_risk",
                    agent_type="risk",
                    instruction=(
                        f"Identify and rank operational and market risks: {question}"
                    ),
                    priority=2,
                    depends_on=[f"{run_id}_fund", f"{run_id}_mkt"],
                )
            )

        return plan

    async def run(
        self,
        question: str | EvalExample,
        *,
        as_of_date: str | None = None,
        example_id: str | None = None,
        **kwargs: Any,
    ) -> RunResult:
        """Execute static 5-agent wave in one round with no gap check."""
        q_text = question.question if isinstance(question, EvalExample) else question
        ex_id = (
            question.id
            if isinstance(question, EvalExample)
            else (example_id or f"v3_{uuid.uuid4().hex[:8]}")
        )
        run_id = f"run_{uuid.uuid4().hex[:8]}"

        static_plan = self._build_static_plan(q_text, run_id)
        state = RunState(
            run_id=run_id,
            question=q_text,
            plan=static_plan,
        )

        budget = RunBudget.from_settings()
        effective_task_runner = self.task_runner or self.create_task_runner(
            state, budget
        )

        # Gap check bypass: immediate done signal without invoking LLM
        no_gapcheck_llm = FakeLLM(
            responses=[
                LLMResponse(
                    text=(
                        '{"action": "done", "follow_up_tasks": [], '
                        '"reasons": ["V3 static mode: gap check bypassed."]}'
                    )
                )
            ]
        )

        engine = OrchestrationEngine(
            max_rounds=1,
            budget=budget,
            llm=self.llm,
            gapcheck_llm=no_gapcheck_llm,
            task_runner=effective_task_runner,
            tool_registry=self.tool_registry,
            run_id=run_id,
        )

        t0 = time.perf_counter()
        outcome: RunOutcome = await engine.run(
            question=q_text,
            initial_state=state,
            available_profiles=self.get_available_profiles(),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        (
            answer,
            claims,
            verdicts,
            conf_score,
        ) = await self._post_process_claims_and_report(outcome, q_text)

        return self.build_run_result(
            example_id=ex_id,
            question=q_text,
            outcome=outcome,
            answer=answer,
            claims=claims,
            verdicts=verdicts,
            elapsed_ms=elapsed_ms,
            confidence_score=conf_score,
        )
