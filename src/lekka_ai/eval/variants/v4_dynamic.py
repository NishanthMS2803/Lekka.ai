"""V4 Dynamic Variant: planner selects specialist agents dynamically, single round."""

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
from lekka_ai.logging import get_logger
from lekka_ai.orchestrator.budget import RunBudget
from lekka_ai.orchestrator.engine import OrchestrationEngine, RunOutcome
from lekka_ai.orchestrator.planner import plan_query
from lekka_ai.orchestrator.state import RunState
from lekka_ai.orchestrator.understand import ParsedQuery, understand_query
from lekka_ai.tools.base import ToolRegistry

logger = get_logger(__name__)


class V4DynamicVariant(BaseEngineVariant):
    """V4 Dynamic: planner selects agents; runs in 1 round without gap check."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        understand_llm: LLMClient | None = None,
        planner_llm: LLMClient | None = None,
        config: VariantConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        name: str = "v4_dynamic",
        task_runner: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize V4DynamicVariant with optional model overrides and config."""
        super().__init__(
            name=name,
            llm=llm,
            config=config,
            tool_registry=tool_registry,
            description=(
                "V4 Dynamic variant: planner selects specialist agents dynamically "
                "based on query understanding, executing in a single round."
            ),
        )
        self.understand_llm = understand_llm
        self.planner_llm = planner_llm
        self.task_runner = task_runner

    async def run(
        self,
        question: str | EvalExample,
        *,
        as_of_date: str | None = None,
        example_id: str | None = None,
        **kwargs: Any,
    ) -> RunResult:
        """Execute dynamic query understanding, planning, and single-round execution."""
        q_text = question.question if isinstance(question, EvalExample) else question
        cutoff = (
            question.as_of_date if isinstance(question, EvalExample) else as_of_date
        )
        ex_id = (
            question.id
            if isinstance(question, EvalExample)
            else (example_id or f"v4_{uuid.uuid4().hex[:8]}")
        )
        run_id = f"run_{uuid.uuid4().hex[:8]}"

        t0 = time.perf_counter()
        budget = RunBudget.from_settings()

        # Step 1: Query understanding
        parsed_query: ParsedQuery
        try:
            parsed_query = await understand_query(
                question=q_text,
                as_of_date=cutoff,
                llm=self.understand_llm or self.llm,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            err_msg = f"V4 query understanding failed: {exc}"
            logger.error(err_msg, exc_info=True)
            return RunResult(
                example_id=ex_id,
                variant=self.name,
                question=q_text,
                status="failed",
                latency_ms=round(elapsed_ms, 2),
                error=err_msg,
            )

        if parsed_query.needs_clarification:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return RunResult(
                example_id=ex_id,
                variant=self.name,
                question=q_text,
                answer="Needs clarification: " + "; ".join(parsed_query.ambiguities),
                status="needs_clarification",
                is_abstention=True,
                latency_ms=round(elapsed_ms, 2),
                metadata={"ambiguities": parsed_query.ambiguities},
            )

        # Step 2: Dynamic planning using available specialist profiles
        available_profiles = self.get_available_profiles()
        try:
            plan = await plan_query(
                query=parsed_query,
                available_profiles=available_profiles,
                llm=self.planner_llm or self.llm,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            err_msg = f"V4 planning failed: {exc}"
            logger.error(err_msg, exc_info=True)
            return RunResult(
                example_id=ex_id,
                variant=self.name,
                question=q_text,
                status="failed",
                latency_ms=round(elapsed_ms, 2),
                error=err_msg,
            )

        state = RunState(
            run_id=run_id,
            question=q_text,
            parsed_query=parsed_query.model_dump(),
            plan=plan,
        )

        effective_task_runner = self.task_runner or self.create_task_runner(
            state, budget
        )

        # Gap check bypass: immediate done signal without invoking LLM (single round)
        no_gapcheck_llm = FakeLLM(
            responses=[
                LLMResponse(
                    text=(
                        '{"action": "done", "follow_up_tasks": [], '
                        '"reasons": ["V4 dynamic mode: 1 round without gap check."]}'
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

        outcome: RunOutcome = await engine.run(
            question=q_text,
            initial_state=state,
            available_profiles=available_profiles,
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
