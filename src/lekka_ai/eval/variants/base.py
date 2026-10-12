"""Base engine variant interface, ablation configuration, and shared dispatch."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lekka_ai.eval.variant import RunResult, Variant
from lekka_ai.harness.models import AgentTask
from lekka_ai.harness.permissions import AgentProfile, ToolView
from lekka_ai.harness.profiles import load_agent_profile
from lekka_ai.harness.runtime import AgentRunResult, AgentRuntime
from lekka_ai.llm.base import LLMClient
from lekka_ai.logging import get_logger
from lekka_ai.orchestrator.budget import RunBudget
from lekka_ai.orchestrator.dispatch import TaskResult, get_relevant_context_facts
from lekka_ai.orchestrator.engine import RunOutcome
from lekka_ai.orchestrator.state import PlannedTask, RunState
from lekka_ai.schemas.claims import Claim
from lekka_ai.tools.base import ToolRegistry, default_registry
from lekka_ai.verify.support import ClaimSupportCheck, verify_claim_support

logger = get_logger(__name__)

ALL_FIVE_SPECIALISTS: list[str] = [
    "fundamental",
    "valuation",
    "market",
    "news",
    "risk",
]

QUANT_TOOLS_SET: set[str] = {
    "run_dcf",
    "run_reverse_dcf",
    "compute_multiples",
    "scenario_table",
    "compute_risk_stats",
}


VariantType = Literal["static", "dynamic", "dynamic_replan"]
QuantToolsMode = Literal["quant_tools", "run_python_only"]


class VariantConfig(BaseModel):
    """Configuration options and ablation switches for engine variants."""

    model_config = ConfigDict(extra="allow")

    variant_type: VariantType = Field(
        default="dynamic",
        description="Architecture style ('static', 'dynamic', 'dynamic_replan')",
    )
    verifier: bool = Field(
        default=True,
        description="Enable claim support verification via verifier",
    )
    risk_agent: bool = Field(
        default=True,
        description="Enable risk specialist agent in planning and execution",
    )
    memory: bool = Field(
        default=True,
        description="Enable fact cache memory (recall_facts tool)",
    )
    quant_tools: bool = Field(
        default=True,
        description="Enable specialized quant tools (vs run_python only)",
    )
    quant_tools_mode: QuantToolsMode = Field(
        default="quant_tools",
        description="Ablation switch: 'quant_tools' vs 'run_python_only'",
    )
    risk_wave: bool = Field(
        default=True,
        description="Whether to execute a dedicated risk assessment wave",
    )
    enable_gap_check: bool = Field(
        default=True,
        description="Whether to run gap check audit and re-planning",
    )
    max_rounds: int = Field(
        default=3,
        description="Maximum re-plan rounds for multi-round variants",
    )
    synthesize_report: bool = Field(
        default=True,
        description="Generate full synthesized report with confidence scoring",
    )


def apply_ablation_to_profile(
    profile: AgentProfile,
    config: VariantConfig,
) -> AgentProfile:
    """Modify agent profile allowed_tools according to ablation switches.

    Args:
        profile: Base agent profile to adjust.
        config: Ablation configuration switches.

    Returns:
        Updated AgentProfile respecting memory and quant_tools ablations.
    """
    tools = list(profile.allowed_tools)

    # 1. Memory ablation: remove recall_facts tool
    if not config.memory:
        tools = [t for t in tools if t != "recall_facts"]

    # 2. Quant tools ablation: remove dedicated tools and keep run_python only
    if not config.quant_tools or config.quant_tools_mode == "run_python_only":
        tools = [t for t in tools if t not in QUANT_TOOLS_SET]
        if "run_python" not in tools:
            tools.append("run_python")

    # Clone profile with updated allowed tools
    return profile.model_copy(update={"allowed_tools": tools})


def collect_claims_from_state(state: RunState) -> list[Claim]:
    """Extract all claims submitted across completed tasks in state."""
    claims: list[Claim] = []
    seen_ids: set[str] = set()

    for res in state.results.values():
        raw_claims: list[Any] = []
        if isinstance(res, TaskResult):
            raw_claims = res.claims
        elif isinstance(res, dict):
            raw_claims = res.get("claims", [])
        elif hasattr(res, "claims"):
            raw_claims = getattr(res, "claims", [])

        for c in raw_claims:
            claim_obj: Claim | None = None
            if isinstance(c, Claim):
                claim_obj = c
            elif isinstance(c, dict):
                try:
                    claim_obj = Claim.model_validate(c)
                except Exception:
                    pass

            if claim_obj and claim_obj.claim_id not in seen_ids:
                seen_ids.add(claim_obj.claim_id)
                claims.append(claim_obj)

    return claims


class BaseEngineVariant(Variant):
    """Abstract base class for engine-backed evaluation variants."""

    def __init__(
        self,
        name: str,
        *,
        llm: LLMClient | None = None,
        config: VariantConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        description: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self.llm = llm
        self.config = config or VariantConfig()
        self.tool_registry = tool_registry or default_registry

    def get_available_profiles(self) -> list[str]:
        """Return list of permissible specialist profiles under config."""
        if self.config.risk_agent:
            return list(ALL_FIVE_SPECIALISTS)
        return [p for p in ALL_FIVE_SPECIALISTS if p != "risk"]

    def create_task_runner(
        self,
        state: RunState,
        budget: RunBudget,
    ) -> Any:
        """Create a dispatch task runner that applies ablation switches to profiles."""

        async def _runner(task: PlannedTask, scratch_dir: Path) -> TaskResult:
            # Load profile and apply ablation switches
            raw_profile = load_agent_profile(task.agent_type)
            profile = apply_ablation_to_profile(raw_profile, self.config)

            tool_view = ToolView(
                registry=self.tool_registry,
                profile=profile,
            )

            agent_task = AgentTask(
                task_id=task.task_id,
                agent_type=task.agent_type,
                instruction=task.instruction,
                context_facts=get_relevant_context_facts(
                    planned_task=task,
                    state=state,
                ),
            )

            limits = budget.allocate(task)
            t0 = time.perf_counter()

            try:
                run_res: AgentRunResult = await AgentRuntime.run(
                    task=agent_task,
                    profile=profile,
                    llm=self.llm,  # type: ignore[arg-type]
                    tool_view=tool_view,
                    limits=limits,
                )
                elapsed = time.perf_counter() - t0
                budget.reconcile(run_res, elapsed_seconds=elapsed)

                claims = list(run_res.submission.claims) if run_res.submission else []
                assumptions = (
                    dict(run_res.submission.assumptions) if run_res.submission else {}
                )
                open_questions = (
                    list(run_res.submission.open_questions)
                    if run_res.submission
                    else []
                )
                conf = run_res.submission.confidence if run_res.submission else None
                rationale = (
                    run_res.submission.confidence_rationale
                    if run_res.submission
                    else None
                )

                task_result = TaskResult(
                    task_id=task.task_id,
                    agent_type=task.agent_type,
                    status="completed",
                    claims=claims,
                    assumptions=assumptions,
                    open_questions=open_questions,
                    usage={
                        "tokens_in": run_res.tokens_in,
                        "tokens_out": run_res.tokens_out,
                        "tool_calls": run_res.tool_calls_count,
                        "elapsed_seconds": round(elapsed, 4),
                    },
                    confidence=conf,
                    confidence_rationale=rationale,
                )
                state.update_task_status(
                    task_id=task.task_id,
                    status="completed",
                    result=task_result.model_dump(),
                )
                return task_result

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                err_msg = f"Task execution failed: {exc}"
                logger.error(err_msg, task_id=task.task_id, exc_info=True)
                failed_res = TaskResult(
                    task_id=task.task_id,
                    agent_type=task.agent_type,
                    status="failed",
                    usage={"elapsed_seconds": round(elapsed, 4)},
                    error=err_msg,
                )
                state.update_task_status(
                    task_id=task.task_id,
                    status="failed",
                    result=failed_res.model_dump(),
                    error=err_msg,
                )
                return failed_res

        return _runner

    async def _post_process_claims_and_report(
        self,
        outcome: RunOutcome,
        question: str,
    ) -> tuple[str, list[Claim], list[ClaimSupportCheck], float | None]:
        """Verify claims if enabled and generate report text and confidence."""
        claims = collect_claims_from_state(outcome.state)
        verdicts: list[ClaimSupportCheck] = []

        # 1. Verifier ablation check
        if self.config.verifier and claims:
            try:
                v_llm = getattr(self, "verifier_llm", None) or self.llm
                verdicts = await verify_claim_support(claims, llm=v_llm)
            except Exception as err:
                logger.warning("Claim verification failed in variant: %s", err)

        # 2. Report generation / answer construction
        answer_parts: list[str] = []
        confidence_score: float | None = None

        if outcome.status == "abstained":
            answer_text = (
                "Abstained: Insufficient evidence to answer the research question."
            )
            return answer_text, claims, verdicts, 0.0

        for res in outcome.state.results.values():
            if isinstance(res, TaskResult) and res.confidence_rationale:
                answer_parts.append(f"[{res.agent_type}] {res.confidence_rationale}")

        if not answer_parts and claims:
            answer_parts = [f"- {c.text}" for c in claims[:10]]

        answer_text = (
            "\n".join(answer_parts)
            if answer_parts
            else f"Completed research with {len(claims)} verified claims."
        )

        return answer_text, claims, verdicts, confidence_score

    def build_run_result(
        self,
        example_id: str,
        question: str,
        outcome: RunOutcome,
        answer: str,
        claims: list[Claim],
        verdicts: Sequence[ClaimSupportCheck],
        elapsed_ms: float,
        confidence_score: float | None = None,
    ) -> RunResult:
        """Construct standard RunResult from executed outcome and measurements."""
        tok_in = 0
        tok_out = 0
        tools = outcome.budget.used_tool_calls if outcome.budget else 0
        cost = outcome.budget.used_cost if outcome.budget else 0.0

        for res in outcome.state.results.values():
            usage = res.usage if isinstance(res, TaskResult) else {}
            tok_in += usage.get("tokens_in", 0)
            tok_out += usage.get("tokens_out", 0)
            if not tools:
                tools += usage.get("tool_calls", 0)

        if cost == 0.0 and (tok_in > 0 or tok_out > 0):
            cost = tok_in * 0.00000015 + tok_out * 0.00000060

        v_map = {v.claim_id: v.supported.value for v in verdicts}
        dispatched = [
            t.agent_type
            for t in outcome.state.plan.tasks
            if t.task_id in outcome.state.task_statuses
        ]

        return RunResult(
            example_id=example_id,
            variant=self.name,
            question=question,
            answer=answer,
            claims=claims,
            cost=round(cost, 6),
            tokens_in=tok_in,
            tokens_out=tok_out,
            tokens_total=tok_in + tok_out,
            latency_ms=round(elapsed_ms, 2),
            tool_calls=tools,
            status=outcome.status,
            is_abstention=(
                outcome.status == "abstained" or outcome.stop_reason == "abstain"
            ),
            confidence_score=confidence_score,
            metadata={
                "rounds": outcome.rounds,
                "stop_reason": outcome.stop_reason,
                "dispatched_agents": dispatched,
                "verifier_verdicts": v_map,
                "config": self.config.model_dump(),
            },
            error=outcome.error,
        )


EngineVariant = BaseEngineVariant
