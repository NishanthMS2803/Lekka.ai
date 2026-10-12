"""Engine-backed evaluation variants V3, V4, and V5 with configurable ablations."""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from typing import Any

from lekka_ai.eval.dataset import EvalExample
from lekka_ai.eval.variant import RunResult, Variant
from lekka_ai.eval.variants.config import VariantConfig
from lekka_ai.harness.permissions import AgentProfile
from lekka_ai.harness.profiles import load_profiles
from lekka_ai.llm.base import LLMClient
from lekka_ai.logging import get_logger
from lekka_ai.orchestrator.budget import RunBudget
from lekka_ai.orchestrator.dispatch import TaskResult
from lekka_ai.orchestrator.engine import TaskRunnerCallable, run_engine
from lekka_ai.orchestrator.state import Plan, PlannedTask, RunState
from lekka_ai.report.confidence import synthesize_report_with_confidence
from lekka_ai.schemas.claims import Claim
from lekka_ai.tools.base import ToolRegistry
from lekka_ai.tracing import Tracer
from lekka_ai.verify.support import verify_claim_support

logger = get_logger(__name__)

QUANT_TOOLS_LIST: list[str] = [
    "run_dcf",
    "run_reverse_dcf",
    "compute_multiples",
    "scenario_table",
    "compute_risk_stats",
]


def build_ablated_profiles(
    config: VariantConfig,
    base_profiles: dict[str, AgentProfile] | None = None,
) -> dict[str, AgentProfile]:
    """Construct agent profiles modified according to ablation config switches.

    Args:
        config: Ablation switches (risk_agent, memory, quant_tools_mode).
        base_profiles: Base specialist profiles dictionary (defaults to loaded).

    Returns:
        Dictionary of agent profiles adapted to active ablations.
    """
    raw_profiles = base_profiles or load_profiles()
    profiles: dict[str, AgentProfile] = {}

    for name, prof in raw_profiles.items():
        # 1. Risk agent ablation switch
        if not config.risk_agent and name == "risk":
            continue

        allowed = list(prof.allowed_tools)

        # 2. Memory ablation switch: remove recall_facts tool
        if not config.memory and "recall_facts" in allowed:
            allowed = [t for t in allowed if t != "recall_facts"]

        # 3. Quant tools ablation switch: run_python only vs specialized quant tools
        if config.quant_tools_mode == "run_python_only":
            allowed = [t for t in allowed if t not in QUANT_TOOLS_LIST]
            if "run_python" not in allowed and name in (
                "valuation",
                "analyst",
                "fundamental",
            ):
                allowed.append("run_python")

        profiles[name] = AgentProfile(
            agent_type=prof.agent_type,
            allowed_tools=allowed,
            system_prompt_path=prof.system_prompt_path,
            answer_extension_schema=prof.answer_extension_schema,
            default_limits=prof.default_limits,
        )

    return profiles


class EngineVariant(Variant):
    """Base class for engine-driven benchmark evaluation variants."""

    def __init__(
        self,
        name: str,
        llm: LLMClient,
        config: VariantConfig | None = None,
        *,
        task_runner: TaskRunnerCallable | None = None,
        tool_registry: ToolRegistry | None = None,
        budget: RunBudget | None = None,
        tracer: Tracer | None = None,
        description: str = "",
    ) -> None:
        self.name = name
        self.llm = llm
        self.config = config or VariantConfig()
        self.task_runner = task_runner
        self.tool_registry = tool_registry
        self.budget = budget
        self.tracer = tracer
        self.description = description

    def _build_static_plan(self, question: str) -> Plan:
        """Construct static plan dispatching all specialists in a single wave."""
        plan = Plan()
        agents = ["fundamental", "valuation", "market", "news"]
        if self.config.risk_agent:
            agents.append("risk")

        for idx, agent in enumerate(agents, start=1):
            plan.add_task(
                PlannedTask(
                    task_id=f"static_{agent}_{idx}",
                    agent_type=agent,
                    instruction=(
                        f"Investigate {agent} aspects and ground claims for: {question}"
                    ),
                    priority=1,
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
        """Execute evaluation candidate run using the orchestration engine."""
        q_text = question.question if isinstance(question, EvalExample) else question
        cutoff = (
            question.as_of_date if isinstance(question, EvalExample) else as_of_date
        )
        if cutoff and f"as of {cutoff}" not in q_text:
            q_text = f"{q_text} (as of {cutoff})"
        ex_id = (
            question.id
            if isinstance(question, EvalExample)
            else (example_id or f"run_{uuid.uuid4().hex[:8]}")
        )
        run_uuid = str(uuid.uuid4())

        # Configure ablated profiles
        profiles = build_ablated_profiles(self.config)

        # Configure initial state based on variant type
        initial_state: RunState | None = None
        max_rounds = 1

        if self.config.variant_type == "static":
            static_plan = self._build_static_plan(q_text)
            initial_state = RunState(
                run_id=run_uuid,
                question=q_text,
                plan=static_plan,
            )
            max_rounds = 1
        elif self.config.variant_type == "dynamic":
            initial_state = None
            max_rounds = 1
        elif self.config.variant_type == "dynamic_replan":
            initial_state = None
            max_rounds = max(2, self.config.max_rounds)

        start_t = time.perf_counter()

        try:
            outcome = await run_engine(
                question=q_text,
                initial_state=initial_state,
                budget=self.budget,
                max_rounds=max_rounds,
                llm=self.llm,
                task_runner=self.task_runner,
                tool_registry=self.tool_registry,
                tracer=self.tracer,
                available_profiles=profiles,
                run_id=run_uuid,
                fault_plan=kwargs.get("fault_plan"),
                enable_gap_check=self.config.enable_gap_check,
            )
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            # Collect claims from all completed tasks
            claims: list[Claim] = []
            total_tokens_in = 0
            total_tokens_out = 0
            total_tool_calls = 0

            if outcome.state and outcome.state.results:
                for res in outcome.state.results.values():
                    c_list: Sequence[Any] = []
                    if isinstance(res, TaskResult):
                        c_list = res.claims
                        u = res.usage or {}
                        total_tokens_in += int(u.get("tokens_in", 0))
                        total_tokens_out += int(u.get("tokens_out", 0))
                        total_tool_calls += int(u.get("tool_calls", 0))
                    elif isinstance(res, dict):
                        c_list = res.get("claims", [])
                    elif hasattr(res, "claims"):
                        c_list = getattr(res, "claims", [])

                    for raw_c in c_list:
                        if isinstance(raw_c, Claim):
                            claims.append(raw_c)
                        elif isinstance(raw_c, dict):
                            try:
                                claims.append(Claim.model_validate(raw_c))
                            except Exception:
                                pass

            # Verifier ablation switch
            verifier_verdicts: dict[str, str] = {}
            if self.config.verifier and claims:
                try:
                    checks = await verify_claim_support(claims, llm=self.llm)
                    for chk in checks:
                        verifier_verdicts[chk.claim_id] = chk.supported.value
                except Exception as ver_err:
                    logger.warning("Variant claim verification skipped: %s", ver_err)

            # Generate synthesis / final answer
            answer_text = ""
            conf_score: float | None = None
            is_abstention = outcome.status == "abstained"

            try:
                report, breakdown = await synthesize_report_with_confidence(
                    claims=claims,
                    question=q_text,
                    llm=self.llm,
                )
                answer_text = report.markdown
                conf_score = breakdown.confidence_score
                if breakdown.is_abstention:
                    is_abstention = True
            except Exception:
                # Fallback to direct claim summary
                if claims:
                    answer_text = "\n".join(f"- {c.text}" for c in claims[:5])
                else:
                    answer_text = outcome.stop_reason

            est_cost = total_tokens_in * 0.00000015 + total_tokens_out * 0.00000060
            if outcome.budget:
                est_cost = max(est_cost, outcome.budget.used_cost)

            metadata: dict[str, Any] = {
                "rounds": outcome.rounds,
                "stop_reason": outcome.stop_reason,
                "verifier_verdicts": verifier_verdicts,
                "config": self.config.model_dump(),
            }

            return RunResult(
                example_id=ex_id,
                variant=self.name,
                question=q_text,
                answer=answer_text,
                claims=claims,
                cost=est_cost,
                tokens_in=total_tokens_in,
                tokens_out=total_tokens_out,
                tokens_total=total_tokens_in + total_tokens_out,
                latency_ms=elapsed_ms,
                tool_calls=total_tool_calls,
                status=outcome.status,
                is_abstention=is_abstention,
                confidence_score=conf_score,
                metadata=metadata,
                raw=outcome,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return RunResult(
                example_id=ex_id,
                variant=self.name,
                question=q_text,
                answer="",
                claims=[],
                cost=0.0,
                latency_ms=elapsed_ms,
                tool_calls=0,
                status="failed",
                error=str(exc),
            )


class V3StaticVariant(EngineVariant):
    """V3 Static Variant: all 5 specialists run in parallel, 1 round."""

    def __init__(
        self,
        llm: LLMClient,
        config: VariantConfig | None = None,
        name: str = "v3_static",
        **kwargs: Any,
    ) -> None:
        cfg = config.model_copy() if config else VariantConfig()
        cfg.variant_type = "static"
        cfg.max_rounds = 1
        cfg.enable_gap_check = False
        if config is None or "verifier" not in config.model_fields_set:
            cfg.verifier = False
        if config is None or "risk_wave" not in config.model_fields_set:
            cfg.risk_wave = False
        super().__init__(
            name=name,
            llm=llm,
            config=cfg,
            description="V3 Static: 5 agents parallel, 1 round, no gap check",
            **kwargs,
        )


class V4DynamicVariant(EngineVariant):
    """V4 Dynamic Variant: planner dynamically selects agents, 1 round."""

    def __init__(
        self,
        llm: LLMClient,
        config: VariantConfig | None = None,
        name: str = "v4_dynamic",
        **kwargs: Any,
    ) -> None:
        cfg = config.model_copy() if config else VariantConfig()
        cfg.variant_type = "dynamic"
        cfg.max_rounds = 1
        cfg.enable_gap_check = False
        if config is None or "verifier" not in config.model_fields_set:
            cfg.verifier = False
        if config is None or "risk_wave" not in config.model_fields_set:
            cfg.risk_wave = False
        super().__init__(
            name=name,
            llm=llm,
            config=cfg,
            description="V4 Dynamic: planner selects agents dynamically, 1 round",
            **kwargs,
        )


class V5DynamicVariant(EngineVariant):
    """V5 Dynamic Variant: dynamic planning + gap check replan + verifier + risk."""

    def __init__(
        self,
        llm: LLMClient,
        config: VariantConfig | None = None,
        name: str = "v5_dynamic",
        **kwargs: Any,
    ) -> None:
        cfg = config.model_copy() if config else VariantConfig()
        cfg.variant_type = "dynamic_replan"
        cfg.max_rounds = max(2, cfg.max_rounds or 3)
        cfg.enable_gap_check = True
        cfg.verifier = True
        cfg.risk_wave = True
        super().__init__(
            name=name,
            llm=llm,
            config=cfg,
            description=(
                "V5 Full: dynamic planner + gap replanning + verifier + risk wave"
            ),
            **kwargs,
        )
