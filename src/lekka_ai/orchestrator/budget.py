"""Run-level budget manager with priority allocation, reconciliation, and tracing."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from lekka_ai.config import Settings, get_settings
from lekka_ai.harness.models import (
    AgentTask,
    Budget,
    BudgetExhaustedError,
    Limits,
)
from lekka_ai.harness.profiles import load_agent_profile
from lekka_ai.harness.runtime import AgentRunResult
from lekka_ai.logging import get_logger
from lekka_ai.orchestrator.state import PlannedTask
from lekka_ai.tracing import Span, Tracer

logger = get_logger(__name__)


class RunBudget(Budget):
    """Run-level resource budget with priority-based allocation and reconciliation.

    Tracks:
    - Tool calls allowance and consumption.
    - Token usage (input, output, and total).
    - Execution time (wall-clock seconds).
    - Cost estimate in USD calculated from token prices in config.
    - Allocation and reconciliation lifecycle with span logging to Tracer.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    input_token_price_per_m: float = Field(
        default=0.15,
        ge=0.0,
        description="USD cost per 1M input tokens",
    )
    output_token_price_per_m: float = Field(
        default=0.60,
        ge=0.0,
        description="USD cost per 1M output tokens",
    )
    tracer: Tracer | None = Field(
        default=None,
        description="Tracer instance for recording allocation and reconciliation spans",
    )

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        tracer: Tracer | None = None,
        **overrides: Any,
    ) -> RunBudget:
        """Instantiate RunBudget using values from application settings."""
        s = settings or get_settings()
        init_kwargs: dict[str, Any] = {
            "tool_calls": s.DEFAULT_RUN_BUDGET_TOOL_CALLS,
            "tokens": s.DEFAULT_RUN_BUDGET_TOKENS,
            "seconds": s.DEFAULT_RUN_BUDGET_SECONDS,
            "cost": s.DEFAULT_RUN_BUDGET_USD,
            "input_token_price_per_m": s.INPUT_TOKEN_PRICE_PER_M,
            "output_token_price_per_m": s.OUTPUT_TOKEN_PRICE_PER_M,
            "tracer": tracer,
        }
        init_kwargs.update(overrides)
        return cls(**init_kwargs)

    def estimate_usd(self, tokens_in: int, tokens_out: int = 0) -> float:
        """Estimate dollar cost from input and output tokens."""
        in_cost = (max(0, tokens_in) / 1_000_000.0) * self.input_token_price_per_m
        out_cost = (max(0, tokens_out) / 1_000_000.0) * self.output_token_price_per_m
        return round(in_cost + out_cost, 6)

    def can_allocate(self, task: PlannedTask | AgentTask | None = None) -> bool:
        """Determine whether the budget has available capacity to launch a task."""
        if self.is_exhausted():
            return False

        rem = self.remaining()
        # Must have positive balance in critical operational resources
        if self.tokens > 0 and rem.tokens <= 0:
            return False
        if self.tool_calls > 0 and rem.tool_calls <= 0:
            return False
        if self.seconds > 0 and rem.seconds <= 0:
            return False
        if self.cost > 0 and rem.cost <= 0:
            return False
        return True

    def stop_launching(self, task: PlannedTask | AgentTask | None = None) -> bool:
        """Stop-launching check: returns True when budget exhaustion rules apply."""
        return not self.can_allocate(task)

    def allocate(
        self,
        task: PlannedTask | AgentTask,
    ) -> Limits:
        """Allocate resource Limits to an agent task based on priority and balance.

        Args:
            task: PlannedTask or AgentTask to allocate resources for.

        Returns:
            Configured Limits instance.

        Raises:
            BudgetExhaustedError: If the run budget is exhausted.
        """
        task_id = task.task_id
        agent_type = task.agent_type
        priority: int = (
            task.priority
            if isinstance(task, PlannedTask)
            else getattr(task, "priority", 1)
        )

        if not self.can_allocate(task):
            depleted = self.exhausted_resources or ["general balance"]
            msg = (
                f"Cannot launch task '{task_id}': run budget is exhausted. "
                f"Depleted resources: {depleted}"
            )
            logger.warning(msg, task_id=task_id, agent_type=agent_type)
            raise BudgetExhaustedError(msg)

        rem = self.remaining()

        # Retrieve profile default limits if available
        profile_limits: Limits | None = None
        try:
            profile = load_agent_profile(agent_type)
            profile_limits = profile.default_limits
        except Exception:
            profile_limits = None

        # Priority-weighted share factor of remaining budget:
        # Priority >= 2 (High): up to 50% of remaining
        # Priority == 1 (Normal): up to 35% of remaining
        # Priority <= 0 (Low): up to 20% of remaining
        p = max(0, priority)
        if p >= 2:
            share = 0.50
        elif p == 1:
            share = 0.35
        else:
            share = 0.20

        # Calculate allocations bounded by remaining balance and profile caps
        avail_tokens = rem.tokens if self.tokens > 0 else 100_000
        avail_tools = rem.tool_calls if self.tool_calls > 0 else 15
        avail_seconds = rem.seconds if self.seconds > 0.0 else 60.0

        target_tokens = int(avail_tokens * share)
        target_tools = int(avail_tools * share)
        target_seconds = round(avail_seconds * share, 1)

        # Enforce minimum viable bounds if budget remaining allows
        alloc_tokens = max(100, min(avail_tokens, target_tokens))
        alloc_tools = max(1, min(avail_tools, target_tools))
        alloc_seconds = max(1.0, min(avail_seconds, target_seconds))

        if profile_limits:
            if profile_limits.max_tokens:
                alloc_tokens = min(alloc_tokens, profile_limits.max_tokens)
            if profile_limits.max_tool_calls:
                alloc_tools = min(alloc_tools, profile_limits.max_tool_calls)
            if profile_limits.max_seconds:
                alloc_seconds = min(alloc_seconds, profile_limits.max_seconds)
            max_iter = profile_limits.max_iterations
        else:
            max_iter = 10

        limits = Limits(
            max_tokens=alloc_tokens,
            max_tool_calls=alloc_tools,
            max_seconds=alloc_seconds,
            max_iterations=max_iter,
        )

        # Log allocation decision to trace and structured logger
        alloc_meta = {
            "task_id": task_id,
            "agent_type": agent_type,
            "priority": priority,
            "allocated_limits": limits.model_dump(),
            "remaining_budget": rem.model_dump(),
        }

        if self.tracer is not None:
            span = Span(
                name="budget_allocate",
                span_type="budget",
                metadata=alloc_meta,
                status="ok",
            )
            self.tracer.spans.append(span)

        logger.info(
            "Allocated budget to agent task",
            task_id=task_id,
            agent_type=agent_type,
            priority=priority,
            limits=limits.model_dump(),
        )

        return limits

    def reconcile(
        self,
        agent_run: AgentRunResult | dict[str, Any] | Any,
        elapsed_seconds: float | None = None,
    ) -> Budget:
        """Reconcile and charge actual usage incurred by a completed agent run.

        Args:
            agent_run: AgentRunResult or mapping containing resource metrics.
            elapsed_seconds: Optional wall-clock duration in seconds.

        Returns:
            Remaining Budget after charging actual consumption.
        """
        # 1. Extract metrics
        tokens_in = 0
        tokens_out = 0
        tool_calls = 0
        agent_run_id = None

        if isinstance(agent_run, AgentRunResult):
            tokens_in = agent_run.tokens_in
            tokens_out = agent_run.tokens_out
            tool_calls = agent_run.tool_calls_count
            agent_run_id = agent_run.agent_run_id
        elif isinstance(agent_run, dict):
            tokens_in = int(
                agent_run.get("tokens_in") or agent_run.get("input_tokens") or 0
            )
            tokens_out = int(
                agent_run.get("tokens_out") or agent_run.get("output_tokens") or 0
            )
            if tokens_in == 0 and tokens_out == 0:
                tokens_in = int(agent_run.get("tokens") or 0)
            tool_calls = int(
                agent_run.get("tool_calls") or agent_run.get("tool_calls_count") or 0
            )
            agent_run_id = agent_run.get("agent_run_id")
            if elapsed_seconds is None and "seconds" in agent_run:
                elapsed_seconds = float(agent_run["seconds"])
        elif hasattr(agent_run, "tokens_in") and hasattr(agent_run, "tokens_out"):
            tokens_in = int(getattr(agent_run, "tokens_in", 0))
            tokens_out = int(getattr(agent_run, "tokens_out", 0))
            tool_calls = int(getattr(agent_run, "tool_calls_count", 0))
            agent_run_id = getattr(agent_run, "agent_run_id", None)

        total_tokens = tokens_in + tokens_out
        seconds = max(0.0, elapsed_seconds or 0.0)
        cost = self.estimate_usd(tokens_in=tokens_in, tokens_out=tokens_out)

        # 2. Charge actual usage to budget
        self.charge(
            tool_calls=tool_calls,
            tokens=total_tokens,
            seconds=seconds,
            cost=cost,
            strict=False,
        )

        rem = self.remaining()

        # 3. Log reconciliation to trace and logger
        reconcile_meta = {
            "agent_run_id": agent_run_id,
            "charged_tokens_in": tokens_in,
            "charged_tokens_out": tokens_out,
            "charged_total_tokens": total_tokens,
            "charged_tool_calls": tool_calls,
            "charged_seconds": seconds,
            "charged_cost_usd": cost,
            "remaining_budget": rem.model_dump(),
        }

        if self.tracer is not None:
            span = Span(
                name="budget_reconcile",
                span_type="budget",
                metadata=reconcile_meta,
                status="ok",
            )
            self.tracer.spans.append(span)

        logger.info(
            "Reconciled agent run resource consumption",
            tool_calls=tool_calls,
            tokens=total_tokens,
            seconds=seconds,
            cost_usd=cost,
        )

        return rem


BudgetManager = RunBudget
