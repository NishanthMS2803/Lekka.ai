"""Unit tests for run-level budget management, allocation, and reconciliation."""

import pytest

from lekka_ai.config import get_settings
from lekka_ai.harness.models import BudgetExhaustedError, Limits
from lekka_ai.harness.runtime import AgentRunResult
from lekka_ai.orchestrator.budget import RunBudget
from lekka_ai.orchestrator.state import PlannedTask
from lekka_ai.tracing import Tracer


def test_budget_initialization_from_settings() -> None:
    """Verify RunBudget initializes with default parameters from settings."""
    settings = get_settings()
    budget = RunBudget.from_settings()

    assert budget.tool_calls == settings.DEFAULT_RUN_BUDGET_TOOL_CALLS
    assert budget.tokens == settings.DEFAULT_RUN_BUDGET_TOKENS
    assert budget.seconds == settings.DEFAULT_RUN_BUDGET_SECONDS
    assert budget.cost == settings.DEFAULT_RUN_BUDGET_USD
    assert budget.input_token_price_per_m == settings.INPUT_TOKEN_PRICE_PER_M
    assert budget.output_token_price_per_m == settings.OUTPUT_TOKEN_PRICE_PER_M
    assert not budget.is_exhausted()


def test_budget_estimate_usd() -> None:
    """Verify USD cost estimate based on configured token pricing."""
    budget = RunBudget(
        input_token_price_per_m=0.15,
        output_token_price_per_m=0.60,
    )
    # 1,000,000 input tokens = $0.15; 1,000,000 output tokens = $0.60 -> $0.75
    cost = budget.estimate_usd(tokens_in=1_000_000, tokens_out=1_000_000)
    assert cost == pytest.approx(0.75, abs=1e-5)

    # 10,000 in, 2,000 out -> (10000*0.15 + 2000*0.60)/1M = (1.5 + 1.2)/1000 = $0.0027
    cost_small = budget.estimate_usd(tokens_in=10_000, tokens_out=2_000)
    assert cost_small == pytest.approx(0.0027, abs=1e-5)


def test_budget_allocation_priority_and_trace_logging() -> None:
    """Verify priority-based limit allocation and trace span logging."""
    tracer = Tracer()
    budget = RunBudget(
        tool_calls=50,
        tokens=100_000,
        seconds=100.0,
        cost=1.00,
        tracer=tracer,
    )

    high_task = PlannedTask(
        task_id="task_high",
        agent_type="valuation",
        instruction="High priority DCF modeling",
        priority=2,
    )
    normal_task = PlannedTask(
        task_id="task_normal",
        agent_type="news",
        instruction="Normal priority news check",
        priority=1,
    )
    low_task = PlannedTask(
        task_id="task_low",
        agent_type="market",
        instruction="Low priority background market check",
        priority=0,
    )

    # High priority allocation (share = 50% of available)
    high_limits = budget.allocate(high_task)
    assert isinstance(high_limits, Limits)
    assert high_limits.max_tokens is not None and high_limits.max_tokens > 0
    assert high_limits.max_tool_calls is not None and high_limits.max_tool_calls > 0
    assert high_limits.max_seconds is not None and high_limits.max_seconds > 0

    # Normal priority allocation (share = 35% of available)
    normal_limits = budget.allocate(normal_task)
    # Low priority allocation (share = 20% of available)
    low_limits = budget.allocate(low_task)

    # Priority differentiation: high gets more or equal tokens compared to low
    assert (high_limits.max_tokens or 0) >= (normal_limits.max_tokens or 0)
    assert (normal_limits.max_tokens or 0) >= (low_limits.max_tokens or 0)

    # Verify trace spans logged for all three allocations
    alloc_spans = [s for s in tracer.spans if s.name == "budget_allocate"]
    assert len(alloc_spans) == 3
    assert alloc_spans[0].metadata["task_id"] == "task_high"
    assert alloc_spans[1].metadata["task_id"] == "task_normal"
    assert alloc_spans[2].metadata["task_id"] == "task_low"


def test_budget_reconciliation() -> None:
    """Verify reconciliation charges actual usage and updates remaining balance."""
    tracer = Tracer()
    budget = RunBudget(
        tool_calls=20,
        tokens=50_000,
        seconds=60.0,
        cost=0.50,
        input_token_price_per_m=0.15,
        output_token_price_per_m=0.60,
        tracer=tracer,
    )

    # Create mock agent run result
    mock_run = AgentRunResult(
        task_id="task_test_rec",
        agent_type="analyst",
        status="completed",
        tokens_in=4_000,
        tokens_out=1_000,
        tool_calls_count=3,
        agent_run_id="run_spec_01",
    )

    rem = budget.reconcile(mock_run, elapsed_seconds=5.2)

    # Verify usage charged against budget
    assert budget.used_tokens == 5_000
    assert budget.used_tool_calls == 3
    assert budget.used_seconds == pytest.approx(5.2, abs=1e-4)
    # Expected cost = (4000*0.15 + 1000*0.60)/1M = 0.0012
    assert budget.used_cost == pytest.approx(0.0012, abs=1e-5)

    # Verify remaining balances
    assert rem.tokens == 45_000
    assert rem.tool_calls == 17
    assert rem.seconds == pytest.approx(54.8, abs=1e-4)
    assert rem.cost == pytest.approx(0.50 - 0.0012, abs=1e-5)

    # Verify reconciliation span recorded in trace
    reconcile_spans = [s for s in tracer.spans if s.name == "budget_reconcile"]
    assert len(reconcile_spans) == 1
    assert reconcile_spans[0].metadata["charged_total_tokens"] == 5_000
    assert reconcile_spans[0].metadata["charged_tool_calls"] == 3


def test_budget_exhaustion_stop_launching() -> None:
    """Verify stop-launching rule triggers BudgetExhaustedError when budget depleted."""
    budget = RunBudget(
        tool_calls=5,
        tokens=2_000,
        seconds=10.0,
        cost=0.10,
    )

    task = PlannedTask(
        task_id="task_will_exhaust",
        agent_type="analyst",
        instruction="Analyze financial filings",
    )

    # Initially not exhausted
    assert not budget.is_exhausted()
    assert budget.can_allocate(task)
    assert not budget.stop_launching(task)

    # Reconcile usage that completely consumes the token budget
    budget.reconcile({"tokens_in": 1_500, "tokens_out": 600, "tool_calls": 2})

    # Now tokens are exhausted (2100 used vs 2000 limit)
    assert budget.is_exhausted()
    assert not budget.can_allocate(task)
    assert budget.stop_launching(task)
    assert "tokens" in budget.exhausted_resources

    # Attempting to allocate raises BudgetExhaustedError
    with pytest.raises(BudgetExhaustedError, match="run budget is exhausted"):
        budget.allocate(task)


def test_budget_exhaustion_by_cost_limit() -> None:
    """Verify exhaustion triggers when USD cost allowance is depleted."""
    budget = RunBudget(
        tool_calls=50,
        tokens=1_000_000,
        seconds=200.0,
        cost=0.01,  # Strict $0.01 cap
        input_token_price_per_m=10.0,  # High token price
        output_token_price_per_m=20.0,
    )

    task = PlannedTask(
        task_id="task_costly",
        agent_type="market",
        instruction="Check market price",
    )

    # 1,000 in, 500 out -> cost = (1000*10 + 500*20)/1M = $0.02 > $0.01 limit
    budget.reconcile({"tokens_in": 1_000, "tokens_out": 500, "tool_calls": 1})

    assert budget.is_exhausted()
    assert "cost" in budget.exhausted_resources
    assert budget.stop_launching(task)

    with pytest.raises(BudgetExhaustedError, match="cost"):
        budget.allocate(task)
