"""Unit tests for AgentTask, Limits, and Budget models."""

from datetime import date

import pytest
from pydantic import ValidationError

from lekka_ai.harness import (
    AgentTask,
    Budget,
    BudgetExhaustedError,
    Limits,
)
from lekka_ai.schemas.claims import Claim, Evidence


def test_agent_task_creation_and_defaults() -> None:
    """Verify AgentTask default ID generation, context handling, and date parsing."""
    task = AgentTask(
        agent_type="analyst",
        instruction="Extract FY24 cash flows",
        as_of_date="2024-09-30",
    )
    assert task.task_id.startswith("task_")
    assert task.agent_type == "analyst"
    assert task.instruction == "Extract FY24 cash flows"
    assert task.as_of_date == "2024-09-30"
    assert task.context_facts == []

    # Custom task_id, context alias, and date object
    claim = Claim(
        text="Operating margin expanded 120bps",
        type="fact",
        confidence=0.9,
        evidence=[Evidence(source_type="filing", tool_call_id="call_123")],
    )
    task_custom = AgentTask(
        task_id="task_custom_99",
        agent_type="quant",
        instruction="Build DCF model",
        context_facts=[claim, {"factor": "macro_rate", "value": 0.045}],
        as_of_date=date(2024, 12, 31),
    )
    assert task_custom.task_id == "task_custom_99"
    assert len(task_custom.context_facts) == 2
    assert task_custom.as_of_date == "2024-12-31"


def test_limits_monitoring_and_to_budget() -> None:
    """Verify Limits thresholds, breached checking, and conversion to Budget."""
    limits = Limits(
        max_iterations=5,
        max_tokens=10_000,
        max_seconds=30.0,
        max_tool_calls=12,
    )
    assert not limits.is_exceeded(
        iterations=4, tokens=9_000, seconds=20.0, tool_calls=10
    )
    assert limits.exceeded_limits(iterations=4, tokens=9_000) == []

    # Breaching iterations and tool calls
    breached = limits.exceeded_limits(iterations=5, tokens=2_000, tool_calls=15)
    assert "max_iterations" in breached
    assert "max_tool_calls" in breached
    assert "max_tokens" not in breached
    assert limits.is_exceeded(iterations=5)

    # Conversion to Budget
    budget = limits.to_budget()
    assert budget.tool_calls == 12
    assert budget.tokens == 10_000
    assert budget.seconds == 30.0
    assert budget.cost == 0.0

    # Negative limits are rejected
    with pytest.raises(ValidationError):
        Limits(max_tokens=-100)


def test_budget_charge_success_and_tracking() -> None:
    """Verify Budget charge deductions and tracking across dimensions."""
    budget = Budget(tool_calls=10, tokens=2000, seconds=60.0, cost=1.50)
    assert budget.used_tool_calls == 0
    assert budget.used_tokens == 0

    # Incremental charges
    budget.charge(tool_calls=2, tokens=350, seconds=5.5, cost=0.15)
    assert budget.used_tool_calls == 2
    assert budget.used_tokens == 350
    assert budget.used_seconds == 5.5
    assert budget.used_cost == 0.15

    rem = budget.remaining()
    assert rem.tool_calls == 8
    assert rem.tokens == 1650
    assert rem.seconds == 54.5
    assert rem.cost == 1.35
    assert rem["tool_calls"] == 8
    assert rem["tokens"] == 1650

    # Charge using another Budget delta object
    delta = Budget(tool_calls=3, tokens=650, seconds=10.0, cost=0.20)
    budget.charge(delta)
    rem2 = budget.remaining()
    assert rem2.tool_calls == 5
    assert rem2.tokens == 1000
    assert rem2.seconds == 44.5
    assert rem2.cost == 1.15


def test_budget_exhaustion() -> None:
    """Verify exhaustion detection across resources and strict charging errors."""
    budget = Budget(tool_calls=2, tokens=500, seconds=10.0, cost=0.50)
    assert not budget.is_exhausted()
    assert not budget.exhausted
    assert budget.exhausted_resources == []

    # Partially deplete tokens
    budget.charge(tokens=500)
    assert budget.is_exhausted()
    assert budget.is_exhausted("tokens")
    assert not budget.is_exhausted("tool_calls")
    assert budget.exhausted_resources == ["tokens"]

    # can_charge checks
    assert budget.can_charge(tool_calls=1)
    assert not budget.can_charge(tokens=1)

    # Strict charge on exhausted resource raises BudgetExhaustedError
    with pytest.raises(BudgetExhaustedError, match="tokens"):
        budget.charge(tokens=1, strict=True)

    # Deplete all resources
    budget.charge(tool_calls=2, seconds=10.0, cost=0.50)
    assert set(budget.exhausted_resources) == {
        "tool_calls",
        "tokens",
        "seconds",
        "cost",
    }

    with pytest.raises(KeyError, match="Unknown budget resource"):
        budget.is_exhausted("invalid_metric")


def test_budget_never_negative() -> None:
    """Verify remaining budget clamps to zero and negative amounts are rejected."""
    budget = Budget(tool_calls=5, tokens=100, seconds=5.0, cost=0.10)

    # Overcharging beyond budget in non-strict mode does not make remaining negative
    budget.charge(tool_calls=15, tokens=500, seconds=20.0, cost=1.0)
    rem = budget.remaining()
    assert rem.tool_calls == 0
    assert rem.tokens == 0
    assert rem.seconds == 0.0
    assert rem.cost == 0.0
    assert rem.tool_calls >= 0
    assert rem.tokens >= 0
    assert rem.seconds >= 0.0
    assert rem.cost >= 0.0

    # Charging negative amounts is strictly rejected with ValueError
    with pytest.raises(ValueError, match="cannot be negative"):
        budget.charge(tool_calls=-1)

    with pytest.raises(ValueError, match="cannot be negative"):
        budget.charge(tokens=-50)

    with pytest.raises(ValueError, match="cannot be negative"):
        budget.charge(seconds=-0.5)

    with pytest.raises(ValueError, match="cannot be negative"):
        budget.charge(cost=-0.01)

    # Constructing Budget with negative values is rejected
    with pytest.raises(ValueError, match="cannot be negative"):
        Budget(tokens=-10)

    with pytest.raises(ValueError, match="cannot be negative"):
        Budget(cost=-1.0)
