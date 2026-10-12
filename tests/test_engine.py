"""Tests for main orchestrator engine loop and stopping rules."""

import uuid
from pathlib import Path

import pytest

from lekka_ai.harness.runtime import AgentRunResult
from lekka_ai.llm.base import LLMResponse
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.orchestrator.budget import RunBudget
from lekka_ai.orchestrator.dispatch import TaskResult
from lekka_ai.orchestrator.engine import (
    OrchestrationEngine,
    RunOutcome,
    run_engine,
)
from lekka_ai.orchestrator.gapcheck import GapCheckAction
from lekka_ai.orchestrator.state import Plan, PlannedTask, RunState


@pytest.mark.asyncio
async def test_engine_stops_when_done() -> None:
    """Verify engine completes cleanly when gap check returns done."""
    state = RunState(
        run_id=str(uuid.uuid4()),
        question="What is the latest closing price and volatility for AAPL?",
        plan=Plan(),
    )
    task_1 = PlannedTask(
        task_id="task_market_01",
        agent_type="market",
        instruction="Evaluate AAPL price action as of 2024-01-12",
        priority=1,
    )
    state.plan.add_task(task_1)

    async def fake_runner(task: PlannedTask, scratch_dir: Path) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            status="completed",
            claims=[
                {
                    "claim_id": "c1",
                    "text": "AAPL closed at $185.92 on 2024-01-12",
                    "type": "fact",
                    "confidence": 0.98,
                }
            ],
            confidence=0.98,
            confidence_rationale="Verified against historical market bars.",
        )

    gap_response = LLMResponse(
        text="""{
  "action": "done",
  "follow_up_tasks": [],
  "reasons": ["All market price metrics verified with high confidence."]
}"""
    )
    fake_gap_llm = FakeLLM(responses=[gap_response])

    outcome = await run_engine(
        question=state.question,
        initial_state=state,
        gapcheck_llm=fake_gap_llm,
        task_runner=fake_runner,
    )

    assert isinstance(outcome, RunOutcome)
    assert outcome.status == "completed"
    assert outcome.stop_reason == "done"
    assert outcome.rounds == 1
    assert outcome.is_successful is True
    assert outcome.state.is_complete() is True
    assert outcome.decision is not None
    assert outcome.decision.action == GapCheckAction.DONE


@pytest.mark.asyncio
async def test_engine_stops_at_max_rounds() -> None:
    """Verify engine halts with max_rounds_reached when max_rounds is exceeded."""
    state = RunState(
        run_id=str(uuid.uuid4()),
        question="Investigate persistent discrepancy in valuations",
        plan=Plan(),
    )
    task_1 = PlannedTask(
        task_id="task_round_1",
        agent_type="fundamental",
        instruction="Initial fundamental extraction",
        priority=1,
    )
    state.plan.add_task(task_1)

    claim_counter = 0

    async def fake_runner(task: PlannedTask, scratch_dir: Path) -> TaskResult:
        nonlocal claim_counter
        claim_counter += 1
        return TaskResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            status="completed",
            claims=[
                {
                    "claim_id": f"claim_{claim_counter}",
                    "text": f"Discovered novel evidence snippet {claim_counter}",
                    "type": "fact",
                    "confidence": 0.90,
                }
            ],
            confidence=0.90,
        )

    # Responses requesting follow-ups across 3 rounds
    gap_resp_1 = LLMResponse(
        text="""{
  "action": "follow_up",
  "follow_up_tasks": [
    {
      "task_id": "task_round_2",
      "agent_type": "valuation",
      "instruction": "Follow-up round 2 DCF check"
    }
  ],
  "reasons": ["Need round 2 DCF check"]
}"""
    )
    gap_resp_2 = LLMResponse(
        text="""{
  "action": "follow_up",
  "follow_up_tasks": [
    {
      "task_id": "task_round_3",
      "agent_type": "news",
      "instruction": "Follow-up round 3 news check"
    }
  ],
  "reasons": ["Need round 3 news check"]
}"""
    )
    gap_resp_3 = LLMResponse(
        text="""{
  "action": "follow_up",
  "follow_up_tasks": [
    {
      "task_id": "task_round_4",
      "agent_type": "fundamental",
      "instruction": "Follow-up round 4 check"
    }
  ],
  "reasons": ["Still unresolved"]
}"""
    )
    fake_gap_llm = FakeLLM(responses=[gap_resp_1, gap_resp_2, gap_resp_3])

    engine = OrchestrationEngine(
        max_rounds=3,
        gapcheck_llm=fake_gap_llm,
        task_runner=fake_runner,
    )
    outcome = await engine.run(
        question=state.question,
        initial_state=state,
    )

    assert outcome.status == "max_rounds_reached"
    assert outcome.stop_reason == "max_rounds"
    assert outcome.rounds == 3
    assert outcome.is_successful is False


@pytest.mark.asyncio
async def test_engine_stops_on_budget() -> None:
    """Verify engine stops task launches when budget is exhausted."""
    state = RunState(
        run_id=str(uuid.uuid4()),
        question="Evaluate valuation scenarios",
        plan=Plan(),
    )
    task_1 = PlannedTask(
        task_id="task_expensive_01",
        agent_type="valuation",
        instruction="Run heavy DCF modeling",
        priority=1,
    )
    state.plan.add_task(task_1)

    # Restrict budget to only 100 tokens
    budget = RunBudget(
        tokens=100,
        tool_calls=5,
        seconds=60.0,
        cost=1.0,
    )

    async def fake_runner(task: PlannedTask, scratch_dir: Path) -> TaskResult:
        # Simulate heavy consumption exhausting the budget
        run_result = AgentRunResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            status="completed",
            tokens_in=80,
            tokens_out=80,  # 160 tokens total > 100 allowance
            tool_calls_count=2,
        )
        budget.reconcile(run_result, elapsed_seconds=1.0)
        return TaskResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            status="completed",
            claims=[
                {
                    "claim_id": "c_exp",
                    "text": "DCF fair value is $195.00",
                    "type": "calculation",
                    "confidence": 0.90,
                }
            ],
            confidence=0.90,
        )

    gap_response = LLMResponse(
        text="""{
  "action": "follow_up",
  "follow_up_tasks": [
    {
      "task_id": "task_follow_up_blocked",
      "agent_type": "valuation",
      "instruction": "Should not be executed due to budget exhaustion"
    }
  ],
  "reasons": ["Requesting another round of valuation checks"]
}"""
    )
    fake_gap_llm = FakeLLM(responses=[gap_response])

    outcome = await run_engine(
        question=state.question,
        initial_state=state,
        budget=budget,
        gapcheck_llm=fake_gap_llm,
        task_runner=fake_runner,
    )

    assert outcome.status == "budget_exhausted"
    assert outcome.stop_reason == "budget_exhausted"
    assert outcome.budget is not None
    assert outcome.budget.is_exhausted() is True
    # The blocked follow-up task was never executed
    assert "task_follow_up_blocked" not in state.task_statuses


@pytest.mark.asyncio
async def test_engine_stops_on_no_new_evidence() -> None:
    """Verify engine stops when follow-up wave yields no novel claims."""
    state = RunState(
        run_id=str(uuid.uuid4()),
        question="Analyze AAPL market catalysts",
        plan=Plan(),
    )
    task_1 = PlannedTask(
        task_id="task_wave_1",
        agent_type="market",
        instruction="Analyze price action",
        priority=1,
    )
    state.plan.add_task(task_1)

    async def repeating_runner(task: PlannedTask, scratch_dir: Path) -> TaskResult:
        # Both round 1 and round 2 return the identical claim text
        return TaskResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            status="completed",
            claims=[
                {
                    "claim_id": f"claim_{task.task_id}",
                    "text": "AAPL closing price was $185.92 on 2024-01-12",
                    "type": "fact",
                    "confidence": 0.95,
                }
            ],
            confidence=0.95,
        )

    # Gap check in round 1 asks for follow-up
    gap_response = LLMResponse(
        text="""{
  "action": "follow_up",
  "follow_up_tasks": [
    {
      "task_id": "task_wave_2",
      "agent_type": "market",
      "instruction": "Re-check price volatility"
    }
  ],
  "reasons": ["Check for additional price evidence"]
}"""
    )
    fake_gap_llm = FakeLLM(responses=[gap_response])

    outcome = await run_engine(
        question=state.question,
        initial_state=state,
        gapcheck_llm=fake_gap_llm,
        task_runner=repeating_runner,
    )

    assert outcome.status == "completed"
    assert outcome.stop_reason == "no_new_evidence"
    assert outcome.rounds == 2
    assert outcome.is_successful is True


@pytest.mark.asyncio
async def test_engine_stops_on_abstain() -> None:
    """Verify engine halts when gap check returns abstain."""
    state = RunState(
        run_id=str(uuid.uuid4()),
        question="Evaluate unresolvable data",
        plan=Plan(),
    )
    task_1 = PlannedTask(
        task_id="task_fail",
        agent_type="fundamental",
        instruction="Check suspended filings",
        priority=1,
    )
    state.plan.add_task(task_1)

    async def fake_runner(task: PlannedTask, scratch_dir: Path) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            status="insufficient_evidence",
            claims=[],
            confidence=0.1,
            error="Filings permanently unavailable",
        )

    gap_response = LLMResponse(
        text="""{
  "action": "abstain",
  "follow_up_tasks": [],
  "reasons": ["Filings unavailable; cannot evaluate."]
}"""
    )
    fake_gap_llm = FakeLLM(responses=[gap_response])

    outcome = await run_engine(
        question=state.question,
        initial_state=state,
        gapcheck_llm=fake_gap_llm,
        task_runner=fake_runner,
    )

    assert outcome.status == "abstained"
    assert outcome.stop_reason == "abstain"
    assert outcome.is_successful is False


@pytest.mark.asyncio
async def test_engine_end_to_end_understand_plan_execute() -> None:
    """Verify full loop: understand -> plan -> run wave -> gap check -> done."""
    # 1. Mock understand response
    understand_resp = LLMResponse(
        text="""{
  "intent": "move_explanation",
  "tickers": ["AAPL"],
  "horizon": "short_term",
  "risk_focus": "price_drop",
  "as_of_date": "2024-01-12",
  "ambiguities": [],
  "needs_clarification": false
}"""
    )
    fake_understand_llm = FakeLLM(responses=[understand_resp])

    # 2. Mock planner response
    plan_resp = LLMResponse(
        text="""{
  "tasks": [
    {
      "task_id": "task_mkt_drop",
      "agent_type": "market",
      "instruction": "Evaluate AAPL 1-day return and volatility on 2024-01-12",
      "depends_on": [],
      "priority": 1,
      "expected_cost": 0.0
    }
  ]
}"""
    )
    fake_planner_llm = FakeLLM(responses=[plan_resp])

    # 3. Runner for wave
    async def fake_runner(task: PlannedTask, scratch_dir: Path) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            status="completed",
            claims=[
                {
                    "claim_id": "c_drop",
                    "text": "AAPL dropped 2.1% on 2024-01-12",
                    "type": "fact",
                    "confidence": 0.97,
                }
            ],
            confidence=0.97,
        )

    # 4. Mock gap check response
    gap_resp = LLMResponse(
        text="""{
  "action": "done",
  "follow_up_tasks": [],
  "reasons": ["Price drop movement explained by market data."]
}"""
    )
    fake_gap_llm = FakeLLM(responses=[gap_resp])

    outcome = await run_engine(
        question="Why did AAPL drop on 2024-01-12?",
        understand_llm=fake_understand_llm,
        planner_llm=fake_planner_llm,
        gapcheck_llm=fake_gap_llm,
        task_runner=fake_runner,
    )

    assert outcome.status == "completed"
    assert outcome.stop_reason == "done"
    assert outcome.rounds == 1
    assert outcome.state.parsed_query is not None
    assert outcome.state.parsed_query.get("intent") == "move_explanation"
    assert len(outcome.state.plan.tasks) == 1
    assert outcome.state.is_complete() is True
