"""Tests for parent orchestrator gap check auditing wave results."""

import uuid
from typing import Any

import pytest

from lekka_ai.llm.base import LLMResponse, Usage
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.orchestrator.dispatch import TaskResult
from lekka_ai.orchestrator.gapcheck import (
    GapCheckAction,
    gap_check,
)
from lekka_ai.orchestrator.state import Plan, RunState


@pytest.mark.asyncio
async def test_gap_check_conflicting_facts_yields_follow_up() -> None:
    """Verify conflict between specialist claims yields follow_up decision."""
    task_1 = TaskResult(
        task_id="task_fundamental_01",
        agent_type="fundamental",
        status="completed",
        confidence=0.98,
        confidence_rationale="Extracted from audited 10-K filing.",
        claims=[
            {
                "claim_id": "claim_fy24_rev_audited",
                "text": "Apple FY2024 total net sales reached $391.0B",
                "type": "fact",
                "confidence": 0.98,
                "evidence": [
                    {
                        "source_type": "sec_filing",
                        "raw_value": "391035000000",
                        "as_of_date": "2024-09-28",
                    }
                ],
            }
        ],
        assumptions={},
        open_questions=[],
    )

    task_2 = TaskResult(
        task_id="task_valuation_01",
        agent_type="valuation",
        status="completed",
        confidence=0.88,
        confidence_rationale="DCF scenario modeling.",
        claims=[
            {
                "claim_id": "claim_dcf_baseline",
                "text": "DCF baseline assumes Apple FY2024 revenue of $383.3B",
                "type": "calculation",
                "confidence": 0.88,
                "evidence": [],
            }
        ],
        assumptions={"fy2024_base_revenue": "$383.3B"},
        open_questions=[],
    )

    conflict_json = """```json
{
  "action": "follow_up",
  "follow_up_tasks": [
    {
      "task_id": "task_reconcile_rev_conflict",
      "agent_type": "valuation",
      "instruction": "Re-run DCF baseline using audited FY2024 sales of $391.0B",
      "depends_on": ["task_fundamental_01", "task_valuation_01"],
      "priority": 3,
      "expected_cost": 0.0
    }
  ],
  "reasons": [
    "Conflicting revenue facts: fundamental verified $391.0B vs valuation $383.3B",
    "Valuation baseline must reconcile with audited 10-K figures"
  ]
}
```"""
    fake_response = LLMResponse(
        text=conflict_json,
        usage=Usage(input_tokens=220, output_tokens=95),
    )

    fake_llm = FakeLLM(responses=[fake_response])

    decision = await gap_check(
        task_results=[task_1, task_2],
        question="What is Apple's intrinsic valuation based on FY2024 results?",
        llm=fake_llm,
    )

    # Core requirement verification: conflict situation yields a follow_up
    assert decision.action == GapCheckAction.FOLLOW_UP
    assert decision.action.value == "follow_up"
    assert len(decision.follow_up_tasks) == 1
    assert any("conflict" in r.lower() for r in decision.reasons)

    # Follow-up task specifications
    follow_up = decision.follow_up_tasks[0]
    assert follow_up.task_id == "task_reconcile_rev_conflict"
    assert follow_up.agent_type == "valuation"
    assert "re-run dcf" in follow_up.instruction.lower()

    # PlannedTask conversion
    planned_tasks = decision.to_planned_tasks()
    assert len(planned_tasks) == 1
    assert planned_tasks[0].task_id == "task_reconcile_rev_conflict"
    assert planned_tasks[0].priority == 3


@pytest.mark.asyncio
async def test_gap_check_clean_consistent_results_yields_done() -> None:
    """Verify consistent high-confidence results yields done decision."""
    task_1 = TaskResult(
        task_id="task_market_01",
        agent_type="market",
        status="completed",
        confidence=0.96,
        confidence_rationale="Market price action verified.",
        claims=[
            {
                "claim_id": "claim_aapl_price",
                "text": "AAPL closing price was $185.92 on 2024-01-12",
                "type": "fact",
                "confidence": 0.99,
                "evidence": [
                    {
                        "source_type": "market_data",
                        "raw_value": "185.92",
                        "as_of_date": "2024-01-12",
                    }
                ],
            }
        ],
        assumptions={},
        open_questions=[],
    )

    task_2 = TaskResult(
        task_id="task_news_01",
        agent_type="news",
        status="completed",
        confidence=0.93,
        confidence_rationale="Verified SEC 8-K product release.",
        claims=[
            {
                "claim_id": "claim_aapl_catalyst",
                "text": "Apple announced leadership transition in 8-K filing",
                "type": "fact",
                "confidence": 0.95,
                "evidence": [
                    {
                        "source_type": "sec_filing",
                        "raw_value": "Luca Maestri CFO transition",
                        "as_of_date": "2024-08-26",
                    }
                ],
            }
        ],
        assumptions={},
        open_questions=[],
    )

    clean_json = """{
  "action": "done",
  "follow_up_tasks": [],
  "reasons": [
    "Market price metrics and news catalysts verified with confidence >0.90.",
    "No factual contradictions or ungrounded assumptions detected.",
    "Evidence is complete to synthesize the response."
  ]
}"""
    fake_response = LLMResponse(
        text=clean_json,
        usage=Usage(input_tokens=180, output_tokens=60),
    )

    fake_llm = FakeLLM(responses=[fake_response])

    decision = await gap_check(
        task_results=[task_1, task_2],
        question="Why did AAPL move and what are recent catalysts?",
        llm=fake_llm,
    )

    assert decision.action == GapCheckAction.DONE
    assert decision.action.value == "done"
    assert len(decision.follow_up_tasks) == 0
    assert any("verified" in r.lower() for r in decision.reasons)


@pytest.mark.asyncio
async def test_gap_check_unresolvable_data_yields_abstain() -> None:
    """Verify permanent data unavailability or unanswerable scenario yields abstain."""
    task_1 = TaskResult(
        task_id="task_fundamental_suspended",
        agent_type="fundamental",
        status="insufficient_evidence",
        confidence=0.15,
        confidence_rationale="No 10-K filings available on SEC EDGAR.",
        claims=[],
        assumptions={},
        open_questions=["SEC reporting suspended following company delisting in 2021"],
        error="Company filings unavailable on SEC EDGAR",
    )

    abstain_json = """{
  "action": "abstain",
  "follow_up_tasks": [],
  "reasons": [
    "Audited SEC disclosures are permanently unavailable due to delisting.",
    "Cannot responsibly complete financial evaluation without SEC reports."
  ]
}"""
    fake_response = LLMResponse(
        text=abstain_json,
        usage=Usage(input_tokens=150, output_tokens=50),
    )

    fake_llm = FakeLLM(responses=[fake_response])

    decision = await gap_check(
        task_results=[task_1],
        question="Provide a fundamental valuation of delisted company XYZ",
        llm=fake_llm,
    )

    assert decision.action == GapCheckAction.ABSTAIN
    assert decision.action.value == "abstain"
    assert len(decision.follow_up_tasks) == 0
    assert any("unavailable" in r.lower() for r in decision.reasons)


@pytest.mark.asyncio
async def test_gap_check_extracts_from_run_state() -> None:
    """Verify gap_check extracts task results from RunState when not passed directly."""
    state = RunState(
        run_id=str(uuid.uuid4()),
        question="Check AAPL market performance",
        plan=Plan(),
    )
    result_dict: dict[str, Any] = {
        "task_id": "task_mkt",
        "agent_type": "market",
        "status": "completed",
        "confidence": 0.95,
        "claims": [
            {
                "claim_id": "c1",
                "text": "AAPL closed at $185.92",
                "type": "fact",
                "confidence": 0.95,
                "evidence": [],
            }
        ],
        "assumptions": {},
        "open_questions": [],
    }
    state.results["task_mkt"] = result_dict

    fake_response = LLMResponse(
        text="""{
  "action": "done",
  "follow_up_tasks": [],
  "reasons": ["Single market task completed with verified data."]
}"""
    )
    fake_llm = FakeLLM(responses=[fake_response])

    decision = await gap_check(state=state, llm=fake_llm)

    assert decision.action == GapCheckAction.DONE
    assert len(fake_llm.calls) == 1
    prompt_sent = fake_llm.calls[0]["messages"][0].content
    assert "AAPL closed at $185.92" in prompt_sent


@pytest.mark.asyncio
async def test_gap_check_empty_results_failsafe() -> None:
    """Verify empty task results returns safe follow_up decision without calling LLM."""
    decision = await gap_check(task_results=[])
    assert decision.action == GapCheckAction.FOLLOW_UP
    assert len(decision.follow_up_tasks) == 0
    assert "no specialist task results" in decision.reasons[0].lower()
