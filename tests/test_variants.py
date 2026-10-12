"""Unit tests for engine evaluation variants V3, V4, V5 and ablation switches."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lekka_ai.eval.dataset import EvalExample
from lekka_ai.eval.runner import EvalRunner
from lekka_ai.eval.variants import (
    V3StaticVariant,
    V4DynamicVariant,
    V5DynamicVariant,
    VariantConfig,
    build_ablated_profiles,
)
from lekka_ai.llm.base import LLMResponse
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.orchestrator.dispatch import TaskResult
from lekka_ai.orchestrator.state import PlannedTask
from lekka_ai.schemas.claims import Claim, ClaimType, Evidence

# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------


def _make_sample_task_result(task: PlannedTask) -> TaskResult:
    """Mock TaskResult generator creating grounded claims per agent type."""
    ev = Evidence(
        source_type="sec_filing",
        document_id="DOC_01",
        tool_call_id="tc_01",
        raw_value={"value": 100.0},
    )
    claim = Claim(
        claim_id=f"c_{task.agent_type}_01",
        text=f"{task.agent_type.capitalize()} finding for {task.task_id}",
        type=ClaimType.FACT,
        confidence=0.95,
        produced_by=task.agent_type,
        evidence=[ev],
    )
    return TaskResult(
        task_id=task.task_id,
        agent_type=task.agent_type,
        status="completed",
        claims=[claim],
        usage={"tool_calls": 2, "tokens_in": 150, "tokens_out": 50},
        confidence=0.95,
    )


# ---------------------------------------------------------------------------
# Ablation Configuration Tests
# ---------------------------------------------------------------------------


def test_build_ablated_profiles_risk_agent_off() -> None:
    """When risk_agent is False, risk profile is excluded from roster."""
    cfg = VariantConfig(risk_agent=False)
    profiles = build_ablated_profiles(cfg)
    assert "risk" not in profiles
    assert "fundamental" in profiles
    assert "valuation" in profiles


def test_build_ablated_profiles_memory_off() -> None:
    """When memory is False, recall_facts tool is stripped from all profiles."""
    cfg = VariantConfig(memory=False)
    profiles = build_ablated_profiles(cfg)
    for name, prof in profiles.items():
        assert "recall_facts" not in prof.allowed_tools, (
            f"Profile {name} should not have recall_facts"
        )


def test_build_ablated_profiles_run_python_only() -> None:
    """When quant_tools_mode is run_python_only, quant tools are removed."""
    cfg = VariantConfig(quant_tools_mode="run_python_only")
    profiles = build_ablated_profiles(cfg)
    quant_tools = {"run_dcf", "run_reverse_dcf", "compute_multiples"}

    val_tools = set(profiles["valuation"].allowed_tools)
    assert not (val_tools & quant_tools)
    assert "run_python" in val_tools


# ---------------------------------------------------------------------------
# Variant V3 Static Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v3_static_variant_runs_all_five_agents() -> None:
    """V3 static variant dispatches all 5 agents simultaneously in 1 round."""
    dispatched_agents: list[str] = []

    async def mock_runner(task: PlannedTask, scratch_dir: Path) -> TaskResult:
        dispatched_agents.append(task.agent_type)
        return _make_sample_task_result(task)

    fake_llm = FakeLLM(responses=[LLMResponse(text="Synthesis report.")])
    v3 = V3StaticVariant(llm=fake_llm, task_runner=mock_runner)

    result = await v3.run("What is Apple valuation and market risk?")
    assert result.variant == "v3_static"
    assert result.status == "completed"
    assert len(result.claims) == 5
    assert set(dispatched_agents) == {
        "fundamental",
        "valuation",
        "market",
        "news",
        "risk",
    }
    assert result.metadata.get("rounds") == 1


@pytest.mark.asyncio
async def test_v3_static_variant_without_risk_agent() -> None:
    """V3 with risk_agent=False runs only 4 non-risk agents."""
    dispatched_agents: list[str] = []

    async def mock_runner(task: PlannedTask, scratch_dir: Path) -> TaskResult:
        dispatched_agents.append(task.agent_type)
        return _make_sample_task_result(task)

    fake_llm = FakeLLM(responses=[LLMResponse(text="Report")])
    cfg = VariantConfig(risk_agent=False)
    v3 = V3StaticVariant(llm=fake_llm, config=cfg, task_runner=mock_runner)

    result = await v3.run("Analyze Apple.")
    assert len(result.claims) == 4
    assert "risk" not in dispatched_agents


# ---------------------------------------------------------------------------
# Variant V4 Dynamic Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v4_dynamic_variant_executes_planner_wave() -> None:
    """V4 dynamic variant plans tasks dynamically and runs 1 round."""
    # Mock planner JSON response selecting fundamental and valuation
    understand_payload = {
        "intent": "company_research",
        "tickers": ["AAPL"],
        "metrics": ["revenue"],
        "needs_clarification": False,
    }
    planner_payload = {
        "tasks": [
            {
                "task_id": "dyn_fund_01",
                "agent_type": "fundamental",
                "instruction": "Fetch AAPL revenues",
                "priority": 1,
            },
            {
                "task_id": "dyn_val_02",
                "agent_type": "valuation",
                "instruction": "Evaluate multiples",
                "priority": 1,
            },
        ]
    }
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(text=json.dumps(understand_payload)),
            LLMResponse(text=json.dumps(planner_payload)),
            LLMResponse(text="V4 Final Synthesis."),
        ]
    )

    dispatched_agents: list[str] = []

    async def mock_runner(task: PlannedTask, scratch_dir: Path) -> TaskResult:
        dispatched_agents.append(task.agent_type)
        return _make_sample_task_result(task)

    v4 = V4DynamicVariant(llm=fake_llm, task_runner=mock_runner)
    result = await v4.run("Analyze Apple revenue multiples.")

    assert result.variant == "v4_dynamic"
    assert result.status == "completed"
    assert set(dispatched_agents) == {"fundamental", "valuation"}
    assert len(result.claims) == 2


# ---------------------------------------------------------------------------
# Variant V5 Dynamic Re-planning Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v5_dynamic_replan_variant() -> None:
    """V5 dynamic variant executes planner + gap check + verifier."""
    understand_payload = {
        "intent": "company_research",
        "tickers": ["MSFT"],
        "metrics": ["growth"],
        "needs_clarification": False,
    }
    planner_payload = {
        "tasks": [
            {
                "task_id": "v5_task_01",
                "agent_type": "market",
                "instruction": "Analyze market price",
                "priority": 1,
            }
        ]
    }
    gap_payload = {
        "action": "done",
        "follow_up_tasks": [],
        "reasons": ["Sufficient market data gathered."],
    }
    verifier_payload = {
        "verdicts": [
            {
                "claim_id": "c_market_01",
                "supported": "yes",
                "reason": "Matches trading feed",
            }
        ]
    }
    fake_llm = FakeLLM(
        responses=[
            LLMResponse(text=json.dumps(understand_payload)),
            LLMResponse(text=json.dumps(planner_payload)),
            LLMResponse(text=json.dumps(gap_payload)),
            LLMResponse(text=json.dumps(verifier_payload)),
            LLMResponse(text="V5 Comprehensive Synthesis."),
        ]
    )

    async def mock_runner(task: PlannedTask, scratch_dir: Path) -> TaskResult:
        return _make_sample_task_result(task)

    v5 = V5DynamicVariant(llm=fake_llm, task_runner=mock_runner)
    result = await v5.run("What is Microsoft market performance?")

    assert result.variant == "v5_dynamic"
    assert result.status == "completed"
    assert len(result.claims) >= 1
    assert "verifier_verdicts" in result.metadata


# ---------------------------------------------------------------------------
# Verification on 2-Question Dataset Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_variants_run_on_two_question_dataset() -> None:
    """Verify V3, V4, and V5 run on a 2-question benchmark dataset."""
    dataset = [
        EvalExample(
            id="q_apple",
            question="What was Apple revenue for 2024?",
            ticker="AAPL",
            as_of_date="2024-09-30",
        ),
        EvalExample(
            id="q_nvidia",
            question="What drove Nvidia Data Center revenue?",
            ticker="NVDA",
            as_of_date="2024-07-31",
        ),
    ]

    async def mock_runner(task: PlannedTask, scratch_dir: Path) -> TaskResult:
        return _make_sample_task_result(task)

    # Prepare FakeLLM responses for query understanding, planning, gapcheck, synthesis
    task_json = json.dumps(
        {
            "tasks": [
                {
                    "task_id": "t1",
                    "agent_type": "fundamental",
                    "instruction": "Check revenue",
                    "priority": 1,
                }
            ]
        }
    )
    gap_json = json.dumps(
        {"action": "done", "follow_up_tasks": [], "reasons": ["Complete"]}
    )
    verdict_json = json.dumps(
        {
            "verdicts": [
                {
                    "claim_id": "c_fundamental_01",
                    "supported": "yes",
                    "reason": "Verified",
                }
            ]
        }
    )
    llm_responses = [
        LLMResponse(text='{"intent": "research", "tickers": ["AAPL"]}'),
        LLMResponse(text=task_json),
        LLMResponse(text=gap_json),
        LLMResponse(text=verdict_json),
        LLMResponse(text="Report answer"),
    ] * 10
    fake_llm = FakeLLM(responses=llm_responses)

    v3 = V3StaticVariant(llm=fake_llm, task_runner=mock_runner)
    v4 = V4DynamicVariant(llm=fake_llm, task_runner=mock_runner)
    v5 = V5DynamicVariant(llm=fake_llm, task_runner=mock_runner)

    runner = EvalRunner()
    summary = await runner.run(dataset=dataset, variants=[v3, v4, v5])

    # 3 variants x 2 questions = 6 total results
    assert len(summary.results) == 6
    assert "v3_static" in summary.variants
    assert "v4_dynamic" in summary.variants
    assert "v5_dynamic" in summary.variants

    for _var_name, agg in summary.variants.items():
        assert agg.count == 2
        assert agg.success_rate == 1.0
