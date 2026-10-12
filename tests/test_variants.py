"""Tests for V3 static, V4 dynamic, and V5 dynamic engine evaluation variants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lekka_ai.eval.dataset import EvalExample
from lekka_ai.eval.runner import EvalRunner
from lekka_ai.eval.variants import (
    V3StaticVariant,
    V4DynamicVariant,
    V5DynamicVariant,
    VariantConfig,
    apply_ablation_to_profile,
)
from lekka_ai.harness.profiles import load_agent_profile
from lekka_ai.llm.base import LLMResponse
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.orchestrator.dispatch import TaskResult
from lekka_ai.orchestrator.planner import PlannedTaskSpec, PlanSpec
from lekka_ai.orchestrator.state import PlannedTask
from lekka_ai.orchestrator.understand import ParsedQuery
from lekka_ai.schemas.claims import Claim, ClaimType, Evidence


def _build_test_claims(agent_type: str) -> list[Claim]:
    """Helper creating valid Grounded claims for unit test runs."""
    c1 = Claim(
        claim_id=f"claim_{agent_type}_1",
        text=f"{agent_type.capitalize()} analysis verified with source evidence.",
        type=ClaimType.FACT,
        evidence=[
            Evidence(
                source_type="sec_filing",
                document_id="10-K",
                tool_call_id=f"call_{agent_type}_01",
            )
        ],
        produced_by=agent_type,
        confidence=0.92,
    )
    c2 = Claim(
        claim_id=f"claim_{agent_type}_2",
        text=f"Derived valuation multiple or trend for {agent_type}.",
        type=ClaimType.INTERPRETATION,
        produced_by=agent_type,
        confidence=0.88,
        depends_on=[c1.claim_id],
    )
    return [c1, c2]


def _build_mock_task_runner() -> Any:
    """Create a task runner returning valid TaskResult with claims."""

    async def _runner(task: PlannedTask, scratch_dir: Path) -> TaskResult:
        claims = _build_test_claims(task.agent_type)
        return TaskResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            status="completed",
            claims=claims,
            usage={
                "tokens_in": 120,
                "tokens_out": 45,
                "tool_calls": 2,
                "elapsed_seconds": 0.05,
            },
            confidence=0.91,
            confidence_rationale=f"Completed research for {task.agent_type}",
        )

    return _runner


def test_apply_ablation_to_profile() -> None:
    """Test ablation switches modifying agent allowed_tools."""
    raw_profile = load_agent_profile("valuation")

    # 1. Default config: tools unmodified
    cfg_default = VariantConfig()
    p_default = apply_ablation_to_profile(raw_profile, cfg_default)
    assert "recall_facts" in p_default.allowed_tools
    assert "run_dcf" in p_default.allowed_tools

    # 2. Memory ablated (memory=False) -> recall_facts removed
    cfg_no_mem = VariantConfig(memory=False)
    p_no_mem = apply_ablation_to_profile(raw_profile, cfg_no_mem)
    assert "recall_facts" not in p_no_mem.allowed_tools
    assert "run_dcf" in p_no_mem.allowed_tools

    # 3. Quant tools ablated (quant_tools=False) ->
    # quant tools removed, run_python added
    cfg_no_quant = VariantConfig(quant_tools=False)
    p_no_quant = apply_ablation_to_profile(raw_profile, cfg_no_quant)
    assert "run_dcf" not in p_no_quant.allowed_tools
    assert "run_reverse_dcf" not in p_no_quant.allowed_tools
    assert "run_python" in p_no_quant.allowed_tools
    assert "recall_facts" in p_no_quant.allowed_tools


@pytest.mark.asyncio
async def test_v3_static_variant_all_five_agents() -> None:
    """Test V3 static variant always executes all 5 specialist agents in 1 round."""
    runner = _build_mock_task_runner()
    v3 = V3StaticVariant(
        config=VariantConfig(verifier=False, risk_agent=True),
        task_runner=runner,
    )

    result = await v3.run("Analyze NVDA balance sheet and risk factors")

    assert result.status == "completed"
    assert result.variant == "v3_static"
    assert result.metadata["rounds"] == 1
    dispatched = result.metadata["dispatched_agents"]
    assert len(dispatched) == 5
    assert set(dispatched) == {"fundamental", "valuation", "market", "news", "risk"}
    assert len(result.claims) > 0


@pytest.mark.asyncio
async def test_v3_static_variant_risk_ablation() -> None:
    """Test V3 static variant respects risk_agent=False ablation."""
    runner = _build_mock_task_runner()
    v3_no_risk = V3StaticVariant(
        config=VariantConfig(verifier=False, risk_agent=False),
        task_runner=runner,
    )

    result = await v3_no_risk.run("Analyze NVDA")

    assert result.status == "completed"
    dispatched = result.metadata["dispatched_agents"]
    assert len(dispatched) == 4
    assert "risk" not in dispatched
    assert set(dispatched) == {"fundamental", "valuation", "market", "news"}


@pytest.mark.asyncio
async def test_v3_static_variant_verifier_ablation() -> None:
    """Test V3 verifier switch: verdicts present when True, empty when False."""
    runner = _build_mock_task_runner()

    # Case A: verifier = False
    v3_no_verif = V3StaticVariant(
        config=VariantConfig(verifier=False),
        task_runner=runner,
    )
    res_no_verif = await v3_no_verif.run("Check MSFT metrics")
    assert len(res_no_verif.metadata["verifier_verdicts"]) == 0

    # Case B: verifier = True with FakeLLM
    verif_llm = FakeLLM(
        responses=[
            LLMResponse(
                text='[{"claim_id": "c1", "supported": "yes", "reason": "Accurate"}]'
            )
        ]
    )
    v3_verif = V3StaticVariant(
        llm=verif_llm,
        config=VariantConfig(verifier=True),
        task_runner=runner,
    )
    res_verif = await v3_verif.run("Check MSFT metrics")
    assert len(res_verif.metadata["verifier_verdicts"]) > 0


@pytest.mark.asyncio
async def test_v4_dynamic_variant_execution() -> None:
    """Test V4 dynamic variant plans subset of agents dynamically in 1 round."""
    runner = _build_mock_task_runner()

    # Mock planner to select only market and news for a price movement question
    understand_fake = FakeLLM(
        responses=[
            LLMResponse(
                structured_output=ParsedQuery(
                    intent="move_explanation",
                    tickers=["AAPL"],
                    needs_clarification=False,
                )
            )
        ]
    )
    planner_fake = FakeLLM(
        responses=[
            LLMResponse(
                structured_output=PlanSpec(
                    tasks=[
                        PlannedTaskSpec(
                            task_id="t_mkt",
                            agent_type="market",
                            instruction="Gather AAPL price history and volatility",
                            priority=1,
                        ),
                        PlannedTaskSpec(
                            task_id="t_news",
                            agent_type="news",
                            instruction="Retrieve AAPL catalyst news",
                            priority=1,
                        ),
                    ]
                )
            )
        ]
    )

    v4 = V4DynamicVariant(
        understand_llm=understand_fake,
        planner_llm=planner_fake,
        config=VariantConfig(verifier=False, risk_agent=True),
        task_runner=runner,
    )

    result = await v4.run("Why did AAPL drop 3% today?")

    assert result.status == "completed"
    assert result.variant == "v4_dynamic"
    assert result.metadata["rounds"] == 1
    dispatched = result.metadata["dispatched_agents"]
    assert set(dispatched) == {"market", "news"}
    assert "fundamental" not in dispatched
    assert "valuation" not in dispatched
    assert "risk" not in dispatched


@pytest.mark.asyncio
async def test_v4_dynamic_variant_needs_clarification() -> None:
    """Test V4 cleanly handles ambiguous questions requiring clarification."""
    understand_fake = FakeLLM(
        responses=[
            LLMResponse(
                structured_output=ParsedQuery(
                    intent="investment_evaluation",
                    tickers=[],
                    needs_clarification=True,
                    ambiguities=["Which company are you evaluating?"],
                )
            )
        ]
    )

    v4 = V4DynamicVariant(
        understand_llm=understand_fake,
        config=VariantConfig(verifier=False),
    )

    result = await v4.run("Is the stock a good buy?")

    assert result.status == "needs_clarification"
    assert result.is_abstention is True
    assert "Which company" in result.answer


@pytest.mark.asyncio
async def test_v5_dynamic_variant_multiround_and_risk_wave() -> None:
    """Test V5 executes dynamic plan, mandatory risk wave, and multi-round gap check."""
    runner = _build_mock_task_runner()

    understand_fake = FakeLLM(
        responses=[
            LLMResponse(
                structured_output=ParsedQuery(
                    intent="investment_evaluation",
                    tickers=["TSLA"],
                    needs_clarification=False,
                )
            )
        ]
    )
    planner_fake = FakeLLM(
        responses=[
            LLMResponse(
                structured_output=PlanSpec(
                    tasks=[
                        PlannedTaskSpec(
                            task_id="t_fund",
                            agent_type="fundamental",
                            instruction="Evaluate TSLA margins",
                            priority=1,
                        ),
                    ]
                )
            )
        ]
    )
    # Round 1 gap check requests follow-up task; Round 2 concludes done
    gapcheck_fake = FakeLLM(
        responses=[
            LLMResponse(
                text=(
                    '{"action": "follow_up", "follow_up_tasks": ['
                    '{"task_id": "t_val_followup", "agent_type": "valuation", '
                    '"instruction": "Model reverse DCF", "priority": 1, '
                    '"depends_on": []}],'
                    '"reasons": ["Need valuation multiple grounding"]}'
                )
            ),
            LLMResponse(
                text=(
                    '{"action": "done", "follow_up_tasks": [], '
                    '"reasons": ["All questions resolved"]}'
                )
            ),
        ]
    )
    verif_fake = FakeLLM(
        responses=[
            LLMResponse(
                text='[{"claim_id": "c1", "supported": "yes", "reason": "Confirmed"}]'
            )
        ]
    )

    v5 = V5DynamicVariant(
        llm=verif_fake,
        understand_llm=understand_fake,
        planner_llm=planner_fake,
        gapcheck_llm=gapcheck_fake,
        verifier_llm=verif_fake,
        config=VariantConfig(verifier=True, risk_agent=True, max_rounds=3),
        task_runner=runner,
    )

    result = await v5.run("Comprehensive valuation and risk analysis for TSLA")

    assert result.status == "completed"
    assert result.variant == "v5_dynamic"
    assert result.metadata["rounds"] == 2
    dispatched = result.metadata["dispatched_agents"]

    # Verify risk wave executed
    assert "risk" in dispatched
    assert "fundamental" in dispatched
    assert "valuation" in dispatched
    # Verify verifier verdicts recorded
    assert len(result.metadata["verifier_verdicts"]) > 0


@pytest.mark.asyncio
async def test_v5_dynamic_variant_risk_ablation() -> None:
    """Test V5 strips risk wave when risk_agent=False."""
    runner = _build_mock_task_runner()

    understand_fake = FakeLLM(
        responses=[
            LLMResponse(
                structured_output=ParsedQuery(
                    intent="investment_evaluation",
                    tickers=["GOOGL"],
                    needs_clarification=False,
                )
            )
        ]
    )
    planner_fake = FakeLLM(
        responses=[
            LLMResponse(
                structured_output=PlanSpec(
                    tasks=[
                        PlannedTaskSpec(
                            task_id="t_fund",
                            agent_type="fundamental",
                            instruction="Evaluate GOOGL margins",
                            priority=1,
                        ),
                    ]
                )
            )
        ]
    )
    gapcheck_fake = FakeLLM(
        responses=[
            LLMResponse(
                text='{"action": "done", "follow_up_tasks": [], "reasons": ["Done"]}'
            )
        ]
    )

    v5_no_risk = V5DynamicVariant(
        understand_llm=understand_fake,
        planner_llm=planner_fake,
        gapcheck_llm=gapcheck_fake,
        config=VariantConfig(verifier=False, risk_agent=False),
        task_runner=runner,
    )

    result = await v5_no_risk.run("Evaluate GOOGL")

    assert result.status == "completed"
    dispatched = result.metadata["dispatched_agents"]
    assert "risk" not in dispatched
    assert "fundamental" in dispatched


@pytest.mark.asyncio
async def test_all_variants_on_two_question_dataset() -> None:
    """Verify that all variants (V3, V4, V5) run on a 2-question dataset."""
    dataset = [
        EvalExample(
            id="q1_nvda",
            question="What factors drove NVDA gross margin expansion in FY24?",
            ticker="NVDA",
            expected="Data center compute demand drove margin expansion.",
        ),
        EvalExample(
            id="q2_aapl",
            question="Evaluate AAPL valuation and downside risk scenarios.",
            ticker="AAPL",
            expected=("Trading multiples are elevated relative to historical median."),
        ),
    ]

    runner = _build_mock_task_runner()

    # V3 variant
    v3 = V3StaticVariant(
        config=VariantConfig(verifier=False),
        task_runner=runner,
    )

    # V4 variant with fake planner
    understand_fake_v4 = FakeLLM(
        responses=[
            LLMResponse(
                structured_output=ParsedQuery(
                    intent="investment_evaluation",
                    tickers=["NVDA"],
                    needs_clarification=False,
                )
            ),
            LLMResponse(
                structured_output=ParsedQuery(
                    intent="investment_evaluation",
                    tickers=["AAPL"],
                    needs_clarification=False,
                )
            ),
        ]
    )
    planner_fake_v4 = FakeLLM(
        responses=[
            LLMResponse(
                structured_output=PlanSpec(
                    tasks=[
                        PlannedTaskSpec(
                            task_id="t1",
                            agent_type="fundamental",
                            instruction="Review NVDA margins",
                            priority=1,
                        )
                    ]
                )
            ),
            LLMResponse(
                structured_output=PlanSpec(
                    tasks=[
                        PlannedTaskSpec(
                            task_id="t2",
                            agent_type="valuation",
                            instruction="Review AAPL valuation",
                            priority=1,
                        )
                    ]
                )
            ),
        ]
    )
    v4 = V4DynamicVariant(
        understand_llm=understand_fake_v4,
        planner_llm=planner_fake_v4,
        config=VariantConfig(verifier=False),
        task_runner=runner,
    )

    # V5 variant with fake planner and gap check
    understand_fake_v5 = FakeLLM(
        responses=[
            LLMResponse(
                structured_output=ParsedQuery(
                    intent="investment_evaluation",
                    tickers=["NVDA"],
                    needs_clarification=False,
                )
            ),
            LLMResponse(
                structured_output=ParsedQuery(
                    intent="investment_evaluation",
                    tickers=["AAPL"],
                    needs_clarification=False,
                )
            ),
        ]
    )
    planner_fake_v5 = FakeLLM(
        responses=[
            LLMResponse(
                structured_output=PlanSpec(
                    tasks=[
                        PlannedTaskSpec(
                            task_id="t1",
                            agent_type="fundamental",
                            instruction="Review NVDA margins",
                            priority=1,
                        )
                    ]
                )
            ),
            LLMResponse(
                structured_output=PlanSpec(
                    tasks=[
                        PlannedTaskSpec(
                            task_id="t2",
                            agent_type="valuation",
                            instruction="Review AAPL valuation",
                            priority=1,
                        )
                    ]
                )
            ),
        ]
    )
    gapcheck_fake_v5 = FakeLLM(
        responses=[
            LLMResponse(
                text=(
                    '{"action": "done", "follow_up_tasks": [], "reasons": ["Verified"]}'
                )
            ),
            LLMResponse(
                text=(
                    '{"action": "done", "follow_up_tasks": [], "reasons": ["Verified"]}'
                )
            ),
        ]
    )
    v5 = V5DynamicVariant(
        understand_llm=understand_fake_v5,
        planner_llm=planner_fake_v5,
        gapcheck_llm=gapcheck_fake_v5,
        config=VariantConfig(verifier=False),
        task_runner=runner,
    )

    from lekka_ai.eval.metrics import success_metric

    eval_runner = EvalRunner(metrics={"success": success_metric})
    summary = await eval_runner.run(
        dataset=dataset,
        variants=[v3, v4, v5],
        print_summary=False,
    )

    # 3 variants x 2 questions = 6 run results
    assert len(summary.results) == 6
    assert all(r.status == "completed" for r in summary.results)
    assert all(not r.error for r in summary.results)
    assert "v3_static" in summary.variants
    assert "v4_dynamic" in summary.variants
    assert "v5_dynamic" in summary.variants

    for _var_name, agg in summary.variants.items():
        assert agg.count == 2
        assert agg.success_rate == 1.0
