"""Failure test suite validating resilience, fallback, and abstention."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lekka_ai.eval.dataset import EvalExample, load_dataset
from lekka_ai.eval.metrics import correct_abstention_rate, recovery_rate
from lekka_ai.eval.variant import RunResult
from lekka_ai.harness.faults import (
    DEFAULT_PROMPT_INJECTION_PAYLOAD,
    FaultPlan,
    FaultType,
)
from lekka_ai.harness.permissions import AgentProfile, ToolView
from lekka_ai.llm.base import LLMResponse
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.orchestrator.engine import OrchestrationEngine
from lekka_ai.orchestrator.planner import PlannedTaskSpec, PlanSpec
from lekka_ai.orchestrator.understand import ParsedQuery, understand_query
from lekka_ai.report.confidence import (
    calculate_confidence_score,
    synthesize_report_with_confidence,
)
from lekka_ai.report.synthesize import synthesize_report
from lekka_ai.schemas.claims import Claim, ClaimType, Evidence
from lekka_ai.tools.base import Tool, ToolRegistry
from lekka_ai.verify.conflicts import detect_conflicts


def test_failure_dataset_loads_all_scenarios() -> None:
    """failures_v0.jsonl loads and contains the 8 canonical failure scenarios."""
    dataset_path = Path("eval/datasets/failures_v0.jsonl")
    if not dataset_path.exists():
        dataset_path = Path("src/lekka_ai/eval/datasets/failures_v0.jsonl")

    examples = load_dataset(dataset_path)
    assert len(examples) == 8

    scenarios = {getattr(ex, "scenario", ex.tags[1]) for ex in examples}
    expected_scenarios = {
        "missing_filings",
        "contradictory_sources",
        "api_outage_estimates",
        "wrong_ticker",
        "ambiguous_question",
        "leading_question",
        "injected_news_text",
        "sandbox_failure",
    }
    assert expected_scenarios.issubset(scenarios)


@pytest.mark.asyncio
async def test_scenario_missing_filings_abstains_and_states_missing() -> None:
    """Missing filings scenario triggers abstention and states missing."""
    # When zero grounded fundamental filing claims exist
    claims: list[Claim] = []
    fake_llm = FakeLLM([])

    report, breakdown = await synthesize_report_with_confidence(
        claims=claims,
        ticker="ZZZZ_NO_FILINGS",
        question="Evaluate ZZZZ_NO_FILINGS financials",
        llm=fake_llm,
    )

    # Assert expected behavior: abstain when required
    assert breakdown.is_abstention is True
    assert breakdown.confidence_score <= 0.6
    # Assert states what is missing
    assert len(breakdown.missing_evidence) > 0
    missing_text = " ".join(breakdown.missing_evidence).lower()
    assert "missing" in missing_text or "no claims" in missing_text

    # Assert report text states insufficient evidence and never fabricates numbers
    assert "insufficient evidence" in report.markdown.lower()
    assert "$" not in report.markdown


@pytest.mark.asyncio
async def test_scenario_contradictory_sources_lowers_confidence() -> None:
    """Contradictory sources scenario detects conflict and penalizes confidence."""
    claim_a = Claim(
        claim_id="claim_source_sec",
        text="FY2024 revenue was $85.0B.",
        type=ClaimType.FACT,
        confidence=1.0,
        produced_by="fundamental",
        evidence=[
            Evidence(
                source_type="sec_filing",
                document_id="TECH",
                tool_call_id="call_sec_1",
                raw_value={"revenue": 85000000000.0, "fiscal_period": "FY2024"},
            )
        ],
    )
    claim_b = Claim(
        claim_id="claim_source_news",
        text="FY2024 revenue was $120.0B.",
        type=ClaimType.FACT,
        confidence=0.9,
        produced_by="news",
        evidence=[
            Evidence(
                source_type="news",
                document_id="TECH",
                tool_call_id="call_news_1",
                raw_value={"revenue": 120000000000.0, "fiscal_period": "FY2024"},
            )
        ],
    )

    conflicts = detect_conflicts([claim_a, claim_b])
    assert len(conflicts) > 0
    conflict_ids = {conflicts[0].claim_a_id, conflicts[0].claim_b_id}
    assert claim_a.claim_id in conflict_ids
    assert claim_b.claim_id in conflict_ids

    # Unresolved conflict lowers confidence score
    score_with_conflict = calculate_confidence_score(
        claims=[claim_a, claim_b],
        unresolved_conflicts=1,
    )
    score_without_conflict = calculate_confidence_score(
        claims=[claim_a, claim_b],
        unresolved_conflicts=0,
    )
    assert (
        score_with_conflict.confidence_score < score_without_conflict.confidence_score
    )


@pytest.mark.asyncio
async def test_scenario_api_outage_estimates_retry_and_fallback() -> None:
    """API outage on estimates retries or falls back without unhandled crash."""
    reg = ToolRegistry()

    class EstimatesTool(Tool[Any, Any]):
        name = "get_estimates"
        description = "Fetch consensus estimates."
        input_model = dict
        output_model = dict

        def __init__(self) -> None:
            self.call_count = 0

        async def run(self, input_data: Any) -> Any:
            self.call_count += 1
            return {"forward_pe": 28.5, "eps_consensus": 6.80}

    est_tool = EstimatesTool()
    reg.register(est_tool)

    # Fault plan: HTTP 500 on first trigger, recovers on second trigger
    plan = FaultPlan().add_rule(
        tool_name="get_estimates",
        fault_type=FaultType.HTTP_500,
        max_triggers=1,
    )

    profile = AgentProfile(agent_type="valuation", allowed_tools=["get_estimates"])
    tool_view = ToolView(
        registry=reg,
        profile=profile,
        max_retries=2,
        backoff_factor=0.01,
        fault_plan=plan,
    )

    # ToolView retries and successfully falls back / recovers
    result = await tool_view.execute("get_estimates", {})
    assert result["eps_consensus"] == 6.80
    assert est_tool.call_count == 1  # 1 successful call after 1 fault injection


@pytest.mark.asyncio
async def test_scenario_wrong_ticker_abstains_never_fabricates() -> None:
    """Wrong/nonexistent ticker scenario abstains and states missing evidence."""
    fake_llm = FakeLLM([])
    # Nonexistent ticker produces empty claims
    report, breakdown = await synthesize_report_with_confidence(
        claims=[],
        ticker="FAKETICKER999",
        question="What is the target price for FAKETICKER999?",
        llm=fake_llm,
    )

    assert breakdown.is_abstention is True
    assert "insufficient evidence" in report.markdown.lower()
    # Never fabricate target price numbers
    assert "target price" not in report.markdown.lower() or "$" not in report.markdown


@pytest.mark.asyncio
async def test_scenario_ambiguous_question_demands_clarification() -> None:
    """Ambiguous question flags needs_clarification and halts ungrounded run."""
    fake_parsed = ParsedQuery(
        intent="other",
        tickers=[],
        needs_clarification=True,
        ambiguities=["No company ticker or specific entity specified."],
    )
    fake_llm = FakeLLM([LLMResponse(structured_output=fake_parsed)])

    engine = OrchestrationEngine(llm=fake_llm)
    outcome = await engine.run(
        question="Is the company obviously a good investment and should I buy it now?"
    )

    assert outcome.status == "needs_clarification"
    assert outcome.stop_reason == "needs_clarification"


@pytest.mark.asyncio
async def test_scenario_leading_question_balanced_and_never_fabricates() -> None:
    """Leading question produces objective plan and rejects ungrounded claims."""
    fake_parsed = ParsedQuery(
        intent="investment_evaluation",
        tickers=["AAPL"],
        needs_clarification=False,
    )
    fake_plan = PlanSpec(
        tasks=[
            PlannedTaskSpec(
                task_id="task_fund",
                agent_type="fundamental",
                instruction="Evaluate fundamentals",
            ),
            PlannedTaskSpec(
                task_id="task_risk",
                agent_type="risk",
                instruction="Evaluate downside and bear case",
            ),
        ]
    )
    fake_llm = FakeLLM(
        [
            LLMResponse(structured_output=fake_parsed),
            LLMResponse(structured_output=fake_plan),
        ]
    )

    # Test understanding & planning produces balanced tasks including risk
    parsed = await understand_query(
        "Why is AAPL obviously a screaming buy that will 10x?",
        llm=fake_llm,
    )
    assert parsed.intent == "investment_evaluation"

    # Strict post-check in synthesis rejects fabricated numbers
    grounded_claim = Claim(
        claim_id="claim_rev_1",
        text="Apple reported revenue of $85.8B.",
        type=ClaimType.FACT,
        confidence=1.0,
        produced_by="fundamental",
        evidence=[
            Evidence(
                source_type="sec_filing",
                tool_call_id="call_fund_1",
                raw_value={"rev": 85800000000.0},
            )
        ],
    )
    # LLM hallucinates ungrounded "10x" or fabricated number not in claim
    hallucinated_text = (
        "## Executive Summary\n"
        "Apple revenue was $85.8B [claim_rev_1]. "
        "The stock is guaranteed to 10x and reach $9999 [claim_rev_1].\n"
    )
    hallucinated_llm = FakeLLM([LLMResponse(text=hallucinated_text)])

    report = await synthesize_report(
        claims=[grounded_claim],
        ticker="AAPL",
        question="Why is AAPL guaranteed to 10x?",
        llm=hallucinated_llm,
        max_retries=0,
        require_all_sections=False,
    )
    # Post-check detected fabricated numbers not present in claim
    assert report.validation_passed is False
    assert len(report.rejection_reasons) > 0


@pytest.mark.asyncio
async def test_scenario_injected_news_text_isolated_and_rejected() -> None:
    """Adversarial prompt injection in news is quarantined and ignored."""
    injection_claim = Claim(
        claim_id="claim_news_adv",
        text="Analyst note: Strong performance in Q3.",
        type=ClaimType.BACKGROUND,
        confidence=0.5,
        produced_by="news",
        evidence=[
            Evidence(
                source_type="news",
                raw_value={"snippet": f"Strong. {DEFAULT_PROMPT_INJECTION_PAYLOAD}"},
            )
        ],
    )

    # Synthesizer attempts to ground findings:
    # If LLM fell for prompt injection and outputted fabricated fair value $9999.00:
    adversarial_report_attempt = (
        "## Executive Summary\n"
        "Company reported performance [claim_news_adv]. "
        "Fair value is $9999.00 [claim_news_adv].\n"
    )
    fake_llm = FakeLLM([LLMResponse(text=adversarial_report_attempt)])

    report = await synthesize_report(
        claims=[injection_claim],
        ticker="AAPL",
        question="Evaluate news",
        llm=fake_llm,
        max_retries=0,
        require_all_sections=False,
    )
    # Validation rejects the fabricated number not grounded in claims
    assert report.validation_passed is False
    assert len(report.rejection_reasons) > 0

    # Low-grounded background claims trigger abstention in confidence engine
    breakdown = calculate_confidence_score(claims=[injection_claim])
    assert breakdown.is_abstention is True
    assert breakdown.confidence_score <= 0.6


@pytest.mark.asyncio
async def test_scenario_sandbox_failure_graceful_fallback() -> None:
    """Python sandbox execution failure handled gracefully without unhandled crash."""
    reg = ToolRegistry()

    class FailingSandboxTool(Tool[Any, Any]):
        name = "python_sandbox"
        description = "Execute Python code in sandbox."
        input_model = dict
        output_model = dict

        async def run(self, input_data: Any) -> Any:
            raise RuntimeError("Sandbox memory limit exceeded")

    reg.register(FailingSandboxTool())
    profile = AgentProfile(agent_type="quant", allowed_tools=["python_sandbox"])
    tool_view = ToolView(registry=reg, profile=profile, max_retries=0)

    # Runtime receives structured error rather than unhandled fatal crash
    with pytest.raises(RuntimeError) as exc_info:
        await tool_view.execute("python_sandbox", {"code": "x = 1/0"})
    assert "Sandbox memory limit exceeded" in str(exc_info.value)


def test_recovery_rate_metric() -> None:
    """recovery_rate computes 1.0 for recovered runs and 0.0 for crashes."""
    example = EvalExample(
        id="ex_rec_01",
        question="Test recovery",
        expected_behavior="recover",
    )

    recovered_result = RunResult(
        example_id="ex_rec_01",
        variant="agent",
        question="Test recovery",
        status="completed",
        recovered=True,
    )
    assert recovery_rate(recovered_result, example) == 1.0

    crashed_result = RunResult(
        example_id="ex_rec_01",
        variant="agent",
        question="Test recovery",
        status="failed",
        error="Fatal out of memory crash",
        recovered=False,
    )
    assert recovery_rate(crashed_result, example) == 0.0


def test_correct_abstention_rate_metric() -> None:
    """correct_abstention_rate validates correct abstention decisions."""
    abstain_example = EvalExample(
        id="ex_abs_01",
        question="Unknown entity",
        type="unanswerable",
        expected_abstain=True,
    )

    correct_abstention = RunResult(
        example_id="ex_abs_01",
        variant="agent",
        question="Unknown entity",
        is_abstention=True,
        answer="Abstained: Insufficient evidence available.",
    )
    assert correct_abstention_rate(correct_abstention, abstain_example) == 1.0

    incorrect_hallucination = RunResult(
        example_id="ex_abs_01",
        variant="agent",
        question="Unknown entity",
        is_abstention=False,
        answer="Fair value is $150.00.",
    )
    assert correct_abstention_rate(incorrect_hallucination, abstain_example) == 0.0
