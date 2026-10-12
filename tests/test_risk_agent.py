"""Tests for Risk specialist agent profile, wave-2 constraints, and live run."""

import os
import uuid
from typing import Any

import pytest

from lekka_ai.agents import load_agent_profile
from lekka_ai.config import get_settings
from lekka_ai.harness.models import AgentTask, Limits
from lekka_ai.harness.permissions import ToolView
from lekka_ai.harness.runtime import AgentRuntime
from lekka_ai.llm.base import LLMResponse, ToolCall, Usage
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.llm.gemini import GeminiClient
from lekka_ai.orchestrator.planner import (
    InvalidDependencyError,
    validate_plan,
)
from lekka_ai.orchestrator.state import Plan, PlannedTask
from lekka_ai.schemas.risk import RiskAnswerExtension
from lekka_ai.tools.base import ToolRegistry, default_registry


def test_risk_profile_configuration() -> None:
    """Verify risk agent profile configuration, allowed tools, and prompt directives."""
    profile = load_agent_profile("risk")
    assert profile.agent_type == "risk"

    # Authorized tools: get_market_data, get_financials, search_filings,
    # get_news, get_industry_data, compute_risk_stats, submit_answer
    expected_tools = [
        "get_market_data",
        "get_financials",
        "search_filings",
        "get_news",
        "get_industry_data",
        "compute_risk_stats",
        "submit_answer",
    ]
    for tool_name in expected_tools:
        assert profile.is_tool_allowed(tool_name), f"Tool {tool_name} should be allowed"

    # Explicitly NO recall_facts or outside tools
    assert not profile.is_tool_allowed("recall_facts")
    assert not profile.is_tool_allowed("run_python")
    assert not profile.is_tool_allowed("run_dcf")
    assert not profile.is_tool_allowed("run_reverse_dcf")
    assert not profile.is_tool_allowed("compute_multiples")

    # Prompt content assertions
    prompt = profile.load_system_prompt()
    assert prompt is not None
    assert "Risk Assessment Specialist System Prompt" in prompt
    assert "ASSUME THE CONCLUSION IS WRONG" in prompt
    assert "FIND EVIDENCE" in prompt
    assert "RANK FAILURE MODES" in prompt
    assert "WHAT WOULD CAUSE A 30-50% DRAWDOWN" in prompt
    assert "NO recall_facts" in prompt or "recall_facts" in prompt

    # Answer extension schema
    ext_model = profile.get_extension_model()
    assert ext_model is RiskAnswerExtension


def test_risk_agent_planner_wave_2_only() -> None:
    """Verify planner enforces that risk agent is Wave 2 only and cannot be Wave 1."""
    # 1. Invalid: Risk agent as Wave 1 (empty depends_on)
    plan_invalid = Plan()
    plan_invalid.add_task(
        PlannedTask(
            task_id="task_risk_01",
            agent_type="risk",
            instruction="Evaluate downside risks",
            depends_on=[],
        )
    )
    with pytest.raises(InvalidDependencyError, match="wave 2 only"):
        validate_plan(plan_invalid, allowed_agent_types=["fundamental", "risk"])

    # 2. Valid: Risk agent as Wave 2 depending on wave-1 fundamental agent
    plan_valid = Plan()
    plan_valid.add_task(
        PlannedTask(
            task_id="task_fundamental_01",
            agent_type="fundamental",
            instruction="Extract FY2024 financials",
            depends_on=[],
        )
    )
    plan_valid.add_task(
        PlannedTask(
            task_id="task_risk_02",
            agent_type="risk",
            instruction=(
                "Review wave-1 conclusions. Assume the conclusion is wrong; "
                "find evidence; rank failure modes; what would cause a 30-50% drawdown."
            ),
            depends_on=["task_fundamental_01"],
        )
    )
    # Passes validation without error
    validate_plan(plan_valid, allowed_agent_types=["fundamental", "risk"])


@pytest.mark.asyncio
async def test_risk_agent_fake_llm(fixture_tool_registry: ToolRegistry) -> None:
    """Verify risk agent execution with FakeLLM and RiskAnswerExtension."""
    profile = load_agent_profile("risk")
    tool_view = ToolView(fixture_tool_registry, profile=profile)

    task = AgentTask(
        task_id="task_risk_aapl_01",
        instruction=(
            "Review wave-1 conclusions for AAPL. Assume the conclusion is wrong; "
            "find evidence; rank failure modes; what would cause a 30-50% drawdown."
        ),
        context_facts=[
            {
                "claim_id": "c_wave1_growth",
                "text": "Wave-1 conclusion: AAPL base case assumes 12% revenue growth.",
                "type": "fact",
            }
        ],
        as_of_date="2024-01-12",
        agent_type="risk",
    )

    # Step 1: Compute risk stats
    step_1 = LLMResponse(
        text="Computing quantitative risk and drawdown metrics.",
        tool_calls=[
            ToolCall(
                id="tc_risk_stats_1",
                name="compute_risk_stats",
                args={
                    "ticker": "AAPL",
                    "benchmark": "SPY",
                    "start": "2024-01-01",
                    "as_of_date": "2024-01-12",
                },
            )
        ],
        usage=Usage(input_tokens=100, output_tokens=30),
    )

    # Step 2: Search filings for antitrust and customer concentration risk
    step_2 = LLMResponse(
        text="Searching SEC filings for regulatory litigation and tail risks.",
        tool_calls=[
            ToolCall(
                id="tc_search_filings_2",
                name="search_filings",
                args={
                    "ticker": "AAPL",
                    "query": "antitrust regulatory litigation risk factors",
                    "top_k": 3,
                    "as_of_date": "2024-01-12",
                },
            )
        ],
        usage=Usage(input_tokens=140, output_tokens=40),
    )

    # Step 3: submit_answer with ranked risks
    submit_payload: dict[str, Any] = {
        "task_id": "task_risk_aapl_01",
        "agent_type": "risk",
        "status": "completed",
        "confidence": 0.92,
        "confidence_rationale": (
            "Empirical risk statistics and 10-K disclosures cross-verified."
        ),
        "claims": [
            {
                "claim_id": "claim_max_dd",
                "text": (
                    "AAPL historical maximum drawdown reached -31.8% during cyclical"
                    " corrections."
                ),
                "type": "fact",
                "confidence": 0.96,
                "evidence": [
                    {
                        "source_type": "quant_engine",
                        "tool_call_id": "tc_risk_stats_1",
                        "raw_value": "-0.318",
                        "as_of_date": "2024-01-12",
                    }
                ],
            },
            {
                "claim_id": "claim_antitrust_dispute",
                "text": (
                    "Apple faces regulatory investigations regarding App Store and "
                    "Search revenue sharing."
                ),
                "type": "fact",
                "confidence": 0.94,
                "evidence": [
                    {
                        "source_type": "sec_filing",
                        "tool_call_id": "tc_search_filings_2",
                        "document_id": "0000320193-24-000106",
                        "raw_value": "Executive transitions and regulatory proceedings",
                        "as_of_date": "2024-01-12",
                    }
                ],
            },
        ],
        "extension": {
            "ranked_risks": [
                {
                    "rank": 1,
                    "risk_factor": "Antitrust Search Revenue Prohibition",
                    "severity": "critical",
                    "failure_mode": (
                        "Regulatory remedy eliminating Google licensing fees ($20B/yr)"
                    ),
                    "potential_drawdown": "30-50%",
                    "evidence_claim_ids": ["claim_antitrust_dispute"],
                    "description": (
                        "High-margin licensing revenue loss would impair operating "
                        "income by ~18%."
                    ),
                },
                {
                    "rank": 2,
                    "risk_factor": "Multiple Compression During Tech Drawdowns",
                    "severity": "high",
                    "failure_mode": "P/E de-rating from 32x to historical median 22x",
                    "potential_drawdown": "31.8%",
                    "evidence_claim_ids": ["claim_max_dd"],
                    "description": (
                        "Valuation vulnerability if hardware upgrade supercycle fails "
                        "to materialize."
                    ),
                },
            ],
            "drawdown_catalysts": [
                "DOJ antitrust ruling prohibiting default search payments",
                "Broad multiple compression in premium consumer tech",
            ],
            "counter_thesis_summary": (
                "Wave-1 base case assumes durable 12% revenue growth and stable 32x"
                " P/E, ignoring $20B search licensing vulnerability and historical"
                " -31.8% drawdown potential."
            ),
        },
    }

    step_3 = LLMResponse(
        text="Submitting completed risk assessment with ranked failure modes.",
        tool_calls=[
            ToolCall(
                id="tc_sub_risk",
                name="submit_answer",
                args=submit_payload,
            )
        ],
        usage=Usage(input_tokens=180, output_tokens=60),
    )

    fake_llm = FakeLLM(responses=[step_1, step_2, step_3])

    result = await AgentRuntime.run(
        task=task,
        profile=profile,
        llm=fake_llm,
        tool_view=tool_view,
        limits=Limits(max_iterations=5),
    )

    assert result.status == "completed"
    assert result.submission is not None
    assert result.submission.agent_type == "risk"
    assert len(result.submission.claims) == 2

    # Verify RiskAnswerExtension
    ext = result.submission.extension
    assert isinstance(ext, RiskAnswerExtension)
    assert len(ext.ranked_risks) == 2

    r1 = ext.ranked_risks[0]
    assert r1.rank == 1
    assert r1.risk_factor == "Antitrust Search Revenue Prohibition"
    assert r1.severity == "critical"
    assert r1.evidence_claim_ids == ["claim_antitrust_dispute"]
    assert r1.potential_drawdown == "30-50%"

    r2 = ext.ranked_risks[1]
    assert r2.rank == 2
    assert r2.severity == "high"
    assert r2.evidence_claim_ids == ["claim_max_dd"]

    assert len(ext.drawdown_catalysts) == 2
    assert "Wave-1 base case" in (ext.counter_thesis_summary or "")


@pytest.mark.live
@pytest.mark.asyncio
async def test_risk_agent_live_aapl() -> None:
    """Live test running risk agent on AAPL verifying ranked risks with evidence."""
    if os.getenv("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE is not set to 1; skipping live risk agent test")

    settings = get_settings()
    if not settings.GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY is not configured; skipping live risk agent test")

    profile = load_agent_profile("risk")
    tool_view = ToolView(default_registry, profile=profile)
    llm = GeminiClient(
        api_key=settings.GEMINI_API_KEY,
        model=settings.GEMINI_MODEL,
    )

    task = AgentTask(
        task_id=f"task_live_aapl_risk_{uuid.uuid4().hex[:6]}",
        instruction=(
            "You are the risk assessment specialist for AAPL as of 2024-09-20. "
            "Wave-1 findings concluded: 'AAPL has solid balance sheet and "
            "iPhone 16 catalysts'. Assume the conclusion is wrong; find evidence; "
            "rank failure modes; what would cause a 30-50% drawdown. "
            "Use compute_risk_stats, search_filings, and get_financials. "
            "Submit answer with ranked risks, severity, and evidence claim ids."
        ),
        as_of_date="2024-09-20",
        agent_type="risk",
    )

    result = await AgentRuntime.run(
        task=task,
        profile=profile,
        llm=llm,
        tool_view=tool_view,
        limits=Limits(max_iterations=8, max_seconds=120.0),
    )

    assert result.status in {"completed", "partial"}
    assert result.submission is not None
    assert len(result.submission.claims) > 0

    # Verify ranked risks with evidence in extension
    ext = result.submission.extension
    assert ext is not None
    assert isinstance(ext, RiskAnswerExtension)
    assert len(ext.ranked_risks) > 0

    first_risk = ext.ranked_risks[0]
    assert first_risk.rank >= 1
    assert first_risk.severity is not None
    assert len(first_risk.evidence_claim_ids) > 0
