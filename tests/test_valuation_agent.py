"""Tests for Valuation specialist agent profile, prompt, extension, and live run."""

import os
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
from lekka_ai.schemas.claims import ClaimType
from lekka_ai.schemas.valuation import ValuationAnswerExtension
from lekka_ai.tools.base import default_registry


def test_valuation_profile_configuration() -> None:
    """Verify valuation profile loads correct tools, prompt, and answer extension."""
    profile = load_agent_profile("valuation")
    assert profile.agent_type == "valuation"

    # Authorized tools
    expected_tools = [
        "get_market_data",
        "get_financials",
        "get_analyst_estimates",
        "run_dcf",
        "run_reverse_dcf",
        "compute_multiples",
        "scenario_table",
        "run_python",
        "submit_answer",
    ]
    for tool_name in expected_tools:
        assert profile.is_tool_allowed(tool_name)

    # Disallowed tools
    assert not profile.is_tool_allowed("search_filings")
    assert not profile.is_tool_allowed("list_filings")
    assert not profile.is_tool_allowed("get_news")

    # Prompt content
    prompt = profile.load_system_prompt()
    assert prompt is not None
    assert "Quantitative Valuation Specialist System Prompt" in prompt
    assert "NEVER COMPUTE WITH MENTAL MATH" in prompt
    assert "BEAR / BASE / BULL SCENARIOS" in prompt
    assert "REVERSE DCF" in prompt

    # Extension schema
    ext_model = profile.get_extension_model()
    assert ext_model is ValuationAnswerExtension


@pytest.mark.asyncio
async def test_valuation_agent_fake_llm() -> None:
    """Test valuation agent runtime flow using FakeLLM and verify typed extension."""
    profile = load_agent_profile("valuation")
    tool_view = ToolView(default_registry, profile=profile)

    task = AgentTask(
        task_id="task_val_01",
        instruction=(
            "Build bear/base/bull DCF valuation scenarios and solve reverse "
            "DCF for AAPL as of 2024-09-20."
        ),
        as_of_date="2024-09-20",
        agent_type="valuation",
    )

    # Turn 1: run_dcf base case
    step_1 = LLMResponse(
        text="Executing baseline DCF valuation model.",
        tool_calls=[
            ToolCall(
                id="tc_dcf_1",
                name="run_dcf",
                args={
                    "ticker": "AAPL",
                    "starting_fcf": 108000000000.0,
                    "fcf_growth_path": [0.08, 0.08, 0.07, 0.06, 0.05],
                    "discount_rate": 0.09,
                    "terminal_growth": 0.025,
                    "shares_outstanding": 15200000000.0,
                    "net_debt": 50000000000.0,
                },
            )
        ],
        usage=Usage(input_tokens=150, output_tokens=40),
    )

    # Turn 2: run_reverse_dcf to infer implied growth
    step_2 = LLMResponse(
        text="Solving reverse DCF for market-implied growth rate.",
        tool_calls=[
            ToolCall(
                id="tc_rev_2",
                name="run_reverse_dcf",
                args={
                    "ticker": "AAPL",
                    "current_price": 228.20,
                    "starting_fcf": 108000000000.0,
                    "discount_rate": 0.09,
                    "terminal_growth": 0.025,
                    "shares_outstanding": 15200000000.0,
                    "net_debt": 50000000000.0,
                },
            )
        ],
        usage=Usage(input_tokens=180, output_tokens=40),
    )

    # Turn 3: scenario_table
    step_3 = LLMResponse(
        text="Constructing scenario table matrix across bear, base, and bull cases.",
        tool_calls=[
            ToolCall(
                id="tc_scen_3",
                name="scenario_table",
                args={
                    "ticker": "AAPL",
                    "current_price": 228.20,
                    "bear": {
                        "name": "Bear Case",
                        "probability": 0.25,
                        "starting_fcf": 108000000000.0,
                        "fcf_growth_path": [0.03, 0.03, 0.02, 0.02, 0.01],
                        "discount_rate": 0.10,
                        "terminal_growth": 0.02,
                        "shares_outstanding": 15200000000.0,
                    },
                    "base": {
                        "name": "Base Case",
                        "probability": 0.50,
                        "starting_fcf": 108000000000.0,
                        "fcf_growth_path": [0.08, 0.08, 0.07, 0.06, 0.05],
                        "discount_rate": 0.09,
                        "terminal_growth": 0.025,
                        "shares_outstanding": 15200000000.0,
                    },
                    "bull": {
                        "name": "Bull Case",
                        "probability": 0.25,
                        "starting_fcf": 108000000000.0,
                        "fcf_growth_path": [0.12, 0.12, 0.10, 0.09, 0.08],
                        "discount_rate": 0.085,
                        "terminal_growth": 0.03,
                        "shares_outstanding": 15200000000.0,
                    },
                },
            )
        ],
        usage=Usage(input_tokens=200, output_tokens=50),
    )

    # Turn 4: submit_answer with scenario_values and implied_growth
    submit_payload: dict[str, Any] = {
        "task_id": "task_val_01",
        "agent_type": "valuation",
        "status": "completed",
        "confidence": 0.94,
        "confidence_rationale": (
            "Model evaluated with tool-grounded DCF, reverse DCF, and "
            "scenario analysis."
        ),
        "assumptions": {
            "discount_rate": 0.09,
            "terminal_growth": 0.025,
            "forecast_years": 5,
            "starting_fcf": 108000000000.0,
            "shares_outstanding": 15200000000.0,
            "net_debt": 50000000000.0,
        },
        "claims": [
            {
                "claim_id": "claim_dcf_base",
                "text": "AAPL baseline DCF fair value per share is $234.10.",
                "type": "calculation",
                "confidence": 0.95,
                "evidence": [
                    {
                        "source_type": "quant_model",
                        "tool_call_id": "tc_dcf_1",
                        "as_of_date": "2024-09-20",
                        "raw_value": 234.10,
                    }
                ],
            },
            {
                "claim_id": "claim_rev_growth",
                "text": "Market-implied growth rate from reverse DCF is 7.8% annually.",
                "type": "calculation",
                "confidence": 0.92,
                "evidence": [
                    {
                        "source_type": "quant_model",
                        "tool_call_id": "tc_rev_2",
                        "as_of_date": "2024-09-20",
                        "raw_value": 0.078,
                    }
                ],
            },
            {
                "claim_id": "claim_scen_expected",
                "text": (
                    "Probability-weighted expected scenario fair value is "
                    "$234.86 per share."
                ),
                "type": "calculation",
                "confidence": 0.94,
                "evidence": [
                    {
                        "source_type": "quant_model",
                        "tool_call_id": "tc_scen_3",
                        "as_of_date": "2024-09-20",
                        "raw_value": 234.86,
                    }
                ],
            },
        ],
        "extension": {
            "scenario_values": {
                "bear": 182.50,
                "base": 234.10,
                "bull": 288.75,
                "expected_value": 234.86,
            },
            "implied_growth": 0.078,
            "current_price": 228.20,
        },
    }

    step_4 = LLMResponse(
        text="Submitting complete valuation analysis with scenarios and extension.",
        tool_calls=[
            ToolCall(
                id="tc_sub_4",
                name="submit_answer",
                args=submit_payload,
            )
        ],
        usage=Usage(input_tokens=220, output_tokens=60),
    )

    fake_llm = FakeLLM(responses=[step_1, step_2, step_3, step_4])

    result = await AgentRuntime.run(
        task=task,
        profile=profile,
        llm=fake_llm,
        tool_view=tool_view,
        limits=Limits(max_iterations=6),
    )

    assert result.status == "completed"
    assert result.iterations == 4
    assert result.submission is not None
    assert result.submission.agent_type == "valuation"
    assert len(result.submission.claims) == 3
    assert all(c.type == ClaimType.CALCULATION for c in result.submission.claims)

    # Assumptions validation
    assert "discount_rate" in result.submission.assumptions
    assert result.submission.assumptions["discount_rate"] == 0.09
    assert result.submission.assumptions["terminal_growth"] == 0.025

    # Extension validation
    ext = result.submission.extension
    assert isinstance(ext, ValuationAnswerExtension)
    assert ext.implied_growth == 0.078
    assert ext.current_price == 228.20

    scen = ext.scenario_values
    scen_dict = scen.model_dump() if hasattr(scen, "model_dump") else scen
    assert scen_dict["bear"] == 182.50
    assert scen_dict["base"] == 234.10
    assert scen_dict["bull"] == 288.75


@pytest.mark.live
@pytest.mark.asyncio
async def test_valuation_agent_live_aapl() -> None:
    """Live test running valuation agent on AAPL verifying tool-computed scenarios."""
    if os.getenv("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE is not set to 1; skipping live valuation test")

    settings = get_settings()
    if not settings.GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY is not configured; skipping live valuation test")

    profile = load_agent_profile("valuation")
    tool_view = ToolView(default_registry, profile=profile)
    llm = GeminiClient(
        api_key=settings.GEMINI_API_KEY,
        model=settings.GEMINI_MODEL,
    )

    task = AgentTask(
        task_id="task_live_aapl_valuation",
        instruction=(
            "Perform a valuation of AAPL as of 2024-09-20. "
            "Retrieve current price with get_market_data. "
            "Use run_dcf, run_reverse_dcf, and scenario_table to build "
            "bear/base/bull scenarios and solve reverse DCF. Never compute with "
            "mental math. Record all explicit assumptions and submit_answer."
        ),
        as_of_date="2024-09-20",
        agent_type="valuation",
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

    # Verify scenario values are present in extension
    ext = result.submission.extension
    assert ext is not None
    assert hasattr(ext, "scenario_values")
    scen = ext.scenario_values
    scen_dict = scen.model_dump() if hasattr(scen, "model_dump") else scen
    assert "bear" in scen_dict or "base" in scen_dict
