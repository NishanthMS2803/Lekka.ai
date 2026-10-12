"""Tests for Market agent profile, prompt, typed answer extension, and live run."""

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
from lekka_ai.schemas.market import MarketAnswerExtension
from lekka_ai.tools.base import default_registry


def test_market_profile_configuration() -> None:
    """Verify market profile loads correct tools, prompt, and answer extension."""
    profile = load_agent_profile("market")
    assert profile.agent_type == "market"

    # Authorized tools
    assert profile.is_tool_allowed("get_market_data")
    assert profile.is_tool_allowed("compute_risk_stats")
    assert profile.is_tool_allowed("submit_answer")

    # Unauthorized tools
    assert not profile.is_tool_allowed("get_financials")
    assert not profile.is_tool_allowed("run_dcf")
    assert not profile.is_tool_allowed("run_python")
    assert not profile.is_tool_allowed("search_filings")

    # Prompt content
    prompt = profile.load_system_prompt()
    assert prompt is not None
    assert "Market Analysis Specialist System Prompt" in prompt
    assert "STRICTLY NO INVESTMENT VERDICTS" in prompt
    assert "Volatility" in prompt
    assert "Drawdown" in prompt
    assert "Beta" in prompt
    assert "Relative Strength" in prompt

    # Extension schema
    ext_model = profile.get_extension_model()
    assert ext_model is MarketAnswerExtension


@pytest.mark.asyncio
async def test_market_agent_fake_llm() -> None:
    """Test market agent runtime flow using FakeLLM and verify typed extension."""
    profile = load_agent_profile("market")
    tool_view = ToolView(default_registry, profile=profile)

    task = AgentTask(
        task_id="task_market_01",
        instruction=(
            "Evaluate market price action and risk stats for AAPL as of 2024-09-20."
        ),
        as_of_date="2024-09-20",
        agent_type="market",
    )

    # Step 1: Tool call get_market_data
    step_1 = LLMResponse(
        text="Retrieving AAPL historical market prices.",
        tool_calls=[
            ToolCall(
                id="tc_mkt_1",
                name="get_market_data",
                args={
                    "ticker": "AAPL",
                    "start": "2024-01-01",
                    "end": "2024-09-20",
                    "as_of_date": "2024-09-20",
                },
            )
        ],
        usage=Usage(input_tokens=100, output_tokens=30),
    )

    # Step 2: Tool call compute_risk_stats
    step_2 = LLMResponse(
        text="Calculating quantitative risk metrics and beta.",
        tool_calls=[
            ToolCall(
                id="tc_risk_2",
                name="compute_risk_stats",
                args={
                    "ticker": "AAPL",
                    "benchmark": "SPY",
                    "start": "2024-01-01",
                    "as_of_date": "2024-09-20",
                },
            )
        ],
        usage=Usage(input_tokens=120, output_tokens=30),
    )

    # Step 3: submit_answer with typed claims and MarketAnswerExtension
    submit_payload: dict[str, Any] = {
        "task_id": "task_market_01",
        "agent_type": "market",
        "status": "completed",
        "confidence": 0.92,
        "confidence_rationale": (
            "Empirical risk statistics and price action computed from market bars."
        ),
        "claims": [
            {
                "claim_id": "claim_mkt_close",
                "text": "AAPL latest closing price was $228.20 on 2024-09-20.",
                "type": "fact",
                "confidence": 0.98,
                "evidence": [
                    {
                        "source_type": "market_data",
                        "tool_call_id": "tc_mkt_1",
                        "as_of_date": "2024-09-20",
                        "raw_value": 228.20,
                    }
                ],
            },
            {
                "claim_id": "claim_mkt_vol",
                "text": (
                    "AAPL annualized volatility over the window was 22.4% with a "
                    "beta of 1.12."
                ),
                "type": "calculation",
                "confidence": 0.95,
                "evidence": [
                    {
                        "source_type": "market_data",
                        "tool_call_id": "tc_risk_2",
                        "as_of_date": "2024-09-20",
                        "raw_value": "volatility=0.224, beta=1.12",
                    }
                ],
            },
            {
                "claim_id": "claim_mkt_trend",
                "text": (
                    "AAPL exhibited an upward bullish trend trading above moving "
                    "averages."
                ),
                "type": "interpretation",
                "confidence": 0.88,
                "evidence": [],
                "depends_on": ["claim_mkt_close"],
            },
        ],
        "extension": {
            "trend_label": "bullish",
            "key_stats": {
                "volatility": 0.224,
                "max_drawdown": -0.145,
                "beta": 1.12,
                "relative_strength": 0.065,
                "latest_close": 228.20,
            },
        },
    }

    step_3 = LLMResponse(
        text="Submitting complete market analysis with typed claims and extension.",
        tool_calls=[
            ToolCall(
                id="tc_sub_3",
                name="submit_answer",
                args=submit_payload,
            )
        ],
        usage=Usage(input_tokens=150, output_tokens=50),
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
    assert result.iterations == 3
    assert result.submission is not None
    assert result.submission.agent_type == "market"
    assert len(result.submission.claims) == 3

    # Verify claim types
    claim_types = [c.type for c in result.submission.claims]
    assert ClaimType.FACT in claim_types
    assert ClaimType.CALCULATION in claim_types
    assert ClaimType.INTERPRETATION in claim_types

    # Verify MarketAnswerExtension
    ext = result.submission.extension
    assert isinstance(ext, MarketAnswerExtension)
    assert ext.trend_label == "bullish"
    assert ext.key_stats["volatility"] == 0.224
    assert ext.key_stats["beta"] == 1.12
    assert ext.key_stats["max_drawdown"] == -0.145


@pytest.mark.live
@pytest.mark.asyncio
async def test_market_agent_live_aapl() -> None:
    """Live test running market agent on AAPL verifying typed claims."""
    if os.getenv("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE is not set to 1; skipping live market agent test")

    settings = get_settings()
    if not settings.GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY is not configured; skipping live market test")

    profile = load_agent_profile("market")
    tool_view = ToolView(default_registry, profile=profile)
    llm = GeminiClient(
        api_key=settings.GEMINI_API_KEY,
        model=settings.GEMINI_MODEL,
    )

    task = AgentTask(
        task_id="task_live_aapl_market",
        instruction=(
            "Examine AAPL price action, volatility, drawdown, beta, "
            "relative strength, and trend over 2024 up to 2024-09-20. "
            "Do not provide any investment verdict. "
            "Use get_market_data and compute_risk_stats, then submit_answer."
        ),
        as_of_date="2024-09-20",
        agent_type="market",
    )

    result = await AgentRuntime.run(
        task=task,
        profile=profile,
        llm=llm,
        tool_view=tool_view,
        limits=Limits(max_iterations=6, max_seconds=120.0),
    )

    assert result.status in {"completed", "partial"}
    assert result.submission is not None
    assert len(result.submission.claims) > 0

    # Verify returned claims are typed
    valid_types = {
        ClaimType.FACT,
        ClaimType.CALCULATION,
        ClaimType.INTERPRETATION,
        ClaimType.BACKGROUND,
    }
    for claim in result.submission.claims:
        assert claim.type in valid_types
        assert isinstance(claim.text, str)
        assert len(claim.text.strip()) > 0
