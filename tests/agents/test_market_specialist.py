"""Golden scenario isolation test for Market specialist agent."""

from typing import Any

import pytest

from lekka_ai.agents import load_agent_profile
from lekka_ai.harness.models import AgentTask, Limits
from lekka_ai.harness.permissions import ToolView
from lekka_ai.harness.runtime import AgentRuntime
from lekka_ai.llm.base import LLMResponse, ToolCall, Usage
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.schemas.market import MarketAnswerExtension
from lekka_ai.tools.base import ToolRegistry
from lekka_ai.tracing import Tracer


@pytest.mark.asyncio
async def test_market_specialist_golden_scenario(
    fixture_tool_registry: ToolRegistry,
) -> None:
    """Golden scenario: Market agent calls expected tools & produces claims."""
    profile = load_agent_profile("market")
    tracer = Tracer()
    tool_view = ToolView(fixture_tool_registry, profile=profile, tracer=tracer)

    task = AgentTask(
        task_id="task_market_golden_01",
        instruction=(
            "Evaluate market price action and risk statistics for AAPL as of"
            " 2024-01-12 relative to SPY. Report volatility, beta, drawdown,"
            " and trend."
        ),
        as_of_date="2024-01-12",
        agent_type="market",
    )

    # Step 1: LLM requests historical market bars
    step_1 = LLMResponse(
        text="Querying historical price bars for AAPL from fixture provider.",
        tool_calls=[
            ToolCall(
                id="tc_mkt_data_1",
                name="get_market_data",
                args={
                    "ticker": "AAPL",
                    "start": "2024-01-01",
                    "end": "2024-01-12",
                    "as_of_date": "2024-01-12",
                },
            )
        ],
        usage=Usage(input_tokens=90, output_tokens=30),
    )

    # Step 2: LLM computes quantitative risk statistics against SPY
    step_2 = LLMResponse(
        text="Computing annualized volatility, beta, and trend classification.",
        tool_calls=[
            ToolCall(
                id="tc_risk_stats_2",
                name="compute_risk_stats",
                args={
                    "ticker": "AAPL",
                    "benchmark": "SPY",
                    "start": "2024-01-01",
                    "as_of_date": "2024-01-12",
                },
            )
        ],
        usage=Usage(input_tokens=120, output_tokens=35),
    )

    # Step 3: LLM submits structured response with claims and extension
    submit_payload: dict[str, Any] = {
        "task_id": "task_market_golden_01",
        "agent_type": "market",
        "status": "completed",
        "confidence": 0.94,
        "confidence_rationale": (
            "Derived directly from recorded price fixtures and quant risk stats."
        ),
        "claims": [
            {
                "claim_id": "claim_close_price",
                "text": "AAPL latest closing price was $185.92 on 2024-01-12.",
                "type": "fact",
                "confidence": 0.99,
                "evidence": [
                    {
                        "source_type": "market_data",
                        "tool_call_id": "tc_mkt_data_1",
                        "as_of_date": "2024-01-12",
                        "raw_value": "185.92",
                    }
                ],
            },
            {
                "claim_id": "claim_beta_stat",
                "text": "AAPL has a market beta of 1.55 relative to benchmark SPY.",
                "type": "fact",
                "confidence": 0.95,
                "evidence": [
                    {
                        "source_type": "quant_engine",
                        "tool_call_id": "tc_risk_stats_2",
                        "as_of_date": "2024-01-12",
                        "raw_value": "1.55",
                    }
                ],
            },
            {
                "claim_id": "claim_trend_interp",
                "text": (
                    "AAPL exhibits a bullish trend characterized by price action"
                    " holding above moving averages."
                ),
                "type": "interpretation",
                "confidence": 0.90,
                "evidence": [],
                "depends_on": ["claim_close_price", "claim_beta_stat"],
            },
        ],
        "extension": {
            "trend_label": "BULLISH",
            "key_stats": {
                "annualized_volatility": 0.177,
                "max_drawdown": -0.024,
                "beta": 1.55,
                "relative_strength": -0.007,
                "moving_averages": {"sma_3": 185.90, "sma_6": 184.93},
                "latest_close": 185.92,
            },
        },
    }

    step_3 = LLMResponse(
        text="Submitting final market specialist answer with typed extension.",
        tool_calls=[
            ToolCall(
                id="tc_submit_3",
                name="submit_answer",
                args=submit_payload,
            )
        ],
        usage=Usage(input_tokens=160, output_tokens=50),
    )

    fake_llm = FakeLLM(responses=[step_1, step_2, step_3])

    result = await AgentRuntime.run(
        task=task,
        profile=profile,
        llm=fake_llm,
        tool_view=tool_view,
        limits=Limits(max_iterations=5),
    )

    # 1. Successful completion and schema validation
    assert result.status == "completed"
    assert result.submission is not None
    assert result.submission.agent_type == "market"
    assert result.submission.confidence == 0.94

    # 2. Expected tools were invoked
    tool_calls_executed = [span.name for span in tracer.spans]
    assert "get_market_data" in tool_calls_executed
    assert "compute_risk_stats" in tool_calls_executed
    assert "submit_answer" in tool_calls_executed

    # 3. All executed tools stayed strictly within permissions
    for tool_name in tool_calls_executed:
        assert profile.is_tool_allowed(tool_name)

    # 4. Valid claims verified
    claims = result.submission.claims
    assert len(claims) == 3
    assert any(c.claim_id == "claim_close_price" and c.type == "fact" for c in claims)
    assert any(c.claim_id == "claim_beta_stat" and c.type == "fact" for c in claims)
    assert any(
        c.claim_id == "claim_trend_interp" and c.type == "interpretation"
        for c in claims
    )

    # 5. Typed answer extension matches MarketAnswerExtension
    ext = result.submission.extension
    assert isinstance(ext, MarketAnswerExtension)
    assert ext.trend_label == "BULLISH"
    assert ext.key_stats["beta"] == 1.55
    assert ext.key_stats["annualized_volatility"] == 0.177
