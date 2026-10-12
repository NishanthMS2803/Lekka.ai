"""Golden scenario isolation test for Valuation specialist agent."""

from typing import Any

import pytest

from lekka_ai.agents import load_agent_profile
from lekka_ai.harness.models import AgentTask, Limits
from lekka_ai.harness.permissions import ToolView
from lekka_ai.harness.runtime import AgentRuntime
from lekka_ai.llm.base import LLMResponse, ToolCall, Usage
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.schemas.valuation import ValuationAnswerExtension
from lekka_ai.tools.base import ToolRegistry
from lekka_ai.tracing import Tracer


@pytest.mark.asyncio
async def test_valuation_specialist_golden_scenario(
    fixture_tool_registry: ToolRegistry,
) -> None:
    """Golden scenario: Valuation agent calls expected tools & produces claims."""
    profile = load_agent_profile("valuation")
    tracer = Tracer()
    tool_view = ToolView(fixture_tool_registry, profile=profile, tracer=tracer)

    task = AgentTask(
        task_id="task_val_golden_01",
        instruction=(
            "Build bear/base/bull DCF valuation scenarios and solve reverse "
            "DCF for AAPL as of 2024-09-20. Use get_financials to retrieve "
            "audited cash flows, run_dcf for intrinsic values, and "
            "run_reverse_dcf for implied growth."
        ),
        as_of_date="2024-09-20",
        agent_type="valuation",
    )

    # Step 1: LLM retrieves audited financials from recorded XBRL facts
    step_1 = LLMResponse(
        text="Retrieving audited financial periods from SEC company facts fixture.",
        tool_calls=[
            ToolCall(
                id="tc_fin_1",
                name="get_financials",
                args={
                    "ticker": "AAPL",
                    "periods": 2,
                    "period_type": "annual",
                    "as_of_date": "2024-09-20",
                },
            )
        ],
        usage=Usage(input_tokens=100, output_tokens=35),
    )

    # Step 2: LLM runs baseline DCF calculation
    step_2 = LLMResponse(
        text="Executing baseline DCF valuation model with explicit assumptions.",
        tool_calls=[
            ToolCall(
                id="tc_dcf_2",
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

    # Step 3: LLM solves reverse DCF for market expectations
    step_3 = LLMResponse(
        text="Solving reverse DCF for implied FCF growth at current market price.",
        tool_calls=[
            ToolCall(
                id="tc_rev_3",
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

    # Step 4: LLM submits structured response with claims and extension
    submit_payload: dict[str, Any] = {
        "task_id": "task_val_golden_01",
        "agent_type": "valuation",
        "status": "completed",
        "confidence": 0.93,
        "confidence_rationale": (
            "Grounded on official SEC XBRL company facts and tool-computed DCF"
            " scenarios."
        ),
        "claims": [
            {
                "claim_id": "claim_base_value",
                "text": "AAPL baseline DCF intrinsic value per share is $242.50.",
                "type": "calculation",
                "confidence": 0.95,
                "evidence": [
                    {
                        "source_type": "quant_engine",
                        "tool_call_id": "tc_dcf_2",
                        "as_of_date": "2024-09-20",
                        "raw_value": "242.50",
                    }
                ],
            },
            {
                "claim_id": "claim_implied_growth",
                "text": (
                    "Reverse DCF indicates market price implies 7.2% annual FCF"
                    " growth over the next 5 years."
                ),
                "type": "fact",
                "confidence": 0.94,
                "evidence": [
                    {
                        "source_type": "quant_engine",
                        "tool_call_id": "tc_rev_3",
                        "as_of_date": "2024-09-20",
                        "raw_value": "0.072",
                    }
                ],
            },
            {
                "claim_id": "claim_val_interp",
                "text": (
                    "Baseline DCF model yields a 6.2% upside against the"
                    " prevailing market price of $228.20."
                ),
                "type": "interpretation",
                "confidence": 0.90,
                "evidence": [],
                "depends_on": ["claim_base_value", "claim_implied_growth"],
            },
        ],
        "extension": {
            "scenario_values": {
                "bear": 185.00,
                "base": 242.50,
                "bull": 295.00,
                "expected_value": 245.20,
            },
            "implied_growth": 0.072,
            "current_price": 228.20,
            "multiples": {
                "assumptions": {
                    "wacc": "9.0%",
                    "terminal_rate": "2.5%",
                    "fcf_trajectory": "8% tapering to 5%",
                }
            },
        },
    }

    step_4 = LLMResponse(
        text="Submitting final valuation specialist answer with typed extension.",
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

    # 1. Successful completion and status check
    assert result.status == "completed"
    assert result.submission is not None
    assert result.submission.agent_type == "valuation"

    # 2. Expected tools were invoked
    tool_calls_executed = [span.name for span in tracer.spans]
    assert "get_financials" in tool_calls_executed
    assert "run_dcf" in tool_calls_executed
    assert "run_reverse_dcf" in tool_calls_executed
    assert "submit_answer" in tool_calls_executed

    # 3. All executed tools stayed strictly within permissions
    for tool_name in tool_calls_executed:
        assert profile.is_tool_allowed(tool_name)

    # 4. Valid claims verified
    claims = result.submission.claims
    assert len(claims) == 3
    assert any(
        c.claim_id == "claim_base_value" and c.type == "calculation" for c in claims
    )
    assert any(
        c.claim_id == "claim_implied_growth" and c.type == "fact" for c in claims
    )
    assert any(
        c.claim_id == "claim_val_interp" and c.type == "interpretation" for c in claims
    )

    # 5. Typed answer extension matches ValuationAnswerExtension
    ext = result.submission.extension
    assert isinstance(ext, ValuationAnswerExtension)
    if isinstance(ext.scenario_values, dict):
        assert ext.scenario_values["base"] == 242.50
    else:
        assert ext.scenario_values.base == 242.50
    assert ext.implied_growth == 0.072
    assert ext.current_price == 228.20
