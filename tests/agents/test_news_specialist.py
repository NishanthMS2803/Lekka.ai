"""Golden scenario isolation test for News & catalyst specialist agent."""

from typing import Any

import pytest

from lekka_ai.agents import load_agent_profile
from lekka_ai.harness.models import AgentTask, Limits
from lekka_ai.harness.permissions import ToolView
from lekka_ai.harness.runtime import AgentRuntime
from lekka_ai.llm.base import LLMResponse, ToolCall, Usage
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.schemas.news import NewsAnswerExtension
from lekka_ai.tools.base import ToolRegistry
from lekka_ai.tracing import Tracer


@pytest.mark.asyncio
async def test_news_specialist_golden_scenario(
    fixture_tool_registry: ToolRegistry,
) -> None:
    """Golden scenario: News agent calls expected tools & produces claims."""
    profile = load_agent_profile("news")
    tracer = Tracer()
    tool_view = ToolView(fixture_tool_registry, profile=profile, tracer=tracer)

    task = AgentTask(
        task_id="task_news_golden_01",
        instruction=(
            "List recent developments with dates for AAPL as of 2024-09-20. "
            "Use get_news and list_filings for 8-K disclosures. Classify each "
            "as positive, negative, neutral or uncertain, and as new vs "
            "already-known. Submit answer with catalyst extension."
        ),
        as_of_date="2024-09-20",
        agent_type="news",
    )

    # Step 1: LLM queries recent news headlines
    step_1 = LLMResponse(
        text="Retrieving recent company news articles from fixture provider.",
        tool_calls=[
            ToolCall(
                id="tc_news_1",
                name="get_news",
                args={
                    "ticker": "AAPL",
                    "as_of_date": "2024-09-20",
                    "limit": 5,
                },
            )
        ],
        usage=Usage(input_tokens=100, output_tokens=35),
    )

    # Step 2: LLM queries SEC Form 8-K filings for material corporate events
    step_2 = LLMResponse(
        text="Listing Form 8-K filings for material corporate disclosures.",
        tool_calls=[
            ToolCall(
                id="tc_filings_2",
                name="list_filings",
                args={
                    "ticker": "AAPL",
                    "form_types": ["8-K"],
                    "as_of_date": "2024-09-20",
                    "limit": 5,
                },
            )
        ],
        usage=Usage(input_tokens=130, output_tokens=35),
    )

    # Step 3: LLM submits structured response with claims and catalyst extension
    submit_payload: dict[str, Any] = {
        "task_id": "task_news_golden_01",
        "agent_type": "news",
        "status": "completed",
        "confidence": 0.93,
        "confidence_rationale": (
            "Grounded on verified press releases and official SEC Form 8-K filings."
        ),
        "claims": [
            {
                "claim_id": "claim_cfo_trans",
                "text": (
                    "Apple announced planned CFO transition for Luca Maestri on"
                    " 2024-08-26."
                ),
                "type": "fact",
                "confidence": 0.98,
                "evidence": [
                    {
                        "source_type": "sec_filings",
                        "tool_call_id": "tc_filings_2",
                        "as_of_date": "2024-09-20",
                        "raw_value": "Form 8-K filed 2024-08-26",
                    }
                ],
            },
            {
                "claim_id": "claim_iphone16_event",
                "text": ("Apple debuted the iPhone 16 product lineup on 2024-09-09."),
                "type": "fact",
                "confidence": 0.96,
                "evidence": [
                    {
                        "source_type": "news",
                        "tool_call_id": "tc_news_1",
                        "as_of_date": "2024-09-20",
                        "raw_value": "Apple Debuts iPhone 16 Lineup",
                    }
                ],
            },
            {
                "claim_id": "claim_catalyst_eval",
                "text": (
                    "The iPhone 16 launch represents a positive new catalyst"
                    " driving fall cycle revenue expectations."
                ),
                "type": "interpretation",
                "confidence": 0.89,
                "evidence": [],
                "depends_on": ["claim_iphone16_event"],
            },
        ],
        "extension": {
            "catalysts": [
                {
                    "headline": "Apple announced planned CFO transition plan",
                    "date": "2024-08-26",
                    "impact": "neutral",
                    "novelty": "new",
                    "source": "SEC Form 8-K",
                },
                {
                    "headline": (
                        "Apple launched iPhone 16 lineup with Apple Intelligence"
                    ),
                    "date": "2024-09-09",
                    "impact": "positive",
                    "novelty": "new",
                    "source": "Reuters",
                },
            ],
            "summary": (
                "Key catalysts include the planned CFO transition and iPhone"
                " 16 launch cycle."
            ),
        },
    }

    step_3 = LLMResponse(
        text="Submitting validated news catalyst analysis with structured extension.",
        tool_calls=[
            ToolCall(
                id="tc_sub_3",
                name="submit_answer",
                args=submit_payload,
            )
        ],
        usage=Usage(input_tokens=170, output_tokens=55),
    )

    fake_llm = FakeLLM(responses=[step_1, step_2, step_3])

    result = await AgentRuntime.run(
        task=task,
        profile=profile,
        llm=fake_llm,
        tool_view=tool_view,
        limits=Limits(max_iterations=5),
    )

    # 1. Successful completion and status check
    assert result.status == "completed"
    assert result.submission is not None
    assert result.submission.agent_type == "news"

    # 2. Expected tools were invoked
    tool_calls_executed = [span.name for span in tracer.spans]
    assert "get_news" in tool_calls_executed
    assert "list_filings" in tool_calls_executed
    assert "submit_answer" in tool_calls_executed

    # 3. All executed tools stayed strictly within permissions
    for tool_name in tool_calls_executed:
        assert profile.is_tool_allowed(tool_name)

    # 4. Valid claims verified
    claims = result.submission.claims
    assert len(claims) == 3
    assert any(c.claim_id == "claim_cfo_trans" and c.type == "fact" for c in claims)
    assert any(
        c.claim_id == "claim_iphone16_event" and c.type == "fact" for c in claims
    )
    assert any(
        c.claim_id == "claim_catalyst_eval" and c.type == "interpretation"
        for c in claims
    )

    # 5. Typed answer extension matches NewsAnswerExtension
    ext = result.submission.extension
    assert isinstance(ext, NewsAnswerExtension)
    assert len(ext.catalysts) == 2
    assert ext.catalysts[0].impact == "neutral"
    assert ext.catalysts[1].impact == "positive"
