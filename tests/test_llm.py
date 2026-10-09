"""Tests for LLM base models and FakeLLM provider."""

import pytest

from lekka_ai.llm import (
    FakeLLM,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)


@pytest.mark.asyncio
async def test_fake_llm_returns_scripted_responses_in_order() -> None:
    """FakeLLM returns predefined responses sequentially."""
    first = LLMResponse(
        text="First response", usage=Usage(input_tokens=10, output_tokens=5)
    )
    second = LLMResponse(
        tool_calls=[ToolCall(id="tc-1", name="sec_query", args={"ticker": "AAPL"})],
        usage=Usage(input_tokens=20, output_tokens=15),
    )

    llm = FakeLLM(responses=[first, second])

    msg = [Message(role="user", content="Hello")]
    resp1 = await llm.generate(msg)
    assert resp1.text == "First response"
    assert resp1.usage.input_tokens == 10
    assert resp1.usage.output_tokens == 5
    assert len(resp1.tool_calls) == 0

    resp2 = await llm.generate(msg)
    assert resp2.text is None
    assert len(resp2.tool_calls) == 1
    assert resp2.tool_calls[0].name == "sec_query"
    assert resp2.tool_calls[0].args == {"ticker": "AAPL"}


@pytest.mark.asyncio
async def test_fake_llm_records_call_arguments() -> None:
    """FakeLLM accurately records calls for assertions."""
    llm = FakeLLM()

    messages = [
        Message(role="system", content="You are a financial analyst."),
        Message(role="user", content="What was MSFT Q3 revenue?"),
    ]
    tools = [
        ToolSpec(
            name="get_revenue",
            description="Fetches company revenue",
            json_schema={
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
            },
        )
    ]

    response = await llm.generate(
        messages=messages,
        tools=tools,
        system="System prompt",
    )

    assert response.text == "default scripted response"
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["messages"] == messages
    assert call["tools"] == tools
    assert call["system"] == "System prompt"
