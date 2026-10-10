"""Live integration tests for GeminiClient."""

import os

import pytest
from pydantic import BaseModel

from lekka_ai.llm.base import Message, ToolSpec
from lekka_ai.llm.gemini import GeminiClient


@pytest.mark.live
@pytest.mark.asyncio
async def test_gemini_live_trivial_generation() -> None:
    """Send a trivial prompt to live Gemini API and verify text and token usage."""
    if os.getenv("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE is not set to 1; skipping live Gemini test")

    client = GeminiClient()
    messages = [
        Message(role="user", content="Reply with exactly the single word 'PONG'.")
    ]
    response = await client.generate(
        messages=messages, system="You are a helpful ping-pong bot."
    )

    assert response.text is not None
    assert len(response.text.strip()) > 0
    assert "PONG" in response.text.upper()
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


@pytest.mark.live
@pytest.mark.asyncio
async def test_gemini_live_tool_calling() -> None:
    """Verify Gemini triggers a tool call matching ToolSpec schema."""
    if os.getenv("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE is not set to 1; skipping live Gemini test")

    client = GeminiClient()
    tool = ToolSpec(
        name="lookup_stock_quote",
        description="Retrieve the real-time market quote for a stock ticker symbol.",
        json_schema={
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., NVDA, AAPL)",
                }
            },
            "required": ["ticker"],
        },
    )

    messages = [
        Message(
            role="user",
            content="Check the current stock quote for NVDA using lookup_stock_quote.",
        )
    ]
    response = await client.generate(messages=messages, tools=[tool])

    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "lookup_stock_quote"
    assert "NVDA" in str(call.args.get("ticker", "")).upper()
    assert response.usage.input_tokens > 0


class FinancialProfile(BaseModel):
    """Structured financial profile schema for test."""

    company: str
    sector: str
    is_public: bool


@pytest.mark.live
@pytest.mark.asyncio
async def test_gemini_live_structured_output() -> None:
    """Verify Gemini returns a validated Pydantic model for response_schema."""
    if os.getenv("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE is not set to 1; skipping live Gemini test")

    client = GeminiClient()
    messages = [
        Message(
            role="user",
            content="Profile for Microsoft: company, sector, is_public.",
        )
    ]
    response = await client.generate(
        messages=messages,
        response_schema=FinancialProfile,
    )

    assert response.structured_output is not None
    assert isinstance(response.structured_output, FinancialProfile)
    assert "MICROSOFT" in response.structured_output.company.upper()
    assert response.structured_output.is_public is True
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
