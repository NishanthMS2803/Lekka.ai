"""Live integration test for GeminiClient."""

import asyncio
import os

import pytest

from lekka_ai.llm.base import Message
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

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = await client.generate(
                messages=messages, system="You are a helpful ping-pong bot."
            )
            assert response.text is not None
            assert len(response.text.strip()) > 0
            assert "PONG" in response.text.upper()
            assert response.usage.input_tokens > 0
            assert response.usage.output_tokens > 0
            return
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(2)

    if last_error is not None:
        raise last_error
