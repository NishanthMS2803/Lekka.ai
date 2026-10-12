"""Live integration test for failure suite scenarios against Gemini API."""

import os

import pytest

from lekka_ai.llm.gemini import GeminiClient
from lekka_ai.orchestrator.understand import understand_query


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_ambiguous_question_demands_clarification() -> None:
    """Live Gemini identifies ambiguous question and demands clarification."""
    if os.getenv("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE is not set to 1; skipping live failure test")

    client = GeminiClient()
    parsed = await understand_query(
        "Is the company obviously a good investment and should I buy it now?",
        llm=client,
    )

    # In live Gemini call, query with no entity/ticker should flag needs_clarification
    assert parsed.needs_clarification is True
    assert len(parsed.ambiguities) > 0
