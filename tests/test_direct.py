"""Unit and integration tests for DirectLLMRunner (V1 baseline)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from lekka_ai.cli import app
from lekka_ai.harness.direct import (
    DirectLLMRunner,
    ExtractedNumericClaims,
    NumericClaimItem,
    _heuristic_extract_numeric_claims,
)
from lekka_ai.harness.models import AgentTask
from lekka_ai.harness.runtime import AgentRunResult
from lekka_ai.llm.base import LLMResponse
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.llm.gemini import GeminiClient
from lekka_ai.schemas.claims import Claim, Evidence
from lekka_ai.schemas.submit import AnswerStatus, SubmitAnswer

runner = CliRunner()


@pytest.mark.asyncio
async def test_direct_llm_runner_with_fakellm_structured_claims() -> None:
    """Verify DirectLLMRunner executes without tools and decodes structured claims."""
    # Step 1: Direct text answer
    resp1 = LLMResponse(
        text=(
            "Apple (AAPL) closed at $185.92 on January 12, 2024 with a volume of 42.5M."
        )
    )
    # Step 2: Structured extraction
    extracted_payload = ExtractedNumericClaims(
        claims=[
            NumericClaimItem(
                text="Apple (AAPL) closed at $185.92 on January 12, 2024",
                type="fact",
                value=185.92,
                metric="closing price",
                as_of_date="2024-01-12",
            ),
            NumericClaimItem(
                text="Trading volume was 42.5M shares",
                type="fact",
                value=42500000.0,
                metric="volume",
                as_of_date="2024-01-12",
            ),
        ]
    )
    resp2 = LLMResponse(
        text="",
        structured_output=extracted_payload,
    )

    fake_llm = FakeLLM(responses=[resp1, resp2])
    task = AgentTask(
        instruction="What was Apple stock price on Jan 12, 2024?",
        as_of_date="2024-01-15",
        agent_type="direct",
    )

    result = await DirectLLMRunner.run(task=task, llm=fake_llm)

    assert result.status == "completed"
    assert result.agent_type == "direct"
    assert result.tool_calls_count == 0
    assert result.final_text == resp1.text
    assert result.submission is not None
    assert len(result.submission.claims) == 2

    c1 = result.submission.claims[0]
    assert "185.92" in c1.text
    assert c1.type == "fact"
    assert c1.produced_by == "direct_llm"
    assert len(c1.evidence) == 1
    assert c1.evidence[0].source_type == "direct_llm"
    assert c1.evidence[0].raw_value == {"value": 185.92, "metric": "closing price"}

    # Verify no tools were passed to the LLM in calls
    assert len(fake_llm.calls) == 2
    assert fake_llm.calls[0]["tools"] is None
    assert fake_llm.calls[1]["tools"] is None
    assert "2024-01-15" in str(fake_llm.calls[0]["system"])


@pytest.mark.asyncio
async def test_direct_llm_runner_with_fakellm_heuristic_fallback() -> None:
    """Verify DirectLLMRunner falls back to heuristic extraction if JSON fails."""
    resp1 = LLMResponse(
        text="Tesla delivered 484,507 vehicles in Q4 2023. Total revenue was $25.17B."
    )
    # Step 2 returns plain text without valid JSON structure
    resp2 = LLMResponse(text="Here are the claims: not a valid json object.")

    fake_llm = FakeLLM(responses=[resp1, resp2])
    task = AgentTask(
        instruction="Summarize Tesla Q4 2023 performance.",
        agent_type="direct",
    )

    result = await DirectLLMRunner.run(task=task, llm=fake_llm)

    assert result.status == "completed"
    assert result.submission is not None
    assert len(result.submission.claims) >= 1
    texts = [c.text for c in result.submission.claims]
    assert any("484,507" in t or "25.17" in t for t in texts)


@pytest.mark.asyncio
async def test_direct_llm_runner_string_task_and_no_extraction() -> None:
    """Verify running with plain string instruction and extract_claims=False."""
    resp1 = LLMResponse(text="NVIDIA market cap exceeded $3.0T.")
    fake_llm = FakeLLM(responses=[resp1])

    result = await DirectLLMRunner.run(
        task="What is Nvidia market cap?",
        llm=fake_llm,
        extract_claims=False,
    )
    assert result.status == "completed"
    assert result.final_text == resp1.text
    assert result.submission is not None
    # With extract_claims=False, claims list is empty or minimal
    assert fake_llm.calls[0]["tools"] is None
    assert len(fake_llm.calls) == 1


def test_heuristic_extraction_helper() -> None:
    """Verify regex-based claim extractor parses numbers correctly."""
    text = (
        "Microsoft revenue reached $62.0 billion. "
        "Operating income grew 33% year over year. "
        "The company remains optimistic."
    )
    claims = _heuristic_extract_numeric_claims(text)
    assert len(claims) >= 2
    assert any(c.value == 62.0 for c in claims)
    assert any(c.value == 33.0 for c in claims)


def test_cli_ask_direct_variant_mocked() -> None:
    """Verify CLI --variant direct runs and prints direct baseline panel."""
    mock_submission = SubmitAnswer(
        task_id="task_direct_001",
        agent_type="direct",
        status=AnswerStatus.COMPLETED,
        confidence=1.0,
        confidence_rationale="Direct LLM completion without external tools.",
        claims=[
            Claim(
                text="Apple closed at $185.92 on 2024-01-12",
                type="fact",
                confidence=1.0,
                produced_by="direct_llm",
                evidence=[
                    Evidence(
                        source_type="direct_llm",
                        as_of_date="2024-01-12",
                        raw_value=185.92,
                    )
                ],
            )
        ],
    )
    mock_result = AgentRunResult(
        task_id="task_direct_001",
        agent_type="direct",
        status="completed",
        final_text="Apple stock closed at $185.92 on January 12, 2024.",
        iterations=1,
        tokens_in=150,
        tokens_out=80,
        tool_calls_count=0,
        submission=mock_submission,
    )

    with (
        patch("lekka_ai.cli._run_ask", new_callable=AsyncMock) as mock_run,
        patch("lekka_ai.cli.get_settings") as mock_settings,
    ):
        mock_settings.return_value.GEMINI_API_KEY = "dummy_key"
        mock_run.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "ask",
                "What was Apple stock price on 2024-01-12?",
                "--variant",
                "direct",
                "--as-of",
                "2024-01-15",
                "-v",
            ],
        )

        assert result.exit_code == 0
        assert "Direct LLM Baseline - V1" in result.stdout
        assert "COMPLETED" in result.stdout
        assert "185.92" in result.stdout
        assert "direct_llm" in result.stdout
        assert "0 tool calls" in result.stdout


@pytest.mark.live
@pytest.mark.asyncio
async def test_direct_llm_runner_live() -> None:
    """Live integration test with real GeminiClient extracting numeric claims."""
    if os.getenv("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE is not set to 1; skipping live direct LLM test")

    client = GeminiClient()
    task = AgentTask(
        instruction="What was Apple (AAPL) closing stock price on 2024-01-12?",
        as_of_date="2024-01-15",
        agent_type="direct",
    )

    result = await DirectLLMRunner.run(task=task, llm=client)
    assert result.status == "completed"
    assert result.agent_type == "direct"
    assert result.tool_calls_count == 0
    assert result.final_text is not None
    assert len(result.final_text) > 0
    assert result.submission is not None
    assert len(result.submission.claims) > 0
    assert any("185" in c.text for c in result.submission.claims)
