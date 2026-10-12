"""Unit tests for the analyst profile and Typer CLI."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from lekka_ai.cli import app
from lekka_ai.harness.profiles import (
    ANALYST_PROFILE,
    create_analyst_profile,
    get_analyst_tools,
)
from lekka_ai.harness.runtime import AgentRunResult
from lekka_ai.schemas.claims import Claim, Evidence
from lekka_ai.schemas.submit import AnswerStatus, SubmitAnswer
from lekka_ai.tools.base import ToolRegistry
from lekka_ai.tools.echo import EchoTool
from lekka_ai.tools.python_sandbox import RunPythonTool

runner = CliRunner(env={"COLUMNS": "160"})


def test_analyst_profile_tools() -> None:
    """Verify analyst profile authorizes non-sandbox tools and excludes run_python."""
    profile = create_analyst_profile()
    assert profile.agent_type == "analyst"
    assert "run_python" not in profile.allowed_tools
    assert "get_market_data" in profile.allowed_tools
    assert "get_financials" in profile.allowed_tools
    assert "submit_answer" in profile.allowed_tools or profile.is_tool_allowed(
        "submit_answer"
    )

    # Custom registry test
    custom_reg = ToolRegistry()
    custom_reg.register(EchoTool())
    custom_reg.register(RunPythonTool())
    tools = get_analyst_tools(custom_reg)
    assert "echo" in tools
    assert "run_python" not in tools


def test_analyst_system_prompt_content() -> None:
    """Verify system prompt enforces numbers from tools, types, and gaps."""
    prompt = ANALYST_PROFILE.load_system_prompt()
    assert prompt is not None
    prompt_lower = prompt.lower()
    assert "numbers only from tools" in prompt_lower or "tools" in prompt_lower
    assert "claim typing" in prompt_lower or "epistemic" in prompt_lower
    assert (
        "insufficient evidence" in prompt_lower
        or "insufficient_evidence" in prompt_lower
    )
    assert "fact" in prompt_lower
    assert "calculation" in prompt_lower
    assert "interpretation" in prompt_lower


def test_cli_help() -> None:
    """Verify CLI --help displays available commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ask" in result.stdout
    assert "version" in result.stdout


def test_cli_version() -> None:
    """Verify CLI version command output."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_cli_ask_help() -> None:
    """Verify CLI ask command --help output."""
    result = runner.invoke(app, ["ask", "--help"])
    assert result.exit_code == 0
    assert "--as-of" in result.stdout
    assert "--max-iterations" in result.stdout


def test_cli_ask_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI prints error and exits when GEMINI_API_KEY is missing."""
    with patch("lekka_ai.cli.get_settings") as mock_settings:
        mock_settings.return_value.GEMINI_API_KEY = ""
        result = runner.invoke(app, ["ask", "What is AAPL stock price?"])
        assert result.exit_code == 1
        assert "GEMINI_API_KEY is not set" in result.stdout


def test_cli_ask_mocked_success() -> None:
    """Verify ask command executes and formats claims and findings correctly."""
    mock_submission = SubmitAnswer(
        task_id="task_test_123",
        agent_type="analyst",
        status=AnswerStatus.COMPLETED,
        confidence=0.95,
        confidence_rationale="Grounded in official market price data.",
        claims=[
            Claim(
                text="AAPL closed at 185.92 on 2024-01-12",
                type="fact",
                confidence=1.0,
                evidence=[
                    Evidence(
                        source_type="market_data",
                        tool_call_id="call_mock_1",
                        as_of_date="2024-01-12",
                        raw_value={"close": 185.92},
                    )
                ],
            )
        ],
    )
    mock_result = AgentRunResult(
        task_id="task_test_123",
        agent_type="analyst",
        status="completed",
        iterations=2,
        tokens_in=1200,
        tokens_out=250,
        tool_calls_count=2,
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
                "What was AAPL price on 2024-01-12?",
                "--as-of",
                "2024-01-15",
                "-v",
            ],
        )
        assert result.exit_code == 0
        assert "COMPLETED" in result.stdout
        assert "95%" in result.stdout
        assert "Grounded in official market price data" in result.stdout
        assert "185.92" in result.stdout
        assert "market_data" in result.stdout
        assert "Execution Metrics" in result.stdout


@pytest.mark.live
def test_cli_ask_live_query() -> None:
    """Live integration test querying a ticker with real Gemini LLM."""
    if os.getenv("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE is not set to 1; skipping live CLI test")

    result = runner.invoke(
        app,
        [
            "ask",
            "What was Apple (AAPL) closing stock price on 2024-01-12?",
            "--as-of",
            "2024-01-15",
        ],
    )
    assert result.exit_code == 0
    assert "COMPLETED" in result.stdout or "PARTIAL" in result.stdout
    assert "AAPL" in result.stdout or "Apple" in result.stdout or "185" in result.stdout
