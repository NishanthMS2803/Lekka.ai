"""Tests for query understanding and structured ParsedQuery extraction."""

import os

import pytest

from lekka_ai.config import get_settings
from lekka_ai.llm.base import LLMResponse
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.llm.gemini import GeminiClient
from lekka_ai.orchestrator.understand import (
    ParsedQuery,
    UnknownTickerError,
    ValidateTickerInput,
    ValidateTickerTool,
    understand_query,
)


@pytest.fixture
def fake_ticker_tool() -> ValidateTickerTool:
    """Provide a ValidateTickerTool with a fixed universe of allowed tickers."""
    return ValidateTickerTool(
        allowed_tickers=["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "GOOGL"]
    )


@pytest.mark.asyncio
async def test_understand_query_investment_evaluation(
    fake_ticker_tool: ValidateTickerTool,
) -> None:
    """Example 1: Investment evaluation query parsing."""
    q = "Is AAPL a solid long-term investment over the next 3 years?"
    fake_parsed = ParsedQuery(
        intent="investment_evaluation",
        tickers=["AAPL"],
        horizon="3 years",
        risk_focus="valuation and growth",
        as_of_date="2024-09-20",
        ambiguities=[],
        needs_clarification=False,
    )
    fake_llm = FakeLLM([LLMResponse(structured_output=fake_parsed)])

    result = await understand_query(
        q,
        llm=fake_llm,
        as_of_date="2024-09-20",
        ticker_tool=fake_ticker_tool,
    )

    assert result.intent == "investment_evaluation"
    assert result.tickers == ["AAPL"]
    assert result.horizon == "3 years"
    assert result.as_of_date == "2024-09-20"
    assert result.needs_clarification is False


@pytest.mark.asyncio
async def test_understand_query_valuation_check(
    fake_ticker_tool: ValidateTickerTool,
) -> None:
    """Example 2: DCF valuation check query parsing."""
    q = "What is Microsoft's intrinsic fair value per share based on DCF?"
    fake_parsed = ParsedQuery(
        intent="valuation_check",
        tickers=["MSFT"],
        horizon="long-term",
        risk_focus=None,
        as_of_date="2024-09-20",
        ambiguities=[],
        needs_clarification=False,
    )
    fake_llm = FakeLLM([LLMResponse(structured_output=fake_parsed)])

    result = await understand_query(
        q,
        llm=fake_llm,
        as_of_date="2024-09-20",
        ticker_tool=fake_ticker_tool,
    )

    assert result.intent == "valuation_check"
    assert result.tickers == ["MSFT"]
    assert result.needs_clarification is False


@pytest.mark.asyncio
async def test_understand_query_move_explanation(
    fake_ticker_tool: ValidateTickerTool,
) -> None:
    """Example 3: Price move explanation query parsing."""
    q = "Why did NVDA drop 8% yesterday following earnings?"
    fake_parsed = ParsedQuery(
        intent="move_explanation",
        tickers=["NVDA"],
        horizon="short-term",
        risk_focus="earnings drop",
        as_of_date="2024-09-20",
        ambiguities=[],
        needs_clarification=False,
    )
    fake_llm = FakeLLM([LLMResponse(structured_output=fake_parsed)])

    result = await understand_query(
        q,
        llm=fake_llm,
        as_of_date="2024-09-20",
        ticker_tool=fake_ticker_tool,
    )

    assert result.intent == "move_explanation"
    assert result.tickers == ["NVDA"]
    assert result.risk_focus == "earnings drop"
    assert result.needs_clarification is False


@pytest.mark.asyncio
async def test_understand_query_risk_review(
    fake_ticker_tool: ValidateTickerTool,
) -> None:
    """Example 4: Quantitative risk and beta review query parsing."""
    q = "Evaluate downside volatility, drawdown, and beta risk for TSLA vs SPY"
    fake_parsed = ParsedQuery(
        intent="risk_review",
        tickers=["TSLA", "SPY"],
        horizon=None,
        risk_focus="volatility, drawdown, and beta",
        as_of_date="2024-09-20",
        ambiguities=[],
        needs_clarification=False,
    )
    fake_llm = FakeLLM([LLMResponse(structured_output=fake_parsed)])

    result = await understand_query(
        q,
        llm=fake_llm,
        as_of_date="2024-09-20",
        ticker_tool=fake_ticker_tool,
    )

    assert result.intent == "risk_review"
    assert result.tickers == ["TSLA", "SPY"]
    assert result.needs_clarification is False


@pytest.mark.asyncio
async def test_understand_query_macro_other(
    fake_ticker_tool: ValidateTickerTool,
) -> None:
    """Example 5: Broad macro / other unclassified query parsing."""
    q = "How are general macroeconomic and interest rate conditions trending?"
    fake_parsed = ParsedQuery(
        intent="other",
        tickers=[],
        horizon=None,
        risk_focus="interest rates",
        as_of_date="2024-09-20",
        ambiguities=["No specific asset or ticker queried"],
        needs_clarification=False,
    )
    fake_llm = FakeLLM([LLMResponse(structured_output=fake_parsed)])

    result = await understand_query(
        q,
        llm=fake_llm,
        as_of_date="2024-09-20",
        ticker_tool=fake_ticker_tool,
    )

    assert result.intent == "other"
    assert result.tickers == []
    assert result.needs_clarification is False


@pytest.mark.asyncio
async def test_understand_query_ambiguous_flags_clarification(
    fake_ticker_tool: ValidateTickerTool,
) -> None:
    """Example 6: Ambiguous question strictly flags needs_clarification."""
    q = "Should I buy it right now or wait for a dip?"
    fake_parsed = ParsedQuery(
        intent="investment_evaluation",
        tickers=[],
        horizon=None,
        risk_focus=None,
        as_of_date="2024-09-20",
        ambiguities=[
            "Target stock or asset is completely unspecified",
            "Investment horizon is not provided",
        ],
        needs_clarification=True,
    )
    fake_llm = FakeLLM([LLMResponse(structured_output=fake_parsed)])

    result = await understand_query(
        q,
        llm=fake_llm,
        as_of_date="2024-09-20",
        ticker_tool=fake_ticker_tool,
    )

    # Verification: ambiguous question flags needs_clarification
    assert result.needs_clarification is True
    assert len(result.ambiguities) >= 1
    assert any("unspecified" in a.lower() for a in result.ambiguities)


@pytest.mark.asyncio
async def test_understand_query_rejects_unknown_ticker(
    fake_ticker_tool: ValidateTickerTool,
) -> None:
    """Verify runs with unknown/unrecognized tickers are rejected via ticker tool."""
    q = "What is the valuation and target price for FAKETICKERXYZ99?"
    fake_parsed = ParsedQuery(
        intent="valuation_check",
        tickers=["FAKETICKERXYZ99"],
        as_of_date="2024-09-20",
        needs_clarification=False,
    )
    fake_llm = FakeLLM([LLMResponse(structured_output=fake_parsed)])

    with pytest.raises(UnknownTickerError, match="Unknown ticker symbol"):
        await understand_query(
            q,
            llm=fake_llm,
            ticker_tool=fake_ticker_tool,
        )


@pytest.mark.asyncio
async def test_validate_ticker_tool_direct() -> None:
    """Verify ValidateTickerTool logic across valid and invalid inputs."""
    tool = ValidateTickerTool(allowed_tickers=["AAPL", "MSFT"])

    # Valid ticker
    res_valid = await tool.run(ValidateTickerInput(ticker="AAPL"))
    assert res_valid.is_valid is True
    assert res_valid.ticker == "AAPL"

    # Case insensitive
    res_case = await tool.run(ValidateTickerInput(ticker="msft"))
    assert res_case.is_valid is True
    assert res_case.ticker == "MSFT"

    # Invalid ticker
    res_invalid = await tool.run(ValidateTickerInput(ticker="NONEXISTENT"))
    assert res_invalid.is_valid is False
    assert "not found" in (res_invalid.error or "").lower()

    # Empty ticker
    res_empty = await tool.run(ValidateTickerInput(ticker=""))
    assert res_empty.is_valid is False


@pytest.mark.live
@pytest.mark.asyncio
async def test_understand_query_live() -> None:
    """Live test running query understanding with real Gemini structured output."""
    if os.getenv("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE is not set to 1; skipping live query understanding test")

    settings = get_settings()
    if not settings.GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY is not configured; skipping live test")

    llm = GeminiClient(
        api_key=settings.GEMINI_API_KEY,
        model=settings.GEMINI_MODEL,
    )

    q = (
        "What is Apple's (AAPL) DCF intrinsic valuation over a 2-year horizon "
        "as of 2024-09-20?"
    )
    parsed = await understand_query(
        q,
        llm=llm,
        as_of_date="2024-09-20",
    )

    assert parsed.intent in {"valuation_check", "investment_evaluation"}
    assert "AAPL" in parsed.tickers
    assert parsed.as_of_date == "2024-09-20"
    assert parsed.needs_clarification is False
