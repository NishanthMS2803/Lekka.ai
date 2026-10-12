"""End-to-end tests for orchestrator dispatch glue and ledger persistence."""

import uuid
from typing import Any

import pytest
from sqlalchemy import text

from lekka_ai.db.ledger import LedgerRepository
from lekka_ai.db.session import get_engine, get_session
from lekka_ai.llm.base import LLMResponse, ToolCall, Usage
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.orchestrator.budget import RunBudget
from lekka_ai.orchestrator.dispatch import (
    TaskResult,
    get_relevant_context_facts,
    run_task,
)
from lekka_ai.orchestrator.state import Plan, PlannedTask, RunState, to_uuid
from lekka_ai.tools.base import ToolRegistry


@pytest.fixture
def test_db_session() -> Any:
    """Provide a database session for testing, skipping if database is offline."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("Database is not reachable")

    gen = get_session()
    session = next(gen)
    try:
        yield session
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


@pytest.mark.asyncio
async def test_dispatch_two_agents_end_to_end_populates_ledger(
    test_db_session: Any,
    fixture_tool_registry: ToolRegistry,
) -> None:
    """Verify two-agent run populates the database ledger with grounded claims."""
    run_id_val = str(uuid.uuid4())
    state = RunState(
        run_id=run_id_val,
        question="Analyze AAPL market price action and recent news catalysts",
        parsed_query={"as_of_date": "2024-01-12"},
        plan=Plan(),
    )

    budget = RunBudget.from_settings()
    ledger = LedgerRepository(test_db_session)

    # -------------------------------------------------------------
    # Agent 1: Market specialist
    # -------------------------------------------------------------
    task_market = PlannedTask(
        task_id="task_market_01",
        agent_type="market",
        instruction=(
            "Evaluate AAPL price action and annualized volatility as of 2024-01-12"
        ),
        priority=2,
    )
    state.plan.add_task(task_market)

    market_task_uuid = str(to_uuid("task_market_01"))

    # Turn 1: Market agent queries historical price data
    market_step_1 = LLMResponse(
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

    # Turn 2: Market agent computes risk stats
    market_step_2 = LLMResponse(
        text="Computing annualized volatility and beta against SPY.",
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
        usage=Usage(input_tokens=110, output_tokens=35),
    )

    # Turn 3: Market agent submits verified claims with typed extension
    market_submit_payload: dict[str, Any] = {
        "task_id": market_task_uuid,
        "agent_type": "market",
        "status": "completed",
        "confidence": 0.95,
        "confidence_rationale": (
            "Historical market closing prices and quant risk verified."
        ),
        "claims": [
            {
                "claim_id": "claim_aapl_close",
                "text": "AAPL closing price was $185.92 on 2024-01-12",
                "type": "fact",
                "confidence": 0.98,
                "produced_by": "market",
                "evidence": [
                    {
                        "source_type": "market_data",
                        "tool_call_id": "tc_mkt_data_1",
                        "raw_value": "185.92",
                        "as_of_date": "2024-01-12",
                    }
                ],
            },
            {
                "claim_id": "claim_aapl_beta",
                "text": "AAPL has a market beta of 1.55 relative to SPY",
                "type": "fact",
                "confidence": 0.92,
                "produced_by": "market",
                "evidence": [
                    {
                        "source_type": "quant_engine",
                        "tool_call_id": "tc_risk_stats_2",
                        "raw_value": "1.55",
                        "as_of_date": "2024-01-12",
                    }
                ],
            },
        ],
        "extension": {
            "trend_label": "BULLISH",
            "key_stats": {
                "latest_close": 185.92,
                "annualized_volatility": 0.177,
                "beta": 1.55,
            },
        },
    }

    market_step_3 = LLMResponse(
        text="Submitting completed market evaluation.",
        tool_calls=[
            ToolCall(
                id="tc_market_sub",
                name="submit_answer",
                args=market_submit_payload,
            )
        ],
        usage=Usage(input_tokens=140, output_tokens=50),
    )

    fake_llm_1 = FakeLLM(responses=[market_step_1, market_step_2, market_step_3])

    result_1 = await run_task(
        planned_task=task_market,
        state=state,
        budget=budget,
        llm=fake_llm_1,
        tool_registry=fixture_tool_registry,
        session=test_db_session,
        ledger=ledger,
    )

    # Assertions on Agent 1 (Market)
    assert isinstance(result_1, TaskResult)
    assert result_1.status == "completed"
    assert len(result_1.claims) == 2
    assert state.task_statuses["task_market_01"] == "completed"
    assert "task_market_01" in state.results
    assert result_1.usage["tool_calls"] >= 3
    assert result_1.usage["total_tokens"] > 0

    # Verify Market claims are stored in the database ledger
    market_db_claims = ledger.get_claims(run_id=to_uuid(state.run_id))
    assert len(market_db_claims) == 2
    claim_texts = {c.text for c in market_db_claims}
    assert "AAPL closing price was $185.92 on 2024-01-12" in claim_texts

    # -------------------------------------------------------------
    # Agent 2: News specialist (depends on Agent 1)
    # -------------------------------------------------------------
    task_news = PlannedTask(
        task_id="task_news_02",
        agent_type="news",
        instruction="Evaluate catalysts explaining market price movement",
        depends_on=["task_market_01"],
        priority=1,
    )
    state.plan.add_task(task_news)

    # Verify context resolution: news agent gets market agent's claims as context facts!
    context_facts = get_relevant_context_facts(
        planned_task=task_news,
        state=state,
        ledger=ledger,
    )
    assert len(context_facts) == 2
    assert any(cf["claim_id"] == "claim_aapl_close" for cf in context_facts)

    news_task_uuid = str(to_uuid("task_news_02"))

    # Turn 1: News agent retrieves news articles
    news_step_1 = LLMResponse(
        text="Querying news articles for AAPL from fixture provider.",
        tool_calls=[
            ToolCall(
                id="tc_news_1",
                name="get_news",
                args={
                    "ticker": "AAPL",
                    "as_of_date": "2024-01-12",
                },
            )
        ],
        usage=Usage(input_tokens=100, output_tokens=30),
    )

    # Turn 2: News agent submits verified catalyst claims
    news_submit_payload: dict[str, Any] = {
        "task_id": news_task_uuid,
        "agent_type": "news",
        "status": "completed",
        "confidence": 0.90,
        "confidence_rationale": "Verified dividend and product announcement news.",
        "claims": [
            {
                "claim_id": "claim_aapl_catalyst",
                "text": "Apple declared cash dividend of $0.25 per share",
                "type": "fact",
                "confidence": 0.94,
                "produced_by": "news",
                "evidence": [
                    {
                        "source_type": "news",
                        "tool_call_id": "tc_news_1",
                        "raw_value": "Cash dividend of $0.25",
                        "as_of_date": "2024-01-12",
                    }
                ],
            }
        ],
        "extension": {
            "catalysts": [
                {
                    "date": "2024-08-01",
                    "headline": "Apple Board Declares Cash Dividend of $0.25",
                    "impact": "positive",
                    "novelty": "already_known",
                    "source": "PR Newswire",
                }
            ]
        },
    }

    news_step_2 = LLMResponse(
        text="Submitting completed news catalyst analysis.",
        tool_calls=[
            ToolCall(
                id="tc_news_sub",
                name="submit_answer",
                args=news_submit_payload,
            )
        ],
        usage=Usage(input_tokens=150, output_tokens=45),
    )

    fake_llm_2 = FakeLLM(responses=[news_step_1, news_step_2])

    result_2 = await run_task(
        planned_task=task_news,
        state=state,
        budget=budget,
        llm=fake_llm_2,
        tool_registry=fixture_tool_registry,
        session=test_db_session,
        ledger=ledger,
    )

    # Assertions on Agent 2 (News)
    assert isinstance(result_2, TaskResult)
    assert result_2.status == "completed"
    assert len(result_2.claims) == 1
    assert state.task_statuses["task_news_02"] == "completed"
    assert "task_news_02" in state.results

    # -------------------------------------------------------------
    # Verification: Two-agent run populates the ledger with claims from BOTH agents
    # -------------------------------------------------------------
    all_db_claims = ledger.get_claims(run_id=to_uuid(state.run_id))
    assert len(all_db_claims) == 3

    producers = {c.produced_by for c in all_db_claims}
    assert "market" in producers
    assert "news" in producers

    # Verify both task results merged into RunState
    assert state.is_complete() is True
    assert len(state.results) == 2


@pytest.mark.asyncio
async def test_context_facts_filtering_without_dependencies() -> None:
    """Verify root tasks with empty depends_on receive empty dependency facts."""
    state = RunState(
        run_id=str(uuid.uuid4()),
        question="Isolated task",
        plan=Plan(),
    )
    standalone_task = PlannedTask(
        task_id="task_standalone",
        agent_type="valuation",
        instruction="DCF calculation without upstream dependencies",
        depends_on=[],
    )
    facts = get_relevant_context_facts(
        planned_task=standalone_task,
        state=state,
        ledger=None,
    )
    assert facts == []
