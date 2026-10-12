"""Unit and integration tests for parent LLM orchestrator planner."""

import os

import pytest

from lekka_ai.config import get_settings
from lekka_ai.llm.base import LLMResponse, Usage
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.llm.gemini import GeminiClient
from lekka_ai.orchestrator.planner import (
    EmptyPlanError,
    GenericInstructionError,
    InvalidDependencyError,
    PlannedTaskSpec,
    PlanSpec,
    UnknownAgentTypeError,
    plan_query,
    validate_plan,
)
from lekka_ai.orchestrator.state import Plan, PlannedTask
from lekka_ai.orchestrator.understand import ParsedQuery, QueryIntent

ALLOWED_PROFILES = [
    "market",
    "news",
    "fundamental",
    "valuation",
    "risk",
    "analyst",
]


@pytest.fixture
def move_explanation_query() -> ParsedQuery:
    """Fixture for a 'why did it drop today' move explanation query."""
    return ParsedQuery(
        intent=QueryIntent.MOVE_EXPLANATION,
        tickers=["AAPL"],
        horizon="1d",
        risk_focus=None,
        as_of_date="2024-10-12",
        ambiguities=[],
        needs_clarification=False,
    )


@pytest.fixture
def investment_evaluation_query() -> ParsedQuery:
    """Fixture for an 'is it attractive' investment evaluation query."""
    return ParsedQuery(
        intent=QueryIntent.INVESTMENT_EVALUATION,
        tickers=["AAPL"],
        horizon="3-5 years",
        risk_focus="downside",
        as_of_date="2024-10-12",
        ambiguities=[],
        needs_clarification=False,
    )


@pytest.mark.asyncio
async def test_plan_move_explanation_fake_llm(
    move_explanation_query: ParsedQuery,
) -> None:
    """Verify 'why did it drop today' plans news + market only, no valuation."""
    fake_spec = PlanSpec(
        tasks=[
            PlannedTaskSpec(
                task_id="task_market_aapl",
                agent_type="market",
                instruction=(
                    "Retrieve intraday and 1-month market price action, volume "
                    "spikes, and moving average breakdown for AAPL as of 2024-10-12."
                ),
                depends_on=[],
                priority=2,
            ),
            PlannedTaskSpec(
                task_id="task_news_aapl",
                agent_type="news",
                instruction=(
                    "Search recent SEC 8-K filings and breaking news catalysts for "
                    "AAPL as of 2024-10-12 to identify earnings misses or downgrades."
                ),
                depends_on=[],
                priority=2,
            ),
        ]
    )

    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                text="",
                structured_output=fake_spec,
                usage=Usage(input_tokens=150, output_tokens=80),
            )
        ]
    )

    plan = await plan_query(
        query=move_explanation_query,
        available_profiles=ALLOWED_PROFILES,
        llm=fake_llm,
    )

    assert len(plan.tasks) == 2
    agent_types = {t.agent_type for t in plan.tasks}
    assert agent_types == {"market", "news"}

    # Must NOT include valuation or fundamental
    assert "valuation" not in agent_types
    assert "fundamental" not in agent_types

    # Both must be wave-1 (no dependencies)
    for task in plan.tasks:
        assert task.depends_on == []
        assert "analyze" not in task.instruction.lower().split()


@pytest.mark.asyncio
async def test_plan_investment_evaluation_fake_llm(
    investment_evaluation_query: ParsedQuery,
) -> None:
    """Verify 'is it attractive' plans fundamental + valuation + news, then risk."""
    fake_spec = PlanSpec(
        tasks=[
            PlannedTaskSpec(
                task_id="task_fundamental_aapl",
                agent_type="fundamental",
                instruction=(
                    "Extract reported 10-K and 10-Q financial statements for AAPL "
                    "as of 2024-10-12. Evaluate revenue growth and margin trends."
                ),
                depends_on=[],
                priority=2,
            ),
            PlannedTaskSpec(
                task_id="task_valuation_aapl",
                agent_type="valuation",
                instruction=(
                    "Model bear, base, and bull DCF fair values and solve reverse "
                    "DCF for market-implied growth on AAPL as of 2024-10-12."
                ),
                depends_on=[],
                priority=2,
            ),
            PlannedTaskSpec(
                task_id="task_news_aapl",
                agent_type="news",
                instruction=(
                    "Review recent SEC filings and market news catalysts for AAPL "
                    "up to 2024-10-12 to assess growth drivers and developments."
                ),
                depends_on=[],
                priority=2,
            ),
            PlannedTaskSpec(
                task_id="task_risk_aapl",
                agent_type="risk",
                instruction=(
                    "Synthesize findings from fundamental, valuation, and news tasks. "
                    "Evaluate downside vulnerabilities and multiple contraction."
                ),
                depends_on=[
                    "task_fundamental_aapl",
                    "task_valuation_aapl",
                    "task_news_aapl",
                ],
                priority=1,
            ),
        ]
    )

    fake_llm = FakeLLM(
        responses=[
            LLMResponse(
                text="",
                structured_output=fake_spec,
                usage=Usage(input_tokens=220, output_tokens=140),
            )
        ]
    )

    plan = await plan_query(
        query=investment_evaluation_query,
        available_profiles=ALLOWED_PROFILES,
        llm=fake_llm,
    )

    assert len(plan.tasks) == 4
    agent_types = {t.agent_type for t in plan.tasks}
    assert agent_types == {"fundamental", "valuation", "news", "risk"}

    # Risk task must depend on the wave-1 tasks
    risk_task = plan["task_risk_aapl"]
    assert set(risk_task.depends_on) == {
        "task_fundamental_aapl",
        "task_valuation_aapl",
        "task_news_aapl",
    }


def test_validate_plan_empty_raises_error() -> None:
    """Verify empty plan raises EmptyPlanError."""
    empty_plan = Plan(tasks=[])
    with pytest.raises(EmptyPlanError, match="at least one task"):
        validate_plan(empty_plan, allowed_agent_types=ALLOWED_PROFILES)


def test_validate_plan_unknown_agent_type_raises_error() -> None:
    """Verify unknown agent type raises UnknownAgentTypeError."""
    bad_plan = Plan(
        tasks=[
            PlannedTask(
                task_id="task_crypto_aapl",
                agent_type="crypto_trader",
                instruction="Inspect on-chain token volume for AAPL.",
                depends_on=[],
            )
        ]
    )
    with pytest.raises(
        UnknownAgentTypeError, match="unknown agent type 'crypto_trader'"
    ):
        validate_plan(bad_plan, allowed_agent_types=ALLOWED_PROFILES)


def test_validate_plan_generic_instruction_raises_error() -> None:
    """Verify generic instructions like 'analyze AAPL' raise GenericInstructionError."""
    vague_plan = Plan(
        tasks=[
            PlannedTask(
                task_id="task_market_aapl",
                agent_type="market",
                instruction="analyze AAPL",
                depends_on=[],
            )
        ]
    )
    with pytest.raises(GenericInstructionError, match="generic instruction"):
        validate_plan(vague_plan, allowed_agent_types=ALLOWED_PROFILES)


def test_validate_plan_risk_without_dependencies_raises_error() -> None:
    """Verify risk agent with empty depends_on raises InvalidDependencyError."""
    bad_risk_plan = Plan(
        tasks=[
            PlannedTask(
                task_id="task_market_aapl",
                agent_type="market",
                instruction="Compute 1-year annualized volatility for AAPL.",
                depends_on=[],
            ),
            PlannedTask(
                task_id="task_risk_aapl",
                agent_type="risk",
                instruction="Stress-test downside scenarios for AAPL.",
                depends_on=[],  # Missing wave-1 dependency!
            ),
        ]
    )
    with pytest.raises(InvalidDependencyError, match="must depend on wave-1 agents"):
        validate_plan(bad_risk_plan, allowed_agent_types=ALLOWED_PROFILES)


def test_validate_plan_risk_depends_on_unknown_task_raises_error() -> None:
    """Verify risk agent with nonexistent dependency raises InvalidDependencyError."""
    bad_risk_plan = Plan(
        tasks=[
            PlannedTask(
                task_id="task_market_aapl",
                agent_type="market",
                instruction="Compute 1-year annualized volatility for AAPL.",
                depends_on=[],
            ),
            PlannedTask(
                task_id="task_risk_aapl",
                agent_type="risk",
                instruction="Stress-test downside scenarios for AAPL.",
                depends_on=["task_nonexistent_01"],
            ),
        ]
    )
    with pytest.raises(InvalidDependencyError, match="DAG validation failed"):
        validate_plan(bad_risk_plan, allowed_agent_types=ALLOWED_PROFILES)


@pytest.mark.live
@pytest.mark.asyncio
async def test_planner_live_move_explanation() -> None:
    """Live Gemini test: 'why did it drop today' plans news + market, no valuation."""
    if os.getenv("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE is not set to 1; skipping live planner test")
    settings = get_settings()
    if not settings.GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY is not configured; skipping live test")

    llm = GeminiClient(
        api_key=settings.GEMINI_API_KEY,
        model=settings.GEMINI_MODEL,
    )

    query = ParsedQuery(
        intent=QueryIntent.MOVE_EXPLANATION,
        tickers=["AAPL"],
        horizon="1d",
        as_of_date="2024-10-12",
        ambiguities=[],
        needs_clarification=False,
    )

    plan = await plan_query(
        query=query,
        available_profiles=ALLOWED_PROFILES,
        llm=llm,
    )

    agent_types = {t.agent_type for t in plan.tasks}
    # Must include news and market
    assert "news" in agent_types
    assert "market" in agent_types
    # Must NOT include valuation
    assert "valuation" not in agent_types

    # Ensure all instructions are concrete
    for t in plan.tasks:
        assert len(t.instruction.strip()) > 20


@pytest.mark.live
@pytest.mark.asyncio
async def test_planner_live_investment_evaluation() -> None:
    """Live Gemini: 'is it attractive' plans fundamental+valuation+news, then risk."""
    if os.getenv("RUN_LIVE") != "1":
        pytest.skip("RUN_LIVE is not set to 1; skipping live planner test")
    settings = get_settings()
    if not settings.GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY is not configured; skipping live test")

    llm = GeminiClient(
        api_key=settings.GEMINI_API_KEY,
        model=settings.GEMINI_MODEL,
    )

    query = ParsedQuery(
        intent=QueryIntent.INVESTMENT_EVALUATION,
        tickers=["AAPL"],
        horizon="3-5 years",
        risk_focus="downside",
        as_of_date="2024-10-12",
        ambiguities=[],
        needs_clarification=False,
    )

    plan = await plan_query(
        query=query,
        available_profiles=ALLOWED_PROFILES,
        llm=llm,
    )

    agent_types = {t.agent_type for t in plan.tasks}
    # Must include wave-1 agents
    assert "fundamental" in agent_types
    assert "valuation" in agent_types
    assert "news" in agent_types
    # Must include risk
    assert "risk" in agent_types

    # Risk must depend on upstream tasks
    risk_tasks = [t for t in plan.tasks if t.agent_type == "risk"]
    assert len(risk_tasks) >= 1
    for rt in risk_tasks:
        assert len(rt.depends_on) >= 1
