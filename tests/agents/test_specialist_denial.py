"""Tests verifying each specialist agent is denied tools outside its profile."""

import pytest

from lekka_ai.agents import load_agent_profile
from lekka_ai.harness.models import AgentTask, Limits
from lekka_ai.harness.permissions import PermissionDenied, ToolView
from lekka_ai.harness.runtime import AgentRuntime
from lekka_ai.llm.base import LLMResponse, ToolCall, Usage
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.tools.base import ToolRegistry
from lekka_ai.tracing import Tracer


@pytest.mark.asyncio
async def test_market_specialist_denied_outside_tools(
    fixture_tool_registry: ToolRegistry,
) -> None:
    """Verify Market agent cannot access valuation, SEC, news, or sandbox tools."""
    profile = load_agent_profile("market")
    tool_view = ToolView(fixture_tool_registry, profile=profile)

    disallowed_tools = [
        "run_dcf",
        "run_reverse_dcf",
        "compute_multiples",
        "scenario_table",
        "run_python",
        "get_financials",
        "get_news",
        "list_filings",
        "search_filings",
        "get_analyst_estimates",
        "echo",
    ]

    exposed_specs = {s.name for s in tool_view.specs()}

    for tool_name in disallowed_tools:
        # 1. Profile authorization check
        assert not profile.is_tool_allowed(tool_name)

        # 2. Excluded from exposed LLM specs
        assert tool_name not in exposed_specs
        assert tool_name not in tool_view

        # 3. get() raises PermissionDenied
        with pytest.raises(
            PermissionDenied,
            match=f"not permitted to access tool '{tool_name}'",
        ):
            tool_view.get(tool_name)

        # 4. execute() raises PermissionDenied
        with pytest.raises(
            PermissionDenied,
            match=f"not permitted to call tool '{tool_name}'",
        ):
            await tool_view.execute(tool_name, {})


@pytest.mark.asyncio
async def test_valuation_specialist_denied_outside_tools(
    fixture_tool_registry: ToolRegistry,
) -> None:
    """Verify Valuation agent cannot access news, SEC filings, or risk stats."""
    profile = load_agent_profile("valuation")
    tool_view = ToolView(fixture_tool_registry, profile=profile)

    disallowed_tools = [
        "get_news",
        "list_filings",
        "search_filings",
        "compute_risk_stats",
        "echo",
    ]

    exposed_specs = {s.name for s in tool_view.specs()}

    for tool_name in disallowed_tools:
        assert not profile.is_tool_allowed(tool_name)
        assert tool_name not in exposed_specs
        assert tool_name not in tool_view

        with pytest.raises(
            PermissionDenied,
            match=f"not permitted to access tool '{tool_name}'",
        ):
            tool_view.get(tool_name)

        with pytest.raises(
            PermissionDenied,
            match=f"not permitted to call tool '{tool_name}'",
        ):
            await tool_view.execute(tool_name, {})


@pytest.mark.asyncio
async def test_news_specialist_denied_outside_tools(
    fixture_tool_registry: ToolRegistry,
) -> None:
    """Verify News agent cannot access quant modeling, Python, or market stats."""
    profile = load_agent_profile("news")
    tool_view = ToolView(fixture_tool_registry, profile=profile)

    disallowed_tools = [
        "run_dcf",
        "run_reverse_dcf",
        "compute_multiples",
        "scenario_table",
        "run_python",
        "compute_risk_stats",
        "get_market_data",
        "get_financials",
        "get_analyst_estimates",
        "echo",
    ]

    exposed_specs = {s.name for s in tool_view.specs()}

    for tool_name in disallowed_tools:
        assert not profile.is_tool_allowed(tool_name)
        assert tool_name not in exposed_specs
        assert tool_name not in tool_view

        with pytest.raises(
            PermissionDenied,
            match=f"not permitted to access tool '{tool_name}'",
        ):
            tool_view.get(tool_name)

        with pytest.raises(
            PermissionDenied,
            match=f"not permitted to call tool '{tool_name}'",
        ):
            await tool_view.execute(tool_name, {})


@pytest.mark.asyncio
async def test_analyst_agent_denied_sandbox_tool(
    fixture_tool_registry: ToolRegistry,
) -> None:
    """Verify baseline Analyst agent is strictly denied arbitrary sandbox execution."""
    profile = load_agent_profile("analyst")
    tool_view = ToolView(fixture_tool_registry, profile=profile)

    assert not profile.is_tool_allowed("run_python")
    assert "run_python" not in tool_view
    assert "run_python" not in {s.name for s in tool_view.specs()}

    with pytest.raises(
        PermissionDenied, match="not permitted to access tool 'run_python'"
    ):
        tool_view.get("run_python")

    with pytest.raises(
        PermissionDenied, match="not permitted to call tool 'run_python'"
    ):
        await tool_view.execute("run_python", {"code": "print('exploit')"})


@pytest.mark.asyncio
async def test_all_profiles_exhaustively_deny_unauthorized_tools(
    fixture_tool_registry: ToolRegistry,
) -> None:
    """Matrix test: every tool not in profile.allowed_tools raises PermissionDenied."""
    all_profiles = ["analyst", "market", "valuation", "news"]
    all_tools = fixture_tool_registry.list()

    for agent_name in all_profiles:
        profile = load_agent_profile(agent_name)
        tool_view = ToolView(fixture_tool_registry, profile=profile)

        for tool in all_tools:
            if tool.name in profile.allowed_tools:
                assert profile.is_tool_allowed(tool.name) is True
                assert tool.name in tool_view
            else:
                assert profile.is_tool_allowed(tool.name) is False
                assert tool.name not in tool_view
                with pytest.raises(
                    PermissionDenied,
                    match=f"not permitted to access tool '{tool.name}'",
                ):
                    tool_view.get(tool.name)
                with pytest.raises(
                    PermissionDenied,
                    match=f"not permitted to call tool '{tool.name}'",
                ):
                    await tool_view.execute(tool.name, {})


@pytest.mark.asyncio
async def test_runtime_blocks_unauthorized_tool_calls_from_llm(
    fixture_tool_registry: ToolRegistry,
) -> None:
    """Verify AgentRuntime intercepts unauthorized calls and returns error."""
    profile = load_agent_profile("market")
    tracer = Tracer()
    tool_view = ToolView(fixture_tool_registry, profile=profile, tracer=tracer)

    task = AgentTask(
        task_id="task_exploit_01",
        instruction="Attempt unauthorized tool calls outside market profile.",
        as_of_date="2024-09-20",
        agent_type="market",
    )

    # Step 1: LLM tries to call run_python and run_dcf (forbidden for market)
    step_1 = LLMResponse(
        text="Attempting to run Python code and DCF modeling.",
        tool_calls=[
            ToolCall(
                id="tc_bad_1",
                name="run_python",
                args={"code": "import os; print('forbidden')"},
            ),
            ToolCall(
                id="tc_bad_2",
                name="run_dcf",
                args={
                    "ticker": "AAPL",
                    "starting_fcf": 1000.0,
                    "fcf_growth_path": [0.05],
                    "discount_rate": 0.08,
                    "terminal_growth": 0.02,
                    "shares_outstanding": 10.0,
                    "net_debt": 0.0,
                },
            ),
        ],
        usage=Usage(input_tokens=80, output_tokens=30),
    )

    # Step 2: LLM receives permission errors, calls authorized submit_answer
    step_2 = LLMResponse(
        text="Recognized unauthorized tool errors, aborting unauthorized requests.",
        tool_calls=[
            ToolCall(
                id="tc_sub_2",
                name="submit_answer",
                args={
                    "task_id": "task_exploit_01",
                    "agent_type": "market",
                    "status": "insufficient_evidence",
                    "confidence": 0.10,
                    "confidence_rationale": "Encountered permission denials.",
                    "claims": [],
                    "extension": {
                        "trend_label": "NEUTRAL",
                        "key_stats": {
                            "annualized_volatility": 0.0,
                            "max_drawdown": 0.0,
                            "beta": 1.0,
                            "relative_strength": 1.0,
                        },
                    },
                },
            )
        ],
        usage=Usage(input_tokens=140, output_tokens=40),
    )

    fake_llm = FakeLLM(responses=[step_1, step_2])

    result = await AgentRuntime.run(
        task=task,
        profile=profile,
        llm=fake_llm,
        tool_view=tool_view,
        limits=Limits(max_iterations=4),
    )

    assert result.status == "insufficient_evidence"
    assert result.submission is not None
    assert result.submission.status == "insufficient_evidence"

    # Verify both unauthorized calls recorded PermissionDenied error spans
    error_spans = [span for span in tracer.spans if span.status == "error"]
    assert len(error_spans) == 2
    assert {s.name for s in error_spans} == {"run_python", "run_dcf"}
    for span in error_spans:
        assert "PermissionDenied" in (span.error or "")

    # Verify tool outputs delivered to LLM reflected PermissionDenied
    tool_messages = [m for m in result.messages if m.role == "tool"]
    assert len(tool_messages) >= 2
    assert any("PermissionDenied" in m.content for m in tool_messages)
