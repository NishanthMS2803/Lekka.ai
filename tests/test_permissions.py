"""Unit tests for AgentProfile, ToolView, permissions enforcement, and audit tracing."""

import io
from typing import Any

import pytest
from pydantic import BaseModel, Field

from lekka_ai.harness import (
    AgentProfile,
    PermissionDenied,
    ToolView,
)
from lekka_ai.logging import configure_logging
from lekka_ai.tools.base import Tool, ToolRegistry
from lekka_ai.tracing import Tracer


class MockInput(BaseModel):
    """Input payload for test tools."""

    message: str = Field(..., description="Message string")


class MockOutput(BaseModel):
    """Output payload for test tools."""

    result: str = Field(..., description="Processed result")


class MockTool(Tool[MockInput, MockOutput]):
    """Simple test tool implementation."""

    input_model: type[MockInput] = MockInput
    output_model: type[MockOutput] = MockOutput

    def __init__(self, name: str, description: str = "Test tool") -> None:
        self.name = name
        self.description = description

    async def run(self, input_data: MockInput) -> MockOutput:
        return MockOutput(result=f"processed:{input_data.message}")


@pytest.fixture
def sample_registry() -> ToolRegistry:
    """Fixture providing a ToolRegistry populated with three tools."""
    registry = ToolRegistry()
    registry.register(MockTool("fetch_sec", "Fetches SEC filing sections"))
    registry.register(MockTool("run_dcf", "Performs DCF calculations"))
    registry.register(
        MockTool("admin_shell", "Executes sensitive administrative tasks")
    )
    return registry


def test_agent_profile_matching(tmp_path: Any) -> None:
    """Test AgentProfile matching rules and system prompt loading."""
    prompt_file = tmp_path / "analyst_prompt.md"
    prompt_file.write_text(
        "You are a professional equity research analyst.", encoding="utf-8"
    )

    profile = AgentProfile(
        agent_type="equity_analyst",
        allowed_tools=["fetch_sec", "run_dcf"],
        system_prompt_path=str(prompt_file),
    )
    assert profile.agent_type == "equity_analyst"
    assert profile.is_tool_allowed("fetch_sec")
    assert profile.is_tool_allowed("run_dcf")
    assert not profile.is_tool_allowed("admin_shell")
    assert (
        profile.load_system_prompt()
        == "You are a professional equity research analyst."
    )

    # Wildcard profile
    super_profile = AgentProfile(agent_type="superuser", allowed_tools=["*"])
    assert super_profile.is_tool_allowed("admin_shell")
    assert super_profile.is_tool_allowed("any_arbitrary_tool")


def test_tool_view_specs_only_exposes_allowed(sample_registry: ToolRegistry) -> None:
    """Verify ToolView.specs() strictly returns ToolSpecs for permitted tools."""
    profile = AgentProfile(
        agent_type="junior_analyst",
        allowed_tools=["fetch_sec"],
    )
    tool_view = ToolView(sample_registry, profile)

    specs = tool_view.specs()
    assert len(specs) == 1
    assert specs[0].name == "fetch_sec"

    # Membership and listing check
    assert "fetch_sec" in tool_view
    assert "run_dcf" not in tool_view
    assert "admin_shell" not in tool_view

    allowed_tools = tool_view.list()
    assert len(allowed_tools) == 1
    assert allowed_tools[0].name == "fetch_sec"


@pytest.mark.asyncio
async def test_tool_view_allows_authorized_execution(
    sample_registry: ToolRegistry,
) -> None:
    """Verify authorized tool execution succeeds and records success span in trace."""
    tracer = Tracer()
    profile = AgentProfile(
        agent_type="financial_analyst",
        allowed_tools=["fetch_sec", "run_dcf"],
    )
    tool_view = ToolView(sample_registry, profile, tracer=tracer)

    # 1. Direct execute via ToolView
    result = await tool_view.execute("fetch_sec", {"message": "10-K FY2023"})
    assert result.result == "processed:10-K FY2023"

    # 2. Execution via permitted tool wrapper
    dcf_tool = tool_view.get("run_dcf")
    dcf_result = await dcf_tool({"message": "discount_rate=0.08"})
    assert dcf_result.result == "processed:discount_rate=0.08"

    # Check trace spans
    assert len(tracer.spans) == 2
    assert all(span.status == "ok" for span in tracer.spans)
    assert [span.name for span in tracer.spans] == ["fetch_sec", "run_dcf"]


@pytest.mark.asyncio
async def test_tool_view_denies_unauthorized_call_and_logs(
    sample_registry: ToolRegistry,
) -> None:
    """Verify disallowed tool calls raise PermissionDenied and log errors."""
    log_stream = io.StringIO()
    configure_logging(env="development", stream=log_stream)

    tracer = Tracer()
    profile = AgentProfile(
        agent_type="restricted_analyst",
        allowed_tools=["fetch_sec"],
    )
    tool_view = ToolView(sample_registry, profile, tracer=tracer)

    # 1. Deny on get()
    with pytest.raises(
        PermissionDenied, match="not permitted to access tool 'admin_shell'"
    ):
        tool_view.get("admin_shell")

    log_output = log_stream.getvalue().lower()
    assert "permission denied" in log_output
    assert "admin_shell" in log_output

    assert len(tracer.spans) == 1
    assert tracer.spans[0].status == "error"
    assert tracer.spans[0].tool_name == "admin_shell"
    assert "PermissionDenied" in (tracer.spans[0].error or "")

    # 2. Deny on execute()
    with pytest.raises(PermissionDenied, match="not permitted to call tool 'run_dcf'"):
        await tool_view.execute("run_dcf", {"message": "unauthorized calc"})

    log_output_after = log_stream.getvalue().lower()
    assert "run_dcf" in log_output_after

    assert len(tracer.spans) == 2
    denied_span = tracer.spans[1]
    assert denied_span.name == "run_dcf"
    assert denied_span.status == "error"
    assert "PermissionDenied" in (denied_span.error or "")
