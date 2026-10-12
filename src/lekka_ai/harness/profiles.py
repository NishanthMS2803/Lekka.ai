"""Standard specialist agent profiles and tool authorization configurations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from lekka_ai.harness.permissions import AgentProfile
from lekka_ai.tools.base import default_registry

if TYPE_CHECKING:
    from lekka_ai.tools.base import ToolRegistry

ANALYST_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "analyst.md"

DEFAULT_ANALYST_TOOLS: list[str] = [
    "echo",
    "get_analyst_estimates",
    "get_industry_data",
    "list_filings",
    "get_financials",
    "get_market_data",
    "get_news",
    "run_dcf",
    "run_reverse_dcf",
    "compute_multiples",
    "scenario_table",
    "compute_risk_stats",
    "search_filings",
    "submit_answer",
]


def get_analyst_tools(registry: ToolRegistry | None = None) -> list[str]:
    """Retrieve all non-sandbox tool names from the registry.

    Excludes 'run_python' to prevent untrusted code execution by the analyst.
    """
    reg = registry or default_registry
    tool_names = [t.name for t in reg.list() if t.name != "run_python"]
    if not tool_names:
        return list(DEFAULT_ANALYST_TOOLS)
    return tool_names


def create_analyst_profile(
    registry: ToolRegistry | None = None,
    system_prompt_path: str | Path | None = None,
) -> AgentProfile:
    """Create a generic 'analyst' AgentProfile authorized for all non-sandbox tools."""
    prompt_path = (
        Path(system_prompt_path) if system_prompt_path else ANALYST_PROMPT_PATH
    )
    return AgentProfile(
        agent_type="analyst",
        allowed_tools=get_analyst_tools(registry),
        system_prompt_path=str(prompt_path),
    )


# Standard singleton analyst profile
ANALYST_PROFILE: AgentProfile = create_analyst_profile()
