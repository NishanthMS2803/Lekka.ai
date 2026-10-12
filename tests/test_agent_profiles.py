"""Unit tests for config-driven agent profiles loader and validation."""

from pathlib import Path
from typing import Any

import pytest

from lekka_ai.agents import (
    load_agent_profile,
    load_profiles,
    load_profiles_from_dict,
    validate_profile_tools,
)
from lekka_ai.harness.models import Limits
from lekka_ai.harness.permissions import AgentProfile
from lekka_ai.tools.base import ToolRegistry


def test_load_profiles_from_repo_config() -> None:
    """Verify loading real agents/profiles.yaml builds analyst profile correctly."""
    profiles = load_profiles()
    assert "analyst" in profiles

    analyst = profiles["analyst"]
    assert analyst.agent_type == "analyst"
    assert "get_market_data" in analyst.allowed_tools
    assert "submit_answer" in analyst.allowed_tools
    assert "run_python" not in analyst.allowed_tools

    # Limits validation
    assert isinstance(analyst.default_limits, Limits)
    assert analyst.default_limits.max_iterations == 10
    assert analyst.default_limits.max_tokens == 150000
    assert analyst.default_limits.max_repair_attempts == 3

    # Prompt loading
    prompt_content = analyst.load_system_prompt()
    assert prompt_content is not None
    assert "Financial Research Analyst System Prompt" in prompt_content


def test_load_agent_profile_direct() -> None:
    """Verify load_agent_profile retrieves specific agent profile."""
    profile = load_agent_profile("analyst")
    assert profile.agent_type == "analyst"
    assert profile.is_tool_allowed("get_financials")
    assert not profile.is_tool_allowed("run_python")


def test_unknown_tool_fails_validation() -> None:
    """Verify that referencing an unknown tool fails validation with ValueError."""
    bad_config: dict[str, Any] = {
        "bad_agent": {
            "allowed_tools": ["echo", "fictional_unregistered_tool"],
            "prompt_file": "agents/prompts/analyst.md",
        }
    }
    with pytest.raises(ValueError, match="Unknown tool 'fictional_unregistered_tool'"):
        load_profiles_from_dict(bad_config, validate_tools=True)


def test_validate_profile_tools_with_custom_registry() -> None:
    """Verify tool validation strictly checks against provided ToolRegistry."""
    empty_registry = ToolRegistry()
    profile = AgentProfile(
        agent_type="strict_agent",
        allowed_tools=["echo"],
    )
    with pytest.raises(ValueError, match="Unknown tool 'echo'"):
        validate_profile_tools(profile, registry=empty_registry)


def test_wildcard_tool_passes_validation() -> None:
    """Verify wildcard tool passes validation."""
    profile = AgentProfile(
        agent_type="superuser",
        allowed_tools=["*"],
    )
    # Should not raise
    validate_profile_tools(profile)
    assert profile.is_tool_allowed("any_tool")


def test_custom_yaml_file_loading(tmp_path: Path) -> None:
    """Verify loading from an explicit custom YAML path."""
    yaml_path = tmp_path / "custom_profiles.yaml"
    yaml_path.write_text(
        """
custom_analyst:
  allowed_tools:
    - echo
    - submit_answer
  prompt_file: null
  default_limits:
    max_iterations: 5
    max_tokens: 50000
""",
        encoding="utf-8",
    )

    profiles = load_profiles(path=yaml_path)
    assert "custom_analyst" in profiles
    custom = profiles["custom_analyst"]
    assert custom.allowed_tools == ["echo", "submit_answer"]
    assert custom.default_limits.max_iterations == 5


def test_nonexistent_profile_raises_key_error() -> None:
    """Verify load_agent_profile with unknown agent_type raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        load_agent_profile("non_existent_role_name")


def test_nonexistent_yaml_file_raises_file_not_found() -> None:
    """Verify invalid path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_profiles(path="/path/that/does/not/exist_at_all.yaml")
