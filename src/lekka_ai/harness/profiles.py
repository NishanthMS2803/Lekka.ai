"""Standard specialist agent profiles, tool authorization, and config loader."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel

from lekka_ai.harness.permissions import AgentProfile
from lekka_ai.schemas.submit import register_answer_extension
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


def resolve_extension_model(schema_path: str | None) -> type[BaseModel] | None:
    """Resolve an extension model from a dotted import path or registered name."""
    if not schema_path:
        return None
    if ":" in schema_path:
        mod_name, cls_name = schema_path.split(":", 1)
    elif "." in schema_path:
        mod_name, cls_name = schema_path.rsplit(".", 1)
    else:
        from lekka_ai.schemas.submit import get_answer_extension

        registered = get_answer_extension(schema_path)
        if registered is not None:
            return registered
        raise ValueError(
            f"Unable to resolve answer extension schema '{schema_path}'. "
            "Expected dotted import path or registered name."
        )

    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise TypeError(
            f"Resolved answer extension '{schema_path}' must be a BaseModel subclass."
        )
    return cls


def validate_profile_tools(
    profile: AgentProfile,
    registry: ToolRegistry | None = None,
) -> None:
    """Validate that all tools referenced in profile.allowed_tools exist."""
    import lekka_ai.tools  # noqa: F401  Ensure built-in tools are registered

    reg = registry or default_registry
    for tool_name in profile.allowed_tools:
        clean = tool_name.strip()
        if clean == "*":
            continue
        if clean not in reg:
            raise ValueError(
                f"Unknown tool '{clean}' referenced in profile '{profile.agent_type}'. "
                "Tool does not exist in registry."
            )


def load_profiles_from_dict(
    data: dict[str, Any],
    registry: ToolRegistry | None = None,
    validate_tools: bool = True,
) -> dict[str, AgentProfile]:
    """Build AgentProfile objects from a configuration mapping dictionary."""
    profiles: dict[str, AgentProfile] = {}
    for agent_type, cfg in data.items():
        if not isinstance(cfg, dict):
            continue
        profile_data = dict(cfg)
        profile_data["agent_type"] = agent_type
        if "limits" in profile_data and "default_limits" not in profile_data:
            profile_data["default_limits"] = profile_data.pop("limits")

        profile = AgentProfile.model_validate(profile_data)

        if validate_tools:
            validate_profile_tools(profile, registry=registry)

        if profile.answer_extension_schema:
            ext_model = resolve_extension_model(profile.answer_extension_schema)
            if ext_model is not None:
                register_answer_extension(profile.agent_type, ext_model)

        profiles[agent_type] = profile

    return profiles


def find_profiles_yaml(custom_path: str | Path | None = None) -> Path:
    """Locate profiles.yaml in explicitly provided path, cwd, or repository root."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(
            f"Profiles configuration file not found at: {custom_path}"
        )

    # 1. Current working directory
    cwd_candidate = Path.cwd() / "agents" / "profiles.yaml"
    if cwd_candidate.is_file():
        return cwd_candidate.resolve()

    # 2. Package repo root
    pkg_root = Path(__file__).resolve().parent.parent
    repo_root = pkg_root.parent.parent
    repo_candidate = repo_root / "agents" / "profiles.yaml"
    if repo_candidate.is_file():
        return repo_candidate.resolve()

    # 3. Direct agents in current dir
    direct_cand = Path.cwd() / "profiles.yaml"
    if direct_cand.is_file():
        return direct_cand.resolve()

    raise FileNotFoundError(
        "Could not locate agents/profiles.yaml in current directory or repository root."
    )


def load_profiles(
    path: str | Path | None = None,
    registry: ToolRegistry | None = None,
    validate_tools: bool = True,
) -> dict[str, AgentProfile]:
    """Load agent profiles from profiles.yaml and validate tools."""
    config_path = find_profiles_yaml(path)
    content = config_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(content) or {}
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Invalid YAML content in {config_path}: expected mapping of agent types."
        )
    return load_profiles_from_dict(
        parsed,
        registry=registry,
        validate_tools=validate_tools,
    )


def load_agent_profile(
    agent_type: str,
    path: str | Path | None = None,
    registry: ToolRegistry | None = None,
) -> AgentProfile:
    """Load a specific AgentProfile by agent_type identifier."""
    profiles = load_profiles(path=path, registry=registry)
    if agent_type not in profiles:
        raise KeyError(
            f"Agent profile '{agent_type}' not found in configuration. "
            f"Available profiles: {list(profiles.keys())}"
        )
    return profiles[agent_type]


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
    """Create or load the standard 'analyst' AgentProfile."""
    try:
        return load_agent_profile("analyst", registry=registry)
    except Exception:
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

DEFAULT_MARKET_TOOLS: list[str] = [
    "get_market_data",
    "compute_risk_stats",
    "submit_answer",
]


def create_market_profile(
    registry: ToolRegistry | None = None,
    system_prompt_path: str | Path | None = None,
) -> AgentProfile:
    """Create or load the standard 'market' AgentProfile."""
    try:
        return load_agent_profile("market", registry=registry)
    except Exception:
        market_prompt_cand = (
            Path(system_prompt_path)
            if system_prompt_path
            else Path("agents/prompts/market.md")
        )
        return AgentProfile(
            agent_type="market",
            allowed_tools=list(DEFAULT_MARKET_TOOLS),
            system_prompt_path=str(market_prompt_cand),
            answer_extension_schema="lekka_ai.schemas.market:MarketAnswerExtension",
        )


MARKET_PROFILE: AgentProfile = create_market_profile()

DEFAULT_VALUATION_TOOLS: list[str] = [
    "get_market_data",
    "get_financials",
    "get_analyst_estimates",
    "run_dcf",
    "run_reverse_dcf",
    "compute_multiples",
    "scenario_table",
    "run_python",
    "submit_answer",
]


def create_valuation_profile(
    registry: ToolRegistry | None = None,
    system_prompt_path: str | Path | None = None,
) -> AgentProfile:
    """Create or load the standard 'valuation' AgentProfile."""
    try:
        return load_agent_profile("valuation", registry=registry)
    except Exception:
        valuation_prompt_cand = (
            Path(system_prompt_path)
            if system_prompt_path
            else Path("agents/prompts/valuation.md")
        )
        return AgentProfile(
            agent_type="valuation",
            allowed_tools=list(DEFAULT_VALUATION_TOOLS),
            system_prompt_path=str(valuation_prompt_cand),
            answer_extension_schema="lekka_ai.schemas.valuation:ValuationAnswerExtension",
        )


VALUATION_PROFILE: AgentProfile = create_valuation_profile()

DEFAULT_NEWS_TOOLS: list[str] = [
    "get_news",
    "list_filings",
    "search_filings",
    "submit_answer",
]


def create_news_profile(
    registry: ToolRegistry | None = None,
    system_prompt_path: str | Path | None = None,
) -> AgentProfile:
    """Create or load the standard 'news' AgentProfile."""
    try:
        return load_agent_profile("news", registry=registry)
    except Exception:
        news_prompt_cand = (
            Path(system_prompt_path)
            if system_prompt_path
            else Path("agents/prompts/news.md")
        )
        return AgentProfile(
            agent_type="news",
            allowed_tools=list(DEFAULT_NEWS_TOOLS),
            system_prompt_path=str(news_prompt_cand),
            answer_extension_schema="lekka_ai.schemas.news:NewsAnswerExtension",
        )


NEWS_PROFILE: AgentProfile = create_news_profile()

DEFAULT_FUNDAMENTAL_TOOLS: list[str] = [
    "get_financials",
    "list_filings",
    "search_filings",
    "get_analyst_estimates",
    "get_industry_data",
    "submit_answer",
]


def create_fundamental_profile(
    registry: ToolRegistry | None = None,
    system_prompt_path: str | Path | None = None,
) -> AgentProfile:
    """Create or load the standard 'fundamental' AgentProfile."""
    try:
        return load_agent_profile("fundamental", registry=registry)
    except Exception:
        fund_prompt_cand = (
            Path(system_prompt_path)
            if system_prompt_path
            else Path("agents/prompts/fundamental.md")
        )
        return AgentProfile(
            agent_type="fundamental",
            allowed_tools=list(DEFAULT_FUNDAMENTAL_TOOLS),
            system_prompt_path=str(fund_prompt_cand),
        )


FUNDAMENTAL_PROFILE: AgentProfile = create_fundamental_profile()

DEFAULT_RISK_TOOLS: list[str] = [
    "compute_risk_stats",
    "get_market_data",
    "submit_answer",
]


def create_risk_profile(
    registry: ToolRegistry | None = None,
    system_prompt_path: str | Path | None = None,
) -> AgentProfile:
    """Create or load the standard 'risk' AgentProfile."""
    try:
        return load_agent_profile("risk", registry=registry)
    except Exception:
        risk_prompt_cand = (
            Path(system_prompt_path)
            if system_prompt_path
            else Path("agents/prompts/risk.md")
        )
        return AgentProfile(
            agent_type="risk",
            allowed_tools=list(DEFAULT_RISK_TOOLS),
            system_prompt_path=str(risk_prompt_cand),
        )


RISK_PROFILE: AgentProfile = create_risk_profile()
