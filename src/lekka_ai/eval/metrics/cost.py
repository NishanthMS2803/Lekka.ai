"""Cost and operational metrics auditing tokens, latency, tools, and agents."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lekka_ai.eval.dataset import EvalExample
from lekka_ai.eval.variant import RunResult
from lekka_ai.schemas.claims import Claim


class CostBreakdown(BaseModel):
    """Resource consumption and efficiency breakdown for an execution run."""

    model_config = ConfigDict(extra="ignore")

    tool_calls: int = Field(
        default=0,
        description="Total tool call invocations executed",
    )
    tokens_in: int = Field(default=0, description="Prompt input token count")
    tokens_out: int = Field(default=0, description="Generated completion token count")
    tokens_total: int = Field(default=0, description="Total tokens consumed")
    latency_ms: float = Field(
        default=0.0,
        description="Execution wall-clock latency in ms",
    )
    usd_estimate: float = Field(
        default=0.0,
        description="Estimated monetary cost in USD",
    )
    unnecessary_agent_calls: int = Field(
        default=0,
        description="Count of dispatched agents whose claims were never cited",
    )
    dispatched_agents: list[str] = Field(
        default_factory=list,
        description="All agents dispatched during the research process",
    )
    used_agents: list[str] = Field(
        default_factory=list,
        description="Agents whose produced claims were cited in the final report",
    )
    unused_agents: list[str] = Field(
        default_factory=list,
        description="Agents whose produced claims were never cited in the final report",
    )


def extract_used_claim_ids(
    report_text_or_used: str | Sequence[str] | dict[str, Any] | None,
) -> set[str]:
    """Extract set of cited claim identifiers from report text or citation maps.

    Args:
        report_text_or_used: Markdown report text, citation mapping, or list of IDs.

    Returns:
        Set of normalized claim ID strings referenced in the report.
    """
    if not report_text_or_used:
        return set()

    if isinstance(report_text_or_used, (list, tuple, set)):
        return {str(cid).strip() for cid in report_text_or_used if str(cid).strip()}

    if isinstance(report_text_or_used, dict):
        cited: set[str] = set()
        for v in report_text_or_used.values():
            if isinstance(v, (list, tuple, set)):
                cited.update(str(x).strip() for x in v if str(x).strip())
            elif isinstance(v, str):
                cited.add(v.strip())
        return cited

    text = str(report_text_or_used)
    # Match bracketed citations e.g. [claim_1], [c_01], [claim_fundamental_10]
    raw_citations = re.findall(
        r"\[\s*([a-zA-Z0-9_\-]+(?:\s*,\s*[a-zA-Z0-9_\-]+)*)\s*\]",
        text,
    )
    extracted: set[str] = set()
    for match in raw_citations:
        for piece in match.split(","):
            clean = piece.strip()
            if clean and not clean.isdigit():
                extracted.add(clean)

    return extracted


def unnecessary_agent_calls(
    claims_or_result: Sequence[Claim | dict[str, Any]] | RunResult,
    report_text_or_claims_used: str | Sequence[str] | None = None,
    dispatched_agents: Sequence[str] | None = None,
) -> int:
    """Count dispatched agents whose produced claims were never used in the report.

    Args:
        claims_or_result: Ledger claims or RunResult execution instance.
        report_text_or_claims_used: Report markdown or explicit list of used claim IDs.
        dispatched_agents: Optional explicit list of dispatched agent names.

    Returns:
        Number of unnecessary agent calls (agents whose claims were never cited).
    """
    count, _, _ = unnecessary_agent_calls_details(
        claims_or_result=claims_or_result,
        report_text_or_claims_used=report_text_or_claims_used,
        dispatched_agents=dispatched_agents,
    )
    return count


def unnecessary_agent_calls_details(
    claims_or_result: Sequence[Claim | dict[str, Any]] | RunResult,
    report_text_or_claims_used: str | Sequence[str] | None = None,
    dispatched_agents: Sequence[str] | None = None,
) -> tuple[int, list[str], list[str]]:
    """Determine unnecessary agent calls with unused and used agent details.

    Returns:
        Tuple of (unnecessary_count, unused_agents_list, used_agents_list).
    """
    raw_claims: list[Any] = []
    resolved_report = report_text_or_claims_used
    explicit_dispatched = list(dispatched_agents) if dispatched_agents else []

    if isinstance(claims_or_result, RunResult):
        raw_claims = list(claims_or_result.claims or [])
        meta = claims_or_result.metadata or {}
        if not resolved_report:
            if meta.get("claims_used"):
                resolved_report = meta["claims_used"]
            elif meta.get("sentence_to_claims"):
                resolved_report = meta["sentence_to_claims"]
            else:
                resolved_report = claims_or_result.answer

        if not explicit_dispatched and meta.get("dispatched_agents"):
            explicit_dispatched = list(meta["dispatched_agents"])
    else:
        raw_claims = list(claims_or_result)

    used_claim_ids = extract_used_claim_ids(resolved_report)

    # Map claims to producing agents
    agent_claims: dict[str, set[str]] = {}
    for c in raw_claims:
        cid = getattr(c, "claim_id", None) or (
            c.get("claim_id") if isinstance(c, dict) else None
        )
        producer = getattr(c, "produced_by", None) or (
            c.get("produced_by") if isinstance(c, dict) else None
        )
        if producer:
            prod_str = str(producer).strip()
            if prod_str not in agent_claims:
                agent_claims[prod_str] = set()
            if cid:
                agent_claims[prod_str].add(str(cid).strip())

    # Form total dispatched agent roster
    all_dispatched: set[str] = set(explicit_dispatched) | set(agent_claims.keys())

    used_agents: set[str] = set()
    for agent, c_ids in agent_claims.items():
        if any(cid in used_claim_ids for cid in c_ids):
            used_agents.add(agent)

    # Agents dispatched that produced zero claims are also unnecessary
    unused_agents = sorted(all_dispatched - used_agents)
    return len(unused_agents), unused_agents, sorted(used_agents)


def tool_calls_metric(result: RunResult, example: EvalExample) -> int:
    """Measure the total number of tool calls executed."""
    return result.tool_calls


def tokens_total_metric(result: RunResult, example: EvalExample) -> int:
    """Measure total tokens consumed across the run."""
    return result.tokens_total


def tokens_in_metric(result: RunResult, example: EvalExample) -> int:
    """Measure prompt input tokens consumed."""
    return result.tokens_in


def tokens_out_metric(result: RunResult, example: EvalExample) -> int:
    """Measure completion output tokens generated."""
    return result.tokens_out


def latency_metric(result: RunResult, example: EvalExample) -> float:
    """Measure execution latency in milliseconds."""
    return round(result.latency_ms, 2)


def usd_estimate_metric(result: RunResult, example: EvalExample) -> float:
    """Measure estimated API cost in USD."""
    return round(result.cost, 6)


def unnecessary_agent_calls_metric(result: RunResult, example: EvalExample) -> int:
    """Evaluate unnecessary agent calls whose claims were not cited in final report."""
    return unnecessary_agent_calls(result)


def compute_cost_breakdown(result: RunResult) -> CostBreakdown:
    """Compute complete CostBreakdown model for an evaluated RunResult."""
    unnecessary, unused, used = unnecessary_agent_calls_details(result)
    all_dispatched = sorted(set(used) | set(unused))

    return CostBreakdown(
        tool_calls=result.tool_calls,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        tokens_total=result.tokens_total,
        latency_ms=round(result.latency_ms, 2),
        usd_estimate=round(result.cost, 6),
        unnecessary_agent_calls=unnecessary,
        dispatched_agents=all_dispatched,
        used_agents=used,
        unused_agents=unused,
    )
