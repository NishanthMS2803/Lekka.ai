"""Deterministic confidence scoring methodology and report abstention path.

This module computes a deterministic confidence score (0.0 to 1.0) for a research run
based on six explicit rules:
1. Share of facts verified: Ratio of fact claims supported by evidence and free of
   contradictions to total fact claims (Weight: 0.25).
2. Number of unresolved conflicts: Penalty of 0.15 per active contradiction or
   unresolved question, capped at -0.45.
3. Evidence recency: Time elapsed between evidence cutoff and anchor date. Evidence
   <= 90 days old receives 1.0, <= 180 days receives 0.8, <= 365 days receives 0.5,
   and older evidence receives 0.2 (Weight: 0.15).
4. Share of background claims: Proportion of narrative background claims versus
   quantitative facts and calculations. Heavy reliance on background claims (>50%)
   applies a proportional penalty (Weight: 0.20).
5. Agent coverage: Breadth of specialist roles (market, fundamental, valuation,
   risk, news) contributing verified claims (Weight: 0.20).
6. Limits hit: Deductions for iteration, token, tool call, or timeout limits
   triggered by subagents, penalizing truncated execution by 0.10 per limit hit.

If the resulting score falls below a configurable threshold (default 0.60) or
required evidence is absent, the system executes the abstention path:
emitting an abstention report that states 'insufficient evidence', enumerates
missing components, and strictly omits all scenario numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lekka_ai.db.models import ClaimModel
from lekka_ai.llm.base import LLMClient
from lekka_ai.logging import get_logger
from lekka_ai.orchestrator.dispatch import TaskResult
from lekka_ai.orchestrator.state import RunState
from lekka_ai.report.synthesize import (
    SynthesizedReport,
    synthesize_report,
)
from lekka_ai.schemas.claims import Claim, ClaimType

logger = get_logger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.60

CORE_SPECIALIST_ROLES = {"market", "fundamental", "valuation", "risk", "news"}


class ConfidenceBreakdown(BaseModel):
    """Detailed deterministic confidence evaluation breakdown and factors."""

    model_config = ConfigDict(extra="ignore")

    confidence_score: float = Field(
        ...,
        description="Deterministic overall confidence score between 0.0 and 1.0",
    )
    threshold: float = Field(
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        description="Configurable minimum score threshold required to proceed",
    )
    is_abstention: bool = Field(
        ...,
        description="True if run fails threshold or required evidence is missing",
    )
    share_facts_verified: float = Field(
        default=0.0,
        description="Ratio of uncontradicted fact claims to total fact claims",
    )
    unresolved_conflicts_count: int = Field(
        default=0,
        description="Number of active claim contradictions or open conflicts",
    )
    evidence_recency_score: float = Field(
        default=0.0,
        description="Average recency score of cited evidence (0.0 to 1.0)",
    )
    share_background_claims: float = Field(
        default=0.0,
        description="Fraction of claims that are background/narrative only",
    )
    agent_coverage_score: float = Field(
        default=0.0,
        description="Fraction of core specialist roles contributing claims",
    )
    limits_hit_count: int = Field(
        default=0,
        description="Number of resource limits hit by subagents during execution",
    )
    missing_evidence: list[str] = Field(
        default_factory=list,
        description="Enumeration of required evidence categories missing from run",
    )
    factors_summary: dict[str, float] = Field(
        default_factory=dict,
        description="Itemized score contributions and deductions",
    )


def _parse_date(date_val: Any) -> date | None:
    """Safely parse date from date, datetime, or ISO string."""
    if date_val is None:
        return None
    if isinstance(date_val, date) and not isinstance(date_val, datetime):
        return date_val
    if isinstance(date_val, datetime):
        return date_val.date()
    try:
        clean = str(date_val).strip()[:10]
        return date.fromisoformat(clean)
    except (ValueError, TypeError):
        return None


def calculate_confidence(
    claims: Sequence[Claim | ClaimModel | dict[str, Any]],
    *,
    state: RunState | None = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    anchor_date: date | str | None = None,
    limits_hit: Sequence[str | dict[str, Any]] | None = None,
    open_questions: Sequence[str] | None = None,
    unresolved_conflicts: int | None = None,
) -> ConfidenceBreakdown:
    """Compute deterministic confidence score based on the 6 stated methodology rules.

    Rules and Weights:
    - share_facts_verified: Weight 0.25 (verified facts / max(1, total facts))
    - unresolved_conflicts: Penalty -0.15 per conflict (capped at -0.45)
    - evidence_recency: Weight 0.15 (recency bucket average 0.2 to 1.0)
    - share_background_claims: Weight 0.20 (penalizes high background claim share)
    - agent_coverage: Weight 0.20 (coverage across core agent profiles)
    - limits_hit: Penalty -0.10 per limit hit (capped at -0.30)
    - claim_confidence_mean: Weight 0.20 (mean confidence of claims)
    """
    total_claims = len(claims)
    missing_evidence: list[str] = []

    if total_claims == 0:
        missing_evidence.append("No claims were produced by research agents.")
        return ConfidenceBreakdown(
            confidence_score=0.0,
            threshold=threshold,
            is_abstention=True,
            missing_evidence=missing_evidence,
            factors_summary={"total_claims": 0.0},
        )

    # 1. Share of facts verified
    fact_claims: list[Any] = []
    verified_facts: list[Any] = []
    background_claims: list[Any] = []
    contradicted_claim_ids: set[str] = set()
    producers: set[str] = set()
    recency_scores: list[float] = []
    claim_confidences: list[float] = []

    # Resolve anchor date for evidence recency calculation
    anchor_d: date
    if anchor_date is not None:
        parsed_anchor = _parse_date(anchor_date)
        anchor_d = parsed_anchor or datetime.now(UTC).date()
    elif state is not None and getattr(state, "created_at", None):
        anchor_d = _parse_date(state.created_at) or datetime.now(UTC).date()
    else:
        anchor_d = datetime.now(UTC).date()

    for c in claims:
        # Extract fields
        cid = getattr(c, "claim_id", None) or (
            c.get("claim_id") if isinstance(c, dict) else "unknown"
        )
        ctype = getattr(c, "type", None) or (
            c.get("type") if isinstance(c, dict) else "fact"
        )
        ctype_str = str(ctype).lower()
        conf = float(
            getattr(c, "confidence", 1.0)
            if not isinstance(c, dict)
            else c.get("confidence", 1.0)
        )
        claim_confidences.append(conf)

        prod = getattr(c, "produced_by", None) or (
            c.get("produced_by") if isinstance(c, dict) else None
        )
        if prod:
            producers.add(str(prod).strip().lower())

        contr = getattr(c, "contradicted_by", None) or (
            c.get("contradicted_by") if isinstance(c, dict) else []
        )
        if contr:
            contradicted_claim_ids.add(str(cid).strip().lower())

        # Evidence evaluation
        ev_list: list[Any] = []
        if isinstance(c, dict):
            ev_list = list(c.get("evidence", []) or [])
        else:
            ev_list = list(getattr(c, "evidence", []) or [])

        # Evidence recency scoring
        if ev_list:
            for ev in ev_list:
                as_of = (
                    ev.get("as_of_date")
                    if isinstance(ev, dict)
                    else getattr(ev, "as_of_date", None)
                )
                retrieved = (
                    ev.get("retrieved_at")
                    if isinstance(ev, dict)
                    else getattr(ev, "retrieved_at", None)
                )
                ev_d = _parse_date(as_of) or _parse_date(retrieved)
                if ev_d:
                    days_old = max(0, (anchor_d - ev_d).days)
                    if days_old <= 90:
                        recency_scores.append(1.0)
                    elif days_old <= 180:
                        recency_scores.append(0.8)
                    elif days_old <= 365:
                        recency_scores.append(0.5)
                    else:
                        recency_scores.append(0.2)
                else:
                    recency_scores.append(0.4)

        if ctype_str in ("fact", str(ClaimType.FACT)):
            fact_claims.append(c)
            # Verified if has evidence, uncontradicted, and reasonable confidence
            if ev_list and not contr and conf >= 0.7:
                verified_facts.append(c)
        elif ctype_str in ("background", str(ClaimType.BACKGROUND)):
            background_claims.append(c)

    # 1. Share of facts verified
    if fact_claims:
        share_facts_verified = len(verified_facts) / len(fact_claims)
    else:
        share_facts_verified = 0.0
        missing_evidence.append(
            "Zero verifiable empirical fact claims found in the ledger."
        )

    # 2. Number of unresolved conflicts
    # Count contradicted claims + conflict questions in open_questions
    conflicts_count = len(contradicted_claim_ids)
    if unresolved_conflicts is not None:
        conflicts_count += max(0, unresolved_conflicts)
    all_open_questions: list[str] = []
    if open_questions:
        all_open_questions.extend(open_questions)
    if state is not None and getattr(state, "open_questions", None):
        all_open_questions.extend(state.open_questions)

    for q in all_open_questions:
        if "conflict" in q.lower() or "reconcile" in q.lower():
            conflicts_count += 1

    if conflicts_count > 0:
        missing_evidence.append(
            f"Detected {conflicts_count} unresolved contradictory metric claims."
        )

    # 3. Evidence recency
    if recency_scores:
        evidence_recency_score = sum(recency_scores) / len(recency_scores)
    else:
        evidence_recency_score = 0.2
        missing_evidence.append("No dated empirical evidence was available.")

    # 4. Share of background claims
    share_background = len(background_claims) / max(1, total_claims)
    if share_background > 0.6:
        missing_evidence.append(
            f"Excessive reliance on ungrounded background narrative "
            f"({share_background * 100:.0f}% background claims)."
        )

    # 5. Agent coverage
    # Check producer roles in claims or task results
    if state is not None and getattr(state, "results", None):
        for res in state.results.values():
            if isinstance(res, TaskResult):
                producers.add(res.agent_type.lower())
            elif isinstance(res, dict) and "agent_type" in res:
                producers.add(str(res["agent_type"]).lower())

    covered_roles = producers.intersection(CORE_SPECIALIST_ROLES)
    agent_coverage_score = len(covered_roles) / len(CORE_SPECIALIST_ROLES)
    missing_roles = sorted(list(CORE_SPECIALIST_ROLES - covered_roles))
    if missing_roles:
        missing_evidence.append(
            f"Incomplete specialist agent coverage "
            f"(missing: {', '.join(missing_roles)})."
        )

    # 6. Limits hit
    all_limits: list[Any] = []
    if limits_hit:
        all_limits.extend(limits_hit)
    if state is not None and getattr(state, "results", None):
        for res in state.results.values():
            lh = (
                getattr(res, "limits_hit", None)
                if not isinstance(res, dict)
                else res.get("limits_hit")
            )
            if lh:
                if isinstance(lh, list):
                    all_limits.extend(lh)
                else:
                    all_limits.append(lh)

    limits_hit_count = len(all_limits)
    if limits_hit_count > 0:
        missing_evidence.append(
            f"Subagent execution hit {limits_hit_count} resource limits prematurely."
        )

    # Check required evidence presence
    has_valuation_or_financials = any(
        "valuation" in p or "fundamental" in p or "financial" in p for p in producers
    )
    if not has_valuation_or_financials:
        missing_evidence.append(
            "Essential fundamental statements and valuation models are missing."
        )

    # ----------------------------------------------------
    # Deterministic Scoring Formula:
    # ----------------------------------------------------
    mean_confidence = (
        sum(claim_confidences) / len(claim_confidences) if claim_confidences else 0.5
    )

    base_score = (
        0.25 * share_facts_verified
        + 0.20 * agent_coverage_score
        + 0.20 * max(0.0, 1.0 - (share_background * 1.5))
        + 0.15 * evidence_recency_score
        + 0.20 * mean_confidence
    )

    # Deductions
    conflict_penalty = min(0.45, conflicts_count * 0.15)
    limits_penalty = min(0.30, limits_hit_count * 0.10)
    total_penalties = conflict_penalty + limits_penalty

    raw_score = base_score - total_penalties
    final_score = round(max(0.0, min(1.0, raw_score)), 3)

    # Determine abstention
    # Low score OR critical evidence completely absent
    is_abstention = (
        final_score < threshold or total_claims < 2 or len(verified_facts) == 0
    )

    factors_summary: dict[str, float] = {
        "share_facts_verified_contrib": round(0.25 * share_facts_verified, 3),
        "agent_coverage_contrib": round(0.20 * agent_coverage_score, 3),
        "narrative_quality_contrib": round(
            0.20 * max(0.0, 1.0 - (share_background * 1.5)), 3
        ),
        "evidence_recency_contrib": round(0.15 * evidence_recency_score, 3),
        "mean_confidence_contrib": round(0.20 * mean_confidence, 3),
        "conflict_penalty": round(-conflict_penalty, 3),
        "limits_penalty": round(-limits_penalty, 3),
        "raw_score": round(raw_score, 3),
        "final_confidence_score": final_score,
    }

    return ConfidenceBreakdown(
        confidence_score=final_score,
        threshold=threshold,
        is_abstention=is_abstention,
        share_facts_verified=round(share_facts_verified, 3),
        unresolved_conflicts_count=conflicts_count,
        evidence_recency_score=round(evidence_recency_score, 3),
        share_background_claims=round(share_background, 3),
        agent_coverage_score=round(agent_coverage_score, 3),
        limits_hit_count=limits_hit_count,
        missing_evidence=missing_evidence,
        factors_summary=factors_summary,
    )


def generate_abstention_report(
    ticker: str | None,
    breakdown: ConfidenceBreakdown,
    *,
    question: str | None = None,
) -> str:
    """Generate an abstention report with NO scenario numbers.

    Strict Requirements:
    1. Must explicitly state 'insufficient evidence'.
    2. Must list what evidence is missing or deficient.
    3. Must give NO scenario numbers (no price targets, DCF fair values,
       or bull/bear multiples).
    """
    target = (ticker or "Target Company").upper()
    mandate = question or "Comprehensive Equity Research Analysis"

    missing_bullets = "\n".join(f"- {item}" for item in breakdown.missing_evidence)
    if not missing_bullets:
        missing_bullets = (
            "- Overall confidence score fell below acceptable threshold.\n"
            "- Available empirical evidence is insufficient for thesis formulation."
        )

    # Note: Deliberately zero scenario numbers or valuations in the text
    report = (
        f"# Investment Research Report: {target} (Abstained)\n\n"
        "## Executive Summary: Insufficient Evidence\n"
        "The research system abstains from issuing an investment conclusion "
        "due to insufficient evidence.\n"
        f"The computed confidence score is {breakdown.confidence_score:.2f}, "
        f"which is below the required threshold of {breakdown.threshold:.2f}.\n"
        "Under system governance policies, no price targets or valuation scenario "
        "numbers are published when evidence standards are not met.\n\n"
        "## Research Mandate\n"
        f"{mandate}\n\n"
        "## Methodology Assessment\n"
        f"- Share of Verified Facts: {breakdown.share_facts_verified * 100:.1f}%\n"
        f"- Unresolved Conflicts: {breakdown.unresolved_conflicts_count}\n"
        f"- Evidence Recency Score: {breakdown.evidence_recency_score:.2f} / 1.00\n"
        f"- Background Share: {breakdown.share_background_claims * 100:.1f}%\n"
        f"- Specialist Agent Coverage: {breakdown.agent_coverage_score * 100:.1f}%\n"
        f"- Limits Hit Count: {breakdown.limits_hit_count}\n\n"
        "## Missing Evidence and Deficiencies\n"
        "The following critical deficiencies prevent thesis synthesis:\n"
        f"{missing_bullets}\n\n"
        "## Valuation Scenarios\n"
        "No valuation scenarios or target prices can be derived due to "
        "insufficient evidence.\n"
        "Scenario numbers (DCF intrinsic value, reverse DCF growth implied, "
        "and scenario tables) are withheld to prevent ungrounded speculation.\n\n"
        "## Bull Case\n"
        "The bull case cannot be formulated without verified growth data.\n\n"
        "## Bear Case\n"
        "The bear case cannot be formulated without verified risk and cost data.\n\n"
        "## Ranked Risks\n"
        "The primary risk is severe informational deficiency and unverified data.\n\n"
        "## Catalysts\n"
        "No actionable catalysts can be confirmed.\n\n"
        "## What Would Invalidate the Thesis\n"
        "Resolution of all missing evidence items and reconciliation of "
        "contradictory claims.\n\n"
        "## Uncertainty and Gaps\n"
        "Comprehensive empirical data collection across all specialist disciplines "
        "is required before a grounded report can be produced."
    )
    return report.strip()


async def synthesize_report_with_confidence(
    claims: Sequence[Claim | ClaimModel | dict[str, Any]],
    *,
    ticker: str | None = None,
    question: str | None = None,
    state: RunState | None = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    llm_client: LLMClient | None = None,
    llm: LLMClient | None = None,
    model_name: str | None = None,
    max_retries: int = 2,
    anchor_date: date | str | None = None,
) -> tuple[SynthesizedReport, ConfidenceBreakdown]:
    """Evaluate confidence and synthesize standard report or emit abstention report.

    If confidence is below threshold or evidence is missing, emits an abstention
    report stating 'insufficient evidence' and giving no scenario numbers.
    """
    effective_llm = llm_client or llm
    breakdown = calculate_confidence(
        claims,
        state=state,
        threshold=threshold,
        anchor_date=anchor_date,
    )

    if breakdown.is_abstention:
        logger.warning(
            "Confidence score below threshold; executing abstention path",
            confidence_score=breakdown.confidence_score,
            threshold=threshold,
            missing=breakdown.missing_evidence,
        )
        abstention_md = generate_abstention_report(
            ticker=ticker,
            breakdown=breakdown,
            question=question,
        )
        report = SynthesizedReport(
            markdown=abstention_md,
            sentence_to_claims={},
            claims_used=[],
            validation_passed=True,
            attempts=1,
            rejection_reasons=[],
            ticker=ticker,
        )
        return report, breakdown

    # High confidence path: execute standard synthesis
    report = await synthesize_report(
        claims=claims,
        ticker=ticker,
        question=question,
        llm_client=effective_llm,
        model_name=model_name,
        max_retries=max_retries,
    )
    return report, breakdown


# Alias for backward compatibility and test suites
calculate_confidence_score = calculate_confidence
