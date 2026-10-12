"""Grounding evaluation metrics auditing citations, evidence support, and validity."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lekka_ai.eval.dataset import EvalExample
from lekka_ai.eval.variant import RunResult
from lekka_ai.schemas.claims import Claim, ClaimType
from lekka_ai.verify.checks import (
    check_background_not_sole_support,
    check_interpretation_supporting_ids,
)
from lekka_ai.verify.support import ClaimSupportCheck, SupportVerdict


def _extract_str_value(obj: Any, default: str = "") -> str:
    """Safely extract string value from enum or object."""
    if obj is None:
        return default
    val = getattr(obj, "value", None)
    if val is not None:
        return str(val).strip().lower()
    return str(obj).strip().lower()


def _extract_claims_and_metadata(
    claims_or_result: Sequence[Claim | dict[str, Any]] | RunResult,
) -> tuple[list[Any], dict[str, Any]]:
    """Extract list of claims and execution metadata from input."""
    if isinstance(claims_or_result, RunResult):
        raw_claims = list(claims_or_result.claims or [])
        metadata = dict(getattr(claims_or_result, "metadata", {}) or {})
        return raw_claims, metadata
    return list(claims_or_result), {}


def _normalize_verdict_map(
    verdicts: (
        Sequence[ClaimSupportCheck | SupportVerdict | dict[str, Any]]
        | dict[str, Any]
        | None
    ),
) -> dict[str, str]:
    """Normalize input verdicts into a standard claim_id -> verdict_string map."""
    vmap: dict[str, str] = {}
    if not verdicts:
        return vmap

    if isinstance(verdicts, dict):
        for cid, val in verdicts.items():
            if isinstance(val, ClaimSupportCheck):
                vmap[str(cid)] = str(val.supported.value).lower()
            elif isinstance(val, dict):
                supp_dict = val.get("supported", "yes")
                vmap[str(cid)] = _extract_str_value(supp_dict, default="yes")
            else:
                vmap[str(cid)] = _extract_str_value(val, default="yes")
        return vmap

    for item in verdicts:
        if isinstance(item, ClaimSupportCheck):
            vmap[str(item.claim_id)] = str(item.supported.value).lower()
        elif isinstance(item, dict) and "claim_id" in item:
            supp = item.get("supported") or item.get("verdict") or "yes"
            vmap[str(item["claim_id"])] = _extract_str_value(supp, default="yes")

    return vmap


def citation_precision(
    claims_or_result: Sequence[Claim | dict[str, Any]] | RunResult,
    verdicts: (
        Sequence[ClaimSupportCheck | SupportVerdict | dict[str, Any]]
        | dict[str, Any]
        | None
    ) = None,
    *,
    partial_weight: float = 0.5,
    default_when_empty: float = 1.0,
) -> float:
    """Compute citation precision: share of cited sources that support the claim.

    Uses independent verifier verdicts to evaluate whether cited evidence
    passages directly support the claims they are attached to:
    - Verdict 'yes': 1.0 point.
    - Verdict 'partial': partial_weight points (default: 0.5).
    - Verdict 'no': 0.0 points.

    Args:
        claims_or_result: Claim sequence or RunResult instance.
        verdicts: Optional mapping of claim_id -> verdict or list of ClaimSupportCheck.
        partial_weight: Credit weight for 'partial' verdicts (0.0 to 1.0).
        default_when_empty: Value returned when zero sources are cited (default 1.0).

    Returns:
        Citation precision score between 0.0 and 1.0.
    """
    if isinstance(claims_or_result, RunResult) and isinstance(verdicts, EvalExample):
        return citation_precision_metric(claims_or_result, verdicts)

    claims, metadata = _extract_claims_and_metadata(claims_or_result)
    active_verdicts = verdicts or metadata.get("verifier_verdicts")
    vmap = _normalize_verdict_map(active_verdicts)

    total_citations = 0
    supported_weight = 0.0

    for c in claims:
        cid = getattr(c, "claim_id", None) or (
            c.get("claim_id") if isinstance(c, dict) else None
        )
        ev_list = getattr(c, "evidence", None) or (
            c.get("evidence", []) if isinstance(c, dict) else []
        )
        if not ev_list:
            continue

        c_verdict = vmap.get(str(cid)) if cid else None

        for ev in ev_list:
            total_citations += 1
            ev_verdict = c_verdict
            if isinstance(ev, dict):
                ev_verdict = ev.get("verdict") or ev.get("supported") or ev_verdict
            else:
                ev_verdict = getattr(ev, "verdict", None) or ev_verdict

            v_str = _extract_str_value(ev_verdict, default="yes")

            if v_str in ("yes", SupportVerdict.YES.value):
                supported_weight += 1.0
            elif v_str in ("partial", SupportVerdict.PARTIAL.value):
                supported_weight += partial_weight

    if total_citations == 0:
        return default_when_empty

    return round(supported_weight / total_citations, 4)


def unsupported_fact_rate(
    claims_or_result: Sequence[Claim | dict[str, Any]] | RunResult,
    verdicts: (
        Sequence[ClaimSupportCheck | SupportVerdict | dict[str, Any]]
        | dict[str, Any]
        | None
    ) = None,
) -> float:
    """Compute the share of empirical fact claims that lack valid evidence.

    A fact claim is considered unsupported if:
    1. It has zero evidence items, OR
    2. All cited evidence entries lack tool_call_id or valid raw_value, OR
    3. The independent verifier verdict for the claim is explicitly 'no'.

    Args:
        claims_or_result: Claim sequence or RunResult instance.
        verdicts: Optional mapping of claim_id -> verdict.

    Returns:
        Unsupported fact rate between 0.0 (all facts supported) and 1.0.
        Returns 0.0 if zero fact claims are present in the set.
    """
    if isinstance(claims_or_result, RunResult) and isinstance(verdicts, EvalExample):
        return unsupported_fact_rate_metric(claims_or_result, verdicts)

    claims, metadata = _extract_claims_and_metadata(claims_or_result)
    active_verdicts = verdicts or metadata.get("verifier_verdicts")
    vmap = _normalize_verdict_map(active_verdicts)

    fact_count = 0
    unsupported_count = 0

    for c in claims:
        cid = getattr(c, "claim_id", None) or (
            c.get("claim_id") if isinstance(c, dict) else None
        )
        raw_type = getattr(c, "type", None) or (
            c.get("type") if isinstance(c, dict) else None
        )
        type_str = _extract_str_value(raw_type, default="")

        if type_str not in ("fact", ClaimType.FACT.value):
            continue

        fact_count += 1
        ev_list = getattr(c, "evidence", None) or (
            c.get("evidence", []) if isinstance(c, dict) else []
        )

        # 1. Zero evidence
        if not ev_list:
            unsupported_count += 1
            continue

        # 2. Check if evidence has valid tool_call_id or raw values
        has_valid_source = False
        for ev in ev_list:
            t_id = (
                getattr(ev, "tool_call_id", None)
                if not isinstance(ev, dict)
                else ev.get("tool_call_id")
            )
            raw = (
                getattr(ev, "raw_value", None)
                if not isinstance(ev, dict)
                else ev.get("raw_value")
            )
            doc = (
                getattr(ev, "document_id", None)
                if not isinstance(ev, dict)
                else ev.get("document_id")
            )
            if t_id or raw is not None or doc:
                has_valid_source = True
                break

        if not has_valid_source:
            unsupported_count += 1
            continue

        # 3. Explicit verifier negative verdict
        c_verdict = vmap.get(str(cid)) if cid else None
        if c_verdict in ("no", SupportVerdict.NO.value, "unsupported"):
            unsupported_count += 1

    if fact_count == 0:
        return 0.0

    return round(unsupported_count / fact_count, 4)


def background_claim_rate(
    claims_or_result: Sequence[Claim | dict[str, Any]] | RunResult,
    example: EvalExample | None = None,
) -> float:
    """Compute the proportion of total claims that are background narrative claims.

    Args:
        claims_or_result: Claim sequence or RunResult instance.
        example: Optional EvalExample when invoked as a MetricFunc adapter.

    Returns:
        Background claim rate between 0.0 and 1.0.
        Returns 0.0 if zero total claims exist.
    """
    claims, _ = _extract_claims_and_metadata(claims_or_result)
    total_claims = len(claims)
    if total_claims == 0:
        return 0.0

    background_count = 0
    for c in claims:
        raw_type = getattr(c, "type", None) or (
            c.get("type") if isinstance(c, dict) else None
        )
        type_str = _extract_str_value(raw_type, default="")
        if type_str in ("background", ClaimType.BACKGROUND.value):
            background_count += 1

    return round(background_count / total_claims, 4)


def interpretation_validity(
    claims_or_result: Sequence[Claim | dict[str, Any]] | RunResult,
    all_claims: Sequence[Claim | dict[str, Any]] | None = None,
    *,
    default_when_empty: float = 1.0,
) -> float:
    """Compute the share of interpretation claims passing the support check.

    An interpretation claim passes the support check if:
    1. It cites existing supporting claim IDs in its depends_on list.
    2. It is not solely supported by background claims (requires >= 1 fact/calc).

    Args:
        claims_or_result: Claim sequence or RunResult instance.
        all_claims: Optional complete claim ledger (defaults to claims_or_result).
        default_when_empty: Value returned when zero interpretations exist
            (default 1.0).

    Returns:
        Share of interpretation claims passing support checks (0.0 to 1.0).
    """
    if isinstance(claims_or_result, RunResult) and isinstance(all_claims, EvalExample):
        return interpretation_validity_metric(claims_or_result, all_claims)

    target_claims, _ = _extract_claims_and_metadata(claims_or_result)
    full_claims = list(all_claims) if all_claims is not None else target_claims

    # Build claim index
    claims_by_id: dict[str, Any] = {}
    known_ids: set[str] = set()
    for c in full_claims:
        cid = getattr(c, "claim_id", None) or (
            c.get("claim_id") if isinstance(c, dict) else None
        )
        if cid:
            str_id = str(cid)
            claims_by_id[str_id] = c
            known_ids.add(str_id)

    interp_count = 0
    valid_count = 0

    for c in target_claims:
        raw_type = getattr(c, "type", None) or (
            c.get("type") if isinstance(c, dict) else None
        )
        type_str = _extract_str_value(raw_type, default="")

        if type_str not in ("interpretation", ClaimType.INTERPRETATION.value):
            continue

        interp_count += 1
        res_ids = check_interpretation_supporting_ids(
            c,
            existing_claim_ids=known_ids,
        )
        res_bg = check_background_not_sole_support(
            c,
            claims_by_id=claims_by_id,
        )

        if res_ids.passed and res_bg.passed:
            valid_count += 1

    if interp_count == 0:
        return default_when_empty

    return round(valid_count / interp_count, 4)


# ---------------------------------------------------------------------------
# MetricFunc Adapters for Evaluation Harness
# ---------------------------------------------------------------------------


def citation_precision_metric(result: RunResult, example: EvalExample) -> float:
    """Evaluate citation precision metric for evaluation variant runs."""
    return citation_precision(result)


def unsupported_fact_rate_metric(result: RunResult, example: EvalExample) -> float:
    """Evaluate unsupported fact rate metric for evaluation variant runs."""
    return unsupported_fact_rate(result)


def background_claim_rate_metric(result: RunResult, example: EvalExample) -> float:
    """Evaluate background claim rate metric for evaluation variant runs."""
    return background_claim_rate(result)


def interpretation_validity_metric(result: RunResult, example: EvalExample) -> float:
    """Evaluate interpretation validity metric for evaluation variant runs."""
    return interpretation_validity(result)
