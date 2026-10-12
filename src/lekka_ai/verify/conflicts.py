"""Cross-agent conflict detection for claims sharing metrics and periods."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lekka_ai.db.ledger import LedgerRepository
from lekka_ai.logging import get_logger
from lekka_ai.orchestrator.state import RunState
from lekka_ai.schemas.claims import Claim

logger = get_logger(__name__)


class Conflict(BaseModel):
    """Detected conflict between two claims reporting differing metric values."""

    model_config = ConfigDict(extra="ignore")

    metric: str = Field(..., description="Canonical metric name, e.g. 'revenue'")
    period: str = Field(..., description="Canonical fiscal period, e.g. 'FY2023'")
    claim_a_id: str = Field(..., description="First claim ID in conflict")
    claim_b_id: str = Field(..., description="Second claim ID in conflict")
    agent_a: str = Field(..., description="Agent or producer of first claim")
    agent_b: str = Field(..., description="Agent or producer of second claim")
    value_a: float = Field(..., description="Normalized numeric value of first claim")
    value_b: float = Field(..., description="Normalized numeric value of second claim")
    raw_text_a: str = Field(..., description="Original statement text of first claim")
    raw_text_b: str = Field(..., description="Original statement text of second claim")
    difference: float = Field(..., description="Absolute numeric difference")
    difference_pct: float = Field(
        ...,
        description="Relative percentage difference between values",
    )
    description: str = Field(..., description="Human-readable summary of conflict")


def normalize_metric(text: str) -> str | None:
    """Normalize free-text financial terms into standard metric identifiers."""
    clean = text.lower().strip()

    # Revenue / Sales
    if re.search(r"\b(revenue|net sales|sales|turnover|topline|top-line)\b", clean):
        return "revenue"

    # Operating Income / EBIT
    if re.search(r"\b(operating income|operating profit|ebit|operating loss)\b", clean):
        return "operating_income"

    # Net Income / Earnings
    if re.search(
        r"\b(net income|net earnings|net profit|bottomline|bottom-line)\b",
        clean,
    ):
        return "net_income"

    # Gross Margin
    if re.search(r"\b(gross margin|gm)\b", clean):
        return "gross_margin"

    # Operating Margin
    if re.search(r"\b(operating margin|op margin)\b", clean):
        return "operating_margin"

    # EPS
    if re.search(r"\b(diluted eps|eps|earnings per share)\b", clean):
        return "eps"

    # Free Cash Flow
    if re.search(r"\b(free cash flow|fcf)\b", clean):
        return "free_cash_flow"

    # EBITDA
    if re.search(r"\b(ebitda)\b", clean):
        return "ebitda"

    # P/E Ratio
    if re.search(r"\b(pe ratio|p/e ratio|pe multiple|p/e multiple|p/e|pe)\b", clean):
        return "pe_ratio"

    # Shares Outstanding
    if re.search(r"\b(shares outstanding|diluted shares)\b", clean):
        return "shares_outstanding"

    return None


def normalize_period(text: str, fallback_date: str | None = None) -> str | None:
    """Normalize fiscal period descriptions into standardized representations."""
    clean = text.strip()

    # Quarters: Q1 2024, 1Q24, Q1 FY24, 2024-Q1
    m_q1 = re.search(
        r"\bQ([1-4])\s*(?:FY|fiscal)?\s*['\"]?(\d{2,4})\b",
        clean,
        re.IGNORECASE,
    )
    if m_q1:
        qtr = m_q1.group(1)
        yr = m_q1.group(2)
        full_yr = f"20{yr}" if len(yr) == 2 else yr
        return f"{full_yr}-Q{qtr}"

    m_q2 = re.search(r"\b(\d{2,4})\s*Q([1-4])\b", clean, re.IGNORECASE)
    if m_q2:
        yr = m_q2.group(1)
        qtr = m_q2.group(2)
        full_yr = f"20{yr}" if len(yr) == 2 else yr
        return f"{full_yr}-Q{qtr}"

    # Fiscal Year: FY2023, FY23, FY 2023, fiscal 2023, fiscal year 2023
    m_fy = re.search(
        r"\b(?:FY|fiscal year|fiscal)\s*['\"]?(\d{2,4})\b",
        clean,
        re.IGNORECASE,
    )
    if m_fy:
        yr = m_fy.group(1)
        full_yr = f"20{yr}" if len(yr) == 2 else yr
        return f"FY{full_yr}"

    # Calendar Year standalone: 2023, 2024
    m_yr = re.search(r"\b(20[1-3]\d)\b", clean)
    if m_yr:
        return f"FY{m_yr.group(1)}"

    # Date format YYYY-MM-DD
    if fallback_date:
        m_dt = re.match(r"^(\d{4})-\d{2}-\d{2}", fallback_date.strip())
        if m_dt:
            return f"FY{m_dt.group(1)}"

    return None


def _extract_metric_and_value(
    claim: Claim | dict[str, Any] | Any,
) -> tuple[str | None, str | None, float | None]:
    """Extract (canonical_metric, canonical_period, normalized_value) from a claim."""
    claim_text = ""
    evidence_list: list[Any] = []
    metadata: dict[str, Any] = {}

    if isinstance(claim, dict):
        claim_text = str(claim.get("text", ""))
        evidence_list = list(claim.get("evidence", []) or [])
        metadata = dict(claim)
    else:
        claim_text = str(getattr(claim, "text", ""))
        evidence_list = list(getattr(claim, "evidence", []) or [])
        if hasattr(claim, "model_dump"):
            try:
                metadata = claim.model_dump()
            except Exception:
                pass

    # 1. Direct explicit metadata / raw_value attributes
    metric = metadata.get("metric")
    period = metadata.get("period")
    value = metadata.get("value") or metadata.get("numeric_value")

    # Check evidence raw_value if not found in metadata
    first_ev_as_of: str | None = None
    if evidence_list:
        first_ev = evidence_list[0]
        raw_val = (
            first_ev.get("raw_value")
            if isinstance(first_ev, dict)
            else getattr(first_ev, "raw_value", None)
        )
        as_of = (
            first_ev.get("as_of_date")
            if isinstance(first_ev, dict)
            else getattr(first_ev, "as_of_date", None)
        )
        if as_of:
            first_ev_as_of = str(as_of)

        if isinstance(raw_val, dict):
            if not metric:
                metric = raw_val.get("metric") or raw_val.get("metric_name")
            if not period:
                period = raw_val.get("period") or raw_val.get("fiscal_period")
            if value is None:
                value = raw_val.get("value") or raw_val.get("raw_value")

            # Check if dict keys match known metrics (e.g. {"revenue": 383285000000})
            if not metric or value is None:
                for k, v in raw_val.items():
                    norm_k = normalize_metric(k)
                    if norm_k and isinstance(v, (int, float)):
                        metric = norm_k
                        value = float(v)
                        break

    # 2. Text-based metric extraction
    canonical_metric = (
        normalize_metric(str(metric)) if metric else normalize_metric(claim_text)
    )

    # 3. Text-based period extraction
    canonical_period = (
        normalize_period(str(period))
        if period
        else normalize_period(claim_text, fallback_date=first_ev_as_of)
    )

    # 4. Text-based value and unit extraction if not found
    normalized_value: float | None = None
    if value is not None:
        try:
            normalized_value = float(value)
        except (ValueError, TypeError):
            pass

    if normalized_value is None and claim_text:
        # Search for values with units: $383.3B, 383.3 billion, $383,285M, 29.8%
        clean_text = claim_text.replace(",", "")

        # Trillions
        m_t = re.search(
            r"\$?\s*(-?\d+(?:\.\d+)?)\s*(?:T|trillion)\b",
            clean_text,
            re.IGNORECASE,
        )
        if m_t:
            normalized_value = float(m_t.group(1)) * 1e12

        # Billions
        if normalized_value is None:
            m_b = re.search(
                r"\$?\s*(-?\d+(?:\.\d+)?)\s*(?:B|billion)\b",
                clean_text,
                re.IGNORECASE,
            )
            if m_b:
                normalized_value = float(m_b.group(1)) * 1e9

        # Millions
        if normalized_value is None:
            m_m = re.search(
                r"\$?\s*(-?\d+(?:\.\d+)?)\s*(?:M|million)\b",
                clean_text,
                re.IGNORECASE,
            )
            if m_m:
                normalized_value = float(m_m.group(1)) * 1e6

        # Thousands
        if normalized_value is None:
            m_k = re.search(
                r"\$?\s*(-?\d+(?:\.\d+)?)\s*(?:K|thousand)\b",
                clean_text,
                re.IGNORECASE,
            )
            if m_k:
                normalized_value = float(m_k.group(1)) * 1e3

        # Percentage
        if normalized_value is None:
            m_pct = re.search(r"(-?\d+(?:\.\d+)?)\s*%", clean_text)
            if m_pct:
                normalized_value = float(m_pct.group(1))

        # Plain number / currency
        if normalized_value is None:
            m_plain = re.search(r"\$?\s*(-?\d+(?:\.\d+)?)", clean_text)
            if m_plain:
                normalized_value = float(m_plain.group(1))

    return canonical_metric, canonical_period, normalized_value


def _format_value_for_display(val: float, metric: str) -> str:
    """Format numeric value in conventional financial units for display."""
    if "margin" in metric or "ratio" in metric:
        if abs(val) < 1.0:
            return f"{val * 100:.1f}%"
        return f"{val:.1f}%"

    abs_v = abs(val)
    if abs_v >= 1e12:
        return f"${val / 1e12:.2f}T"
    if abs_v >= 1e9:
        return f"${val / 1e9:.2f}B"
    if abs_v >= 1e6:
        return f"${val / 1e6:.1f}M"
    return f"${val:,.2f}"


def _get_claim_attribute(claim: Any, attr: str, default: Any = None) -> Any:
    """Safely get attribute from Claim instance or dictionary."""
    if isinstance(claim, dict):
        return claim.get(attr, default)
    return getattr(claim, attr, default)


def _mark_claim_contradicted_by(claim: Any, contr_id: str) -> None:
    """Append contradicted_by ID to claim in memory."""
    if isinstance(claim, dict):
        current = list(claim.get("contradicted_by", []) or [])
        if contr_id not in current:
            current.append(contr_id)
            claim["contradicted_by"] = current
    else:
        current_list = list(getattr(claim, "contradicted_by", []) or [])
        if contr_id not in current_list:
            current_list.append(contr_id)
            try:
                claim.contradicted_by = current_list
            except Exception:
                pass


def detect_conflicts(
    claims: Sequence[Claim | dict[str, Any] | Any],
    *,
    state: RunState | None = None,
    ledger: LedgerRepository | None = None,
    session: Any | None = None,
    tolerance: float = 0.01,
) -> list[Conflict]:
    """Detect claims reporting differing values for the same metric and period.

    Normalizes metric names and units across specialist agents and sources.
    When a conflict is detected:
    1. Marks 'contradicted_by' reciprocally on the claims and in the ledger.
    2. Adds conflict questions into 'state.open_questions' for the gap check.

    Args:
        claims: List or sequence of Claim objects or claim dictionaries to inspect.
        state: Optional RunState to populate with open questions for gap check.
        ledger: Optional LedgerRepository to persist contradiction relations.
        session: Optional SQLAlchemy session.
        tolerance: Permitted relative difference margin (default 0.01 = 1%).

    Returns:
        List of detected Conflict objects.
    """
    effective_ledger = ledger
    if effective_ledger is None and session is not None:
        effective_ledger = LedgerRepository(session)

    # 1. Parse and extract (metric, period, value) for each claim
    parsed_claims: list[dict[str, Any]] = []

    for c in claims:
        cid = str(_get_claim_attribute(c, "claim_id") or "")
        agent = str(
            _get_claim_attribute(c, "produced_by")
            or _get_claim_attribute(c, "agent_type")
            or "agent"
        )
        text = str(_get_claim_attribute(c, "text") or "")
        metric, period, val = _extract_metric_and_value(c)

        if metric and period and val is not None and cid:
            parsed_claims.append(
                {
                    "claim": c,
                    "claim_id": cid,
                    "agent": agent,
                    "text": text,
                    "metric": metric,
                    "period": period,
                    "value": val,
                }
            )

    # 2. Group by (metric, period)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for p in parsed_claims:
        key = (p["metric"], p["period"])
        groups.setdefault(key, []).append(p)

    conflicts: list[Conflict] = []
    seen_pairs: set[frozenset[str]] = set()

    # 3. Compare pairs within each group
    for (metric, period), items in groups.items():
        if len(items) < 2:
            continue

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                item_a = items[i]
                item_b = items[j]

                cid_a = item_a["claim_id"]
                cid_b = item_b["claim_id"]

                pair_key = frozenset([cid_a, cid_b])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                val_a = item_a["value"]
                val_b = item_b["value"]

                # Calculate absolute and relative differences
                diff = abs(val_a - val_b)
                denom = max(abs(val_a), abs(val_b), 1e-9)
                diff_pct = diff / denom

                if diff_pct > tolerance:
                    # Conflict confirmed!
                    agent_a = item_a["agent"]
                    agent_b = item_b["agent"]

                    val_a_str = _format_value_for_display(val_a, metric)
                    val_b_str = _format_value_for_display(val_b, metric)

                    desc = (
                        f"Conflict on {metric.upper()} ({period}): "
                        f"{agent_a} reported {val_a_str} (claim '{cid_a}'), but "
                        f"{agent_b} reported {val_b_str} (claim '{cid_b}'). "
                        f"Difference is {diff_pct:.1%} ({diff:,.2f})."
                    )

                    conflict_obj = Conflict(
                        metric=metric,
                        period=period,
                        claim_a_id=cid_a,
                        claim_b_id=cid_b,
                        agent_a=agent_a,
                        agent_b=agent_b,
                        value_a=val_a,
                        value_b=val_b,
                        raw_text_a=item_a["text"],
                        raw_text_b=item_b["text"],
                        difference=round(diff, 4),
                        difference_pct=round(diff_pct, 4),
                        description=desc,
                    )
                    conflicts.append(conflict_obj)

                    logger.warning(
                        "Contradiction detected between claims",
                        metric=metric,
                        period=period,
                        claim_a=cid_a,
                        claim_b=cid_b,
                        diff_pct=round(diff_pct, 4),
                    )

                    # 4. Mark contradicted_by on in-memory claims
                    _mark_claim_contradicted_by(item_a["claim"], cid_b)
                    _mark_claim_contradicted_by(item_b["claim"], cid_a)

                    # 5. Mark contradicted_by in ledger repository
                    if effective_ledger is not None:
                        try:
                            effective_ledger.mark_contradiction(cid_a, cid_b)
                        except Exception as exc:
                            logger.debug(
                                "Failed to mark contradiction in ledger",
                                error=str(exc),
                            )
                        try:
                            effective_ledger.mark_contradiction(cid_b, cid_a)
                        except Exception as exc:
                            logger.debug(
                                "Failed to mark contradiction in ledger",
                                error=str(exc),
                            )

                    # 6. Add to RunState.open_questions for parent gap check
                    if state is not None:
                        question = (
                            f"Reconcile conflict on {metric} ({period}): {agent_a} "
                            f"claims {val_a_str} vs {agent_b} claims {val_b_str} "
                            f"(claims: '{cid_a}', '{cid_b}')."
                        )
                        if question not in state.open_questions:
                            state.open_questions.append(question)

    return conflicts
