"""Risk recall metric evaluating coverage of known risk themes via LLM judge."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lekka_ai.eval.dataset import EvalExample
from lekka_ai.eval.variant import RunResult
from lekka_ai.llm.base import LLMClient, Message
from lekka_ai.logging import get_logger
from lekka_ai.schemas.claims import Claim

logger = get_logger(__name__)

RISK_JUDGE_SYSTEM_PROMPT = """\
You are an expert equity research evaluator assessing whether an investment research
report or risk analysis successfully identified and flagged expected risk themes.

Given:
1. Flagged Themes: Risk themes or vulnerabilities raised in the model output.
2. Labeled Themes: Historical risk drivers labeled as "should have been flagged".

Rubric for each labeled theme:
- MATCH (matched: true): The flagged themes explicitly identify the underlying causal
  risk driver, financial vulnerability, or structural exposure specified by the labeled
  theme. Specific terminology, sector-appropriate synonyms, or quantitative breakdowns
  satisfy this criteria.
- NO MATCH (matched: false): The model omitted the risk topic entirely, only made
  generic boilerplate disclaimers (e.g. "general market risks"), or discussed an
  unrelated issue.

You must respond with valid JSON matching this schema:
{
  "matches": [
    {
      "theme": "Exact labeled theme text",
      "matched": true,
      "matched_by": "Flagged theme or text excerpt that matched, or null",
      "reasoning": "Concise explanation of the match or omission"
    }
  ]
}\
"""


class RiskRecallMatch(BaseModel):
    """Evaluation match verdict for an individual labeled risk theme."""

    model_config = ConfigDict(extra="ignore")

    theme: str = Field(..., description="Target labeled risk theme evaluated")
    matched: bool = Field(..., description="Whether the theme was flagged by the model")
    matched_by: str | None = Field(
        default=None,
        description="Flagged theme or excerpt that satisfied the labeled theme",
    )
    reasoning: str = Field(
        default="",
        description="Evaluator reasoning explaining the determination",
    )


class RiskRecallResult(BaseModel):
    """Aggregate result of evaluating risk recall against expected themes."""

    model_config = ConfigDict(extra="ignore")

    recall: float = Field(
        ...,
        description="Share of labeled risk themes flagged (0.0 to 1.0)",
    )
    matched_count: int = Field(
        ...,
        description="Total number of matched labeled themes",
    )
    total_labeled: int = Field(
        ...,
        description="Total number of labeled themes evaluated",
    )
    matches: list[RiskRecallMatch] = Field(
        default_factory=list,
        description="Per-theme grading determinations",
    )
    flagged_themes: list[str] = Field(
        default_factory=list,
        description="List of risk themes extracted from the model run",
    )
    labeled_themes: list[str] = Field(
        default_factory=list,
        description="List of labeled reference risk themes",
    )


def extract_flagged_themes(
    candidate: RunResult | str | Sequence[Claim | dict[str, Any] | str],
) -> list[str]:
    """Extract candidate risk themes from a RunResult, text report, or claims.

    Args:
        candidate: Evaluation run result, markdown report text, or claim sequence.

    Returns:
        List of distinct flagged risk theme strings.
    """
    themes: list[str] = []

    if isinstance(candidate, RunResult):
        # 1. Check metadata overrides
        meta = candidate.metadata or {}
        if meta.get("risk_themes"):
            return [str(t).strip() for t in meta["risk_themes"] if str(t).strip()]
        if meta.get("flagged_themes"):
            return [str(t).strip() for t in meta["flagged_themes"] if str(t).strip()]

        # 2. Extract from claims produced by risk agent
        for c in candidate.claims or []:
            producer = getattr(c, "produced_by", None) or (
                c.get("produced_by") if isinstance(c, dict) else None
            )
            text = getattr(c, "text", None) or (
                c.get("text") if isinstance(c, dict) else None
            )
            if str(producer).lower() in ("risk", "risk_agent") and text:
                themes.append(str(text).strip())

        # 3. Fallback to parsing answer markdown
        if not themes and candidate.answer:
            return extract_flagged_themes(candidate.answer)

        return themes

    if isinstance(candidate, str):
        # Parse Ranked Risks section if present
        risk_section_match = re.search(
            r"##\s*(?:Ranked Risks|Key Risks|Risks)(.*?)(?=##|\Z)",
            candidate,
            re.DOTALL | re.IGNORECASE,
        )
        body = risk_section_match.group(1) if risk_section_match else candidate

        for line in body.splitlines():
            line_str = line.strip()
            # Bullet point or numbered list item
            bullet_match = re.match(
                r"^(?:[-*•]|\d+[\.\)])\s*(?:\*\*)?(.*?)(?:\*\*)?$",
                line_str,
            )
            if bullet_match:
                item = bullet_match.group(1).strip()
                if item and len(item) > 3:
                    themes.append(item)
            elif (
                not risk_section_match
                and any(w in line_str.lower() for w in ("risk", "loss", "exposure"))
                and len(line_str) > 5
            ):
                themes.append(line_str)

        return themes

    # Sequence of claims or string themes
    for item in candidate:
        if isinstance(item, str):
            if item.strip():
                themes.append(item.strip())
        elif isinstance(item, (Claim, dict)):
            txt = getattr(item, "text", None) or (
                item.get("text") if isinstance(item, dict) else None
            )
            if txt:
                themes.append(str(txt).strip())

    return themes


def _clean_tokens(text: str) -> set[str]:
    """Extract normalized alphanumeric words for token matching."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def evaluate_risk_recall_rule_based(
    flagged_themes: Sequence[str],
    labeled_themes: Sequence[str],
) -> RiskRecallResult:
    """Evaluate risk recall using deterministic keyword and token rubric matching.

    Args:
        flagged_themes: Risk themes identified by the model.
        labeled_themes: Labeled reference risk themes expected to be flagged.

    Returns:
        Structured RiskRecallResult with per-theme verdicts.
    """
    total = len(labeled_themes)
    if total == 0:
        return RiskRecallResult(
            recall=1.0,
            matched_count=0,
            total_labeled=0,
            matches=[],
            flagged_themes=list(flagged_themes),
            labeled_themes=[],
        )

    clean_flagged = [t.strip() for t in flagged_themes if t.strip()]
    matches: list[RiskRecallMatch] = []
    matched_count = 0

    for labeled in labeled_themes:
        labeled_str = labeled.strip()
        labeled_tokens = _clean_tokens(labeled_str)
        found_match: str | None = None

        for flagged in clean_flagged:
            flagged_tokens = _clean_tokens(flagged)
            # 1. Substring containment
            if (
                labeled_str.lower() in flagged.lower()
                or flagged.lower() in labeled_str.lower()
            ):
                found_match = flagged
                break

            # 2. Significant token intersection (>= 50% of labeled keywords)
            if labeled_tokens:
                overlap = labeled_tokens & flagged_tokens
                if len(overlap) / len(labeled_tokens) >= 0.5:
                    found_match = flagged
                    break

        if found_match:
            matched_count += 1
            matches.append(
                RiskRecallMatch(
                    theme=labeled_str,
                    matched=True,
                    matched_by=found_match,
                    reasoning=f"Matched keywords with candidate theme '{found_match}'",
                )
            )
        else:
            matches.append(
                RiskRecallMatch(
                    theme=labeled_str,
                    matched=False,
                    matched_by=None,
                    reasoning="No sufficiently matching candidate risk theme found",
                )
            )

    recall = round(matched_count / total, 4)
    return RiskRecallResult(
        recall=recall,
        matched_count=matched_count,
        total_labeled=total,
        matches=matches,
        flagged_themes=clean_flagged,
        labeled_themes=list(labeled_themes),
    )


def _parse_llm_json_response(raw_text: str) -> dict[str, Any] | None:
    """Safely parse JSON response from LLM output."""
    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
    try:
        data = json.loads(clean)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", clean, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                pass
    return None


async def evaluate_risk_recall_llm(
    flagged_themes: Sequence[str],
    labeled_themes: Sequence[str],
    judge: LLMClient,
) -> RiskRecallResult:
    """Evaluate risk recall using an LLM judge prompted with rubric standards.

    Args:
        flagged_themes: Risk themes identified by the model.
        labeled_themes: Labeled reference risk themes expected to be flagged.
        judge: LLMClient instance executing the evaluation rubric.

    Returns:
        Structured RiskRecallResult with per-theme verdicts.
    """
    total = len(labeled_themes)
    if total == 0:
        return RiskRecallResult(
            recall=1.0,
            matched_count=0,
            total_labeled=0,
            matches=[],
            flagged_themes=list(flagged_themes),
            labeled_themes=[],
        )

    user_prompt = (
        "Flagged Themes from Model:\n"
        + "\n".join(f"- {t}" for t in flagged_themes)
        + "\n\nLabeled Reference Themes to Audit:\n"
        + "\n".join(f"- {t}" for t in labeled_themes)
        + "\n\nEvaluate each labeled theme against the rubric and return JSON."
    )

    messages = [
        Message(role="system", content=RISK_JUDGE_SYSTEM_PROMPT),
        Message(role="user", content=user_prompt),
    ]

    try:
        response = await judge.generate(messages)
        if response.text:
            parsed = _parse_llm_json_response(response.text)
            if parsed and "matches" in parsed and isinstance(parsed["matches"], list):
                match_list: list[RiskRecallMatch] = []
                matched_count = 0
                for item in parsed["matches"]:
                    if isinstance(item, dict) and "theme" in item:
                        is_m = bool(item.get("matched", False))
                        if is_m:
                            matched_count += 1
                        match_list.append(
                            RiskRecallMatch(
                                theme=str(item["theme"]),
                                matched=is_m,
                                matched_by=item.get("matched_by"),
                                reasoning=str(item.get("reasoning", "")),
                            )
                        )
                return RiskRecallResult(
                    recall=round(matched_count / total, 4),
                    matched_count=matched_count,
                    total_labeled=total,
                    matches=match_list,
                    flagged_themes=list(flagged_themes),
                    labeled_themes=list(labeled_themes),
                )
    except Exception as err:
        logger.warning(
            "LLM judge failed for risk recall, falling back to rule-based: %s",
            err,
        )

    # Fallback to rule-based rubric evaluation
    return evaluate_risk_recall_rule_based(flagged_themes, labeled_themes)


def risk_recall(
    flagged: RunResult | str | Sequence[Claim | dict[str, Any] | str],
    labeled: Sequence[str] | EvalExample,
    judge: LLMClient | None = None,
) -> float:
    """Compute the risk recall score (matched themes / labeled themes).

    Args:
        flagged: Candidate output (RunResult, string, or claims sequence).
        labeled: Labeled reference themes sequence or EvalExample.
        judge: Optional LLMClient to act as judge (defaults to rubric rules).

    Returns:
        Risk recall score as a float between 0.0 and 1.0.
    """
    flagged_themes = extract_flagged_themes(flagged)

    if isinstance(labeled, EvalExample):
        labeled_themes = getattr(labeled, "risk_themes", None) or (
            labeled.expected
            if isinstance(labeled.expected, list)
            else [str(labeled.expected)]
            if labeled.expected
            else []
        )
    else:
        labeled_themes = list(labeled)

    if not labeled_themes:
        return 1.0

    result = evaluate_risk_recall_rule_based(flagged_themes, labeled_themes)
    return result.recall


def risk_recall_metric(result: RunResult, example: EvalExample) -> float:
    """Evaluation harness MetricFunc adapter for risk recall.

    Args:
        result: Evaluated run output.
        example: Benchmark item with labeled risk themes.

    Returns:
        Risk recall score between 0.0 and 1.0.
    """
    return risk_recall(result, example)
