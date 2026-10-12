"""Report synthesizer generating grounded reports strictly from ledger claims."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lekka_ai.config import get_settings
from lekka_ai.db.models import ClaimModel
from lekka_ai.llm.base import LLMClient, Message
from lekka_ai.llm.gemini import GeminiClient
from lekka_ai.logging import get_logger
from lekka_ai.schemas.claims import Claim

logger = get_logger(__name__)

REQUIRED_SECTIONS: list[str] = [
    "Executive Summary",
    "Business Quality",
    "Valuation Scenarios",
    "Bull Case",
    "Bear Case",
    "Ranked Risks",
    "Catalysts",
    "What Would Invalidate the Thesis",
    "Uncertainty and Gaps",
]

SYNTHESIZE_SYSTEM_PROMPT = """You are the Senior Research Report Synthesizer.
Your goal is to write a rigorous equity research report based STRICTLY on ledger claims.

CRITICAL RULES:
1. LEDGER CLAIMS ONLY: Use ONLY facts/conclusions from the provided claims.
   Never introduce external numbers or assumptions not present in the claims.
2. CITATION REQUIREMENT: EVERY SINGLE SENTENCE in the body MUST cite the
   supporting claim ID(s) in brackets at the end, e.g. [claim_1].
3. STRICT NUMBER GROUNDING: Every number, metric, percentage, multiple, or currency
   figure in a sentence MUST be present in the claim(s) cited by that sentence.
   DO NOT INVENT, ESTIMATE, OR ROUND NUMBERS beyond the cited claims.
4. REQUIRED STRUCTURE: Organize the report into these EXACT 9 sections:
   ## Executive Summary
   ## Business Quality
   ## Valuation Scenarios
   ## Bull Case
   ## Bear Case
   ## Ranked Risks
   ## Catalysts
   ## What Would Invalidate the Thesis
   ## Uncertainty and Gaps

Any sentence that lacks a claim citation or contains an ungrounded or invented
number will cause the report to be rejected immediately.
"""

NUMERIC_PATTERN = re.compile(
    r"(?<![a-zA-Z_])"
    r"(\$|€|£)?"
    r"\s*"
    r"(-?\d+(?:,\d{3})*(?:\.\d+)?)"
    r"\s*"
    r"(%|x|T|B|M|K|trillion|billion|million|thousand)?"
    r"(?![a-zA-Z_])",
    re.IGNORECASE,
)


class ValidationCheck(BaseModel):
    """Detailed validation check result for a single sentence."""

    model_config = ConfigDict(extra="ignore")

    sentence: str
    claims_cited: list[str] = Field(default_factory=list)
    passed: bool = True
    error: str | None = None
    numbers_found: list[str] = Field(default_factory=list)


class ReportValidationResult(BaseModel):
    """Result of post-check validation evaluating citations, numbers, and sections."""

    model_config = ConfigDict(extra="ignore")

    passed: bool = True
    errors: list[str] = Field(default_factory=list)
    sentence_to_claims: dict[str, list[str]] = Field(default_factory=dict)
    checks: list[ValidationCheck] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)


class SynthesizedReport(BaseModel):
    """Final synthesized research report and sentence citation mapping."""

    model_config = ConfigDict(extra="ignore")

    markdown: str
    sentence_to_claims: dict[str, list[str]]
    claims_used: list[str] = Field(default_factory=list)
    validation_passed: bool = True
    attempts: int = 1
    rejection_reasons: list[str] = Field(default_factory=list)
    ticker: str | None = None


class ReportValidationError(Exception):
    """Raised when report synthesis fails validation checks after max retries."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Report validation failed with errors: {errors}")


def _extract_numbers_from_text(text: str) -> list[tuple[str, float, set[float]]]:
    """Extract numeric tokens, base floats, and scaled floats from text.

    Returns a list of tuples: (raw_matched_string, base_float, set_of_scaled_floats).
    """
    results: list[tuple[str, float, set[float]]] = []
    # Strip claim citations first so digits in claim_1 are not treated as numbers
    clean = re.sub(
        r"\[\s*claim_[a-zA-Z0-9_\-,\s]+\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\(\s*claim_[a-zA-Z0-9_\-,\s]+\)",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"\bclaim_[a-zA-Z0-9_\-]+\b", "", clean, flags=re.IGNORECASE)

    for match in NUMERIC_PATTERN.finditer(clean):
        num_str = match.group(2).replace(",", "")
        unit = (match.group(3) or "").lower()
        full_tok = match.group(0).strip()

        try:
            val = float(num_str)
        except ValueError:
            continue

        scaled: set[float] = {val}
        if unit in ("b", "billion"):
            scaled.add(val * 1e9)
        elif unit in ("m", "million"):
            scaled.add(val * 1e6)
        elif unit in ("t", "trillion"):
            scaled.add(val * 1e12)
        elif unit in ("k", "thousand"):
            scaled.add(val * 1e3)
        elif unit == "%":
            scaled.add(val / 100.0)
            scaled.add(val)

        results.append((full_tok, val, scaled))
    return results


def _extract_claim_numbers(
    claim: Claim | ClaimModel | dict[str, Any],
) -> tuple[set[float], set[str]]:
    """Extract all float values and string tokens from a claim and its evidence."""
    floats: set[float] = set()
    strings: set[str] = set()

    text_val = str(getattr(claim, "text", None) or "")
    if not text_val and isinstance(claim, dict):
        text_val = str(claim.get("text") or "")

    for tok, base_val, scaled_set in _extract_numbers_from_text(text_val):
        floats.update(scaled_set)
        floats.add(base_val)
        clean_tok = tok.replace("$", "").replace("%", "").strip()
        strings.add(clean_tok)
        strings.add(tok)

    # Check evidence raw_value
    evidence_list: list[Any] = []
    if isinstance(claim, dict):
        evidence_list = list(claim.get("evidence", []) or [])
    else:
        evidence_list = list(getattr(claim, "evidence", []) or [])

    for ev in evidence_list:
        raw_val = (
            ev.get("raw_value")
            if isinstance(ev, dict)
            else getattr(ev, "raw_value", None)
        )
        if raw_val is not None:
            if isinstance(raw_val, (int, float)):
                f = float(raw_val)
                floats.add(f)
                strings.add(str(raw_val))
                if abs(f) >= 1e9:
                    floats.add(f / 1e9)
                    strings.add(f"{f / 1e9:.1f}")
                    strings.add(f"{f / 1e9:.2f}")
                elif abs(f) >= 1e6:
                    floats.add(f / 1e6)
                    strings.add(f"{f / 1e6:.1f}")
            elif isinstance(raw_val, dict):
                for v in raw_val.values():
                    if isinstance(v, (int, float)):
                        f = float(v)
                        floats.add(f)
                        strings.add(str(v))
                        if abs(f) >= 1e9:
                            floats.add(f / 1e9)
                            strings.add(f"{f / 1e9:.1f}")
                            strings.add(f"{f / 1e9:.2f}")
                    elif isinstance(v, str):
                        for tok, base_v, s_set in _extract_numbers_from_text(v):
                            floats.update(s_set)
                            floats.add(base_v)
                            strings.add(tok)
            elif isinstance(raw_val, str):
                for tok, base_v, s_set in _extract_numbers_from_text(raw_val):
                    floats.update(s_set)
                    floats.add(base_v)
                    strings.add(tok)

    return floats, strings


def _extract_sentences_from_markdown(markdown: str) -> list[str]:
    """Split markdown report into substantive sentences while ignoring headers."""
    sentences: list[str] = []
    lines = markdown.splitlines()
    in_code_block = False
    current_para: list[str] = []

    def flush_para(para_lines: list[str]) -> None:
        if not para_lines:
            return
        para_text = " ".join(para_lines).strip()
        if not para_text:
            return
        # Split on sentence terminals . ! ? followed by space or capitalized word
        raw_sents = re.split(r"(?<=[.!?\]])\s+(?=[A-Z0-9\-\*\"'])", para_text)
        for s in raw_sents:
            clean_s = s.strip()
            if len(clean_s) > 5 and not clean_s.startswith("#"):
                sentences.append(clean_s)

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        # Header lines and horizontal rules flush paragraphs and are skipped
        if stripped.startswith("#") or stripped in ("---", "***", "___"):
            flush_para(current_para)
            current_para = []
            continue

        if not stripped:
            flush_para(current_para)
            current_para = []
            continue

        # List bullet points flush previous and start new item
        if re.match(r"^(\*|-|\d+\.)\s+", stripped):
            flush_para(current_para)
            clean_item = re.sub(r"^(\*|-|\d+\.)\s+", "", stripped)
            current_para = [clean_item]
            continue

        current_para.append(stripped)

    flush_para(current_para)
    return sentences


def _extract_cited_claim_ids(sentence: str) -> list[str]:
    """Extract claim IDs cited in brackets or parentheses in a sentence."""
    cited: list[str] = []
    # Pattern: [claim_xyz, claim_abc] or (claim_xyz)
    bracket_matches = re.findall(r"\[([^\]]+)\]|\(([^\)]+)\)", sentence)
    for m in bracket_matches:
        content = m[0] or m[1]
        ids = re.findall(
            r"\b(claim_[a-zA-Z0-9_\-]+|[a-zA-Z0-9_\-]+_claim)\b",
            content,
            re.IGNORECASE,
        )
        cited.extend(ids)

    # Bare fallback
    if not cited:
        bare_ids = re.findall(r"\b(claim_[a-zA-Z0-9_\-]+)\b", sentence, re.IGNORECASE)
        cited.extend(bare_ids)

    seen: set[str] = set()
    result: list[str] = []
    for c in cited:
        clean = c.strip()
        if clean.lower() not in seen:
            seen.add(clean.lower())
            result.append(clean)
    return result


def _is_number_grounded(
    num_str: str,
    base_val: float,
    scaled_floats: set[float],
    claim_floats: set[float],
    claim_strings: set[str],
) -> bool:
    """Determine if a sentence numeric token is grounded in cited claim numbers."""
    clean_num = num_str.replace("$", "").replace("%", "").strip()

    # 1. Exact string match
    if clean_num in claim_strings or num_str in claim_strings:
        return True

    # Check substring match in any claim string
    for cs in claim_strings:
        if clean_num in cs or cs in clean_num:
            return True

    # 2. Float value match within tolerance
    for sf in scaled_floats:
        for cf in claim_floats:
            diff = abs(sf - cf)
            if diff < 1e-4:
                return True
            # Relative error within 1%
            denom = max(abs(cf), 1e-6)
            if (diff / denom) <= 0.01:
                return True

    for cf in claim_floats:
        diff = abs(base_val - cf)
        if diff < 1e-4:
            return True
        denom = max(abs(cf), 1e-6)
        if (diff / denom) <= 0.01:
            return True

    return False


def validate_report(
    markdown: str,
    claims: Sequence[Claim | ClaimModel | dict[str, Any]],
    *,
    require_all_sections: bool = True,
) -> ReportValidationResult:
    """Perform post-check validation on generated report markdown.

    Verifies:
    1. All 9 required sections are present.
    2. Every sentence references at least one claim ID.
    3. All numbers in each sentence are grounded in its cited claims.
    """
    errors: list[str] = []
    checks: list[ValidationCheck] = []
    sentence_to_claims: dict[str, list[str]] = {}

    # Build claim lookup map and extracted numbers
    claims_by_id: dict[str, Any] = {}
    claim_data: dict[str, tuple[set[float], set[str]]] = {}

    for c in claims:
        cid = getattr(c, "claim_id", None)
        if not cid and isinstance(c, dict):
            cid = c.get("claim_id")
        if cid:
            clean_id = str(cid).strip().lower()
            claims_by_id[clean_id] = c
            claim_data[clean_id] = _extract_claim_numbers(c)

    # 1. Check required section headings
    missing_sections: list[str] = []
    heading_lines = [
        line.strip() for line in markdown.splitlines() if line.strip().startswith("#")
    ]
    heading_corpus = " \n ".join(heading_lines).lower()

    if require_all_sections:
        for sec in REQUIRED_SECTIONS:
            if sec.lower() not in heading_corpus:
                missing_sections.append(sec)

        if missing_sections:
            errors.append(f"Report is missing required sections: {missing_sections}")

    # 2. Extract and validate sentences
    sentences = _extract_sentences_from_markdown(markdown)
    if not sentences:
        errors.append("Report contains no substantive text sentences.")
        return ReportValidationResult(
            passed=False,
            errors=errors,
            missing_sections=missing_sections,
        )

    for sent in sentences:
        cited_ids = _extract_cited_claim_ids(sent)
        numbers_found = [tok for tok, _, _ in _extract_numbers_from_text(sent)]
        sent_check = ValidationCheck(
            sentence=sent,
            claims_cited=cited_ids,
            numbers_found=numbers_found,
        )

        # Rule A: Must cite at least one claim ID
        if not cited_ids:
            sent_check.passed = False
            err_msg = f"Sentence has no claim citations: '{sent}'"
            sent_check.error = err_msg
            errors.append(err_msg)
            checks.append(sent_check)
            continue

        # Check for unknown claim IDs
        unknown_ids = [cid for cid in cited_ids if cid.lower() not in claims_by_id]
        if unknown_ids:
            sent_check.passed = False
            err_msg = f"Sentence cites unknown claim ID(s) {unknown_ids}: '{sent}'"
            sent_check.error = err_msg
            errors.append(err_msg)
            checks.append(sent_check)
            continue

        sentence_to_claims[sent] = cited_ids

        # Rule B: All numbers must be grounded in cited claims
        sentence_num_tokens = _extract_numbers_from_text(sent)
        if sentence_num_tokens:
            # Aggregate available floats and strings from cited claims
            combined_floats: set[float] = set()
            combined_strings: set[str] = set()
            for cid in cited_ids:
                c_floats, c_strings = claim_data.get(cid.lower(), (set(), set()))
                combined_floats.update(c_floats)
                combined_strings.update(c_strings)

            ungrounded_numbers: list[str] = []
            for tok, base_val, scaled_set in sentence_num_tokens:
                if not _is_number_grounded(
                    tok,
                    base_val,
                    scaled_set,
                    combined_floats,
                    combined_strings,
                ):
                    ungrounded_numbers.append(tok)

            if ungrounded_numbers:
                sent_check.passed = False
                err_msg = (
                    f"Sentence contains invented number(s) {ungrounded_numbers} "
                    f"not in cited claims {cited_ids}: '{sent}'"
                )
                sent_check.error = err_msg
                errors.append(err_msg)

        checks.append(sent_check)

    passed = len(errors) == 0
    return ReportValidationResult(
        passed=passed,
        errors=errors,
        sentence_to_claims=sentence_to_claims,
        checks=checks,
        missing_sections=missing_sections,
    )


def _format_claims_for_prompt(
    claims: Sequence[Claim | ClaimModel | dict[str, Any]],
) -> str:
    """Format ledger claims for injection into the report synthesizer prompt."""
    lines: list[str] = []
    for c in claims:
        cid = getattr(c, "claim_id", None) or (
            c.get("claim_id") if isinstance(c, dict) else "unknown"
        )
        text_v = getattr(c, "text", None) or (
            c.get("text") if isinstance(c, dict) else ""
        )
        ctype = getattr(c, "type", None) or (
            c.get("type") if isinstance(c, dict) else "fact"
        )
        conf = getattr(c, "confidence", 1.0)
        if isinstance(c, dict):
            conf = c.get("confidence", 1.0)

        lines.append(f"[{cid}] ({ctype}, confidence: {conf}): {text_v}")
    return "\n".join(lines)


async def synthesize_report(
    claims: Sequence[Claim | ClaimModel | dict[str, Any]],
    *,
    ticker: str | None = None,
    question: str | None = None,
    llm: LLMClient | None = None,
    llm_client: LLMClient | None = None,
    model_name: str | None = None,
    max_retries: int = 2,
    raise_on_failure: bool = False,
    require_all_sections: bool = True,
) -> SynthesizedReport:
    """Synthesize the final research report from ledger claims with regeneration.

    Parameters:
        claims: Verified ledger claims to ground the report.
        ticker: Optional target stock ticker symbol.
        question: Optional research question or mandate.
        llm: LLM client alias.
        llm_client: LLM client (GeminiClient or FakeLLM).
        model_name: Specific Gemini model identifier.
        max_retries: Maximum number of regeneration attempts (default 2).
        raise_on_failure: If True, raises ReportValidationError when checks fail.
        require_all_sections: Enforce all 9 required markdown sections.
    """
    settings = get_settings()
    effective_llm = llm_client or llm
    client = effective_llm or GeminiClient(model=model_name or settings.GEMINI_MODEL)

    claims_text = _format_claims_for_prompt(claims)
    mandate_str = question or "Comprehensive Equity Research Analysis"
    target_str = ticker or "the target company"
    user_prompt = (
        f"Synthesize the final equity research report for {target_str} "
        f"based STRICTLY on the ledger claims below.\n\n"
        f"RESEARCH MANDATE: {mandate_str}\n\n"
        f"VERIFIED LEDGER CLAIMS:\n{claims_text}\n\n"
        f"REMINDER: Every sentence must reference supporting claim IDs, "
        f"e.g. [claim_1]. Do not invent any numbers or outside facts."
    )

    messages: list[Message] = [Message(role="user", content=user_prompt)]
    rejection_history: list[str] = []
    latest_markdown = ""
    latest_result: ReportValidationResult | None = None

    total_attempts = 1 + max_retries

    for attempt in range(1, total_attempts + 1):
        logger.info(
            "Synthesizing report attempt",
            attempt=attempt,
            total_attempts=total_attempts,
            ticker=ticker,
        )

        response = await client.generate(
            messages=messages,
            system=SYNTHESIZE_SYSTEM_PROMPT,
        )
        latest_markdown = response.text or ""

        validation = validate_report(
            latest_markdown,
            claims,
            require_all_sections=require_all_sections,
        )
        latest_result = validation

        if validation.passed:
            logger.info(
                "Report synthesis validation PASSED",
                attempt=attempt,
                sentences=len(validation.sentence_to_claims),
            )
            claims_used = sorted(
                list(
                    {
                        cid
                        for cids in validation.sentence_to_claims.values()
                        for cid in cids
                    }
                )
            )
            return SynthesizedReport(
                markdown=latest_markdown,
                sentence_to_claims=validation.sentence_to_claims,
                claims_used=claims_used,
                validation_passed=True,
                attempts=attempt,
                rejection_reasons=rejection_history,
                ticker=ticker,
            )

        # Record validation errors
        err_summary = f"Attempt {attempt} rejected: {'; '.join(validation.errors[:3])}"
        rejection_history.append(err_summary)
        logger.warning(
            "Report synthesis validation REJECTED",
            attempt=attempt,
            errors=validation.errors,
        )

        # Prepare repair feedback message for regeneration
        if attempt < total_attempts:
            formatted_errors = "\n".join(f"- {err}" for err in validation.errors[:5])
            repair_prompt = (
                f"The previous report draft was REJECTED by validation:\n"
                f"{formatted_errors}\n\n"
                f"Please regenerate the report correcting all violations:\n"
                f"1. EVERY body sentence must cite valid claim ID(s) [claim_id].\n"
                f"2. Remove/correct all invented numbers to match cited claims.\n"
                f"3. Ensure all 9 required sections are present.\n"
            )
            messages.append(Message(role="assistant", content=latest_markdown))
            messages.append(Message(role="user", content=repair_prompt))

    # All attempts exhausted
    if raise_on_failure and latest_result and not latest_result.passed:
        raise ReportValidationError(latest_result.errors)

    active_map = latest_result.sentence_to_claims if latest_result else {}
    claims_used = sorted(list({cid for cids in active_map.values() for cid in cids}))

    return SynthesizedReport(
        markdown=latest_markdown,
        sentence_to_claims=active_map,
        claims_used=claims_used,
        validation_passed=False,
        attempts=total_attempts,
        rejection_reasons=rejection_history,
        ticker=ticker,
    )
