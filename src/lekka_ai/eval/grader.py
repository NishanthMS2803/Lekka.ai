"""Programmatic numeric grader for financial facts and epistemic claims."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from lekka_ai.eval.dataset import EvalExample
from lekka_ai.eval.variant import RunResult
from lekka_ai.logging import get_logger

logger = get_logger(__name__)

# Unit scale multipliers
UNIT_MULTIPLIERS: dict[str, float] = {
    "t": 1e12,
    "tn": 1e12,
    "trillion": 1e12,
    "trillions": 1e12,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "billions": 1e9,
    "m": 1e6,
    "mn": 1e6,
    "million": 1e6,
    "millions": 1e6,
    "k": 1e3,
    "thousand": 1e3,
    "thousands": 1e3,
    "%": 0.01,
    "pct": 0.01,
    "percent": 0.01,
    "percentage": 0.01,
}

# Regex to capture numbers with optional currencies and unit suffixes
NUMBER_PATTERN = re.compile(
    r"(\(?\s*[-+]?\s*(?:\$|€|£|¥|USD\s*)?\s*[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?|\.\d+)\s*(\)?)\s*"
    r"(trillions?|billions?|millions?|thousands?|percent(?:age)?|%|[tbkm]n?|[tbkm])?"
    r"(?!\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExtractedNumber:
    """A parsed numeric quantity with extracted unit and normalized scale."""

    raw_text: str
    value: float
    base_value: float
    unit: str | None = None
    scale: float = 1.0
    is_percentage: bool = False


class GradeResult(BaseModel):
    """Detailed result of grading an evaluated output against ground truth."""

    is_correct: bool = Field(..., description="Whether candidate matches target")
    score: float = Field(..., description="Numeric accuracy score between 0.0 and 1.0")
    matched_number: ExtractedNumber | None = Field(
        default=None, description="The winning matched candidate number"
    )
    target_number: ExtractedNumber | None = Field(
        default=None, description="The parsed ground truth expected number"
    )
    relative_error: float | None = Field(
        default=None, description="Calculated relative error if matched"
    )
    source: str = Field(
        default="none",
        description="Source of match ('evidence_raw', 'claim_text', 'answer_text')",
    )
    explanation: str = Field(default="", description="Human-readable explanation")


class NumericGrader:
    """Programmatic grader extracting numbers and comparing within tolerance."""

    def __init__(self, default_tolerance: float = 0.05) -> None:
        self.default_tolerance = default_tolerance

    def parse_number_text(self, text: str) -> list[ExtractedNumber]:
        """Extract numeric quantities from text, resolving units and scales."""
        extracted: list[ExtractedNumber] = []
        clean_text = text.strip()
        if not clean_text:
            return extracted

        # Check for sentence context multipliers (e.g. "in millions")
        context_multiplier = 1.0
        context_unit: str | None = None
        lower_text = clean_text.lower()
        if "in billions" in lower_text:
            context_multiplier = 1e9
            context_unit = "B"
        elif "in millions" in lower_text:
            context_multiplier = 1e6
            context_unit = "M"
        elif "in thousands" in lower_text:
            context_multiplier = 1e3
            context_unit = "K"

        for match in NUMBER_PATTERN.finditer(clean_text):
            num_part = match.group(1).strip()
            closing_paren = match.group(2).strip()
            unit_part = match.group(3)

            is_negative = False
            if num_part.startswith("(") or (
                num_part.startswith("-") and not num_part.startswith("-.")
            ):
                is_negative = True
            elif "-" in num_part and not num_part.startswith("+"):
                is_negative = True

            # Clean punctuation and currency symbols
            cleaned_num = num_part
            for char in ["(", ")", "$", "€", "£", "¥", "USD", "+", "-", " "]:
                cleaned_num = cleaned_num.replace(char, "")
            cleaned_num = cleaned_num.replace(",", "")

            if not cleaned_num:
                continue

            try:
                raw_val = float(cleaned_num)
            except ValueError:
                continue

            if is_negative or (closing_paren == ")" and num_part.startswith("(")):
                raw_val = -abs(raw_val)

            # Determine unit scale
            scale = 1.0
            unit_str: str | None = None
            is_pct = False

            if unit_part:
                u_key = unit_part.lower()
                scale = UNIT_MULTIPLIERS.get(u_key, 1.0)
                unit_str = u_key.upper() if len(u_key) <= 2 else u_key.capitalize()
                if u_key in ("%", "pct", "percent", "percentage"):
                    is_pct = True
                    unit_str = "%"
            elif context_multiplier != 1.0:
                scale = context_multiplier
                unit_str = context_unit

            base_val = raw_val * scale
            extracted.append(
                ExtractedNumber(
                    raw_text=match.group(0).strip(),
                    value=raw_val,
                    base_value=base_val,
                    unit=unit_str,
                    scale=scale,
                    is_percentage=is_pct,
                )
            )

        return extracted

    def parse_target_value(self, target: Any) -> ExtractedNumber | None:
        """Parse ground truth expectation into normalized ExtractedNumber."""
        if target is None:
            return None

        if isinstance(target, (int, float)):
            val = float(target)
            return ExtractedNumber(
                raw_text=str(target),
                value=val,
                base_value=val,
                unit=None,
                scale=1.0,
            )

        if isinstance(target, dict):
            val_raw = target.get("expected_value")
            if val_raw is None:
                val_raw = target.get("value")
            if val_raw is None:
                return None
            try:
                val = float(val_raw)
            except (ValueError, TypeError):
                return None

            unit = target.get("unit")
            scale = 1.0
            is_pct = False
            if isinstance(unit, str):
                u_lower = unit.lower()
                scale = UNIT_MULTIPLIERS.get(u_lower, 1.0)
                if u_lower in ("%", "pct", "percent", "percentage"):
                    is_pct = True

            return ExtractedNumber(
                raw_text=str(val_raw),
                value=val,
                base_value=val * scale,
                unit=str(unit) if unit else None,
                scale=scale,
                is_percentage=is_pct,
            )

        if isinstance(target, str):
            parsed = self.parse_number_text(target)
            if parsed:
                return parsed[0]

        return None

    def compare_values(
        self,
        candidate: ExtractedNumber,
        target: ExtractedNumber,
        tolerance: float,
    ) -> tuple[bool, float | None]:
        """Compare a candidate number against the target across normalized scales.

        Returns:
            Tuple of (is_matched, relative_error).
        """

        def is_close(c_val: float, t_val: float) -> tuple[bool, float]:
            if t_val == 0.0:
                diff = abs(c_val - t_val)
                return diff <= max(tolerance, 1e-6), diff
            rel_err = abs(c_val - t_val) / abs(t_val)
            return rel_err <= tolerance, rel_err

        # 1. Base-to-base comparison (normalizes millions vs billions vs raw dollars)
        matched, err = is_close(candidate.base_value, target.base_value)
        if matched:
            return True, err

        # 2. Raw value comparison (handles unscaled values, e.g. 72.7 vs 72.7)
        matched, err = is_close(candidate.value, target.value)
        if matched:
            return True, err

        # 3. Percentage conversions (e.g. 15.4% [0.154] vs 0.154, or points vs ratio)
        if candidate.is_percentage or target.is_percentage:
            # Candidate as ratio (0.154) vs target
            matched, err = is_close(candidate.base_value, target.value)
            if matched:
                return True, err
            # Candidate as percentage points (15.4) vs target base
            matched, err = is_close(candidate.value, target.base_value)
            if matched:
                return True, err

        # 4. Cross-scale comparison (e.g. unscaled 391.0 vs 391e9 or 391M)
        standard_scales = [1e12, 1e9, 1e6, 1e3]
        for s in standard_scales:
            # Candidate in base vs target * scale
            matched, err = is_close(candidate.base_value, target.value * s)
            if matched:
                return True, err
            # Candidate scaled down vs target
            matched, err = is_close(candidate.base_value / s, target.value)
            if matched:
                return True, err

        return False, None

    def grade(
        self,
        result: RunResult,
        example: EvalExample,
        tolerance: float | None = None,
    ) -> GradeResult:
        """Evaluate RunResult accuracy against ground truth example."""
        # Determine target number
        target_obj: Any = None
        if example.expected_value is not None:
            target_obj = example.expected_value
        elif isinstance(example.expected, dict):
            target_obj = example.expected
        else:
            target_obj = example.expected

        target = self.parse_target_value(target_obj)
        if target is None:
            # Fallback to non-empty check if no numeric target
            has_ans = bool(result.answer.strip()) and not result.error
            return GradeResult(
                is_correct=has_ans,
                score=1.0 if has_ans else 0.0,
                explanation="Non-numeric target: evaluated non-empty answer.",
            )

        # Determine tolerance
        tol = self.default_tolerance
        if tolerance is not None:
            tol = tolerance
        elif example.tolerance is not None:
            tol = example.tolerance
        elif isinstance(example.expected, dict) and "tolerance" in example.expected:
            try:
                tol = float(example.expected["tolerance"])
            except (ValueError, TypeError):
                pass

        # Phase 1: Search structured claims evidence raw_values
        for claim in result.claims:
            evidences: Sequence[Any] = []
            if isinstance(claim, dict):
                evidences = claim.get("evidence", [])
            else:
                evidences = getattr(claim, "evidence", [])

            for ev in evidences:
                raw_val: Any = None
                if isinstance(ev, dict):
                    raw_val = ev.get("raw_value")
                else:
                    raw_val = getattr(ev, "raw_value", None)

                if raw_val is not None:
                    candidates: list[ExtractedNumber] = []
                    if isinstance(raw_val, (int, float)):
                        candidates.append(
                            ExtractedNumber(
                                raw_text=str(raw_val),
                                value=float(raw_val),
                                base_value=float(raw_val),
                            )
                        )
                    elif isinstance(raw_val, dict) and "value" in raw_val:
                        try:
                            v = float(raw_val["value"])
                            candidates.append(
                                ExtractedNumber(
                                    raw_text=str(v),
                                    value=v,
                                    base_value=v,
                                )
                            )
                        except (ValueError, TypeError):
                            pass
                    elif isinstance(raw_val, str):
                        candidates.extend(self.parse_number_text(raw_val))

                    for cand in candidates:
                        matched, err = self.compare_values(cand, target, tol)
                        if matched:
                            return GradeResult(
                                is_correct=True,
                                score=1.0,
                                matched_number=cand,
                                target_number=target,
                                relative_error=err,
                                source="evidence_raw",
                                explanation=(
                                    f"Matched evidence raw value {cand.raw_text} "
                                    f"against target {target.raw_text} "
                                    f"(tol={tol * 100:.1f}%)"
                                ),
                            )

        # Phase 2: Search structured claim text statements
        for claim in result.claims:
            claim_text = (
                claim.get("text", "")
                if isinstance(claim, dict)
                else getattr(claim, "text", "")
            )
            for cand in self.parse_number_text(claim_text):
                matched, err = self.compare_values(cand, target, tol)
                if matched:
                    return GradeResult(
                        is_correct=True,
                        score=1.0,
                        matched_number=cand,
                        target_number=target,
                        relative_error=err,
                        source="claim_text",
                        explanation=(
                            f"Matched claim statement number {cand.raw_text} "
                            f"against target {target.raw_text} (tol={tol * 100:.1f}%)"
                        ),
                    )

        # Phase 3: Fallback search in freeform answer text
        for cand in self.parse_number_text(result.answer):
            matched, err = self.compare_values(cand, target, tol)
            if matched:
                return GradeResult(
                    is_correct=True,
                    score=1.0,
                    matched_number=cand,
                    target_number=target,
                    relative_error=err,
                    source="answer_text",
                    explanation=(
                        f"Matched answer text number {cand.raw_text} "
                        f"against target {target.raw_text} (tol={tol * 100:.1f}%)"
                    ),
                )

        return GradeResult(
            is_correct=False,
            score=0.0,
            target_number=target,
            explanation=(
                f"No candidate numbers matched target {target.raw_text} "
                f"within tolerance {tol * 100:.1f}%."
            ),
        )


def factual_accuracy_metric(result: RunResult, example: EvalExample) -> float:
    """Evaluate factual accuracy using the programmatic NumericGrader."""
    grader = NumericGrader()
    grade = grader.grade(result, example)
    return grade.score
