"""Pluggable evaluation metric functions and registry."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from lekka_ai.eval.dataset import EvalExample
from lekka_ai.eval.variant import RunResult

# Pluggable metric function signature: (RunResult, EvalExample) -> float | int | bool
MetricFunc = Callable[[RunResult, EvalExample], Any]


def latency_metric(result: RunResult, example: EvalExample) -> float:
    """Measure total execution latency in milliseconds."""
    return round(result.latency_ms, 2)


def cost_metric(result: RunResult, example: EvalExample) -> float:
    """Measure estimated API cost in USD."""
    return round(result.cost, 6)


def tool_calls_metric(result: RunResult, example: EvalExample) -> int:
    """Count the total number of tool calls executed."""
    return result.tool_calls


def claim_count_metric(result: RunResult, example: EvalExample) -> int:
    """Count the total number of grounded or extracted claims."""
    return len(result.claims)


def success_metric(result: RunResult, example: EvalExample) -> float:
    """Binary indicator of whether the run completed without fatal errors."""
    return 1.0 if (not result.error and bool(result.answer.strip())) else 0.0


def numeric_accuracy_metric(
    result: RunResult,
    example: EvalExample,
    tolerance: float = 0.05,
) -> float:
    """Evaluate numeric accuracy against expected ground truth value.

    Args:
        result: Evaluated run output containing answer and claims.
        example: Benchmark item with expected value.
        tolerance: Relative tolerance threshold (default: 5%).

    Returns:
        1.0 if matched within tolerance, 0.0 otherwise.
    """
    if example.expected is None:
        return 1.0 if result.answer.strip() else 0.0

    target_val: float | None = None
    target_tol: float = tolerance

    if isinstance(example.expected, (int, float)):
        target_val = float(example.expected)
    elif isinstance(example.expected, dict):
        if "value" in example.expected and example.expected["value"] is not None:
            try:
                target_val = float(example.expected["value"])
            except (ValueError, TypeError):
                target_val = None
        if "tolerance" in example.expected:
            try:
                target_tol = float(example.expected["tolerance"])
            except (ValueError, TypeError):
                pass
    elif isinstance(example.expected, str):
        try:
            target_val = float(example.expected.replace("$", "").replace(",", ""))
        except ValueError:
            target_val = None

    if target_val is None:
        # String match fallback
        exp_str = str(example.expected).strip().lower()
        ans_str = result.answer.strip().lower()
        return 1.0 if exp_str in ans_str else 0.0

    # Search extracted claims for matching numeric value
    for claim in result.claims:
        claim_vals: list[float] = []
        if isinstance(claim, dict):
            for ev in claim.get("evidence", []):
                raw = ev.get("raw_value") if isinstance(ev, dict) else None
                if (
                    isinstance(raw, dict)
                    and "value" in raw
                    and raw["value"] is not None
                ):
                    try:
                        claim_vals.append(float(raw["value"]))
                    except (ValueError, TypeError):
                        pass
                elif isinstance(raw, (int, float)):
                    claim_vals.append(float(raw))
        else:
            # Pydantic Claim model
            for ev in getattr(claim, "evidence", []):
                raw = getattr(ev, "raw_value", None)
                if (
                    isinstance(raw, dict)
                    and "value" in raw
                    and raw["value"] is not None
                ):
                    try:
                        claim_vals.append(float(raw["value"]))
                    except (ValueError, TypeError):
                        pass
                elif isinstance(raw, (int, float)):
                    claim_vals.append(float(raw))

        for v in claim_vals:
            if math.isclose(v, target_val, rel_tol=target_tol, abs_tol=1e-4):
                return 1.0

    # Text search fallback for the target number
    target_pattern = f"{target_val:.2f}"
    if target_pattern in result.answer or str(int(target_val)) in result.answer:
        return 1.0

    return 0.0


def factual_accuracy_metric(result: RunResult, example: EvalExample) -> float:
    """Evaluate factual accuracy using programmatic NumericGrader."""
    from lekka_ai.eval.grader import NumericGrader

    grader = NumericGrader()
    grade = grader.grade(result, example)
    return grade.score


DEFAULT_METRICS: dict[str, MetricFunc] = {
    "success": success_metric,
    "accuracy": numeric_accuracy_metric,
    "factual_accuracy": factual_accuracy_metric,
    "claims": claim_count_metric,
    "tool_calls": tool_calls_metric,
    "latency_ms": latency_metric,
    "cost": cost_metric,
}
