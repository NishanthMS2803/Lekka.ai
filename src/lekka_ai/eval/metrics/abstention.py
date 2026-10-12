"""Abstention quality metrics auditing precision and recall of abstentions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lekka_ai.eval.dataset import EvalExample
from lekka_ai.eval.variant import RunResult


class AbstentionQualityResult(BaseModel):
    """Confusion matrix and precision/recall evaluation for abstention decisions."""

    model_config = ConfigDict(extra="ignore")

    precision: float = Field(
        ...,
        description="Precision of abstaining (TP / (TP + FP))",
    )
    recall: float = Field(
        ...,
        description="Recall of abstaining when expected (TP / (TP + FN))",
    )
    f1: float = Field(
        ...,
        description="Harmonic mean of abstention precision and recall",
    )
    accuracy: float = Field(
        ...,
        description="Overall decision accuracy ((TP + TN) / total)",
    )
    tp: int = Field(default=0, description="Correctly abstained (expected & actual)")
    fp: int = Field(default=0, description="Abstained unnecessarily (false alarm)")
    fn: int = Field(default=0, description="Failed to abstain when expected")
    tn: int = Field(default=0, description="Correctly answered when expected")
    total: int = Field(default=0, description="Total cases evaluated")


def _resolve_did_and_should_abstain(
    record: tuple[bool, bool] | RunResult | dict[str, Any],
    example: EvalExample | None = None,
) -> tuple[bool, bool]:
    """Extract (did_abstain, should_abstain) boolean pair from input record."""
    if isinstance(record, tuple):
        return bool(record[0]), bool(record[1])

    # 1. Resolve did_abstain
    did_abstain = False
    if isinstance(record, RunResult):
        did_abstain = record.is_abstention or (record.status == "abstained")
        if not did_abstain and record.answer:
            lower = record.answer.lower()
            if any(
                phrase in lower
                for phrase in (
                    "insufficient evidence",
                    "abstained",
                    "cannot answer",
                    "unable to answer",
                    "abstain from",
                )
            ):
                did_abstain = True
    elif isinstance(record, dict):
        did_abstain = bool(
            record.get("did_abstain")
            or record.get("is_abstention")
            or (record.get("status") == "abstained")
        )

    # 2. Resolve should_abstain
    should_abstain = False
    if example is not None:
        if example.expected_abstain is not None:
            should_abstain = bool(example.expected_abstain)
        elif getattr(example, "expected_behavior", None) == "abstain":
            should_abstain = True
        elif example.type in ("unanswerable", "abstain"):
            should_abstain = True
        elif isinstance(example.expected, dict) and "abstain" in example.expected:
            should_abstain = bool(example.expected["abstain"])
        elif "abstain" in getattr(example, "tags", []):
            should_abstain = True
    elif isinstance(record, dict):
        should_abstain = bool(
            record.get("should_abstain")
            or record.get("expected_abstain")
            or (record.get("expected_behavior") == "abstain")
        )

    return did_abstain, should_abstain


def compute_abstention_quality(
    records: Sequence[tuple[bool, bool] | RunResult | dict[str, Any]],
    examples: Sequence[EvalExample] | None = None,
) -> AbstentionQualityResult:
    """Compute precision, recall, F1, and accuracy of system abstentions.

    Args:
        records: Collection of evaluated runs, dicts, or (did, should) tuples.
        examples: Optional parallel list of EvalExample benchmark items.

    Returns:
        Structured AbstentionQualityResult with complete confusion matrix.
    """
    total = len(records)
    if total == 0:
        return AbstentionQualityResult(
            precision=1.0,
            recall=1.0,
            f1=1.0,
            accuracy=1.0,
            tp=0,
            fp=0,
            fn=0,
            tn=0,
            total=0,
        )

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i, rec in enumerate(records):
        ex = examples[i] if examples and i < len(examples) else None
        did, should = _resolve_did_and_should_abstain(rec, ex)

        if did and should:
            tp += 1
        elif did and not should:
            fp += 1
        elif not did and should:
            fn += 1
        else:
            tn += 1

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 1.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 1.0

    if (precision + recall) > 0:
        f1 = round((2 * precision * recall) / (precision + recall), 4)
    else:
        f1 = 0.0

    accuracy = round((tp + tn) / total, 4)

    return AbstentionQualityResult(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        total=total,
    )


def abstention_precision(
    records: Sequence[tuple[bool, bool] | RunResult | dict[str, Any]],
    examples: Sequence[EvalExample] | None = None,
) -> float:
    """Compute the precision of abstaining when predicted."""
    res = compute_abstention_quality(records, examples)
    return res.precision


def abstention_recall(
    records: Sequence[tuple[bool, bool] | RunResult | dict[str, Any]],
    examples: Sequence[EvalExample] | None = None,
) -> float:
    """Compute the recall of abstaining when required."""
    res = compute_abstention_quality(records, examples)
    return res.recall


def abstention_quality(
    records: Sequence[tuple[bool, bool] | RunResult | dict[str, Any]],
    examples: Sequence[EvalExample] | None = None,
) -> float:
    """Compute overall abstention quality (F1 harmonic score)."""
    res = compute_abstention_quality(records, examples)
    return res.f1


def abstention_quality_metric(result: RunResult, example: EvalExample) -> float:
    """Evaluation harness MetricFunc adapter assessing single-item abstention quality.

    Returns 1.0 if the abstention decision perfectly matched expectation, 0.0 otherwise.
    """
    did, should = _resolve_did_and_should_abstain(result, example)
    return 1.0 if (did == should) else 0.0
