"""Calibration metrics evaluating confidence versus empirical correctness."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lekka_ai.eval.dataset import EvalExample
from lekka_ai.eval.variant import RunResult


class ReliabilityBin(BaseModel):
    """A single confidence partition in a binned reliability diagram."""

    model_config = ConfigDict(extra="ignore")

    bin_lower: float = Field(..., description="Lower confidence boundary (inclusive)")
    bin_upper: float = Field(..., description="Upper confidence boundary (exclusive)")
    count: int = Field(default=0, description="Number of predictions in bin")
    mean_confidence: float = Field(
        default=0.0,
        description="Average predicted confidence in bin",
    )
    accuracy: float = Field(
        default=0.0,
        description="Empirical accuracy / success rate in bin",
    )
    calibration_error: float = Field(
        default=0.0,
        description="Absolute deviation between mean confidence and accuracy",
    )


class CalibrationResult(BaseModel):
    """Aggregate calibration report with Brier score, ECE, MCE, and reliability bins."""

    model_config = ConfigDict(extra="ignore")

    brier_score: float = Field(
        ...,
        description="Mean squared difference between confidence and correctness",
    )
    ece: float = Field(
        ...,
        description="Expected Calibration Error weighted across bins",
    )
    mce: float = Field(
        ...,
        description="Maximum Calibration Error across all non-empty bins",
    )
    count: int = Field(..., description="Total number of evaluated predictions")
    bins: list[ReliabilityBin] = Field(
        default_factory=list,
        description="Binned reliability table",
    )


def _extract_confidence_and_correctness(
    record: tuple[float, float | bool] | RunResult | dict[str, Any],
) -> tuple[float, float]:
    """Extract normalized (confidence, correctness) pair from record."""
    if isinstance(record, tuple):
        conf = float(record[0])
        raw_corr = record[1]
        if raw_corr is True:
            corr = 1.0
        elif raw_corr is False:
            corr = 0.0
        else:
            corr = float(raw_corr)
        return max(0.0, min(1.0, conf)), max(0.0, min(1.0, corr))

    if isinstance(record, RunResult):
        conf = (
            float(record.confidence_score)
            if record.confidence_score is not None
            else 1.0
        )
        if "accuracy" in record.metrics:
            corr = float(record.metrics["accuracy"])
        elif "success" in record.metrics:
            corr = float(record.metrics["success"])
        else:
            corr = 1.0 if not record.error and bool(record.answer.strip()) else 0.0
        return max(0.0, min(1.0, conf)), max(0.0, min(1.0, corr))

    if isinstance(record, dict):
        conf_val = record.get("confidence") or record.get("confidence_score") or 1.0
        conf = float(conf_val)
        corr_raw = (
            record.get("correctness") or record.get("accuracy") or record.get("score")
        )
        if corr_raw is None:
            corr_raw = 1.0 if record.get("is_correct") else 0.0

        if corr_raw is True:
            corr = 1.0
        elif corr_raw is False:
            corr = 0.0
        else:
            corr = float(corr_raw)
        return max(0.0, min(1.0, conf)), max(0.0, min(1.0, corr))

    return 1.0, 1.0


def brier_score(
    records: Sequence[tuple[float, float | bool] | RunResult | dict[str, Any]],
) -> float:
    """Compute the empirical Brier score: mean((confidence - correctness)^2).

    Args:
        records: Collection of prediction records or (confidence, outcome) tuples.

    Returns:
        Brier score between 0.0 (perfect) and 1.0 (completely miscalibrated).
        Returns 0.0 if input is empty.
    """
    if not records:
        return 0.0

    total_sq_err = 0.0
    for r in records:
        conf, corr = _extract_confidence_and_correctness(r)
        total_sq_err += (conf - corr) ** 2

    return round(total_sq_err / len(records), 4)


def compute_calibration(
    records: Sequence[tuple[float, float | bool] | RunResult | dict[str, Any]],
    n_bins: int = 5,
) -> CalibrationResult:
    """Compute Brier score, Expected Calibration Error, and binned reliability table.

    Args:
        records: Collection of evaluated prediction items.
        n_bins: Number of confidence bins (default 5, corresponding to 0.2 intervals).

    Returns:
        Structured CalibrationResult.
    """
    total = len(records)
    if total == 0:
        return CalibrationResult(
            brier_score=0.0,
            ece=0.0,
            mce=0.0,
            count=0,
            bins=[],
        )

    bin_width = 1.0 / n_bins
    bin_items: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    total_sq_err = 0.0

    for r in records:
        conf, corr = _extract_confidence_and_correctness(r)
        total_sq_err += (conf - corr) ** 2

        # Bin index mapping with upper edge inclusion
        idx = min(int(conf / bin_width), n_bins - 1)
        bin_items[idx].append((conf, corr))

    reliability_bins: list[ReliabilityBin] = []
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        lower = round(i * bin_width, 4)
        upper = round((i + 1) * bin_width, 4)
        items = bin_items[i]
        b_count = len(items)

        if b_count == 0:
            reliability_bins.append(
                ReliabilityBin(
                    bin_lower=lower,
                    bin_upper=upper,
                    count=0,
                    mean_confidence=round((lower + upper) / 2, 4),
                    accuracy=0.0,
                    calibration_error=0.0,
                )
            )
            continue

        mean_conf = sum(p[0] for p in items) / b_count
        acc = sum(p[1] for p in items) / b_count
        cal_err = abs(mean_conf - acc)

        ece += (b_count / total) * cal_err
        mce = max(mce, cal_err)

        reliability_bins.append(
            ReliabilityBin(
                bin_lower=lower,
                bin_upper=upper,
                count=b_count,
                mean_confidence=round(mean_conf, 4),
                accuracy=round(acc, 4),
                calibration_error=round(cal_err, 4),
            )
        )

    brier = round(total_sq_err / total, 4)
    return CalibrationResult(
        brier_score=brier,
        ece=round(ece, 4),
        mce=round(mce, 4),
        count=total,
        bins=reliability_bins,
    )


def calibration_metric(result: RunResult, example: EvalExample) -> float:
    """Evaluation harness MetricFunc computing individual Brier score contribution.

    Returns the calibration accuracy score 1.0 - (confidence - outcome)^2, where
    1.0 is perfectly calibrated.
    """
    conf = (
        float(result.confidence_score) if result.confidence_score is not None else 1.0
    )
    conf = max(0.0, min(1.0, conf))

    # Determine empirical correctness outcome
    if "accuracy" in result.metrics:
        corr = float(result.metrics["accuracy"])
    elif "factual_accuracy" in result.metrics:
        corr = float(result.metrics["factual_accuracy"])
    elif result.error or not result.answer.strip():
        corr = 0.0
    else:
        corr = 1.0

    sq_err = (conf - corr) ** 2
    return round(1.0 - sq_err, 4)
