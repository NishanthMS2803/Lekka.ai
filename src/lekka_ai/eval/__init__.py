"""Evaluation harness for benchmarking specialist agent variants against baselines."""

from lekka_ai.eval.dataset import EvalExample, load_dataset, save_dataset
from lekka_ai.eval.grader import (
    ExtractedNumber,
    GradeResult,
    NumericGrader,
    factual_accuracy_metric,
)
from lekka_ai.eval.metrics import (
    DEFAULT_METRICS,
    MetricFunc,
    background_claim_rate,
    citation_precision,
    claim_count_metric,
    correct_abstention_rate,
    cost_metric,
    interpretation_validity,
    latency_metric,
    numeric_accuracy_metric,
    recovery_rate,
    success_metric,
    tool_calls_metric,
    unsupported_fact_rate,
)
from lekka_ai.eval.runner import EvalRunner, EvalSummary, VariantAggregate
from lekka_ai.eval.schemas import (
    AdversarialExample,
    AdversarialExpectedBehavior,
    RiskRecallExample,
)
from lekka_ai.eval.variant import (
    AgentVariant,
    DirectVariant,
    RunResult,
    StubVariant,
    Variant,
)

__all__ = [
    "DEFAULT_METRICS",
    "AdversarialExample",
    "AdversarialExpectedBehavior",
    "AgentVariant",
    "DirectVariant",
    "EvalExample",
    "EvalRunner",
    "EvalSummary",
    "ExtractedNumber",
    "GradeResult",
    "MetricFunc",
    "NumericGrader",
    "RiskRecallExample",
    "RunResult",
    "StubVariant",
    "Variant",
    "VariantAggregate",
    "background_claim_rate",
    "citation_precision",
    "claim_count_metric",
    "correct_abstention_rate",
    "cost_metric",
    "factual_accuracy_metric",
    "interpretation_validity",
    "latency_metric",
    "load_dataset",
    "numeric_accuracy_metric",
    "recovery_rate",
    "save_dataset",
    "success_metric",
    "tool_calls_metric",
    "unsupported_fact_rate",
]
