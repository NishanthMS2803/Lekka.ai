"""Evaluation harness for benchmarking specialist agent variants against baselines."""

from lekka_ai.eval.dataset import EvalExample, load_dataset, save_dataset
from lekka_ai.eval.metrics import (
    DEFAULT_METRICS,
    MetricFunc,
    claim_count_metric,
    cost_metric,
    latency_metric,
    numeric_accuracy_metric,
    success_metric,
    tool_calls_metric,
)
from lekka_ai.eval.runner import EvalRunner, EvalSummary, VariantAggregate
from lekka_ai.eval.variant import (
    AgentVariant,
    DirectVariant,
    RunResult,
    StubVariant,
    Variant,
)

__all__ = [
    "DEFAULT_METRICS",
    "AgentVariant",
    "DirectVariant",
    "EvalExample",
    "EvalRunner",
    "EvalSummary",
    "MetricFunc",
    "RunResult",
    "StubVariant",
    "Variant",
    "VariantAggregate",
    "claim_count_metric",
    "cost_metric",
    "latency_metric",
    "load_dataset",
    "numeric_accuracy_metric",
    "save_dataset",
    "success_metric",
    "tool_calls_metric",
]
