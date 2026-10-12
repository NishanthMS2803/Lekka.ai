"""Evaluation runner executing benchmark datasets over variants."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from lekka_ai.db.models import EvalResultModel
from lekka_ai.eval.dataset import EvalExample, load_dataset
from lekka_ai.eval.metrics import DEFAULT_METRICS, MetricFunc
from lekka_ai.eval.variant import RunResult, Variant
from lekka_ai.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = get_logger(__name__)
console = Console()


class VariantAggregate(BaseModel):
    """Aggregated benchmark scores for a single variant."""

    variant: str = Field(..., description="Variant name")
    count: int = Field(default=0, description="Number of evaluated examples")
    success_rate: float = Field(default=0.0, description="Ratio of successful runs")
    accuracy: float = Field(default=0.0, description="Average numeric accuracy score")
    avg_claims: float = Field(default=0.0, description="Average number of claims")
    avg_tool_calls: float = Field(default=0.0, description="Average tool invocations")
    avg_latency_ms: float = Field(default=0.0, description="Average latency in ms")
    total_cost: float = Field(default=0.0, description="Total estimated cost in USD")
    metric_averages: dict[str, float] = Field(
        default_factory=dict,
        description="Averages for all evaluated metrics",
    )


class EvalSummary(BaseModel):
    """Complete summary of an evaluation run across all candidate variants."""

    eval_run_id: str = Field(..., description="Unique evaluation execution identifier")
    results: list[RunResult] = Field(default_factory=list)
    variants: dict[str, VariantAggregate] = Field(default_factory=dict)


class EvalRunner:
    """Orchestrates benchmark dataset execution against one or more variants."""

    def __init__(
        self,
        metrics: dict[str, MetricFunc] | None = None,
        session: Session | None = None,
    ) -> None:
        self.metrics: dict[str, MetricFunc] = (
            dict(metrics) if metrics is not None else dict(DEFAULT_METRICS)
        )
        self.session = session

    async def run(
        self,
        dataset: Sequence[EvalExample] | str | Path,
        variants: Sequence[Variant],
        *,
        eval_run_id: str | None = None,
        print_summary: bool = True,
    ) -> EvalSummary:
        """Execute dataset over candidate variants, compute metrics, and persist.

        Args:
            dataset: Sequence of EvalExample objects or path to JSONL dataset.
            variants: Sequence of Variant systems to evaluate.
            eval_run_id: Optional ID to identify this evaluation run.
            print_summary: Whether to print Rich summary table to stdout.

        Returns:
            EvalSummary containing detailed results and aggregated metrics.
        """
        # Load dataset if file path provided
        examples: list[EvalExample]
        if isinstance(dataset, (str, Path)):
            examples = load_dataset(dataset)
        else:
            examples = list(dataset)

        run_id = eval_run_id or f"eval_{uuid4().hex[:10]}"
        all_results: list[RunResult] = []

        logger.info(
            "Starting evaluation benchmark run",
            eval_run_id=run_id,
            example_count=len(examples),
            variant_count=len(variants),
        )

        for variant in variants:
            for example in examples:
                try:
                    res = await variant.run(example)
                except Exception as exc:
                    logger.error(
                        "Variant execution crashed",
                        variant=variant.name,
                        example_id=example.id,
                        error=str(exc),
                    )
                    res = RunResult(
                        example_id=example.id,
                        variant=variant.name,
                        question=example.question,
                        answer="",
                        claims=[],
                        cost=0.0,
                        latency_ms=0.0,
                        tool_calls=0,
                        error=str(exc),
                    )

                # Compute pluggable metrics
                for metric_name, metric_fn in self.metrics.items():
                    try:
                        res.metrics[metric_name] = metric_fn(res, example)
                    except Exception as m_exc:
                        logger.warning(
                            "Metric computation failed",
                            metric=metric_name,
                            error=str(m_exc),
                        )
                        res.metrics[metric_name] = 0.0

                all_results.append(res)

                # Persist to eval_results database table if session available
                if self.session is not None:
                    serialized_claims = [
                        c.model_dump() if isinstance(c, BaseModel) else c
                        for c in res.claims
                    ]
                    db_row = EvalResultModel(
                        eval_run_id=run_id,
                        example_id=example.id,
                        variant=variant.name,
                        question=example.question,
                        answer=res.answer,
                        claims=serialized_claims,
                        tool_calls=res.tool_calls,
                        latency_ms=res.latency_ms,
                        cost=res.cost,
                        metrics=res.metrics,
                        error=res.error,
                    )
                    self.session.add(db_row)

        if self.session is not None:
            self.session.commit()

        # Compute variant aggregates
        variant_aggregates: dict[str, VariantAggregate] = {}
        for variant in variants:
            v_results = [r for r in all_results if r.variant == variant.name]
            count = len(v_results)
            if count == 0:
                continue

            success_sum = sum(float(r.metrics.get("success", 0.0)) for r in v_results)
            accuracy_sum = sum(float(r.metrics.get("accuracy", 0.0)) for r in v_results)
            claims_sum = sum(len(r.claims) for r in v_results)
            tools_sum = sum(r.tool_calls for r in v_results)
            latency_sum = sum(r.latency_ms for r in v_results)
            cost_sum = sum(r.cost for r in v_results)

            metric_averages: dict[str, float] = {}
            for m_name in self.metrics:
                m_vals = [
                    float(r.metrics[m_name])
                    for r in v_results
                    if m_name in r.metrics
                    and isinstance(r.metrics[m_name], (int, float))
                ]
                if m_vals:
                    metric_averages[m_name] = round(sum(m_vals) / len(m_vals), 4)

            variant_aggregates[variant.name] = VariantAggregate(
                variant=variant.name,
                count=count,
                success_rate=round(success_sum / count, 4),
                accuracy=round(accuracy_sum / count, 4),
                avg_claims=round(claims_sum / count, 2),
                avg_tool_calls=round(tools_sum / count, 2),
                avg_latency_ms=round(latency_sum / count, 2),
                total_cost=round(cost_sum, 6),
                metric_averages=metric_averages,
            )

        summary = EvalSummary(
            eval_run_id=run_id,
            results=all_results,
            variants=variant_aggregates,
        )

        if print_summary:
            self.print_summary_table(summary)

        return summary

    def print_summary_table(self, summary: EvalSummary) -> None:
        """Render Rich comparison summary table across evaluated variants."""
        table = Table(
            title=f"Evaluation Benchmark Summary (Run ID: {summary.eval_run_id})",
            title_style="bold cyan",
            header_style="bold magenta",
            show_header=True,
            expand=True,
        )
        table.add_column("Variant", style="bold white", width=18)
        table.add_column("Count", justify="right", width=7)
        table.add_column("Success Rate", justify="right", width=14)
        table.add_column("Accuracy", justify="right", width=11)
        table.add_column("Avg Claims", justify="right", width=12)
        table.add_column("Avg Tools", justify="right", width=11)
        table.add_column("Avg Latency", justify="right", width=14)
        table.add_column("Total Cost", justify="right", width=13)

        for agg in summary.variants.values():
            succ_pct = f"{int(agg.success_rate * 100)}%"
            acc_pct = f"{int(agg.accuracy * 100)}%"
            lat_str = f"{agg.avg_latency_ms:.1f}ms"
            cost_str = f"${agg.total_cost:.4f}"

            table.add_row(
                agg.variant,
                str(agg.count),
                succ_pct,
                acc_pct,
                f"{agg.avg_claims:.1f}",
                f"{agg.avg_tool_calls:.1f}",
                lat_str,
                cost_str,
            )

        console.print()
        console.print(table)
        console.print()
