"""Evaluation CLI for executing benchmark runs and generating reports."""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from lekka_ai.config import get_settings
from lekka_ai.eval.dataset import EvalExample, load_dataset
from lekka_ai.eval.metrics import DEFAULT_METRICS
from lekka_ai.eval.runner import EvalRunner, EvalSummary
from lekka_ai.eval.variant import (
    AgentVariant,
    DirectVariant,
    StubVariant,
    Variant,
)
from lekka_ai.harness.models import Limits
from lekka_ai.llm.gemini import GeminiClient
from lekka_ai.logging import get_logger

logger = get_logger(__name__)
console = Console()


def save_results_csv(
    summary: EvalSummary,
    examples_map: dict[str, EvalExample],
    output_path: Path,
) -> None:
    """Export benchmark run results to a structured CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "eval_run_id",
        "example_id",
        "ticker",
        "metric",
        "expected_value",
        "variant",
        "answer",
        "factual_accuracy",
        "accuracy",
        "tool_calls",
        "tokens_in",
        "tokens_out",
        "tokens_total",
        "latency_ms",
        "cost",
        "error",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in summary.results:
            ex = examples_map.get(res.example_id)
            exp_val: Any = None
            met_name: str | None = None
            tick: str | None = None
            if ex:
                exp_val = ex.expected_value
                if exp_val is None and isinstance(ex.expected, dict):
                    exp_val = ex.expected.get("expected_value") or ex.expected.get(
                        "value"
                    )
                met_name = ex.metric
                tick = ex.ticker

            writer.writerow(
                {
                    "eval_run_id": summary.eval_run_id,
                    "example_id": res.example_id,
                    "ticker": tick or "",
                    "metric": met_name or "",
                    "expected_value": str(exp_val) if exp_val is not None else "",
                    "variant": res.variant,
                    "answer": res.answer.replace("\n", " ").strip(),
                    "factual_accuracy": res.metrics.get("factual_accuracy", 0.0),
                    "accuracy": res.metrics.get("accuracy", 0.0),
                    "tool_calls": res.tool_calls,
                    "tokens_in": res.tokens_in,
                    "tokens_out": res.tokens_out,
                    "tokens_total": res.tokens_total,
                    "latency_ms": round(res.latency_ms, 2),
                    "cost": round(res.cost, 6),
                    "error": res.error or "",
                }
            )


def save_summary_markdown(
    summary: EvalSummary,
    examples: list[EvalExample],
    output_path: Path,
    model_name: str,
) -> None:
    """Export evaluation summary and comparisons to a GitHub-flavored Markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = [
        f"# Evaluation Benchmark Summary: Baseline Results ({summary.eval_run_id})",
        "",
        f"- **Date**: {now_utc}",
        f"- **Model**: `{model_name}`",
        f"- **Dataset**: `facts_v0` ({len(examples)} examples)",
        f"- **Evaluated Variants**: {', '.join(summary.variants.keys())}",
        "",
        "## 1. Executive Summary Table",
        "",
        "| Variant | Examples | Success Rate | Factual Accuracy | "
        "Avg Tools | Avg Tokens | Total Tokens | Avg Latency (s) | Cost ($) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for agg in summary.variants.values():
        succ_pct = f"{int(agg.success_rate * 100)}%"
        fact_acc = agg.metric_averages.get("factual_accuracy", agg.accuracy)
        acc_pct = f"{int(fact_acc * 100)}%"
        avg_lat_s = f"{agg.avg_latency_ms / 1000.0:.2f}s"
        cost_str = f"${agg.total_cost:.4f}"

        lines.append(
            f"| **{agg.variant}** | {agg.count} | {succ_pct} | {acc_pct} | "
            f"{agg.avg_tool_calls:.1f} | {agg.avg_tokens:.0f} | {agg.total_tokens} | "
            f"{avg_lat_s} | {cost_str} |"
        )

    lines.extend(
        [
            "",
            "## 2. Per-Question Results Breakdown",
            "",
            "| ID | Ticker | Metric | Expected Ground Truth | "
            "Variant | Match | Tokens | Tools | Latency (s) |",
            "| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
        ]
    )

    # Organize per question
    for ex in examples:
        target_val = ex.expected_value
        if target_val is None and isinstance(ex.expected, dict):
            target_val = ex.expected.get("expected_value") or ex.expected.get("value")
        target_display = str(target_val) if target_val is not None else ""

        ex_results = [r for r in summary.results if r.example_id == ex.id]
        for res in ex_results:
            is_match = bool(res.metrics.get("factual_accuracy", 0.0) >= 0.5)
            match_badge = "✅ PASS" if is_match else "❌ FAIL"
            lat_sec = f"{res.latency_ms / 1000.0:.2f}s"

            lines.append(
                f"| `{ex.id}` | **{ex.ticker}** | `{ex.metric}` | `{target_display}` | "
                f"`{res.variant}` | {match_badge} | {res.tokens_total} | "
                f"{res.tool_calls} | {lat_sec} |"
            )

    lines.extend(
        [
            "",
            "## 3. Observations & Notes",
            "",
            "- **V1 (Direct LLM Baseline)**: Direct prompts without tools rely solely "
            "on model internal weights and memory.",
            "- **V2 (Single Agent Analyst Baseline)**: Uses autonomous tool calls "
            "against SEC filing search, XBRL facts, and market data.",
            "- **Status**: Baseline calibration complete. Zero tuning applied.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run_eval_cli(
    dataset_path: str = "eval/datasets/facts_v0.jsonl",
    variants_str: str = "direct,single",
    output_dir: str = "eval/results",
    eval_run_id: str | None = None,
    delay_between_requests: float = 1.0,
    limit: int | None = None,
) -> EvalSummary:
    """Execute evaluation benchmark and export CSV + Markdown outputs."""
    settings = get_settings()
    model_name = settings.GEMINI_MODEL

    console.print(
        Panel.fit(
            f"[bold cyan]Lekka AI Evaluation Benchmark Runner[/bold cyan]\n"
            f"[white]Dataset:[/white] {dataset_path}\n"
            f"[white]Variants:[/white] {variants_str}\n"
            f"[white]Model:[/white] {model_name}\n"
            f"[white]Output Directory:[/white] {output_dir}",
            title="[bold green]Eval Engine[/bold green]",
            border_style="cyan",
        )
    )

    # 1. Load dataset
    examples = load_dataset(dataset_path)
    if limit is not None and limit > 0:
        examples = examples[:limit]
    examples_map = {ex.id: ex for ex in examples}
    console.print(
        f"[green]Loaded {len(examples)} benchmark examples from {dataset_path}[/green]"
    )

    # 2. Build variants
    variant_names = [v.strip().lower() for v in variants_str.split(",") if v.strip()]
    variants: list[Variant] = []

    llm: GeminiClient | None = None
    if any(
        name in ("direct", "single", "direct_llm", "analyst_agent")
        for name in variant_names
    ):
        if not settings.GEMINI_API_KEY:
            console.print(
                "[bold red]Error: GEMINI_API_KEY required for live variants.[/bold red]"
            )
            raise RuntimeError("GEMINI_API_KEY missing")
        llm = GeminiClient(api_key=settings.GEMINI_API_KEY, model=model_name)

    for name in variant_names:
        if name in ("direct", "direct_llm", "v1"):
            assert llm is not None
            variants.append(DirectVariant(llm=llm, name="direct"))
        elif name in ("single", "analyst", "analyst_agent", "v2"):
            assert llm is not None
            variants.append(
                AgentVariant(
                    llm=llm,
                    name="single",
                    limits=Limits(max_iterations=6, max_seconds=60.0),
                )
            )
        elif name in ("stub", "stub_variant"):
            variants.append(StubVariant(name="stub"))
        else:
            console.print(
                f"[yellow]Warning: Unknown variant '{name}', skipping.[/yellow]"
            )

    if not variants:
        raise ValueError(f"No valid variants parsed from '{variants_str}'")

    # 3. Setup runner with database session if available
    db_session = None
    try:
        from sqlalchemy import text

        from lekka_ai.db.session import get_session

        session_gen = get_session()
        s = next(session_gen)
        s.execute(text("SELECT 1"))
        db_session = s
    except Exception:
        db_session = None

    runner = EvalRunner(metrics=DEFAULT_METRICS, session=db_session)
    run_id = eval_run_id or f"baseline_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

    # 4. Execute evaluation
    console.print(
        f"[bold cyan]Running {len(variants)} variants on "
        f"{len(examples)} questions...[/bold cyan]"
    )
    summary = await runner.run(
        dataset=examples,
        variants=variants,
        eval_run_id=run_id,
        print_summary=True,
    )

    # 5. Save outputs
    out_dir = Path(output_dir)
    csv_primary = out_dir / "baseline_results.csv"
    csv_alias = out_dir / "results.csv"
    md_primary = out_dir / "baseline_summary.md"
    md_alias = out_dir / "summary.md"

    save_results_csv(summary, examples_map, csv_primary)
    save_results_csv(summary, examples_map, csv_alias)
    console.print(f"[green]✓ Saved CSV results to {csv_primary}[/green]")

    save_summary_markdown(summary, examples, md_primary, model_name=model_name)
    save_summary_markdown(summary, examples, md_alias, model_name=model_name)
    console.print(f"[green]✓ Saved Markdown summary to {md_primary}[/green]")

    return summary


def main() -> None:
    """CLI entry point for running evaluation benchmarks."""
    parser = argparse.ArgumentParser(
        description="Lekka AI Evaluation Harness Benchmark Runner"
    )
    parser.add_argument(
        "--dataset",
        default="eval/datasets/facts_v0.jsonl",
        help="Path to JSONL benchmark dataset file",
    )
    parser.add_argument(
        "--variants",
        default="direct,single",
        help="Comma-separated variants to evaluate (direct, single, stub)",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/results",
        help="Output directory for CSV and Markdown reports",
    )
    parser.add_argument(
        "--eval-run-id",
        default=None,
        help="Optional unique evaluation run identifier",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of benchmark questions evaluated",
    )

    args = parser.parse_args()

    asyncio.run(
        run_eval_cli(
            dataset_path=args.dataset,
            variants_str=args.variants,
            output_dir=args.output_dir,
            eval_run_id=args.eval_run_id,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
