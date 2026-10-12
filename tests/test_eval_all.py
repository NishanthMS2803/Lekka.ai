"""Unit tests for full evaluation runner and report pipeline (Step 87)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lekka_ai.eval.eval_all import (
    build_evaluation_variants,
    generate_evaluation_plots,
    generate_evaluation_report_markdown,
    generate_findings_section,
    load_all_eval_datasets,
    run_full_evaluation,
)
from lekka_ai.eval.runner import EvalSummary, VariantAggregate
from lekka_ai.llm.base import LLMResponse
from lekka_ai.llm.fake import FakeLLM


def test_load_all_eval_datasets() -> None:
    """Verify load_all_eval_datasets consolidates rows from benchmark datasets."""
    datasets = load_all_eval_datasets()
    assert len(datasets) >= 20
    ids = {d.id for d in datasets}
    # Check that entries from different datasets are present
    assert any("fact" in i for i in ids)
    assert any("risk" in i for i in ids)
    assert any("fail" in i for i in ids)


def test_build_evaluation_variants() -> None:
    """Verify build_evaluation_variants constructs V1-V5 and all ablations."""
    fake_llm = FakeLLM(responses=[LLMResponse(text="Test")])
    variants = build_evaluation_variants(llm=fake_llm, memory_enabled=False)
    v_names = {v.name for v in variants}

    expected_variants = {
        "v1_direct",
        "v2_agent",
        "v3_static",
        "v4_dynamic",
        "v5_dynamic",
        "v5_no_verifier",
        "v5_no_risk",
        "v5_python_only",
    }
    assert v_names == expected_variants


def test_generate_evaluation_plots(tmp_path: Path) -> None:
    """Verify matplotlib plots are generated and saved correctly."""
    aggs = {
        "v1_direct": VariantAggregate(
            variant="v1_direct",
            count=5,
            accuracy=0.6,
            total_cost=0.002,
            avg_latency_ms=150.0,
        ),
        "v5_dynamic": VariantAggregate(
            variant="v5_dynamic",
            count=5,
            accuracy=0.9,
            total_cost=0.015,
            avg_latency_ms=850.0,
        ),
    }
    summary = EvalSummary(eval_run_id="test_plots_run", variants=aggs)
    q_plot, cal_plot = generate_evaluation_plots(summary, tmp_path)

    assert q_plot.exists()
    assert cal_plot.exists()
    assert q_plot.stat().st_size > 1000
    assert cal_plot.stat().st_size > 1000


def test_generate_findings_section() -> None:
    """Verify automated findings section analyzes diffs between components."""
    aggs = {
        "v3_static": VariantAggregate(
            variant="v3_static",
            count=5,
            accuracy=0.75,
            total_cost=0.05,
            avg_latency_ms=900.0,
            metric_averages={"unnecessary_agent_calls": 2.4},
        ),
        "v4_dynamic": VariantAggregate(
            variant="v4_dynamic",
            count=5,
            accuracy=0.80,
            total_cost=0.02,
            avg_latency_ms=500.0,
            metric_averages={"unnecessary_agent_calls": 0.2, "recovery_rate": 0.5},
        ),
        "v5_dynamic": VariantAggregate(
            variant="v5_dynamic",
            count=5,
            accuracy=0.90,
            total_cost=0.03,
            avg_latency_ms=800.0,
            avg_tokens=4500,
            metric_averages={
                "recovery_rate": 0.85,
                "citation_precision": 0.95,
                "unsupported_fact_rate": 0.05,
                "risk_recall": 0.92,
            },
        ),
        "v5_no_verifier": VariantAggregate(
            variant="v5_no_verifier",
            count=5,
            accuracy=0.88,
            total_cost=0.025,
            avg_latency_ms=700.0,
            metric_averages={
                "citation_precision": 0.70,
                "unsupported_fact_rate": 0.30,
            },
        ),
        "v5_no_risk": VariantAggregate(
            variant="v5_no_risk",
            count=5,
            accuracy=0.85,
            total_cost=0.022,
            avg_latency_ms=650.0,
            metric_averages={"risk_recall": 0.40},
        ),
        "v5_python_only": VariantAggregate(
            variant="v5_python_only",
            count=5,
            accuracy=0.84,
            total_cost=0.038,
            avg_latency_ms=1100.0,
            avg_tokens=7200,
        ),
    }
    summary = EvalSummary(eval_run_id="test_findings_run", variants=aggs)
    findings = generate_findings_section(summary)

    assert "Dynamic Query Planning (V4 vs. V3 Static)" in findings
    assert "Multi-Wave Gap Check & Re-planning (V5 vs. V4)" in findings
    assert "Claim Support Verification (V5 Full vs. No-Verifier)" in findings
    assert "Dedicated Risk Specialist Wave (V5 Full vs. No-Risk)" in findings
    assert "Specialized Quant Tools vs. General Python Sandbox" in findings


def test_generate_evaluation_report_markdown(tmp_path: Path) -> None:
    """Verify REPORT.md content contains tables, metrics, and plot links."""
    aggs = {
        "v1_direct": VariantAggregate(
            variant="v1_direct",
            count=2,
            accuracy=0.5,
            total_cost=0.001,
            avg_latency_ms=100.0,
        ),
        "v5_dynamic": VariantAggregate(
            variant="v5_dynamic",
            count=2,
            accuracy=1.0,
            total_cost=0.010,
            avg_latency_ms=600.0,
        ),
    }
    summary = EvalSummary(eval_run_id="test_report_run", variants=aggs)
    report_md = generate_evaluation_report_markdown(
        summary=summary,
        output_dir=tmp_path,
        dataset_count=2,
        budget_cap=5.0,
        total_spent=0.011,
    )

    assert "# Comprehensive Evaluation & Benchmark Report (V1–V5)" in report_md
    assert "## Quality Metrics per Variant" in report_md
    assert "## Cost & Resource Utilization per Variant" in report_md
    assert "## Latency Performance per Variant" in report_md
    assert "![Quality vs Cost](plots/quality_vs_cost.png)" in report_md
    assert "![Calibration Curve](plots/calibration_curve.png)" in report_md


@pytest.mark.asyncio
async def test_run_full_evaluation_mock(tmp_path: Path) -> None:
    """Verify run_full_evaluation executes end-to-end with mock engine."""
    sample_dataset = tmp_path / "sample.jsonl"
    sample_row = (
        '{"id": "q1", "question": "What is revenue?", '
        '"type": "factual", "expected": 100.0}\n'
    )
    sample_dataset.write_text(sample_row, encoding="utf-8")

    summary = await run_full_evaluation(
        dataset_paths=[str(sample_dataset)],
        output_dir=str(tmp_path / "results"),
        budget_cap_usd=5.0,
        use_mock=True,
    )

    assert (tmp_path / "results" / "raw_eval_results.json").exists()
    assert (tmp_path / "results" / "REPORT.md").exists()
    assert (tmp_path / "results" / "plots" / "quality_vs_cost.png").exists()
    assert (tmp_path / "results" / "plots" / "calibration_curve.png").exists()
    assert len(summary.variants) == 8
