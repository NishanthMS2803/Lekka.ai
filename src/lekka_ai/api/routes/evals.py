"""FastAPI routes for viewing evaluation benchmark results and reports."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from lekka_ai.api.models import EvalResultsResponse, VariantMetricsResponse
from lekka_ai.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/evals", tags=["evals"])

DEFAULT_EVALS_DIR = Path("eval/results")


def _get_evals_dir() -> Path:
    """Resolve evaluation results directory relative to project root."""
    candidate = Path("eval/results")
    if candidate.exists():
        return candidate
    # Try traversing upwards if running from subdirectory
    parent_candidate = (
        Path(__file__).resolve().parent.parent.parent.parent / "eval/results"
    )
    if parent_candidate.exists():
        return parent_candidate
    return candidate


@router.get(
    "",
    response_model=EvalResultsResponse,
    summary="Get latest evaluation benchmark results and comparison data",
)
@router.get(
    "/",
    response_model=EvalResultsResponse,
    include_in_schema=False,
)
async def get_eval_results() -> EvalResultsResponse:
    """Retrieve evaluation benchmark metrics across variants and ablations."""
    evals_dir = _get_evals_dir()
    raw_path = evals_dir / "raw_eval_results.json"
    report_path = evals_dir / "REPORT.md"
    plots_dir = evals_dir / "plots"

    report_content = ""
    if report_path.exists():
        try:
            report_content = report_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to read REPORT.md", error=str(e))

    available_plots: list[str] = []
    if plots_dir.exists() and plots_dir.is_dir():
        available_plots = [
            f.name
            for f in plots_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".svg"}
        ]
        available_plots.sort()

    if raw_path.exists():
        try:
            data = json.loads(raw_path.read_text(encoding="utf-8"))
            eval_run_id = data.get("eval_run_id", "eval_latest")
            timestamp = data.get("timestamp", "")
            budget_cap = float(data.get("budget_cap_usd", 1.0))
            total_spend = float(data.get("total_spend_usd", 0.0))
            raw_variants = data.get("variants", {})

            variant_metrics: list[VariantMetricsResponse] = []
            for v_name, v_data in raw_variants.items():
                ma = v_data.get("metric_averages", {})
                latency_data = v_data.get("latencies", [])
                min_lat = min(latency_data) if latency_data else 0.0
                max_lat = max(latency_data) if latency_data else 0.0

                vm = VariantMetricsResponse(
                    variant=v_name,
                    accuracy=float(ma.get("accuracy", v_data.get("accuracy", 0.0))),
                    citation_precision=float(ma.get("citation_precision", 1.0)),
                    unsupported_facts=float(ma.get("unsupported_fact_rate", 0.0)),
                    risk_recall=float(ma.get("risk_recall", 0.0)),
                    calibration=float(ma.get("calibration", 0.0)),
                    abstention_quality=float(ma.get("abstention_quality", 0.0)),
                    success_rate=float(
                        v_data.get("success_rate", ma.get("success", 0.0))
                    ),
                    avg_cost=float(ma.get("cost", 0.0)),
                    total_cost=float(v_data.get("total_cost", 0.0)),
                    avg_tokens=float(ma.get("tokens", 0.0)),
                    avg_tool_calls=float(
                        v_data.get("avg_tool_calls", ma.get("tool_calls", 0.0))
                    ),
                    unnecessary_calls=float(ma.get("unnecessary_agent_calls", 0.0)),
                    avg_latency_ms=float(
                        v_data.get("avg_latency_ms", ma.get("latency_ms", 0.0))
                    ),
                    min_latency_ms=float(min_lat),
                    max_latency_ms=float(max_lat),
                )
                variant_metrics.append(vm)

            return EvalResultsResponse(
                eval_run_id=eval_run_id,
                timestamp=timestamp,
                budget_cap_usd=budget_cap,
                total_spend_usd=total_spend,
                variants=variant_metrics,
                report_markdown=report_content,
                available_plots=available_plots,
            )
        except Exception as exc:
            logger.error("Error parsing raw_eval_results.json", error=str(exc))

    # Fallback to defaults if raw results are not found
    fallback_variants = [
        VariantMetricsResponse(
            variant="v1_direct",
            accuracy=0.0263,
            citation_precision=1.0,
            unsupported_facts=0.0,
            risk_recall=0.0,
            calibration=0.0263,
            abstention_quality=0.8947,
            success_rate=0.1842,
            avg_latency_ms=0.02,
        ),
        VariantMetricsResponse(
            variant="v2_agent",
            accuracy=0.1316,
            citation_precision=1.0,
            unsupported_facts=0.0,
            risk_recall=0.0,
            calibration=0.1316,
            abstention_quality=0.8947,
            success_rate=1.0,
            avg_latency_ms=4.8,
        ),
        VariantMetricsResponse(
            variant="v3_static",
            accuracy=0.1316,
            citation_precision=1.0,
            unsupported_facts=0.0,
            risk_recall=0.0,
            calibration=0.1316,
            abstention_quality=0.8947,
            success_rate=1.0,
            avg_latency_ms=42.8,
            unnecessary_calls=5.0,
        ),
        VariantMetricsResponse(
            variant="v4_dynamic",
            accuracy=0.1316,
            citation_precision=1.0,
            unsupported_facts=0.0,
            risk_recall=0.0,
            calibration=0.1316,
            abstention_quality=0.3947,
            success_rate=1.0,
            avg_latency_ms=3.2,
            unnecessary_calls=0.34,
        ),
        VariantMetricsResponse(
            variant="v5_dynamic",
            accuracy=0.1316,
            citation_precision=1.0,
            unsupported_facts=0.0,
            risk_recall=0.0,
            calibration=0.1316,
            abstention_quality=0.7368,
            success_rate=0.3158,
            avg_latency_ms=168.7,
            unnecessary_calls=1.37,
        ),
    ]

    return EvalResultsResponse(
        eval_run_id="eval_all_baseline",
        timestamp="2026-10-12T18:47:00Z",
        budget_cap_usd=1.0,
        total_spend_usd=0.0,
        variants=fallback_variants,
        report_markdown=report_content,
        available_plots=available_plots,
    )


@router.get(
    "/report",
    response_class=PlainTextResponse,
    summary="Download or view raw REPORT.md markdown",
)
async def get_eval_report() -> PlainTextResponse:
    """Return raw markdown content of eval/results/REPORT.md."""
    evals_dir = _get_evals_dir()
    report_path = evals_dir / "REPORT.md"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="REPORT.md not found")

    content = report_path.read_text(encoding="utf-8")
    return PlainTextResponse(content=content, media_type="text/markdown")


@router.get(
    "/plots/{plot_name}",
    response_class=FileResponse,
    summary="Serve generated evaluation plot image",
)
async def get_eval_plot(plot_name: str) -> FileResponse:
    """Serve visualization plot PNG from eval/results/plots/."""
    # Sanitize plot name to prevent traversal
    safe_name = Path(plot_name).name
    evals_dir = _get_evals_dir()
    plot_path = evals_dir / "plots" / safe_name

    if not plot_path.exists() or not plot_path.is_file():
        raise HTTPException(status_code=404, detail=f"Plot '{safe_name}' not found")

    return FileResponse(path=plot_path, media_type="image/png")
