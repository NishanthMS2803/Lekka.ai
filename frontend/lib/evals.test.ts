import { describe, it } from "node:test";
import assert from "node:assert";
import type { EvalResultsResponse, VariantMetricsResponse } from "./api";
import { ApiClient } from "./api";
import { getVariantLabel } from "../components/EvalComparisonTable";

describe("Evaluation Benchmark Results & Variants Verification", () => {
  const mockVariantV1: VariantMetricsResponse = {
    variant: "v1_direct",
    accuracy: 0.026,
    citation_precision: 1.0,
    unsupported_facts: 0.0,
    risk_recall: 0.0,
    calibration: 0.0263,
    abstention_quality: 0.895,
    success_rate: 0.184,
    avg_cost: 0.0,
    total_cost: 0.0,
    avg_tokens: 0,
    avg_tool_calls: 0,
    unnecessary_calls: 0.18,
    avg_latency_ms: 0.1,
    min_latency_ms: 0.0,
    max_latency_ms: 0.2,
  };

  const mockVariantV5: VariantMetricsResponse = {
    variant: "v5_dynamic",
    accuracy: 0.132,
    citation_precision: 1.0,
    unsupported_facts: 0.0,
    risk_recall: 0.0,
    calibration: 0.1316,
    abstention_quality: 0.737,
    success_rate: 0.316,
    avg_cost: 0.0,
    total_cost: 0.0,
    avg_tokens: 0,
    avg_tool_calls: 0,
    unnecessary_calls: 1.37,
    avg_latency_ms: 168.7,
    min_latency_ms: 0.0,
    max_latency_ms: 598.4,
  };

  const mockVariantAblation: VariantMetricsResponse = {
    variant: "v5_no_verifier",
    accuracy: 0.132,
    citation_precision: 1.0,
    unsupported_facts: 0.0,
    risk_recall: 0.0,
    calibration: 0.1316,
    abstention_quality: 0.105,
    success_rate: 1.0,
    avg_cost: 0.0,
    total_cost: 0.0,
    avg_tokens: 0,
    avg_tool_calls: 0,
    unnecessary_calls: 0.0,
    avg_latency_ms: 0.1,
    min_latency_ms: 0.0,
    max_latency_ms: 0.1,
  };

  const mockEvalResults: EvalResultsResponse = {
    eval_run_id: "eval_all_20261012_184651",
    timestamp: "2026-10-12T18:47:00Z",
    budget_cap_usd: 1.0,
    total_spend_usd: 0.0,
    variants: [mockVariantV1, mockVariantV5, mockVariantAblation],
    report_markdown: "# Benchmark Report\n\nQuality vs Cost",
    available_plots: ["quality_vs_cost.png", "calibration_curve.png"],
  };

  it("classifies variants correctly into core vs ablation categories", () => {
    const v1Meta = getVariantLabel("v1_direct");
    assert.strictEqual(v1Meta.name, "V1 Direct");
    assert.strictEqual(v1Meta.category, "core");

    const v5Meta = getVariantLabel("v5_dynamic");
    assert.strictEqual(v5Meta.name, "V5 Orchestrator");
    assert.strictEqual(v5Meta.category, "core");

    const ablationMeta = getVariantLabel("v5_no_verifier");
    assert.strictEqual(ablationMeta.name, "V5 No Verifier");
    assert.strictEqual(ablationMeta.category, "ablation");

    const pythonMeta = getVariantLabel("v5_python_only");
    assert.strictEqual(pythonMeta.category, "ablation");
  });

  it("handles ApiClient plot URL generation safely", () => {
    const client = new ApiClient({ baseUrl: "http://localhost:8000" });
    const plotUrl = client.getEvalPlotUrl("quality_vs_cost.png");
    assert.strictEqual(plotUrl, "http://localhost:8000/evals/plots/quality_vs_cost.png");

    const calibUrl = client.getEvalPlotUrl("calibration_curve.png");
    assert.strictEqual(calibUrl, "http://localhost:8000/evals/plots/calibration_curve.png");
  });

  it("validates evaluation results structure and variant metrics", () => {
    assert.strictEqual(mockEvalResults.eval_run_id, "eval_all_20261012_184651");
    assert.strictEqual(mockEvalResults.variants?.length, 3);
    assert.strictEqual(mockEvalResults.budget_cap_usd, 1.0);
    assert.ok(mockEvalResults.available_plots?.includes("quality_vs_cost.png"));
    assert.ok(mockEvalResults.available_plots?.includes("calibration_curve.png"));

    const v5 = mockEvalResults.variants?.find((v) => v.variant === "v5_dynamic");
    assert.ok(v5);
    assert.strictEqual(v5.accuracy, 0.132);
    assert.strictEqual(v5.citation_precision, 1.0);
    assert.strictEqual(v5.unsupported_facts, 0.0);
    assert.strictEqual(v5.calibration, 0.1316);
  });
});
