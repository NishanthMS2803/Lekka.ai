"use client";

import React, { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  ScatterChart,
  Scatter,
  ZAxis,
  Cell,
} from "recharts";
import type { VariantMetricsResponse } from "@/lib/api";
import { apiClient } from "@/lib/api";
import { getVariantLabel } from "./EvalComparisonTable";

export interface EvalChartsProps {
  variants: VariantMetricsResponse[];
  availablePlots?: string[];
}

export function EvalCharts({
  variants,
  availablePlots = ["quality_vs_cost.png", "calibration_curve.png"],
}: EvalChartsProps) {
  const [activeTab, setActiveTab] = useState<"interactive" | "plots">("interactive");
  const [selectedPlot, setSelectedPlot] = useState<string | null>(null);

  // Prepare data for Quality vs Cost
  const qualityCostData = variants.map((v) => {
    const meta = getVariantLabel(v.variant);
    return {
      variant: meta.name,
      variantKey: v.variant,
      accuracyPct: +(v.accuracy * 100).toFixed(1),
      citationPrecisionPct: +(v.citation_precision * 100).toFixed(1),
      costUsd: v.avg_cost,
      // Scaled cost for visual representation or display in cents
      costCents: +(v.avg_cost * 100).toFixed(4),
      latencyMs: +(v.avg_latency_ms).toFixed(1),
      calibrationBrier: +(v.calibration).toFixed(4),
      abstentionPct: +(v.abstention_quality * 100).toFixed(1),
      category: meta.category,
    };
  });

  const getVariantColor = (variantKey: string) => {
    switch (variantKey) {
      case "v5_dynamic":
        return "#6366f1"; // Indigo
      case "v4_dynamic":
        return "#3b82f6"; // Blue
      case "v3_static":
        return "#0ea5e9"; // Sky
      case "v2_agent":
        return "#14b8a6"; // Teal
      case "v1_direct":
        return "#64748b"; // Slate
      default:
        return "#f59e0b"; // Amber for ablations
    }
  };

  return (
    <div className="w-full bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-5 shadow-sm space-y-6">
      {/* Header with Tab Navigation */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-zinc-100 dark:border-zinc-800">
        <div>
          <h3 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">
            Evaluation Visualizations
          </h3>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            Empirical quality frontier, probabilistic calibration, and resource trade-offs
          </p>
        </div>

        <div className="flex items-center gap-1.5 p-1 bg-zinc-100 dark:bg-zinc-800/80 rounded-xl text-xs font-medium">
          <button
            type="button"
            onClick={() => setActiveTab("interactive")}
            className={`px-3 py-1 rounded-lg transition-colors ${
              activeTab === "interactive"
                ? "bg-white dark:bg-zinc-900 text-indigo-600 dark:text-indigo-400 font-semibold shadow-xs"
                : "text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200"
            }`}
          >
            Interactive Charts
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("plots")}
            className={`px-3 py-1 rounded-lg transition-colors ${
              activeTab === "plots"
                ? "bg-white dark:bg-zinc-900 text-indigo-600 dark:text-indigo-400 font-semibold shadow-xs"
                : "text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200"
            }`}
          >
            Generated Benchmark Plots ({availablePlots.length})
          </button>
        </div>
      </div>

      {activeTab === "interactive" ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Chart 1: Quality vs Cost Trade-off */}
          <div className="bg-zinc-50/60 dark:bg-zinc-950/40 p-4 rounded-xl border border-zinc-200/70 dark:border-zinc-800">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h4 className="text-xs font-semibold text-zinc-900 dark:text-zinc-100 uppercase tracking-wide">
                  Quality vs. Cost / Latency Frontier
                </h4>
                <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
                  Accuracy (%) vs. Average Query Latency (ms) across pipeline variants
                </p>
              </div>
              <span className="text-[10px] px-2 py-0.5 rounded bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 font-medium border border-indigo-200/50 dark:border-indigo-800/50">
                Pareto Analysis
              </span>
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart
                  margin={{ top: 15, right: 20, bottom: 20, left: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" strokeOpacity={0.6} />
                  <XAxis
                    type="number"
                    dataKey="latencyMs"
                    name="Latency"
                    unit="ms"
                    tick={{ fontSize: 10, fill: "#71717a" }}
                    domain={[0, "auto"]}
                  />
                  <YAxis
                    type="number"
                    dataKey="accuracyPct"
                    name="Accuracy"
                    unit="%"
                    tick={{ fontSize: 10, fill: "#71717a" }}
                    domain={[0, 100]}
                  />
                  <ZAxis type="category" dataKey="variant" name="Variant" />
                  <Tooltip
                    content={({ payload }) => {
                      if (!payload || !payload.length) return null;
                      const item = payload[0].payload;
                      return (
                        <div className="bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900 p-2.5 rounded-lg text-xs shadow-lg space-y-1">
                          <div className="font-semibold text-indigo-300 dark:text-indigo-600">
                            {item.variant}
                          </div>
                          <div>Accuracy: <span className="font-mono">{item.accuracyPct}%</span></div>
                          <div>Citation Precision: <span className="font-mono">{item.citationPrecisionPct}%</span></div>
                          <div>Latency: <span className="font-mono">{item.latencyMs} ms</span></div>
                          <div>Avg Cost: <span className="font-mono">${item.costUsd.toFixed(4)}</span></div>
                        </div>
                      );
                    }}
                  />
                  <Scatter name="Variants" data={qualityCostData}>
                    {qualityCostData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={getVariantColor(entry.variantKey)}
                      />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            {/* Variant Legend Tags */}
            <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-zinc-200/50 dark:border-zinc-800 text-[10px]">
              {qualityCostData.map((d) => (
                <span
                  key={d.variantKey}
                  className="flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-zinc-700 dark:text-zinc-300"
                >
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: getVariantColor(d.variantKey) }}
                  />
                  {d.variant}: <span className="font-mono">{d.accuracyPct}%</span>
                </span>
              ))}
            </div>
          </div>

          {/* Chart 2: Calibration (Brier Score) & Abstention */}
          <div className="bg-zinc-50/60 dark:bg-zinc-950/40 p-4 rounded-xl border border-zinc-200/70 dark:border-zinc-800">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h4 className="text-xs font-semibold text-zinc-900 dark:text-zinc-100 uppercase tracking-wide">
                  Probabilistic Calibration & Abstention
                </h4>
                <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
                  Brier score calibration (lower = better) and abstention safety rate
                </p>
              </div>
              <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-50 dark:bg-emerald-950/60 text-emerald-600 dark:text-emerald-400 font-medium border border-emerald-200/50 dark:border-emerald-800/50">
                Safety & Reliability
              </span>
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={qualityCostData}
                  margin={{ top: 15, right: 10, bottom: 25, left: -10 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" strokeOpacity={0.6} />
                  <XAxis
                    dataKey="variant"
                    tick={{ fontSize: 9, fill: "#71717a" }}
                    angle={-25}
                    textAnchor="end"
                    interval={0}
                  />
                  <YAxis
                    yAxisId="left"
                    orientation="left"
                    tick={{ fontSize: 10, fill: "#71717a" }}
                    domain={[0, "auto"]}
                  />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    unit="%"
                    tick={{ fontSize: 10, fill: "#71717a" }}
                    domain={[0, 100]}
                  />
                  <Tooltip
                    content={({ payload }) => {
                      if (!payload || !payload.length) return null;
                      const item = payload[0].payload;
                      return (
                        <div className="bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900 p-2.5 rounded-lg text-xs shadow-lg space-y-1">
                          <div className="font-semibold text-indigo-300 dark:text-indigo-600">
                            {item.variant}
                          </div>
                          <div>Brier Score: <span className="font-mono">{item.calibrationBrier}</span> (lower is better)</div>
                          <div>Abstention Quality: <span className="font-mono">{item.abstentionPct}%</span></div>
                          <div>Citation Precision: <span className="font-mono">{item.citationPrecisionPct}%</span></div>
                        </div>
                      );
                    }}
                  />
                  <Legend
                    verticalAlign="top"
                    wrapperStyle={{ fontSize: "11px", paddingBottom: "8px" }}
                  />
                  <Bar
                    yAxisId="left"
                    dataKey="calibrationBrier"
                    name="Brier Score (Lower is better)"
                    fill="#f59e0b"
                    radius={[4, 4, 0, 0]}
                  />
                  <Bar
                    yAxisId="right"
                    dataKey="abstentionPct"
                    name="Abstention Quality (%)"
                    fill="#10b981"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="pt-2 border-t border-zinc-200/50 dark:border-zinc-800 text-[10px] text-zinc-500 dark:text-zinc-400">
              * Lower Brier score indicates confidence values accurately match empirical empirical correctness rates.
            </div>
          </div>
        </div>
      ) : (
        /* Generated Benchmark Matplotlib Plots */
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {availablePlots.map((plotName) => {
              const plotUrl = apiClient.getEvalPlotUrl(plotName);
              const isQuality = plotName.includes("quality");
              const title = isQuality
                ? "Quality vs. Cost Trade-off Plot"
                : "Confidence Calibration Curve";
              const desc = isQuality
                ? "Static matplotlib benchmark analysis mapping factual accuracy frontier against token cost."
                : "Reliability curve comparing model confidence predictions against observed empirical accuracy.";

              return (
                <div
                  key={plotName}
                  className="group bg-zinc-50/70 dark:bg-zinc-950/60 rounded-xl border border-zinc-200/80 dark:border-zinc-800 overflow-hidden shadow-xs hover:border-indigo-300 dark:hover:border-indigo-700 transition-all"
                >
                  <div className="p-4 border-b border-zinc-200/60 dark:border-zinc-800/80 flex items-center justify-between">
                    <div>
                      <h4 className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                        {title}
                      </h4>
                      <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
                        {desc}
                      </p>
                    </div>
                    <a
                      href={plotUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-2.5 py-1 text-[11px] font-medium rounded-lg bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-zinc-700 dark:text-zinc-300 hover:text-indigo-600 dark:hover:text-indigo-400 hover:border-indigo-300 transition-colors"
                    >
                      Open Raw ↗
                    </a>
                  </div>

                  <div
                    className="p-3 bg-zinc-100/50 dark:bg-zinc-900/50 flex items-center justify-center cursor-pointer"
                    onClick={() => setSelectedPlot(plotUrl)}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={plotUrl}
                      alt={title}
                      className="max-h-72 w-auto object-contain rounded-lg shadow-xs group-hover:scale-[1.01] transition-transform"
                    />
                  </div>

                  <div className="px-4 py-2 bg-zinc-50 dark:bg-zinc-950/80 text-[10px] text-zinc-400 font-mono flex items-center justify-between">
                    <span>eval/results/plots/{plotName}</span>
                    <button
                      type="button"
                      onClick={() => setSelectedPlot(plotUrl)}
                      className="text-indigo-600 dark:text-indigo-400 hover:underline"
                    >
                      Click to expand
                    </button>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Fullscreen Preview Modal */}
          {selectedPlot && (
            <div
              className="fixed inset-0 z-50 bg-black/75 backdrop-blur-xs flex items-center justify-center p-4"
              onClick={() => setSelectedPlot(null)}
            >
              <div
                className="max-w-4xl max-h-[90vh] bg-white dark:bg-zinc-900 rounded-2xl p-4 shadow-2xl border border-zinc-200 dark:border-zinc-800 relative flex flex-col"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex items-center justify-between pb-3 border-b border-zinc-200 dark:border-zinc-800">
                  <h4 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100 font-mono">
                    {selectedPlot}
                  </h4>
                  <button
                    type="button"
                    onClick={() => setSelectedPlot(null)}
                    className="text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 text-sm font-bold p-1"
                  >
                    ✕ Close
                  </button>
                </div>
                <div className="p-2 flex-1 overflow-auto flex items-center justify-center">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={selectedPlot}
                    alt="Expanded plot preview"
                    className="max-h-[75vh] w-auto object-contain rounded-lg"
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
