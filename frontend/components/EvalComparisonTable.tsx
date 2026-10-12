"use client";

import React, { useState, useMemo } from "react";
import type { VariantMetricsResponse } from "@/lib/api";

export interface EvalComparisonTableProps {
  variants: VariantMetricsResponse[];
}

type SortField =
  | "variant"
  | "accuracy"
  | "citation_precision"
  | "unsupported_facts"
  | "risk_recall"
  | "calibration"
  | "abstention_quality"
  | "success_rate"
  | "avg_latency_ms"
  | "avg_cost";

type SortDirection = "asc" | "desc";
type FilterCategory = "all" | "core" | "ablations";

export function getVariantLabel(variantKey: string): {
  name: string;
  badge: string;
  category: "core" | "ablation";
  description: string;
} {
  switch (variantKey) {
    case "v1_direct":
      return {
        name: "V1 Direct",
        badge: "Direct Prompting",
        category: "core",
        description: "Zero-shot direct LLM prompting with no tool calls",
      };
    case "v2_agent":
      return {
        name: "V2 ReAct",
        badge: "Sequential Loop",
        category: "core",
        description: "Single-agent tool-augmented ReAct execution loop",
      };
    case "v3_static":
      return {
        name: "V3 Plan-Solve",
        badge: "Static 5-Agent DAG",
        category: "core",
        description: "Fixed topology dispatching all 5 specialist agents in parallel",
      };
    case "v4_dynamic":
      return {
        name: "V4 Dynamic",
        badge: "Selective Dispatch",
        category: "core",
        description: "Dynamic query decomposition & selective agent routing",
      };
    case "v5_dynamic":
      return {
        name: "V5 Orchestrator",
        badge: "Full Multi-Wave",
        category: "core",
        description: "Multi-wave dispatch with gap check, risk wave & claim verifier",
      };
    case "v5_no_verifier":
      return {
        name: "V5 No Verifier",
        badge: "Ablation",
        category: "ablation",
        description: "V5 pipeline with claim verification and citation audits disabled",
      };
    case "v5_no_risk":
      return {
        name: "V5 No Risk Wave",
        badge: "Ablation",
        category: "ablation",
        description: "V5 pipeline omitting dedicated contrarian risk analysis wave",
      };
    case "v5_python_only":
      return {
        name: "V5 Python Only",
        badge: "Ablation",
        category: "ablation",
        description: "V5 pipeline replacing quant tool primitives with raw python execution",
      };
    default:
      return {
        name: variantKey,
        badge: "Variant",
        category: variantKey.includes("no_") || variantKey.includes("python")
          ? "ablation"
          : "core",
        description: `Pipeline configuration ${variantKey}`,
      };
  }
}

export function EvalComparisonTable({ variants }: EvalComparisonTableProps) {
  const [filter, setFilter] = useState<FilterCategory>("all");
  const [sortField, setSortField] = useState<SortField>("accuracy");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");

  const filteredVariants = useMemo(() => {
    return variants.filter((v) => {
      const meta = getVariantLabel(v.variant);
      if (filter === "core") return meta.category === "core";
      if (filter === "ablations") return meta.category === "ablation";
      return true;
    });
  }, [variants, filter]);

  const sortedVariants = useMemo(() => {
    return [...filteredVariants].sort((a, b) => {
      const mult = sortDirection === "asc" ? 1 : -1;
      if (sortField === "variant") {
        return mult * a.variant.localeCompare(b.variant);
      }
      const valA = (a[sortField] as number) ?? 0;
      const valB = (b[sortField] as number) ?? 0;
      return mult * (valA - valB);
    });
  }, [filteredVariants, sortField, sortDirection]);

  // Find best values for visual highlighting
  const bestAccuracy = Math.max(...variants.map((v) => v.accuracy || 0));
  const bestCitationPrecision = Math.max(
    ...variants.map((v) => v.citation_precision || 0)
  );
  const minUnsupported = Math.min(...variants.map((v) => v.unsupported_facts || 0));
  const minCalibration = Math.min(...variants.map((v) => v.calibration ?? 1));

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      // For calibration and latency, lower is typically preferred
      if (field === "calibration" || field === "avg_latency_ms" || field === "unsupported_facts") {
        setSortDirection("asc");
      } else {
        setSortDirection("desc");
      }
    }
  };

  const getSortIcon = (field: SortField) => {
    if (sortField !== field) return "↕";
    return sortDirection === "asc" ? "▲" : "▼";
  };

  return (
    <div className="w-full bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-5 shadow-sm space-y-4">
      {/* Table Header Controls */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-zinc-100 dark:border-zinc-800">
        <div>
          <h3 className="text-base font-semibold text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
            <span>Variant Performance Comparison</span>
            <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300">
              {variants.length} configurations
            </span>
          </h3>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            Empirical benchmark metrics evaluated across finance questions with ground-truth verification
          </p>
        </div>

        {/* Filter Pills */}
        <div className="flex items-center gap-1.5 p-1 bg-zinc-100 dark:bg-zinc-800/80 rounded-xl text-xs font-medium">
          <button
            type="button"
            onClick={() => setFilter("all")}
            className={`px-3 py-1 rounded-lg transition-colors ${
              filter === "all"
                ? "bg-white dark:bg-zinc-900 text-indigo-600 dark:text-indigo-400 font-semibold shadow-xs"
                : "text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200"
            }`}
          >
            All Variants ({variants.length})
          </button>
          <button
            type="button"
            onClick={() => setFilter("core")}
            className={`px-3 py-1 rounded-lg transition-colors ${
              filter === "core"
                ? "bg-white dark:bg-zinc-900 text-indigo-600 dark:text-indigo-400 font-semibold shadow-xs"
                : "text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200"
            }`}
          >
            Core (V1–V5)
          </button>
          <button
            type="button"
            onClick={() => setFilter("ablations")}
            className={`px-3 py-1 rounded-lg transition-colors ${
              filter === "ablations"
                ? "bg-white dark:bg-zinc-900 text-indigo-600 dark:text-indigo-400 font-semibold shadow-xs"
                : "text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200"
            }`}
          >
            Ablations
          </button>
        </div>
      </div>

      {/* Table Container */}
      <div className="overflow-x-auto rounded-xl border border-zinc-200/80 dark:border-zinc-800">
        <table className="w-full text-left text-xs divide-y divide-zinc-200 dark:divide-zinc-800">
          <thead className="bg-zinc-50/90 dark:bg-zinc-950/60 text-zinc-600 dark:text-zinc-400 font-medium">
            <tr>
              <th
                scope="col"
                className="py-3 px-3.5 cursor-pointer hover:text-zinc-900 dark:hover:text-zinc-100 select-none whitespace-nowrap"
                onClick={() => handleSort("variant")}
              >
                Variant <span className="text-[10px] opacity-70">{getSortIcon("variant")}</span>
              </th>
              <th
                scope="col"
                className="py-3 px-3 text-right cursor-pointer hover:text-zinc-900 dark:hover:text-zinc-100 select-none whitespace-nowrap"
                onClick={() => handleSort("accuracy")}
              >
                Accuracy <span className="text-[10px] opacity-70">{getSortIcon("accuracy")}</span>
              </th>
              <th
                scope="col"
                className="py-3 px-3 text-right cursor-pointer hover:text-zinc-900 dark:hover:text-zinc-100 select-none whitespace-nowrap"
                onClick={() => handleSort("citation_precision")}
              >
                Citation Prec. <span className="text-[10px] opacity-70">{getSortIcon("citation_precision")}</span>
              </th>
              <th
                scope="col"
                className="py-3 px-3 text-right cursor-pointer hover:text-zinc-900 dark:hover:text-zinc-100 select-none whitespace-nowrap"
                onClick={() => handleSort("unsupported_facts")}
              >
                Unsupp. Facts <span className="text-[10px] opacity-70">{getSortIcon("unsupported_facts")}</span>
              </th>
              <th
                scope="col"
                className="py-3 px-3 text-right cursor-pointer hover:text-zinc-900 dark:hover:text-zinc-100 select-none whitespace-nowrap"
                onClick={() => handleSort("risk_recall")}
              >
                Risk Recall <span className="text-[10px] opacity-70">{getSortIcon("risk_recall")}</span>
              </th>
              <th
                scope="col"
                className="py-3 px-3 text-right cursor-pointer hover:text-zinc-900 dark:hover:text-zinc-100 select-none whitespace-nowrap"
                onClick={() => handleSort("calibration")}
              >
                Brier Score <span className="text-[10px] opacity-70">{getSortIcon("calibration")}</span>
              </th>
              <th
                scope="col"
                className="py-3 px-3 text-right cursor-pointer hover:text-zinc-900 dark:hover:text-zinc-100 select-none whitespace-nowrap"
                onClick={() => handleSort("abstention_quality")}
              >
                Abstention <span className="text-[10px] opacity-70">{getSortIcon("abstention_quality")}</span>
              </th>
              <th
                scope="col"
                className="py-3 px-3 text-right cursor-pointer hover:text-zinc-900 dark:hover:text-zinc-100 select-none whitespace-nowrap"
                onClick={() => handleSort("success_rate")}
              >
                Success <span className="text-[10px] opacity-70">{getSortIcon("success_rate")}</span>
              </th>
              <th
                scope="col"
                className="py-3 px-3 text-right cursor-pointer hover:text-zinc-900 dark:hover:text-zinc-100 select-none whitespace-nowrap"
                onClick={() => handleSort("avg_latency_ms")}
              >
                Latency <span className="text-[10px] opacity-70">{getSortIcon("avg_latency_ms")}</span>
              </th>
              <th
                scope="col"
                className="py-3 px-3.5 text-right cursor-pointer hover:text-zinc-900 dark:hover:text-zinc-100 select-none whitespace-nowrap"
                onClick={() => handleSort("avg_cost")}
              >
                Cost / Q <span className="text-[10px] opacity-70">{getSortIcon("avg_cost")}</span>
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/80 bg-white dark:bg-zinc-900 font-mono">
            {sortedVariants.map((item) => {
              const meta = getVariantLabel(item.variant);
              const isHighlight = item.variant === "v5_dynamic";
              const isAccuracyLeader = item.accuracy >= bestAccuracy && bestAccuracy > 0;
              const isCitationLeader =
                item.citation_precision >= bestCitationPrecision && bestCitationPrecision > 0;
              const isZeroUnsupported = item.unsupported_facts === minUnsupported;

              return (
                <tr
                  key={item.variant}
                  className={`hover:bg-zinc-50/70 dark:hover:bg-zinc-800/50 transition-colors ${
                    isHighlight
                      ? "bg-indigo-50/40 dark:bg-indigo-950/20 font-medium"
                      : ""
                  }`}
                >
                  {/* Variant Column */}
                  <td className="py-3 px-3.5 whitespace-nowrap">
                    <div className="flex items-center gap-2 font-sans">
                      <span className="font-semibold text-zinc-900 dark:text-zinc-100 text-xs">
                        {meta.name}
                      </span>
                      <span
                        className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                          meta.category === "core"
                            ? isHighlight
                              ? "bg-indigo-100 text-indigo-700 dark:bg-indigo-900/60 dark:text-indigo-300 font-bold"
                              : "bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300"
                            : "bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300"
                        }`}
                      >
                        {meta.badge}
                      </span>
                    </div>
                    <div className="text-[10px] font-sans text-zinc-500 dark:text-zinc-400 truncate max-w-xs">
                      {meta.description}
                    </div>
                  </td>

                  {/* Accuracy */}
                  <td className="py-3 px-3 text-right">
                    <span
                      className={`inline-flex items-center gap-1 ${
                        isAccuracyLeader
                          ? "text-emerald-600 dark:text-emerald-400 font-semibold"
                          : "text-zinc-700 dark:text-zinc-300"
                      }`}
                    >
                      {(item.accuracy * 100).toFixed(1)}%
                      {isAccuracyLeader && <span className="text-[10px]">★</span>}
                    </span>
                  </td>

                  {/* Citation Precision */}
                  <td className="py-3 px-3 text-right">
                    <span
                      className={`${
                        isCitationLeader
                          ? "text-emerald-600 dark:text-emerald-400 font-semibold"
                          : "text-zinc-700 dark:text-zinc-300"
                      }`}
                    >
                      {(item.citation_precision * 100).toFixed(1)}%
                    </span>
                  </td>

                  {/* Unsupported Facts */}
                  <td className="py-3 px-3 text-right">
                    <span
                      className={`${
                        isZeroUnsupported
                          ? "text-emerald-600 dark:text-emerald-400 font-semibold"
                          : item.unsupported_facts > 0.05
                          ? "text-rose-600 dark:text-rose-400 font-semibold"
                          : "text-zinc-700 dark:text-zinc-300"
                      }`}
                    >
                      {(item.unsupported_facts * 100).toFixed(1)}%
                    </span>
                  </td>

                  {/* Risk Recall */}
                  <td className="py-3 px-3 text-right text-zinc-700 dark:text-zinc-300">
                    {(item.risk_recall * 100).toFixed(1)}%
                  </td>

                  {/* Calibration (Brier Score) */}
                  <td className="py-3 px-3 text-right">
                    <span
                      className={`${
                        item.calibration <= minCalibration
                          ? "text-emerald-600 dark:text-emerald-400 font-semibold"
                          : "text-zinc-700 dark:text-zinc-300"
                      }`}
                    >
                      {item.calibration.toFixed(4)}
                    </span>
                  </td>

                  {/* Abstention Quality */}
                  <td className="py-3 px-3 text-right text-zinc-700 dark:text-zinc-300">
                    {(item.abstention_quality * 100).toFixed(1)}%
                  </td>

                  {/* Success Rate */}
                  <td className="py-3 px-3 text-right text-zinc-700 dark:text-zinc-300">
                    {(item.success_rate * 100).toFixed(1)}%
                  </td>

                  {/* Latency */}
                  <td className="py-3 px-3 text-right text-zinc-600 dark:text-zinc-400">
                    {item.avg_latency_ms.toFixed(1)} ms
                  </td>

                  {/* Cost */}
                  <td className="py-3 px-3.5 text-right font-medium text-zinc-800 dark:text-zinc-200">
                    ${item.avg_cost.toFixed(4)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="flex flex-wrap items-center justify-between text-[11px] text-zinc-500 dark:text-zinc-400 pt-1">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="text-emerald-600 dark:text-emerald-400 font-bold">★</span> Top accuracy / precision
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full bg-indigo-500" /> V5 Multi-wave production pipeline
          </span>
        </div>
        <div className="text-zinc-400 dark:text-zinc-500">
          * Brier score measures probabilistic calibration (lower is better, 0.00 = perfect calibration)
        </div>
      </div>
    </div>
  );
}
