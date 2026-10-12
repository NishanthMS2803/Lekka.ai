"use client";

import React, { useEffect, useState, useCallback, Suspense } from "react";
import Link from "next/link";
import { apiClient } from "@/lib/api";
import type { EvalResultsResponse } from "@/lib/api";
import { EvalComparisonTable } from "@/components/EvalComparisonTable";
import { EvalCharts } from "@/components/EvalCharts";
import { EvalReportViewer } from "@/components/EvalReportViewer";

function EvalsPageContent() {
  const [data, setData] = useState<EvalResultsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchEvalResults = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await apiClient.getEvals();
      setData(res);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    let isMounted = true;

    async function loadInitialData() {
      try {
        const res = await apiClient.getEvals();
        if (isMounted) {
          setData(res);
          setError(null);
          setIsLoading(false);
        }
      } catch (err: unknown) {
        if (isMounted) {
          const msg = err instanceof Error ? err.message : String(err);
          setError(msg);
          setIsLoading(false);
        }
      }
    }

    loadInitialData();

    return () => {
      isMounted = false;
    };
  }, []);

  if (isLoading) {
    return (
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-8 sm:px-6 lg:px-8 space-y-6">
        <div className="p-12 rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-center space-y-3">
          <div className="inline-block h-6 w-6 rounded-full border-2 border-indigo-600 border-t-transparent animate-spin" />
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
            Loading Evaluation Results...
          </h3>
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            Fetching benchmark runs, variant comparison metrics, and calibration curves
          </p>
        </div>
      </main>
    );
  }

  if (error || !data) {
    return (
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-8 sm:px-6 lg:px-8 space-y-6">
        <div className="p-8 rounded-2xl bg-rose-50/50 dark:bg-rose-950/20 border border-rose-200 dark:border-rose-900 text-rose-900 dark:text-rose-200 space-y-4">
          <div className="flex items-center gap-2">
            <span className="text-lg">⚠️</span>
            <h3 className="text-sm font-semibold">
              Unable to load evaluation benchmark results
            </h3>
          </div>
          <p className="text-xs text-rose-700 dark:text-rose-300 font-mono">
            {error || "No data received from evaluation endpoint."}
          </p>
          <div className="flex items-center gap-3 pt-2">
            <button
              type="button"
              onClick={fetchEvalResults}
              className="px-3.5 py-1.5 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-xs font-medium transition-colors"
            >
              Retry Load
            </button>
            <Link
              href="/"
              className="px-3.5 py-1.5 rounded-xl bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 transition-colors"
            >
              Return Home
            </Link>
          </div>
        </div>
      </main>
    );
  }

  const variants = data.variants || [];
  const availablePlots = data.available_plots || ["quality_vs_cost.png", "calibration_curve.png"];

  // Calculate summary metrics
  const coreVariantsCount = variants.filter(
    (v) => !v.variant.includes("no_") && !v.variant.includes("python")
  ).length;
  const ablationsCount = variants.length - coreVariantsCount;

  return (
    <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-8 sm:px-6 lg:px-8 space-y-6">
      {/* Page Title & Evaluation Run Metadata Banner */}
      <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-6 shadow-sm space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-400 border border-indigo-200/60 dark:border-indigo-800/60">
                Evaluation Benchmark
              </span>
              <span className="text-xs text-zinc-400 dark:text-zinc-500 font-mono">
                Read-Only
              </span>
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50 mt-1">
              Autonomous Pipeline Evaluations
            </h1>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1 max-w-3xl leading-relaxed">
              Empirical performance benchmarking across architectures (V1 Direct to V5 Dynamic Orchestrator)
              and ablation variants. Evaluating factual grounding, calibration, latency, and cost frontiers.
            </p>
          </div>

          {/* Quick Action Links */}
          <div className="flex items-center gap-2 shrink-0">
            <a
              href={`${apiClient.baseUrl}/evals/report`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-medium bg-zinc-900 dark:bg-zinc-100 hover:bg-zinc-800 dark:hover:bg-zinc-200 text-white dark:text-zinc-900 shadow-sm transition-colors"
            >
              <span>View REPORT.md</span>
              <span className="text-[10px]">↗</span>
            </a>
            <Link
              href="/"
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-medium bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-300 border border-zinc-200 dark:border-zinc-700 transition-colors"
            >
              <span>← Run Research</span>
            </Link>
          </div>
        </div>

        {/* Evaluation Run Stat Badges */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-3 border-t border-zinc-100 dark:border-zinc-800/80">
          <div className="p-3 rounded-xl bg-zinc-50/70 dark:bg-zinc-950/40 border border-zinc-200/60 dark:border-zinc-800/60">
            <span className="text-[10px] uppercase tracking-wider font-semibold text-zinc-400">
              Run Identifier
            </span>
            <div className="text-xs font-mono font-medium text-zinc-800 dark:text-zinc-200 truncate mt-0.5" title={data.eval_run_id}>
              {data.eval_run_id}
            </div>
          </div>

          <div className="p-3 rounded-xl bg-zinc-50/70 dark:bg-zinc-950/40 border border-zinc-200/60 dark:border-zinc-800/60">
            <span className="text-[10px] uppercase tracking-wider font-semibold text-zinc-400">
              Variants Tested
            </span>
            <div className="text-xs font-medium text-zinc-800 dark:text-zinc-200 mt-0.5">
              {variants.length} total ({coreVariantsCount} core, {ablationsCount} ablations)
            </div>
          </div>

          <div className="p-3 rounded-xl bg-zinc-50/70 dark:bg-zinc-950/40 border border-zinc-200/60 dark:border-zinc-800/60">
            <span className="text-[10px] uppercase tracking-wider font-semibold text-zinc-400">
              Allocated Budget Cap
            </span>
            <div className="text-xs font-medium text-zinc-800 dark:text-zinc-200 mt-0.5">
              ${data.budget_cap_usd.toFixed(2)} (Spent: ${data.total_spend_usd.toFixed(4)})
            </div>
          </div>

          <div className="p-3 rounded-xl bg-zinc-50/70 dark:bg-zinc-950/40 border border-zinc-200/60 dark:border-zinc-800/60">
            <span className="text-[10px] uppercase tracking-wider font-semibold text-zinc-400">
              Benchmarked At
            </span>
            <div className="text-xs font-medium text-zinc-800 dark:text-zinc-200 mt-0.5 truncate" title={data.timestamp}>
              {data.timestamp ? new Date(data.timestamp).toLocaleString() : "Latest Release"}
            </div>
          </div>
        </div>
      </div>

      {/* 1. Variant Comparison Table */}
      <section aria-labelledby="comparison-heading">
        <h2 id="comparison-heading" className="sr-only">
          Variant Comparison Table
        </h2>
        <EvalComparisonTable variants={variants} />
      </section>

      {/* 2. Visualizations and Charts (Quality vs Cost, Calibration) */}
      <section aria-labelledby="visualizations-heading">
        <h2 id="visualizations-heading" className="sr-only">
          Evaluation Charts
        </h2>
        <EvalCharts variants={variants} availablePlots={availablePlots} />
      </section>

      {/* 3. REPORT.md Findings & Raw Markdown Viewer */}
      <section aria-labelledby="report-heading">
        <h2 id="report-heading" className="sr-only">
          Evaluation Report & Findings
        </h2>
        <EvalReportViewer
          reportMarkdown={data.report_markdown}
          evalRunId={data.eval_run_id}
        />
      </section>
    </main>
  );
}

export default function EvalsPage() {
  return (
    <Suspense
      fallback={
        <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-8 sm:px-6 lg:px-8 space-y-6">
          <div className="p-12 rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-center">
            <span className="inline-block h-5 w-5 rounded-full bg-indigo-500 animate-ping mb-3" />
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Loading evaluation results...
            </p>
          </div>
        </main>
      }
    >
      <EvalsPageContent />
    </Suspense>
  );
}
