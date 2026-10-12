"use client";

import React, { useEffect, useState, useCallback, Suspense } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  apiClient,
  RunTraceResponse,
  RunStatusResponse,
} from "@/lib/api";
import { TraceTimeline } from "@/components/TraceTimeline";

function TracePageContent() {
  const params = useParams();
  const runId = (params?.id as string) || "";

  const [traceData, setTraceData] = useState<RunTraceResponse | null>(null);
  const [statusData, setStatusData] = useState<RunStatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchTrace = useCallback(async () => {
    if (!runId) return;
    try {
      const [trace, status] = await Promise.all([
        apiClient.getTrace(runId),
        apiClient.getRun(runId),
      ]);
      setTraceData(trace);
      setStatusData(status);
      setError(null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    let isMounted = true;

    async function loadData() {
      if (!runId) return;
      try {
        const [trace, status] = await Promise.all([
          apiClient.getTrace(runId),
          apiClient.getRun(runId),
        ]);
        if (isMounted) {
          setTraceData(trace);
          setStatusData(status);
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

    loadData();

    // Poll every 3 seconds while mounted
    const interval = setInterval(loadData, 3000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [runId]);

  const cost = traceData?.cost_totals;
  const spans = traceData?.spans || [];

  const errorCount = spans.filter(
    (s) => s.status === "error" || Boolean(s.error)
  ).length;

  const denialCount = spans.filter(
    (s) =>
      s.error &&
      (s.error.toLowerCase().includes("permission") ||
        s.error.toLowerCase().includes("denied") ||
        s.error.toLowerCase().includes("disallowed"))
  ).length;

  const toolSpansCount = spans.filter(
    (s) => s.span_type === "tool" || Boolean(s.tool_name)
  ).length;

  const budgetSpansCount = spans.filter(
    (s) => s.span_type === "budget" || s.name.startsWith("budget_")
  ).length;

  return (
    <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-8 sm:px-6">
      {/* Navigation & Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 mb-6 border-b border-zinc-200 dark:border-zinc-800 gap-4">
        <div className="flex items-center gap-3">
          <Link
            href={`/runs/${runId}`}
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-zinc-600 dark:text-zinc-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
          >
            <span>←</span>
            <span>Back to Run Report</span>
          </Link>
          <span className="text-zinc-300 dark:text-zinc-700">|</span>
          <Link
            href="/"
            className="text-xs font-semibold text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200 transition-colors"
          >
            New Query
          </Link>
        </div>

        <div className="flex items-center gap-3">
          <span className="font-mono text-xs text-zinc-400 dark:text-zinc-500">
            Run: {runId}
          </span>
          {statusData && (
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase tracking-wider ${
                statusData.status === "completed"
                  ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800"
                  : statusData.status === "abstained"
                  ? "bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300 border border-amber-200 dark:border-amber-800"
                  : statusData.status === "failed"
                  ? "bg-red-50 text-red-700 dark:bg-red-950/60 dark:text-red-300 border border-red-200 dark:border-red-800"
                  : "bg-indigo-50 text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800"
              }`}
            >
              {statusData.status}
            </span>
          )}
          <button
            type="button"
            onClick={fetchTrace}
            className="p-1.5 rounded-lg text-xs bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
            title="Refresh Trace"
          >
            🔄
          </button>
        </div>
      </div>

      {/* Query Banner */}
      {statusData && (
        <div className="mb-6 p-4 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 block mb-1">
            Research Investigation Query
          </span>
          <p className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
            &ldquo;
            {((statusData.outcome as Record<string, unknown> | undefined)?.state as Record<string, unknown> | undefined)?.question
              ? String(
                  ((statusData.outcome as Record<string, unknown>)?.state as Record<string, unknown>).question
                )
              : `Research Run [${runId}]`}
            &rdquo;
          </p>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="mb-6 p-4 rounded-xl bg-rose-50 dark:bg-rose-950/50 border border-rose-200 dark:border-rose-900 text-rose-800 dark:text-rose-200 text-xs">
          <div className="font-bold mb-1">Error Loading Trace</div>
          <div>{error}</div>
        </div>
      )}

      {/* Cumulative Resource & Metric Totals Banner */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-8">
        {/* Total Tokens */}
        <div className="p-3.5 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-xs">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 block mb-1">
            Total Tokens
          </span>
          <div className="text-lg font-bold font-mono text-zinc-900 dark:text-zinc-100">
            {(cost?.total_tokens ?? 0).toLocaleString()}
          </div>
          <div className="text-[10px] text-zinc-400 font-mono mt-0.5">
            in: {((cost?.tokens_in ?? 0)).toLocaleString()} • out: {((cost?.tokens_out ?? 0)).toLocaleString()}
          </div>
        </div>

        {/* Estimated Cost */}
        <div className="p-3.5 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-xs">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 block mb-1">
            Estimated Cost
          </span>
          <div className="text-lg font-bold font-mono text-emerald-600 dark:text-emerald-400">
            ${(cost?.estimated_cost_usd ?? 0).toFixed(4)}
          </div>
          <div className="text-[10px] text-zinc-400 font-mono mt-0.5">
            USD API Spend
          </div>
        </div>

        {/* Latency */}
        <div className="p-3.5 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-xs">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 block mb-1">
            Execution Time
          </span>
          <div className="text-lg font-bold font-mono text-indigo-600 dark:text-indigo-400">
            {(cost?.elapsed_seconds ?? 0).toFixed(2)}s
          </div>
          <div className="text-[10px] text-zinc-400 font-mono mt-0.5">
            Wall-Clock Duration
          </div>
        </div>

        {/* Tool Calls Count */}
        <div className="p-3.5 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-xs">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 block mb-1">
            Tool Invocations
          </span>
          <div className="text-lg font-bold font-mono text-zinc-900 dark:text-zinc-100">
            {cost?.tool_calls ?? toolSpansCount}
          </div>
          <div className="text-[10px] text-zinc-400 font-mono mt-0.5">
            {toolSpansCount} recorded spans
          </div>
        </div>

        {/* Budget Allocation Decisions */}
        <div className="p-3.5 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-xs">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 block mb-1">
            Budget Decisions
          </span>
          <div className="text-lg font-bold font-mono text-amber-600 dark:text-amber-400">
            {budgetSpansCount}
          </div>
          <div className="text-[10px] text-zinc-400 font-mono mt-0.5">
            Planner Allocations
          </div>
        </div>

        {/* Errors & Permission Denials */}
        <div className="p-3.5 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-xs">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 block mb-1">
            Errors & Denials
          </span>
          <div
            className={`text-lg font-bold font-mono ${
              errorCount > 0 ? "text-rose-600 dark:text-rose-400" : "text-zinc-400"
            }`}
          >
            {errorCount}
          </div>
          <div className="text-[10px] text-zinc-400 font-mono mt-0.5">
            {denialCount > 0 ? `${denialCount} denials` : "0 denials"}
          </div>
        </div>
      </div>

      {/* Main Execution Timeline Waterfall */}
      {isLoading ? (
        <div className="py-20 text-center">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-indigo-600 border-r-transparent" />
          <p className="mt-3 text-xs text-zinc-500 font-medium">Loading execution trace spans...</p>
        </div>
      ) : (
        <TraceTimeline spans={spans} />
      )}
    </main>
  );
}

export default function TracePage() {
  return (
    <Suspense
      fallback={
        <div className="py-20 text-center text-xs text-zinc-500 font-medium">
          Loading execution trace...
        </div>
      }
    >
      <TracePageContent />
    </Suspense>
  );
}
