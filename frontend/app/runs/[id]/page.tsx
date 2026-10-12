"use client";

import React, { useEffect, useState, Suspense } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  apiClient,
  RunStatusResponse,
  RunReportResponse,
  createRunEventSource,
  RunEventPayload,
} from "@/lib/api";

function RunDetailContent() {
  const params = useParams();
  const runId = (params?.id as string) || "";

  const [statusData, setStatusData] = useState<RunStatusResponse | null>(null);
  const [reportData, setReportData] = useState<RunReportResponse | null>(null);
  const [events, setEvents] = useState<RunEventPayload[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Poll status periodically or fetch initial state
  useEffect(() => {
    if (!runId) return;

    let isMounted = true;

    async function fetchInitial() {
      try {
        const status = await apiClient.getRun(runId);
        if (isMounted) {
          setStatusData(status);
          setIsLoading(false);
          if (status.status === "completed") {
            try {
              const rep = await apiClient.getReport(runId);
              if (isMounted) setReportData(rep);
            } catch {
              // Report may not be ready yet
            }
          }
        }
      } catch (err: unknown) {
        if (isMounted) {
          setError(err instanceof Error ? err.message : "Failed to load run");
          setIsLoading(false);
        }
      }
    }

    fetchInitial();

    // Subscribe to real-time events via SSE
    const unsubscribe = createRunEventSource(runId, {
      onEvent: (event) => {
        if (!isMounted) return;
        setEvents((prev) => [...prev, event]);
      },
      onReportReady: async () => {
        try {
          const rep = await apiClient.getReport(runId);
          if (isMounted) setReportData(rep);
          const st = await apiClient.getRun(runId);
          if (isMounted) setStatusData(st);
        } catch {
          // ignore
        }
      },
      onComplete: async () => {
        try {
          const st = await apiClient.getRun(runId);
          if (isMounted) setStatusData(st);
        } catch {
          // ignore
        }
      },
      onError: () => {
        // Fallback polling if SSE is not connected
      },
    });

    // Fallback polling every 3s if still pending or running
    const interval = setInterval(async () => {
      try {
        const current = await apiClient.getRun(runId);
        if (isMounted) {
          setStatusData(current);
          if (current.status === "completed" && !reportData) {
            const rep = await apiClient.getReport(runId);
            setReportData(rep);
          }
        }
      } catch {
        // ignore polling errors
      }
    }, 3000);

    return () => {
      isMounted = false;
      unsubscribe();
      clearInterval(interval);
    };
  }, [runId, reportData]);

  const currentStatus = statusData?.status || "pending";
  const isClarification = currentStatus === "needs_clarification";

  return (
    <main className="flex-1 max-w-5xl mx-auto w-full px-4 py-8 sm:px-6">
      {/* Navigation Header */}
      <div className="flex items-center justify-between mb-8 pb-4 border-b border-zinc-200 dark:border-zinc-800">
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-xs font-semibold text-zinc-600 dark:text-zinc-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
        >
          <span>←</span>
          <span>New Research Query</span>
        </Link>
        <span className="font-mono text-xs text-zinc-400 dark:text-zinc-500">
          ID: {runId}
        </span>
      </div>

      {/* Loading Banner */}
      {isLoading && !statusData && (
        <div className="mb-6 p-4 rounded-xl bg-indigo-50 dark:bg-indigo-950/40 border border-indigo-200 dark:border-indigo-900/60 text-indigo-800 dark:text-indigo-300 text-xs flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-indigo-500 animate-ping" />
          <span>Connecting to orchestrator and loading run status...</span>
        </div>
      )}

      {/* Error Banner */}
      {error && (
        <div className="mb-6 p-4 rounded-xl bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900/60 text-red-800 dark:text-red-300 text-sm">
          <p className="font-semibold">Unable to fetch run status</p>
          <p className="text-xs mt-1">{error}</p>
        </div>
      )}

      {/* Clarification Alert */}
      {isClarification && (
        <div className="mb-8 p-6 rounded-2xl bg-amber-50 dark:bg-amber-950/40 border border-amber-300 dark:border-amber-800 text-amber-900 dark:text-amber-200 shadow-sm">
          <div className="flex items-start gap-4">
            <span className="text-2xl">⚠️</span>
            <div className="flex-1">
              <h2 className="text-base font-bold text-amber-900 dark:text-amber-100">
                Research Stopped: Clarification Required
              </h2>
              <p className="text-xs text-amber-800 dark:text-amber-300 mt-1 leading-relaxed">
                The orchestrator query parser flagged that the question does not specify a target
                ticker symbol or company name. Multi-agent execution was stopped before executing
                unbounded tool calls.
              </p>
              <div className="mt-4 flex items-center gap-3">
                <Link
                  href="/"
                  className="px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-700 text-white text-xs font-semibold transition-colors"
                >
                  Return to Ask Page to Clarify
                </Link>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Status Header Card */}
      <div className="p-6 rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-sm mb-8">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400 dark:text-zinc-500">
              Orchestrator Run Status
            </span>
            <h1 className="text-xl sm:text-2xl font-bold text-zinc-900 dark:text-zinc-100 mt-1">
              Run <span className="font-mono text-base font-normal text-zinc-500">#{runId.slice(0, 8)}</span>
            </h1>
          </div>

          <div className="flex items-center gap-3">
            <span
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold uppercase tracking-wider ${
                currentStatus === "completed"
                  ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800"
                  : currentStatus === "needs_clarification"
                  ? "bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300 border border-amber-200 dark:border-amber-800"
                  : currentStatus === "failed"
                  ? "bg-red-50 text-red-700 dark:bg-red-950/60 dark:text-red-300 border border-red-200 dark:border-red-800"
                  : "bg-indigo-50 text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800"
              }`}
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  currentStatus === "completed"
                    ? "bg-emerald-500"
                    : currentStatus === "needs_clarification"
                    ? "bg-amber-500"
                    : currentStatus === "failed"
                    ? "bg-red-500"
                    : "bg-indigo-500 animate-pulse"
                }`}
              />
              {currentStatus}
            </span>
          </div>
        </div>

        {statusData && (
          <div className="mt-6 pt-4 border-t border-zinc-100 dark:border-zinc-800 grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
            <div>
              <span className="text-zinc-400 dark:text-zinc-500 block">Rounds Executed</span>
              <span className="font-semibold text-zinc-800 dark:text-zinc-200">
                {statusData.rounds ?? 0}
              </span>
            </div>
            <div>
              <span className="text-zinc-400 dark:text-zinc-500 block">Stop Reason</span>
              <span className="font-semibold text-zinc-800 dark:text-zinc-200">
                {statusData.stop_reason || (currentStatus === "running" ? "Active" : "None")}
              </span>
            </div>
            <div>
              <span className="text-zinc-400 dark:text-zinc-500 block">Streamed Events</span>
              <span className="font-semibold text-zinc-800 dark:text-zinc-200">
                {events.length}
              </span>
            </div>
            <div>
              <span className="text-zinc-400 dark:text-zinc-500 block">Report Status</span>
              <span className="font-semibold text-zinc-800 dark:text-zinc-200">
                {reportData ? "Synthesized" : "In Progress"}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Synthesized Report Content if Ready */}
      {reportData && (
        <div className="mb-8 p-6 sm:p-8 rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-sm">
          <div className="flex items-center justify-between mb-4 pb-2 border-b border-zinc-100 dark:border-zinc-800">
            <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
              Synthesized Research Report
            </h2>
            {reportData.confidence_score !== null && reportData.confidence_score !== undefined && (
              <span className="text-xs px-2.5 py-1 rounded-md bg-zinc-100 dark:bg-zinc-800 font-semibold text-zinc-700 dark:text-zinc-300">
                Confidence: {(reportData.confidence_score * 100).toFixed(0)}%
              </span>
            )}
          </div>
          <div className="prose dark:prose-invert max-w-none text-xs sm:text-sm leading-relaxed whitespace-pre-wrap font-sans">
            {reportData.markdown}
          </div>
        </div>
      )}

      {/* Live Event Stream Log */}
      <div className="p-6 rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-sm">
        <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100 mb-3">
          Execution Event Feed ({events.length})
        </h3>
        {events.length === 0 ? (
          <p className="text-xs text-zinc-500 dark:text-zinc-400 italic">
            Waiting for orchestration events from backend...
          </p>
        ) : (
          <div className="space-y-2 max-h-80 overflow-y-auto font-mono text-[11px]">
            {events.map((evt, idx) => (
              <div
                key={idx}
                className="p-2.5 rounded-lg bg-zinc-50 dark:bg-zinc-950 border border-zinc-200/60 dark:border-zinc-800 flex items-start justify-between gap-2"
              >
                <div>
                  <span className="text-indigo-600 dark:text-indigo-400 font-bold mr-2">
                    {evt.event}
                  </span>
                  <span className="text-zinc-600 dark:text-zinc-400">
                    {JSON.stringify(evt.data)}
                  </span>
                </div>
                <span className="text-zinc-400 dark:text-zinc-600 shrink-0">
                  {evt.timestamp ? new Date(evt.timestamp).toLocaleTimeString() : ""}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}

export default function RunDetailPage() {
  return (
    <Suspense
      fallback={
        <main className="flex-1 max-w-5xl mx-auto w-full px-4 py-8 sm:px-6">
          <div className="p-8 rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-center">
            <span className="inline-block h-4 w-4 rounded-full bg-indigo-500 animate-ping mb-3" />
            <p className="text-xs text-zinc-500 dark:text-zinc-400">Loading research run...</p>
          </div>
        </main>
      }
    >
      <RunDetailContent />
    </Suspense>
  );
}
