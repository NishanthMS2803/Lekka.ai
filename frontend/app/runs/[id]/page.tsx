"use client";

import React, { useEffect, useState, useRef, Suspense } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  apiClient,
  RunStatusResponse,
  RunReportResponse,
  RunTraceResponse,
  ClaimResponse,
  createRunEventSource,
  RunEventPayload,
} from "@/lib/api";
import { ResourceCounters } from "@/components/ResourceCounters";
import {
  StageChecklist,
  StageDetails,
  AgentTaskItem,
} from "@/components/StageChecklist";
import { AbstentionBanner } from "@/components/AbstentionBanner";
import { InteractiveReport } from "@/components/InteractiveReport";
import { ClaimDrillDownPanel } from "@/components/ClaimDrillDownPanel";
import { ScenarioChart, ScenarioDataPoint } from "@/components/ScenarioChart";
import { RisksTable, RiskItem } from "@/components/RisksTable";

function extractScenarioData(statusData: RunStatusResponse | null): ScenarioDataPoint[] {
  const outcome = statusData?.outcome as Record<string, unknown> | undefined;
  const state = outcome?.state as Record<string, unknown> | undefined;
  const results = state?.results as Array<Record<string, unknown>> | undefined;

  if (results && Array.isArray(results)) {
    for (const res of results) {
      const ext = res.extensions as Record<string, unknown> | undefined;
      if (ext && ext.valuation) {
        const valExt = ext.valuation as Record<string, unknown>;
        const sv = valExt.scenario_values as Record<string, unknown> | undefined;
        if (
          sv &&
          typeof sv.bear === "number" &&
          typeof sv.base === "number" &&
          typeof sv.bull === "number"
        ) {
          return [
            { scenario: "Bear Case", value: sv.bear, notes: "Macro headwinds & margin compression" },
            { scenario: "Base Case", value: sv.base, notes: "Consensus DCF fundamental expectation" },
            { scenario: "Bull Case", value: sv.bull, notes: "Multiple expansion & margin upside" },
          ];
        }
      }
    }
  }
  return [];
}

function extractRisksData(statusData: RunStatusResponse | null): RiskItem[] {
  const outcome = statusData?.outcome as Record<string, unknown> | undefined;
  const state = outcome?.state as Record<string, unknown> | undefined;
  const results = state?.results as Array<Record<string, unknown>> | undefined;

  if (results && Array.isArray(results)) {
    for (const res of results) {
      const ext = res.extensions as Record<string, unknown> | undefined;
      if (ext && ext.risk) {
        const riskExt = ext.risk as Record<string, unknown>;
        const ranked = (riskExt.ranked_risks || riskExt.risks) as
          | Array<Record<string, unknown>>
          | undefined;
        if (ranked && Array.isArray(ranked) && ranked.length > 0) {
          return ranked.map((r, i) => ({
            rank: (r.rank as number) || i + 1,
            riskFactor: (r.risk_factor || r.title || r.name || "Risk Factor") as string,
            severity: (r.severity as string) || "medium",
            failureMode: (r.failure_mode as string) || null,
            potentialDrawdown: (r.potential_drawdown as string | number) || null,
            evidenceClaimIds: (r.evidence_claim_ids as string[]) || [],
            description: (r.description as string) || null,
          }));
        }
      }
    }
  }
  return [];
}

function getPrimaryTicker(stages: StageDetails, statusData: RunStatusResponse | null): string {
  if (stages.understanding.tickers?.[0]) {
    return stages.understanding.tickers[0];
  }
  const outcome = statusData?.outcome as Record<string, unknown> | undefined;
  const state = outcome?.state as Record<string, unknown> | undefined;
  const parsed = state?.parsed_query as Record<string, unknown> | undefined;
  const tickers = parsed?.tickers as string[] | undefined;
  if (tickers && tickers.length > 0) {
    return tickers[0];
  }
  return "AAPL";
}

function RunDetailContent() {
  const params = useParams();
  const runId = (params?.id as string) || "";

  // Core run state
  const [statusData, setStatusData] = useState<RunStatusResponse | null>(null);
  const [reportData, setReportData] = useState<RunReportResponse | null>(null);
  const [traceData, setTraceData] = useState<RunTraceResponse | null>(null);
  const [events, setEvents] = useState<RunEventPayload[]>([]);
  const [claimsList, setClaimsList] = useState<ClaimResponse[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Claim drill-down panel state
  const [selectedClaimId, setSelectedClaimId] = useState<string | null>(null);
  const [selectedSentence, setSelectedSentence] = useState<string | null>(null);
  const [relatedClaimIds, setRelatedClaimIds] = useState<string[]>([]);
  const [isPanelOpen, setIsPanelOpen] = useState(false);

  // Live timer state
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const startTimeRef = useRef<number | null>(null);
  const timerIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Live checklist stage state
  const [stages, setStages] = useState<StageDetails>({
    understanding: { status: "running" },
    planning: { status: "pending" },
    agents: { status: "pending", tasks: [] },
    gapCheck: { status: "pending" },
    verification: { status: "pending" },
    report: { status: "pending" },
  });

  // Resource counters
  const [counters, setCounters] = useState({
    toolCalls: 0,
    tokensTotal: 0,
    tokensIn: 0,
    tokensOut: 0,
    estimatedCostUsd: 0,
  });

  const currentStatus = statusData?.status || "pending";
  const isRunning = currentStatus === "pending" || currentStatus === "running";
  const isAbstention =
    currentStatus === "abstained" ||
    reportData?.is_abstention === true ||
    stages.verification.isAbstention === true;
  const isClarification = currentStatus === "needs_clarification";

  // Wall-clock timer effect
  useEffect(() => {
    if (isRunning) {
      if (startTimeRef.current === null) {
        startTimeRef.current = Date.now();
      }
      timerIntervalRef.current = setInterval(() => {
        if (startTimeRef.current !== null) {
          const sec = (Date.now() - startTimeRef.current) / 1000;
          setElapsedSeconds(sec);
        }
      }, 200);
    } else {
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
    }

    return () => {
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
      }
    };
  }, [isRunning]);

  // Main lifecycle & SSE subscription
  useEffect(() => {
    if (!runId) return;

    let isMounted = true;

    // Fetch initial status, report, trace, and claims
    async function fetchInitialData() {
      try {
        const status = await apiClient.getRun(runId);
        if (!isMounted) return;
        setStatusData(status);
        setIsLoading(false);

        // Fetch claims
        try {
          const claims = await apiClient.getClaims(runId);
          if (isMounted) setClaimsList(claims);
        } catch {
          // Claims not yet generated
        }

        // Fetch report if ready
        if (status.status === "completed" || status.status === "abstained") {
          try {
            const rep = await apiClient.getReport(runId);
            if (isMounted) setReportData(rep);
          } catch {
            // Report not ready yet
          }
        }

        try {
          const tr = await apiClient.getTrace(runId);
          if (isMounted && tr.cost_totals) {
            setTraceData(tr);
            setCounters({
              toolCalls: tr.cost_totals.tool_calls,
              tokensTotal: tr.cost_totals.total_tokens,
              tokensIn: tr.cost_totals.tokens_in,
              tokensOut: tr.cost_totals.tokens_out,
              estimatedCostUsd: tr.cost_totals.estimated_cost_usd,
            });
            if (tr.cost_totals.elapsed_seconds > 0) {
              setElapsedSeconds(tr.cost_totals.elapsed_seconds);
            }
          }
        } catch {
          // Trace not available yet
        }
      } catch (err: unknown) {
        if (isMounted) {
          setError(err instanceof Error ? err.message : "Failed to load run");
          setIsLoading(false);
        }
      }
    }

    fetchInitialData();

    // Subscribe to SSE events stream
    const unsubscribe = createRunEventSource(runId, {
      onEvent: (evt) => {
        if (!isMounted) return;
        setEvents((prev) => [...prev, evt]);

        const rawData = (evt.data || {}) as Record<string, unknown>;
        const data = (
          rawData.data && typeof rawData.data === "object"
            ? rawData.data
            : rawData
        ) as Record<string, unknown>;

        switch (evt.event) {
          case "run_started":
            setStages((prev) => ({
              ...prev,
              understanding: {
                ...prev.understanding,
                status: "running",
                asOfDate: (data.as_of_date as string) || null,
              },
            }));
            break;

          case "plan_created": {
            const rawTasks = (data.tasks as Array<Record<string, unknown>>) || [];
            const newTasks: AgentTaskItem[] = rawTasks.map((t) => ({
              taskId: (t.task_id as string) || "task",
              agentType: (t.agent_type as string) || "analyst",
              instruction: (t.instruction as string) || "",
              priority: (t.priority as number) || 2,
              status: "pending",
              toolCalls: 0,
              claimsCount: 0,
            }));

            setStages((prev) => ({
              ...prev,
              understanding: {
                ...prev.understanding,
                status: "completed",
                tickers: (data.tickers as string[]) || [],
                intent: (data.intent as string) || undefined,
              },
              planning: {
                status: "completed",
                taskCount: (data.task_count as number) || newTasks.length,
                intent: (data.intent as string) || undefined,
              },
              agents: {
                status: newTasks.length > 0 ? "running" : "completed",
                tasks: newTasks,
              },
            }));
            break;
          }

          case "task_started": {
            const taskId = data.task_id as string;
            const agentType = data.agent_type as string;

            setStages((prev) => {
              const existing = prev.agents.tasks;
              const idx = existing.findIndex((t) => t.taskId === taskId);
              let updatedTasks: AgentTaskItem[];

              if (idx >= 0) {
                updatedTasks = existing.map((t, i) =>
                  i === idx ? { ...t, status: "running" } : t
                );
              } else {
                updatedTasks = [
                  ...existing,
                  {
                    taskId: taskId || "task",
                    agentType: agentType || "analyst",
                    instruction: (data.instruction as string) || "",
                    status: "running",
                    toolCalls: 0,
                    claimsCount: 0,
                  },
                ];
              }

              return {
                ...prev,
                agents: {
                  status: "running",
                  tasks: updatedTasks,
                },
              };
            });
            break;
          }

          case "tool_called": {
            const count = (data.count as number) || 1;
            const taskId = data.task_id as string;

            setCounters((prev) => ({
              ...prev,
              toolCalls: prev.toolCalls + count,
            }));

            if (taskId) {
              setStages((prev) => ({
                ...prev,
                agents: {
                  ...prev.agents,
                  tasks: prev.agents.tasks.map((t) =>
                    t.taskId === taskId
                      ? { ...t, toolCalls: t.toolCalls + count }
                      : t
                  ),
                },
              }));
            }
            break;
          }

          case "task_completed": {
            const taskId = data.task_id as string;
            const taskStatus: "completed" | "failed" =
              (data.status as string) === "failed" ? "failed" : "completed";
            const claims = (data.claims_count as number) || 0;

            setStages((prev) => {
              const updatedTasks = prev.agents.tasks.map((t) =>
                t.taskId === taskId
                  ? { ...t, status: taskStatus, claimsCount: claims }
                  : t
              );
              const allDone = updatedTasks.every(
                (t) => t.status === "completed" || t.status === "failed"
              );

              return {
                ...prev,
                agents: {
                  status: allDone ? "completed" : "running",
                  tasks: updatedTasks,
                },
              };
            });
            break;
          }

          case "gap_check": {
            setStages((prev) => ({
              ...prev,
              gapCheck: {
                status: "completed",
                round: data.round as number,
                action: data.action as string,
                reasons: (data.reasons as string[]) || [],
              },
            }));
            break;
          }

          case "verification_done": {
            const conf = data.confidence_score as number | null;
            const abstained = (data.is_abstention as boolean) || false;
            const missing = (data.missing_evidence as string[]) || [];

            setStages((prev) => ({
              ...prev,
              verification: {
                status: "completed",
                confidenceScore: conf,
                isAbstention: abstained,
                missingEvidence: missing,
              },
            }));
            break;
          }

          case "report_ready": {
            const passed = (data.validation_passed as boolean) ?? true;
            const abstained = (data.is_abstention as boolean) || false;
            const conf = data.confidence_score as number | null;

            setStages((prev) => ({
              ...prev,
              understanding: { ...prev.understanding, status: "completed" },
              planning: { ...prev.planning, status: "completed" },
              agents: {
                ...prev.agents,
                status: prev.agents.status === "pending" ? "completed" : prev.agents.status,
              },
              gapCheck: {
                ...prev.gapCheck,
                status: prev.gapCheck.status === "pending" ? "completed" : prev.gapCheck.status,
              },
              verification: {
                ...prev.verification,
                status: "completed",
                confidenceScore: conf,
                isAbstention: abstained,
              },
              report: {
                status: abstained ? "abstained" : "completed",
                validationPassed: passed,
                isAbstention: abstained,
                confidenceScore: conf,
              },
            }));

            // Fetch synthesized report
            apiClient
              .getReport(runId)
              .then((rep) => {
                if (isMounted) setReportData(rep);
              })
              .catch(() => {});

            // Fetch final claims
            apiClient
              .getClaims(runId)
              .then((claims) => {
                if (isMounted) setClaimsList(claims);
              })
              .catch(() => {});

            // Update terminal status
            apiClient
              .getRun(runId)
              .then((st) => {
                if (isMounted) setStatusData(st);
              })
              .catch(() => {});

            // Update final trace
            apiClient
              .getTrace(runId)
              .then((tr) => {
                if (isMounted && tr.cost_totals) {
                  setTraceData(tr);
                  setCounters({
                    toolCalls: tr.cost_totals.tool_calls,
                    tokensTotal: tr.cost_totals.total_tokens,
                    tokensIn: tr.cost_totals.tokens_in,
                    tokensOut: tr.cost_totals.tokens_out,
                    estimatedCostUsd: tr.cost_totals.estimated_cost_usd,
                  });
                  if (tr.cost_totals.elapsed_seconds > 0) {
                    setElapsedSeconds(tr.cost_totals.elapsed_seconds);
                  }
                }
              })
              .catch(() => {});
            break;
          }

          case "run_failed": {
            const errMsg = (data.error as string) || "Run failed";
            setError(errMsg);
            setStatusData((prev) =>
              prev ? { ...prev, status: "failed", error: errMsg } : null
            );
            setStages((prev) => ({
              ...prev,
              agents: { ...prev.agents, status: "failed" },
              report: { ...prev.report, status: "failed" },
            }));
            break;
          }
        }
      },
      onError: () => {
        // Fallback polling will handle updates
      },
    });

    // Periodic trace and claims sync every 2.5s while active
    const pollInterval = setInterval(async () => {
      if (!isMounted) return;
      try {
        const tr = await apiClient.getTrace(runId);
        if (isMounted && tr.cost_totals) {
          setTraceData(tr);
          setCounters({
            toolCalls: tr.cost_totals.tool_calls,
            tokensTotal: tr.cost_totals.total_tokens,
            tokensIn: tr.cost_totals.tokens_in,
            tokensOut: tr.cost_totals.tokens_out,
            estimatedCostUsd: tr.cost_totals.estimated_cost_usd,
          });
          if (!isRunning && tr.cost_totals.elapsed_seconds > 0) {
            setElapsedSeconds(tr.cost_totals.elapsed_seconds);
          }
        }

        if (claimsList.length === 0) {
          const claims = await apiClient.getClaims(runId);
          if (isMounted && claims.length > 0) {
            setClaimsList(claims);
          }
        }
      } catch {
        // ignore polling errors
      }
    }, 2500);

    return () => {
      isMounted = false;
      unsubscribe();
      clearInterval(pollInterval);
    };
  }, [runId, isRunning, claimsList.length]);

  // Open drill-down handler
  const handleSelectClaim = async (
    claimId: string,
    sentence?: string,
    allClaimIds: string[] = []
  ) => {
    setSelectedClaimId(claimId);
    setSelectedSentence(sentence || null);
    setRelatedClaimIds(allClaimIds.length > 0 ? allClaimIds : [claimId]);
    setIsPanelOpen(true);

    // If claim not already in claimsList, fetch it individually
    if (!claimsList.some((c) => c.claim_id === claimId)) {
      try {
        const fetched = await apiClient.getClaim(claimId);
        setClaimsList((prev) => [...prev, fetched]);
      } catch {
        // ignore
      }
    }
  };

  const selectedClaim = selectedClaimId
    ? claimsList.find((c) => c.claim_id === selectedClaimId) || null
    : null;

  // Extract Scenario Values from Outcome State or Default
  const scenarioData = extractScenarioData(statusData);

  // Extract Risks from Outcome State or Default
  const risksData = extractRisksData(statusData);

  const primaryTicker = getPrimaryTicker(stages, statusData);

  return (
    <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-8 sm:px-6">
      {/* Navigation Header */}
      <div className="flex items-center justify-between mb-8 pb-4 border-b border-zinc-200 dark:border-zinc-800">
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-xs font-semibold text-zinc-600 dark:text-zinc-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
        >
          <span>←</span>
          <span>New Research Query</span>
        </Link>
        <div className="flex items-center gap-3">
          <Link
            href={`/runs/${runId}/trace`}
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-950/60 hover:bg-indigo-100 dark:hover:bg-indigo-900 border border-indigo-200 dark:border-indigo-800 transition-colors"
            title="Inspect latency timeline, tool calls, retries, and budget allocation decisions"
          >
            <span>⏱️</span>
            <span>Trace Viewer</span>
          </Link>
          <span className="font-mono text-xs text-zinc-400 dark:text-zinc-500">
            Run ID: {runId}
          </span>
          <span
            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider ${
              currentStatus === "completed"
                ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800"
                : currentStatus === "needs_clarification"
                ? "bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300 border border-amber-200 dark:border-amber-800"
                : currentStatus === "abstained"
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
                  : currentStatus === "abstained" || currentStatus === "needs_clarification"
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

      {/* Loading Banner */}
      {isLoading && !statusData && (
        <div className="mb-6 p-4 rounded-xl bg-indigo-50 dark:bg-indigo-950/40 border border-indigo-200 dark:border-indigo-900/60 text-indigo-800 dark:text-indigo-300 text-xs flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-indigo-500 animate-ping" />
          <span>Connecting to orchestrator SSE feed and streaming live run events...</span>
        </div>
      )}

      {/* Abstention, Clarification & Error Banner */}
      <AbstentionBanner
        isAbstention={isAbstention}
        confidenceScore={
          reportData?.confidence_score ?? stages.verification.confidenceScore
        }
        missingEvidence={stages.verification.missingEvidence}
        error={error || statusData?.error}
        isClarification={isClarification}
      />

      {/* Live Resource Counters (Tool calls, tokens, latency, cost) */}
      <ResourceCounters
        toolCalls={counters.toolCalls}
        tokensTotal={counters.tokensTotal}
        tokensIn={counters.tokensIn}
        tokensOut={counters.tokensOut}
        elapsedSeconds={elapsedSeconds}
        estimatedCostUsd={counters.estimatedCostUsd}
        isRunning={isRunning}
      />

      {/* Live Orchestration Stages Checklist */}
      <StageChecklist stages={stages} />

      {/* Synthesized Research Report Section with Sentence Drill-down */}
      {reportData && (
        <div className="mb-8 p-6 sm:p-8 rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-xs">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 mb-5 border-b border-zinc-100 dark:border-zinc-800 gap-2">
            <div>
              <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400 dark:text-zinc-500">
                Audited Final Deliverable
              </span>
              <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
                {reportData.is_abstention
                  ? "Audit Abstention Report"
                  : "Synthesized Research Memorandum"}
              </h2>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-zinc-400 dark:text-zinc-500 hidden sm:inline">
                Click any sentence to inspect audited evidence ↗
              </span>
              {reportData.confidence_score !== null &&
                reportData.confidence_score !== undefined && (
                  <span className="text-xs px-2.5 py-1 rounded-md bg-zinc-100 dark:bg-zinc-800 font-semibold font-mono text-zinc-700 dark:text-zinc-300">
                    Confidence: {(reportData.confidence_score * 100).toFixed(0)}%
                  </span>
                )}
              {reportData.validation_passed && (
                <span className="text-xs px-2.5 py-1 rounded-md bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 font-semibold">
                  ✓ Verified Grounding
                </span>
              )}
            </div>
          </div>

          {/* Interactive Report View */}
          <InteractiveReport
            markdown={reportData.markdown}
            sentenceToClaims={reportData.sentence_to_claims}
            onSelectClaim={handleSelectClaim}
            selectedClaimId={selectedClaimId}
          />
        </div>
      )}

      {/* Scenario Values Chart & Downside Risks Table */}
      {reportData && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <ScenarioChart
            scenarios={scenarioData}
            currentPrice={215.0}
            ticker={primaryTicker}
          />
          <RisksTable
            risks={risksData}
            onSelectClaim={(cid) => handleSelectClaim(cid)}
          />
        </div>
      )}

      {/* Live Stream Event Feed */}
      <div className="p-6 rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-xs">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
            Live Event Feed ({events.length})
          </h3>
          <span className="text-[11px] font-mono text-zinc-400 dark:text-zinc-500">
            EventSource SSE Stream
          </span>
        </div>

        {events.length === 0 ? (
          <p className="text-xs text-zinc-400 dark:text-zinc-500 italic">
            Connecting to event stream...
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
                  <span className="text-zinc-600 dark:text-zinc-400 break-all">
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

      {/* Execution Spans & Tracer Instrumentation */}
      {traceData?.spans && traceData.spans.length > 0 && (
        <div className="mt-8 p-6 rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-xs">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
              Execution Spans ({traceData.spans.length})
            </h3>
            <span className="text-[11px] font-mono text-zinc-400 dark:text-zinc-500">
              Tracer Latency Audit
            </span>
          </div>

          <div className="space-y-2 max-h-64 overflow-y-auto font-mono text-[11px]">
            {traceData.spans.map((span, idx) => (
              <div
                key={span.id || idx}
                className="p-2.5 rounded-lg bg-zinc-50 dark:bg-zinc-950 border border-zinc-200/60 dark:border-zinc-800 flex items-center justify-between gap-3"
              >
                <div className="flex items-center gap-2 truncate">
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${
                      span.status === "error" ? "bg-red-500" : "bg-emerald-500"
                    }`}
                  />
                  <span className="font-semibold text-zinc-800 dark:text-zinc-200 truncate">
                    {span.name}
                  </span>
                  {span.tool_name && (
                    <span className="text-indigo-600 dark:text-indigo-400">
                      [{span.tool_name}]
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3 shrink-0 text-zinc-500 dark:text-zinc-400">
                  {span.latency_ms !== null && span.latency_ms !== undefined && (
                    <span>{span.latency_ms}ms</span>
                  )}
                  <span className="uppercase text-[10px] font-semibold text-zinc-400">
                    {span.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Claim Drill-Down Side Panel Flyout */}
      <ClaimDrillDownPanel
        isOpen={isPanelOpen}
        onClose={() => setIsPanelOpen(false)}
        claim={selectedClaim}
        sentence={selectedSentence}
        allReferencedClaimIds={relatedClaimIds}
        onSwitchClaim={(cid) =>
          handleSelectClaim(cid, selectedSentence || undefined, relatedClaimIds)
        }
      />
    </main>
  );
}

export default function RunDetailPage() {
  return (
    <Suspense
      fallback={
        <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-8 sm:px-6">
          <div className="p-8 rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-center">
            <span className="inline-block h-4 w-4 rounded-full bg-indigo-500 animate-ping mb-3" />
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Loading live research run view...
            </p>
          </div>
        </main>
      }
    >
      <RunDetailContent />
    </Suspense>
  );
}
