"use client";

import React from "react";

export type StageStatus = "pending" | "running" | "completed" | "failed" | "abstained";

export interface AgentTaskItem {
  taskId: string;
  agentType: string;
  instruction: string;
  priority?: number;
  status: "pending" | "running" | "completed" | "failed";
  toolCalls: number;
  claimsCount: number;
}

export interface StageDetails {
  understanding: {
    status: StageStatus;
    intent?: string;
    tickers?: string[];
    asOfDate?: string | null;
  };
  planning: {
    status: StageStatus;
    taskCount?: number;
    intent?: string;
  };
  agents: {
    status: StageStatus;
    tasks: AgentTaskItem[];
  };
  gapCheck: {
    status: StageStatus;
    round?: number;
    action?: string;
    reasons?: string[];
  };
  verification: {
    status: StageStatus;
    confidenceScore?: number | null;
    isAbstention?: boolean;
    missingEvidence?: string[];
  };
  report: {
    status: StageStatus;
    validationPassed?: boolean;
    isAbstention?: boolean;
    confidenceScore?: number | null;
  };
}

export interface StageChecklistProps {
  stages: StageDetails;
}

const AGENT_LABELS: Record<string, { name: string; icon: string; color: string }> = {
  filings: {
    name: "Fundamental & Filings",
    icon: "🏛️",
    color: "border-blue-200 dark:border-blue-900 bg-blue-50/50 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300",
  },
  quant: {
    name: "Quantitative Valuation",
    icon: "📐",
    color: "border-purple-200 dark:border-purple-900 bg-purple-50/50 dark:bg-purple-950/40 text-purple-700 dark:text-purple-300",
  },
  market: {
    name: "Market Microstructure",
    icon: "📈",
    color: "border-emerald-200 dark:border-emerald-900 bg-emerald-50/50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300",
  },
  news: {
    name: "News & Sentiment",
    icon: "📰",
    color: "border-amber-200 dark:border-amber-900 bg-amber-50/50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300",
  },
  risk: {
    name: "Contrarian Risk Wave",
    icon: "🛡️",
    color: "border-rose-200 dark:border-rose-900 bg-rose-50/50 dark:bg-rose-950/40 text-rose-700 dark:text-rose-300",
  },
  synthesis: {
    name: "Synthesis & Verification",
    icon: "📝",
    color: "border-indigo-200 dark:border-indigo-900 bg-indigo-50/50 dark:bg-indigo-950/40 text-indigo-700 dark:text-indigo-300",
  },
};

export function StageChecklist({ stages }: StageChecklistProps) {
  const renderStatusBadge = (status: StageStatus) => {
    switch (status) {
      case "completed":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800">
            <span className="text-xs">✓</span>
            <span>Completed</span>
          </span>
        );
      case "running":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-indigo-50 text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800">
            <span className="h-1.5 w-1.5 rounded-full bg-indigo-500 animate-pulse" />
            <span>Active</span>
          </span>
        );
      case "abstained":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300 border border-amber-200 dark:border-amber-800">
            <span className="text-xs">✋</span>
            <span>Abstained</span>
          </span>
        );
      case "failed":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-red-50 text-red-700 dark:bg-red-950/60 dark:text-red-300 border border-red-200 dark:border-red-800">
            <span className="text-xs">✕</span>
            <span>Failed</span>
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-zinc-100 text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400">
            <span>Waiting</span>
          </span>
        );
    }
  };

  return (
    <div className="rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 p-6 shadow-xs mb-8">
      <div className="flex items-center justify-between mb-6 pb-3 border-b border-zinc-100 dark:border-zinc-800">
        <div>
          <h2 className="text-base font-bold text-zinc-900 dark:text-zinc-100">
            Orchestration Stages & Checklist
          </h2>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            Step-by-step audit trail of autonomous multi-agent execution
          </p>
        </div>
      </div>

      <div className="space-y-6">
        {/* Stage 1: Understanding */}
        <div className="flex items-start gap-4">
          <div
            className={`h-9 w-9 rounded-xl flex items-center justify-center font-bold text-sm shrink-0 ${
              stages.understanding.status === "completed"
                ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300"
                : stages.understanding.status === "running"
                ? "bg-indigo-100 text-indigo-800 dark:bg-indigo-950/80 dark:text-indigo-300 animate-pulse"
                : "bg-zinc-100 text-zinc-400 dark:bg-zinc-800 dark:text-zinc-500"
            }`}
          >
            1
          </div>
          <div className="flex-1">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                Query Understanding & Entity Resolution
              </h3>
              {renderStatusBadge(stages.understanding.status)}
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              Extracts target ticker symbols, dates, financial intent, and checks for ambiguity.
            </p>
            {stages.understanding.status !== "pending" && (
              <div className="mt-2.5 flex flex-wrap gap-2 text-[11px]">
                {stages.understanding.tickers && stages.understanding.tickers.length > 0 && (
                  <span className="px-2 py-0.5 rounded-md bg-zinc-100 dark:bg-zinc-800 font-mono font-medium text-zinc-800 dark:text-zinc-200">
                    Tickers: {stages.understanding.tickers.join(", ")}
                  </span>
                )}
                {stages.understanding.intent && (
                  <span className="px-2 py-0.5 rounded-md bg-zinc-100 dark:bg-zinc-800 font-medium text-zinc-700 dark:text-zinc-300">
                    Intent: {stages.understanding.intent}
                  </span>
                )}
                {stages.understanding.asOfDate && (
                  <span className="px-2 py-0.5 rounded-md bg-zinc-100 dark:bg-zinc-800 font-medium text-zinc-700 dark:text-zinc-300">
                    As-of Cutoff: {stages.understanding.asOfDate}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Stage 2: Planning */}
        <div className="flex items-start gap-4">
          <div
            className={`h-9 w-9 rounded-xl flex items-center justify-center font-bold text-sm shrink-0 ${
              stages.planning.status === "completed"
                ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300"
                : stages.planning.status === "running"
                ? "bg-indigo-100 text-indigo-800 dark:bg-indigo-950/80 dark:text-indigo-300 animate-pulse"
                : "bg-zinc-100 text-zinc-400 dark:bg-zinc-800 dark:text-zinc-500"
            }`}
          >
            2
          </div>
          <div className="flex-1">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                Research Planning & Strategy Decomposition
              </h3>
              {renderStatusBadge(stages.planning.status)}
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              Generates directed acyclic task graph across specialist agents with budget limits.
            </p>
            {stages.planning.status === "completed" && (
              <div className="mt-2 text-[11px] text-zinc-600 dark:text-zinc-400">
                <span>
                  ✓ Planned {stages.planning.taskCount ?? stages.agents.tasks.length} specialized task(s)
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Stage 3: Agents Running (with Per-Agent Status Cards) */}
        <div className="flex items-start gap-4">
          <div
            className={`h-9 w-9 rounded-xl flex items-center justify-center font-bold text-sm shrink-0 ${
              stages.agents.status === "completed"
                ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300"
                : stages.agents.status === "running"
                ? "bg-indigo-100 text-indigo-800 dark:bg-indigo-950/80 dark:text-indigo-300 animate-pulse"
                : "bg-zinc-100 text-zinc-400 dark:bg-zinc-800 dark:text-zinc-500"
            }`}
          >
            3
          </div>
          <div className="flex-1">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                  Specialist Agents Execution
                </h3>
                <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                  Autonomous agents query SEC filings, run Python sandboxes, and extract factual claims.
                </p>
              </div>
              {renderStatusBadge(stages.agents.status)}
            </div>

            {/* Per-Agent Status List */}
            {stages.agents.tasks.length > 0 && (
              <div className="mt-3.5 space-y-2.5">
                {stages.agents.tasks.map((task) => {
                  const meta = AGENT_LABELS[task.agentType] || {
                    name: task.agentType,
                    icon: "🤖",
                    color: "border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300",
                  };

                  return (
                    <div
                      key={task.taskId}
                      className="p-3 rounded-xl border border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/60 dark:bg-zinc-950/60 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs"
                    >
                      <div className="flex items-start gap-3">
                        <span className="text-lg shrink-0">{meta.icon}</span>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-zinc-900 dark:text-zinc-100">
                              {meta.name}
                            </span>
                            <span className="font-mono text-[10px] text-zinc-400 dark:text-zinc-500">
                              #{task.taskId.replace("task_", "")}
                            </span>
                          </div>
                          <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-0.5 line-clamp-2 max-w-xl">
                            {task.instruction}
                          </p>
                        </div>
                      </div>

                      <div className="flex items-center gap-2.5 self-end sm:self-center shrink-0">
                        {task.toolCalls > 0 && (
                          <span className="px-2 py-0.5 rounded-md bg-zinc-100 dark:bg-zinc-800 text-[10px] font-mono text-zinc-600 dark:text-zinc-400">
                            🛠️ {task.toolCalls} call{task.toolCalls !== 1 ? "s" : ""}
                          </span>
                        )}
                        {task.claimsCount > 0 && (
                          <span className="px-2 py-0.5 rounded-md bg-zinc-100 dark:bg-zinc-800 text-[10px] font-mono text-zinc-600 dark:text-zinc-400">
                            📄 {task.claimsCount} claim{task.claimsCount !== 1 ? "s" : ""}
                          </span>
                        )}
                        {task.status === "running" && (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-indigo-50 text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800">
                            <span className="h-1.5 w-1.5 rounded-full bg-indigo-500 animate-pulse" />
                            <span>Running</span>
                          </span>
                        )}
                        {task.status === "completed" && (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800">
                            <span>✓ Done</span>
                          </span>
                        )}
                        {task.status === "failed" && (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-red-50 text-red-700 dark:bg-red-950/60 dark:text-red-300 border border-red-200 dark:border-red-800">
                            <span>✕ Failed</span>
                          </span>
                        )}
                        {task.status === "pending" && (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-zinc-100 text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400">
                            <span>Queued</span>
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Stage 4: Gap Check */}
        <div className="flex items-start gap-4">
          <div
            className={`h-9 w-9 rounded-xl flex items-center justify-center font-bold text-sm shrink-0 ${
              stages.gapCheck.status === "completed"
                ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300"
                : stages.gapCheck.status === "running"
                ? "bg-indigo-100 text-indigo-800 dark:bg-indigo-950/80 dark:text-indigo-300 animate-pulse"
                : "bg-zinc-100 text-zinc-400 dark:bg-zinc-800 dark:text-zinc-500"
            }`}
          >
            4
          </div>
          <div className="flex-1">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                Evidence Gap Analysis & Re-Planning Audit
              </h3>
              {renderStatusBadge(stages.gapCheck.status)}
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              Audits claims against prompt requirements; triggers targeted follow-up tasks if gaps exist.
            </p>
            {stages.gapCheck.action && (
              <div className="mt-2 text-[11px] text-zinc-600 dark:text-zinc-400 flex items-center gap-2">
                <span className="font-semibold text-zinc-800 dark:text-zinc-200">
                  Decision: {stages.gapCheck.action}
                </span>
                {stages.gapCheck.round !== undefined && (
                  <span>· Round {stages.gapCheck.round}</span>
                )}
                {stages.gapCheck.reasons && stages.gapCheck.reasons.length > 0 && (
                  <span className="text-zinc-500">
                    ({stages.gapCheck.reasons[0]})
                  </span>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Stage 5: Verification */}
        <div className="flex items-start gap-4">
          <div
            className={`h-9 w-9 rounded-xl flex items-center justify-center font-bold text-sm shrink-0 ${
              stages.verification.status === "completed"
                ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300"
                : stages.verification.status === "running"
                ? "bg-indigo-100 text-indigo-800 dark:bg-indigo-950/80 dark:text-indigo-300 animate-pulse"
                : "bg-zinc-100 text-zinc-400 dark:bg-zinc-800 dark:text-zinc-500"
            }`}
          >
            5
          </div>
          <div className="flex-1">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                Claim Verification & Confidence Grounding
              </h3>
              {renderStatusBadge(stages.verification.status)}
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              Strict numeric check against ground-truth tables, cross-agent contradiction verification.
            </p>
            {stages.verification.confidenceScore !== undefined &&
              stages.verification.confidenceScore !== null && (
                <div className="mt-2 text-[11px] flex items-center gap-3">
                  <span className="font-semibold text-zinc-800 dark:text-zinc-200">
                    Confidence: {(stages.verification.confidenceScore * 100).toFixed(0)}%
                  </span>
                  {stages.verification.isAbstention && (
                    <span className="text-amber-600 dark:text-amber-400 font-semibold">
                      (Abstention criteria triggered)
                    </span>
                  )}
                </div>
              )}
          </div>
        </div>

        {/* Stage 6: Report */}
        <div className="flex items-start gap-4">
          <div
            className={`h-9 w-9 rounded-xl flex items-center justify-center font-bold text-sm shrink-0 ${
              stages.report.status === "completed"
                ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300"
                : stages.report.status === "abstained"
                ? "bg-amber-100 text-amber-800 dark:bg-amber-950/80 dark:text-amber-300"
                : stages.report.status === "running"
                ? "bg-indigo-100 text-indigo-800 dark:bg-indigo-950/80 dark:text-indigo-300 animate-pulse"
                : "bg-zinc-100 text-zinc-400 dark:bg-zinc-800 dark:text-zinc-500"
            }`}
          >
            6
          </div>
          <div className="flex-1">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                Report Synthesis & Grounded Citations
              </h3>
              {renderStatusBadge(stages.report.status)}
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              Generates executive memorandum with sentence-level claim references and sensitivity tables.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
