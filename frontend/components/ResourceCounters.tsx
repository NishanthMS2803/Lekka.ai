"use client";

import React from "react";

export interface ResourceCountersProps {
  toolCalls: number;
  tokensTotal: number;
  tokensIn?: number;
  tokensOut?: number;
  elapsedSeconds: number;
  estimatedCostUsd: number;
  isRunning: boolean;
}

export function ResourceCounters({
  toolCalls,
  tokensTotal,
  tokensIn,
  tokensOut,
  elapsedSeconds,
  estimatedCostUsd,
  isRunning,
}: ResourceCountersProps) {
  const formatSeconds = (sec: number) => {
    if (sec < 60) return `${sec.toFixed(1)}s`;
    const mins = Math.floor(sec / 60);
    const remainder = (sec % 60).toFixed(0);
    return `${mins}m ${remainder}s`;
  };

  const formatTokens = (t: number) => {
    if (t >= 1_000_000) return `${(t / 1_000_000).toFixed(2)}M`;
    if (t >= 1_000) return `${(t / 1_000).toFixed(1)}k`;
    return t.toLocaleString();
  };

  const formatCost = (usd: number) => {
    if (usd === 0) return "$0.0000";
    if (usd < 0.01) return `$${usd.toFixed(4)}`;
    return `$${usd.toFixed(3)}`;
  };

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-8">
      {/* Tool Calls Card */}
      <div className="p-4 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-xs">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
            Tool Invocations
          </span>
          <span className="text-base">🛠️</span>
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold font-mono text-zinc-900 dark:text-zinc-100">
            {toolCalls}
          </span>
          {isRunning && (
            <span className="h-1.5 w-1.5 rounded-full bg-indigo-500 animate-ping" />
          )}
        </div>
        <p className="text-[11px] text-zinc-400 dark:text-zinc-500 mt-1">
          SEC filings, Python & Market APIs
        </p>
      </div>

      {/* Tokens Card */}
      <div className="p-4 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-xs">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
            Tokens Processed
          </span>
          <span className="text-base">🪙</span>
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold font-mono text-zinc-900 dark:text-zinc-100">
            {formatTokens(tokensTotal)}
          </span>
        </div>
        <p className="text-[11px] text-zinc-400 dark:text-zinc-500 mt-1">
          {tokensIn !== undefined && tokensOut !== undefined
            ? `In: ${formatTokens(tokensIn)} · Out: ${formatTokens(tokensOut)}`
            : "Input + Output Tokens"}
        </p>
      </div>

      {/* Wall-Clock Elapsed Time */}
      <div className="p-4 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-xs">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
            Execution Time
          </span>
          <span className="text-base">⏱️</span>
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold font-mono text-zinc-900 dark:text-zinc-100">
            {formatSeconds(elapsedSeconds)}
          </span>
          {isRunning && (
            <span className="text-[10px] font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider animate-pulse">
              Live
            </span>
          )}
        </div>
        <p className="text-[11px] text-zinc-400 dark:text-zinc-500 mt-1">
          Wall-clock orchestration latency
        </p>
      </div>

      {/* Estimated Cost */}
      <div className="p-4 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-xs">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
            Estimated Cost
          </span>
          <span className="text-base">💵</span>
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold font-mono text-emerald-600 dark:text-emerald-400">
            {formatCost(estimatedCostUsd)}
          </span>
        </div>
        <p className="text-[11px] text-zinc-400 dark:text-zinc-500 mt-1">
          Gemini token pricing model
        </p>
      </div>
    </div>
  );
}
