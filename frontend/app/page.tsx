import React from "react";
import { HealthIndicator } from "@/components/HealthIndicator";

export default function Home() {
  const apiUrl =
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  return (
    <main className="flex-1 flex flex-col items-center px-4 py-12 sm:px-6 lg:px-8 max-w-7xl mx-auto w-full">
      {/* Hero Section */}
      <div className="w-full text-center max-w-3xl mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 dark:bg-indigo-950/50 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800 mb-6">
          <span className="h-1.5 w-1.5 rounded-full bg-indigo-500 animate-pulse" />
          Autonomous Multi-Agent Intelligence
        </div>
        <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-zinc-900 dark:text-zinc-50 mb-4">
          Deep Financial Research,{" "}
          <span className="bg-gradient-to-r from-blue-600 via-indigo-600 to-violet-600 bg-clip-text text-transparent">
            Grounded & Audited
          </span>
        </h1>
        <p className="text-base sm:text-lg text-zinc-600 dark:text-zinc-400 max-w-2xl mx-auto leading-relaxed">
          Coordinated specialist agents investigate SEC 10-K filings, run DCF intrinsic
          valuations, evaluate real-time market microstructure, and audit assumptions with
          zero-hallucination verification.
        </p>
      </div>

      {/* System Status Banner */}
      <div className="w-full max-w-4xl p-6 rounded-2xl bg-white dark:bg-zinc-900/60 border border-zinc-200/80 dark:border-zinc-800 shadow-sm mb-10 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-4 text-left">
          <div className="h-12 w-12 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 border border-indigo-200 dark:border-indigo-800/80 flex items-center justify-center text-indigo-600 dark:text-indigo-400 font-bold text-xl">
            ⚡
          </div>
          <div>
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
              Backend Orchestration Engine
            </h3>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 font-mono mt-0.5">
              Target API: {apiUrl}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <HealthIndicator />
          <a
            href={`${apiUrl}/health`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 font-mono underline"
          >
            /health ↗
          </a>
        </div>
      </div>

      {/* Architecture Cards Grid */}
      <div className="w-full max-w-5xl grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        <div className="p-6 rounded-2xl bg-white dark:bg-zinc-900/40 border border-zinc-200/80 dark:border-zinc-800">
          <div className="text-2xl mb-3">🏛️</div>
          <h3 className="font-bold text-zinc-900 dark:text-zinc-100 text-base mb-1">
            Fundamental & Filings
          </h3>
          <p className="text-xs leading-relaxed text-zinc-600 dark:text-zinc-400">
            XBRL-grounded balance sheet metrics, GAAP vs. non-GAAP reconciliation, and
            point-in-time financial audit.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-white dark:bg-zinc-900/40 border border-zinc-200/80 dark:border-zinc-800">
          <div className="text-2xl mb-3">📐</div>
          <h3 className="font-bold text-zinc-900 dark:text-zinc-100 text-base mb-1">
            Quantitative Valuation
          </h3>
          <p className="text-xs leading-relaxed text-zinc-600 dark:text-zinc-400">
            Monte Carlo DCFs, reverse DCF growth solving, enterprise multiple regressions,
            and scenario trees.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-white dark:bg-zinc-900/40 border border-zinc-200/80 dark:border-zinc-800">
          <div className="text-2xl mb-3">🛡️</div>
          <h3 className="font-bold text-zinc-900 dark:text-zinc-100 text-base mb-1">
            Contrarian Risk Wave
          </h3>
          <p className="text-xs leading-relaxed text-zinc-600 dark:text-zinc-400">
            Wave-2 contrarian stress testing, solvency drawdown risk ranking, and claim
            support audit verification.
          </p>
        </div>
      </div>
    </main>
  );
}
