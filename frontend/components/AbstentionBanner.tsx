"use client";

import React from "react";
import Link from "next/link";

export interface AbstentionBannerProps {
  isAbstention: boolean;
  confidenceScore?: number | null;
  missingEvidence?: string[];
  error?: string | null;
  isClarification?: boolean;
}

export function AbstentionBanner({
  isAbstention,
  confidenceScore,
  missingEvidence,
  error,
  isClarification,
}: AbstentionBannerProps) {
  if (isClarification) {
    return (
      <div className="mb-8 p-6 rounded-2xl bg-amber-50 dark:bg-amber-950/40 border border-amber-300 dark:border-amber-800 text-amber-900 dark:text-amber-200 shadow-sm">
        <div className="flex items-start gap-4">
          <span className="text-2xl">⚠️</span>
          <div className="flex-1">
            <h2 className="text-base font-bold text-amber-900 dark:text-amber-100">
              Query Requires User Clarification
            </h2>
            <p className="text-xs text-amber-800 dark:text-amber-300 mt-1 leading-relaxed">
              The query understanding parser detected ambiguities (such as a missing company or
              ticker symbol). Execution stopped early to avoid ungrounded research.
            </p>
            <div className="mt-4 flex items-center gap-3">
              <Link
                href="/"
                className="px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-700 text-white text-xs font-semibold transition-colors shadow-xs"
              >
                ← Return to Ask Page to Clarify
              </Link>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (isAbstention) {
    return (
      <div className="mb-8 p-6 rounded-2xl bg-amber-50 dark:bg-amber-950/40 border border-amber-300 dark:border-amber-800 text-amber-900 dark:text-amber-200 shadow-sm">
        <div className="flex items-start gap-4">
          <span className="text-2xl">🛑</span>
          <div className="flex-1">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-bold text-amber-900 dark:text-amber-100">
                Autonomous Abstention: Insufficient Factual Grounding
              </h2>
              {confidenceScore !== null && confidenceScore !== undefined && (
                <span className="text-xs font-mono font-bold px-2 py-0.5 rounded-md bg-amber-200/80 dark:bg-amber-900/60 text-amber-900 dark:text-amber-200">
                  Confidence: {(confidenceScore * 100).toFixed(0)}% (threshold: 60%)
                </span>
              )}
            </div>
            <p className="text-xs text-amber-800 dark:text-amber-300 mt-1.5 leading-relaxed">
              In accordance with zero-hallucination protocols, the synthesis engine refused to
              generate unsubstantiated financial claims because required supporting evidence could
              not be verified.
            </p>

            {missingEvidence && missingEvidence.length > 0 && (
              <div className="mt-3 p-3 rounded-xl bg-amber-100/60 dark:bg-amber-900/40 text-[11px]">
                <span className="font-semibold block mb-1">Missing Evidence Factors:</span>
                <ul className="list-disc list-inside space-y-0.5 text-amber-900 dark:text-amber-200">
                  {missingEvidence.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mb-8 p-6 rounded-2xl bg-red-50 dark:bg-red-950/40 border border-red-300 dark:border-red-800 text-red-900 dark:text-red-200 shadow-sm">
        <div className="flex items-start gap-4">
          <span className="text-2xl">❌</span>
          <div className="flex-1">
            <h2 className="text-base font-bold text-red-900 dark:text-red-100">
              Run Execution Encountered Error
            </h2>
            <p className="text-xs text-red-800 dark:text-red-300 mt-1 font-mono break-all leading-relaxed">
              {error}
            </p>
            <div className="mt-4 flex items-center gap-3">
              <Link
                href="/"
                className="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-700 text-white text-xs font-semibold transition-colors shadow-xs"
              >
                ← Back to Home
              </Link>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return null;
}
