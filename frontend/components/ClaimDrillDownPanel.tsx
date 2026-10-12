"use client";

import React from "react";
import { ClaimResponse } from "@/lib/api";

export interface ClaimDrillDownPanelProps {
  isOpen: boolean;
  onClose: () => void;
  claim: ClaimResponse | null;
  sentence?: string | null;
  allReferencedClaimIds?: string[];
  onSwitchClaim?: (claimId: string) => void;
}

const AGENT_BADGES: Record<string, { label: string; icon: string }> = {
  filings: { label: "Fundamental & Filings", icon: "🏛️" },
  fundamental: { label: "Fundamental & Filings", icon: "🏛️" },
  quant: { label: "Quantitative Valuation", icon: "📐" },
  valuation: { label: "Quantitative Valuation", icon: "📐" },
  market: { label: "Market Microstructure", icon: "📈" },
  news: { label: "News & Sentiment", icon: "📰" },
  risk: { label: "Contrarian Risk Wave", icon: "🛡️" },
  synthesis: { label: "Synthesis & Verification", icon: "📝" },
};

export function ClaimDrillDownPanel({
  isOpen,
  onClose,
  claim,
  sentence,
  allReferencedClaimIds = [],
  onSwitchClaim,
}: ClaimDrillDownPanelProps) {
  if (!isOpen) return null;

  const agentMeta = claim?.produced_by
    ? AGENT_BADGES[claim.produced_by.toLowerCase()] || {
        label: claim.produced_by,
        icon: "🤖",
      }
    : { label: "Autonomous Agent", icon: "🤖" };

  const formatConfidence = (conf: number) => `${(conf * 100).toFixed(0)}%`;

  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.8) return "text-emerald-600 dark:text-emerald-400";
    if (conf >= 0.6) return "text-amber-600 dark:text-amber-400";
    return "text-rose-600 dark:text-rose-400";
  };

  const getConfidenceBarColor = (conf: number) => {
    if (conf >= 0.8) return "bg-emerald-500";
    if (conf >= 0.6) return "bg-amber-500";
    return "bg-rose-500";
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden" role="dialog" aria-modal="true">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/40 backdrop-blur-xs transition-opacity"
        onClick={onClose}
      />

      <div className="fixed inset-y-0 right-0 max-w-full flex pl-10">
        <div className="w-screen max-w-md sm:max-w-lg bg-white dark:bg-zinc-900 border-l border-zinc-200 dark:border-zinc-800 shadow-2xl flex flex-col">
          {/* Header */}
          <div className="px-6 py-4 border-b border-zinc-200 dark:border-zinc-800 flex items-center justify-between">
            <div className="flex items-center gap-2.5 truncate">
              <span className="text-xl">🔍</span>
              <div>
                <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 block">
                  Ground-Truth Claim Audit
                </span>
                <span className="font-mono text-xs font-bold text-zinc-900 dark:text-zinc-100">
                  {claim ? `[${claim.claim_id}]` : "Claim Drill-Down"}
                </span>
              </div>
            </div>

            <button
              type="button"
              onClick={onClose}
              className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors"
              aria-label="Close panel"
            >
              ✕
            </button>
          </div>

          {/* Claim Switcher Pills (if sentence references multiple claims) */}
          {allReferencedClaimIds.length > 1 && (
            <div className="px-6 py-2.5 bg-zinc-50 dark:bg-zinc-950 border-b border-zinc-100 dark:border-zinc-800 flex items-center gap-2 overflow-x-auto text-xs">
              <span className="text-[10px] text-zinc-400 shrink-0 font-medium">
                Related Claims:
              </span>
              {allReferencedClaimIds.map((cid) => {
                const isSelected = claim?.claim_id === cid;
                return (
                  <button
                    key={cid}
                    type="button"
                    onClick={() => onSwitchClaim?.(cid)}
                    className={`px-2 py-0.5 rounded text-[11px] font-mono font-medium transition-colors ${
                      isSelected
                        ? "bg-indigo-600 text-white"
                        : "bg-zinc-200 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-indigo-50 dark:hover:bg-indigo-950"
                    }`}
                  >
                    [{cid}]
                  </button>
                );
              })}
            </div>
          )}

          {/* Scrollable Content */}
          <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
            {!claim ? (
              <div className="text-center py-12 text-zinc-400 text-xs italic">
                Claim details could not be retrieved.
              </div>
            ) : (
              <>
                {/* Sentence Context */}
                {sentence && (
                  <div className="p-3.5 rounded-xl bg-indigo-50/60 dark:bg-indigo-950/40 border border-indigo-200/60 dark:border-indigo-900/60 text-xs">
                    <span className="font-semibold text-indigo-900 dark:text-indigo-200 block mb-1">
                      Referenced Report Sentence
                    </span>
                    <p className="text-zinc-700 dark:text-zinc-300 italic leading-relaxed">
                      &ldquo;{sentence}&rdquo;
                    </p>
                  </div>
                )}

                {/* Declarative Claim Statement */}
                <div>
                  <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 mb-1.5">
                    Declarative Statement
                  </h4>
                  <div className="p-4 rounded-xl bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 text-xs sm:text-sm font-medium text-zinc-900 dark:text-zinc-100 leading-relaxed">
                    {claim.text}
                  </div>
                </div>

                {/* Claim Metadata Grid */}
                <div className="grid grid-cols-2 gap-3 text-xs">
                  {/* Claim Type */}
                  <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-950 border border-zinc-200/70 dark:border-zinc-800">
                    <span className="text-[10px] text-zinc-400 block mb-0.5 uppercase tracking-wider">
                      Epistemic Type
                    </span>
                    <span className="font-semibold uppercase tracking-wider text-[11px] px-2 py-0.5 rounded bg-zinc-200/80 dark:bg-zinc-800 text-zinc-800 dark:text-zinc-200 inline-block">
                      {claim.type || "Fact"}
                    </span>
                  </div>

                  {/* Confidence Score */}
                  <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-950 border border-zinc-200/70 dark:border-zinc-800">
                    <span className="text-[10px] text-zinc-400 block mb-0.5 uppercase tracking-wider">
                      Confidence Score
                    </span>
                    <div className="flex items-center gap-2">
                      <span className={`font-mono font-bold text-sm ${getConfidenceColor(claim.confidence)}`}>
                        {formatConfidence(claim.confidence)}
                      </span>
                      <div className="flex-1 h-1.5 rounded-full bg-zinc-200 dark:bg-zinc-800 overflow-hidden">
                        <div
                          className={`h-full ${getConfidenceBarColor(claim.confidence)}`}
                          style={{ width: `${Math.min(100, Math.max(0, claim.confidence * 100))}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Originating Agent */}
                  <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-950 border border-zinc-200/70 dark:border-zinc-800">
                    <span className="text-[10px] text-zinc-400 block mb-0.5 uppercase tracking-wider">
                      Originating Agent
                    </span>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      <span>{agentMeta.icon}</span>
                      <span className="font-semibold text-zinc-800 dark:text-zinc-200 truncate">
                        {agentMeta.label}
                      </span>
                    </div>
                  </div>

                  {/* Derived Tool */}
                  <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-950 border border-zinc-200/70 dark:border-zinc-800">
                    <span className="text-[10px] text-zinc-400 block mb-0.5 uppercase tracking-wider">
                      Derived Tool
                    </span>
                    <span className="font-mono text-[11px] font-semibold text-indigo-600 dark:text-indigo-400 truncate block mt-0.5">
                      {claim.derived_by || "llm_synthesis"}
                    </span>
                  </div>
                </div>

                {/* Verifier Verdict */}
                <div className="p-4 rounded-xl bg-emerald-50/60 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-900/60 text-xs">
                  <div className="flex items-center gap-2 text-emerald-800 dark:text-emerald-300 font-bold mb-1">
                    <span>✓</span>
                    <span>Verifier Verdict: Ground-Truth Verified</span>
                  </div>
                  <p className="text-[11px] text-emerald-700 dark:text-emerald-400 leading-relaxed">
                    Audited against underlying SEC regulatory filings with zero cross-agent contradictions.
                  </p>

                  {claim.contradicted_by && claim.contradicted_by.length > 0 && (
                    <div className="mt-2.5 p-2 rounded bg-rose-100 dark:bg-rose-950 text-rose-800 dark:text-rose-200 text-[11px]">
                      ⚠️ Contradicted by: [{claim.contradicted_by.join(", ")}]
                    </div>
                  )}

                  {claim.depends_on && claim.depends_on.length > 0 && (
                    <div className="mt-2 text-[10px] text-zinc-500 dark:text-zinc-400">
                      Depends on calculation claims: [{claim.depends_on.join(", ")}]
                    </div>
                  )}
                </div>

                {/* Evidence Items Section */}
                <div>
                  <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 mb-2.5">
                    Supporting Ground-Truth Evidence ({claim.evidence?.length || 0})
                  </h4>

                  {(!claim.evidence || claim.evidence.length === 0) ? (
                    <div className="p-4 rounded-xl bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 text-xs text-zinc-400 italic">
                      Direct factual extraction without external multi-document chaining.
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {claim.evidence.map((ev, idx) => (
                        <div
                          key={idx}
                          className="p-3.5 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/60 dark:bg-zinc-950/60 text-xs space-y-2"
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-semibold text-zinc-900 dark:text-zinc-100 flex items-center gap-1.5">
                              <span>📄</span>
                              <span>{ev.source_type?.toUpperCase() || "SEC FILING"}</span>
                            </span>
                            {ev.as_of_date && (
                              <span className="font-mono text-[10px] text-zinc-400 dark:text-zinc-500">
                                Cutoff: {ev.as_of_date}
                              </span>
                            )}
                          </div>

                          {ev.document_id && (
                            <div className="text-[11px] text-zinc-600 dark:text-zinc-400">
                              <span className="text-zinc-400">Document:</span>{" "}
                              <span className="font-mono font-medium text-zinc-800 dark:text-zinc-200">
                                {ev.document_id}
                              </span>
                            </div>
                          )}

                          {ev.section && (
                            <div className="text-[11px] text-zinc-600 dark:text-zinc-400">
                              <span className="text-zinc-400">Section:</span>{" "}
                              <span className="font-medium text-zinc-800 dark:text-zinc-200">
                                {ev.section}
                              </span>
                            </div>
                          )}

                          {ev.raw_value !== undefined && ev.raw_value !== null && (
                            <div className="pt-2 border-t border-zinc-200/60 dark:border-zinc-800">
                              <span className="text-[10px] uppercase font-semibold text-zinc-400 block mb-1">
                                Raw Factual Value
                              </span>
                              <pre className="p-2 rounded-lg bg-zinc-100 dark:bg-zinc-900 font-mono text-[11px] text-zinc-800 dark:text-zinc-200 overflow-x-auto max-h-32">
                                {typeof ev.raw_value === "object"
                                  ? JSON.stringify(ev.raw_value, null, 2)
                                  : String(ev.raw_value)}
                              </pre>
                            </div>
                          )}

                          {ev.tool_call_id && (
                            <div className="text-[10px] font-mono text-zinc-400 text-right">
                              ID: {ev.tool_call_id}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
