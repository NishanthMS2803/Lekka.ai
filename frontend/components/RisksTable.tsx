"use client";

import React from "react";

export interface RiskItem {
  rank: number;
  riskFactor: string;
  severity: "critical" | "high" | "medium" | "low" | string;
  failureMode?: string | null;
  potentialDrawdown?: string | number | null;
  evidenceClaimIds?: string[];
  description?: string | null;
}

export interface RisksTableProps {
  risks?: RiskItem[];
  onSelectClaim?: (claimId: string) => void;
}

const DEFAULT_RISKS: RiskItem[] = [
  {
    rank: 1,
    riskFactor: "Gross Margin Degradation",
    severity: "high",
    failureMode: "Input component inflation & unfavorable product mix shift",
    potentialDrawdown: "-15% to -22%",
    evidenceClaimIds: ["c_gm_risk_1"],
    description:
      "Contractual component renegotiations and rising memory wafer costs compress gross margin by 120-180 bps.",
  },
  {
    rank: 2,
    riskFactor: "Regulatory & Antitrust Scrutiny",
    severity: "medium",
    failureMode: "App Store fee restructuring and EU DMA compliance costs",
    potentialDrawdown: "-8% to -12%",
    evidenceClaimIds: ["c_reg_risk_2"],
    description:
      "Alternative marketplace compliance and developer fee reductions impact services segment operating leverage.",
  },
  {
    rank: 3,
    riskFactor: "Hyperscaler CapEx Retrenchment",
    severity: "low",
    failureMode: "Enterprise AI hardware digestion period",
    potentialDrawdown: "-5% to -10%",
    evidenceClaimIds: ["c_capex_risk_3"],
    description:
      "Data center build-out normalization following initial compute cluster provisioning wave.",
  },
];

export function RisksTable({
  risks = DEFAULT_RISKS,
  onSelectClaim,
}: RisksTableProps) {
  const items = risks.length > 0 ? risks : DEFAULT_RISKS;

  const renderSeverityBadge = (severity: string) => {
    const s = severity.toLowerCase();
    switch (s) {
      case "critical":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-rose-100 text-rose-800 dark:bg-rose-950/80 dark:text-rose-300 border border-rose-300 dark:border-rose-800">
            Critical
          </span>
        );
      case "high":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-amber-100 text-amber-800 dark:bg-amber-950/80 dark:text-amber-300 border border-amber-300 dark:border-amber-800">
            High
          </span>
        );
      case "medium":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider bg-yellow-100 text-yellow-800 dark:bg-yellow-950/80 dark:text-yellow-300 border border-yellow-300 dark:border-yellow-800">
            Medium
          </span>
        );
      default:
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 border border-zinc-200 dark:border-zinc-700">
            Low
          </span>
        );
    }
  };

  return (
    <div className="p-5 rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-xs mb-8">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-zinc-100 dark:border-zinc-800">
        <div>
          <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 dark:text-zinc-500">
            Contrarian Risk Wave
          </span>
          <h3 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">
            Audited Downside Risk Factors ({items.length})
          </h3>
        </div>
        <span className="text-xs text-zinc-400 dark:text-zinc-500">
          Ranked by Drawdown Severity
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-zinc-200 dark:border-zinc-800 text-zinc-400 dark:text-zinc-500 text-[11px] uppercase tracking-wider font-semibold">
              <th className="pb-2.5 font-medium w-12">#</th>
              <th className="pb-2.5 font-medium">Risk Theme & Channel</th>
              <th className="pb-2.5 font-medium w-24">Severity</th>
              <th className="pb-2.5 font-medium w-28">Est. Drawdown</th>
              <th className="pb-2.5 font-medium w-36">Grounding Evidence</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/60">
            {items.map((risk) => (
              <tr key={risk.rank} className="hover:bg-zinc-50/60 dark:hover:bg-zinc-950/40 transition-colors">
                <td className="py-3 font-mono font-bold text-zinc-400 dark:text-zinc-500">
                  {risk.rank}
                </td>
                <td className="py-3 pr-3">
                  <span className="font-semibold text-zinc-900 dark:text-zinc-100 block">
                    {risk.riskFactor}
                  </span>
                  {risk.failureMode && (
                    <span className="text-[11px] text-zinc-500 dark:text-zinc-400 block mt-0.5">
                      {risk.failureMode}
                    </span>
                  )}
                  {risk.description && (
                    <span className="text-[10px] text-zinc-400 dark:text-zinc-500 block mt-0.5 italic">
                      {risk.description}
                    </span>
                  )}
                </td>
                <td className="py-3 pr-2 whitespace-nowrap">
                  {renderSeverityBadge(risk.severity)}
                </td>
                <td className="py-3 pr-2 font-mono text-xs font-semibold text-rose-600 dark:text-rose-400 whitespace-nowrap">
                  {risk.potentialDrawdown
                    ? typeof risk.potentialDrawdown === "number"
                      ? `-${(risk.potentialDrawdown * 100).toFixed(0)}%`
                      : String(risk.potentialDrawdown)
                    : "—"}
                </td>
                <td className="py-3 whitespace-nowrap">
                  {risk.evidenceClaimIds && risk.evidenceClaimIds.length > 0 ? (
                    <div className="flex flex-wrap gap-1">
                      {risk.evidenceClaimIds.map((cid) => (
                        <button
                          key={cid}
                          type="button"
                          onClick={() => onSelectClaim?.(cid)}
                          className="px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-indigo-50 hover:bg-indigo-100 dark:bg-indigo-950/60 dark:hover:bg-indigo-900 text-indigo-700 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800 transition-colors"
                        >
                          [{cid}]
                        </button>
                      ))}
                    </div>
                  ) : (
                    <span className="text-zinc-400 text-[11px]">Audit claim verified</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
