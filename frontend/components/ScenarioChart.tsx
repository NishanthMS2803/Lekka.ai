"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";

export interface ScenarioDataPoint {
  scenario: string;
  value: number;
  probability?: number;
  impliedGrowth?: string;
  notes?: string;
}

export interface ScenarioChartProps {
  scenarios?: ScenarioDataPoint[];
  currentPrice?: number | null;
  ticker?: string;
}

const DEFAULT_SCENARIOS: ScenarioDataPoint[] = [
  {
    scenario: "Bear Case",
    value: 178.5,
    probability: 0.25,
    impliedGrowth: "3.5%",
    notes: "Macro headwinds & margin compression",
  },
  {
    scenario: "Base Case",
    value: 228.0,
    probability: 0.5,
    impliedGrowth: "7.8%",
    notes: "Consensus revenue & steady gross margins",
  },
  {
    scenario: "Bull Case",
    value: 268.5,
    probability: 0.25,
    impliedGrowth: "12.0%",
    notes: "Accelerating services & multiple expansion",
  },
];

export function ScenarioChart({
  scenarios = DEFAULT_SCENARIOS,
  currentPrice = 215.0,
  ticker = "Asset",
}: ScenarioChartProps) {
  const data = scenarios.length > 0 ? scenarios : DEFAULT_SCENARIOS;

  const getBarColor = (scenario: string) => {
    const s = scenario.toLowerCase();
    if (s.includes("bear")) return "#ef4444"; // red-500
    if (s.includes("bull")) return "#10b981"; // emerald-500
    return "#6366f1"; // indigo-500
  };

  const calculateSpread = (val: number) => {
    if (!currentPrice || currentPrice <= 0) return null;
    const diff = ((val - currentPrice) / currentPrice) * 100;
    return `${diff >= 0 ? "+" : ""}${diff.toFixed(1)}%`;
  };

  return (
    <div className="p-5 rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-xs mb-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4 pb-3 border-b border-zinc-100 dark:border-zinc-800">
        <div>
          <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 dark:text-zinc-500">
            Quantitative DCF Modeling
          </span>
          <h3 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">
            Intrinsic Fair Value Scenarios {ticker ? `(${ticker})` : ""}
          </h3>
        </div>
        {currentPrice !== null && currentPrice !== undefined && (
          <div className="flex items-center gap-2 text-xs">
            <span className="text-zinc-500 dark:text-zinc-400">Current Market Price:</span>
            <span className="font-mono font-bold text-zinc-900 dark:text-zinc-100">
              ${currentPrice.toFixed(2)}
            </span>
          </div>
        )}
      </div>

      {/* Chart Canvas */}
      <div className="h-56 w-full pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <XAxis
              dataKey="scenario"
              stroke="#888888"
              fontSize={11}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke="#888888"
              fontSize={11}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => `$${v}`}
            />
            {currentPrice !== null && currentPrice !== undefined && (
              <ReferenceLine
                y={currentPrice}
                stroke="#64748b"
                strokeDasharray="3 3"
                label={{
                  value: `Price: $${currentPrice}`,
                  fill: "#64748b",
                  fontSize: 10,
                  position: "insideTopRight",
                }}
              />
            )}
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const pt = payload[0].payload as ScenarioDataPoint;
                  const spread = calculateSpread(pt.value);
                  return (
                    <div className="p-3 rounded-xl bg-zinc-900 text-zinc-100 border border-zinc-700 shadow-xl text-xs space-y-1">
                      <p className="font-bold text-sm">{pt.scenario}</p>
                      <p className="font-mono text-indigo-300 font-semibold text-base">
                        ${pt.value.toFixed(2)}
                      </p>
                      {spread && (
                        <p className="text-[11px] text-zinc-300">
                          Spread vs Current: <span className="font-semibold">{spread}</span>
                        </p>
                      )}
                      {pt.impliedGrowth && (
                        <p className="text-[11px] text-zinc-400">
                          Implied Growth: {pt.impliedGrowth}
                        </p>
                      )}
                      {pt.notes && (
                        <p className="text-[10px] text-zinc-400 italic pt-1 border-t border-zinc-800">
                          {pt.notes}
                        </p>
                      )}
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar dataKey="value" radius={[6, 6, 0, 0]} barSize={42}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getBarColor(entry.scenario)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Scenario Spread Pills */}
      <div className="grid grid-cols-3 gap-2 mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800 text-center text-xs">
        {data.map((item, idx) => {
          const spread = calculateSpread(item.value);
          return (
            <div key={idx} className="p-2 rounded-lg bg-zinc-50 dark:bg-zinc-950/60">
              <span className="block text-[10px] text-zinc-400 dark:text-zinc-500 font-medium">
                {item.scenario}
              </span>
              <span className="font-mono font-bold text-zinc-800 dark:text-zinc-200 block mt-0.5">
                ${item.value.toFixed(2)}
              </span>
              {spread && (
                <span
                  className={`text-[10px] font-semibold ${
                    spread.startsWith("+")
                      ? "text-emerald-600 dark:text-emerald-400"
                      : "text-rose-600 dark:text-rose-400"
                  }`}
                >
                  {spread}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
