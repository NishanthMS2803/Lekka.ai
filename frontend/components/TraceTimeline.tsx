"use client";

import React, { useState, useMemo } from "react";
import { SpanResponse } from "@/lib/api";

export interface TraceTimelineProps {
  spans: SpanResponse[];
}

type FilterCategory = "all" | "agents" | "tools" | "budget" | "errors";

export function TraceTimeline({ spans }: TraceTimelineProps) {
  const [selectedCategory, setSelectedCategory] = useState<FilterCategory>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedSpanIds, setExpandedSpanIds] = useState<Set<string>>(new Set());

  const toggleExpand = (id: string) => {
    setExpandedSpanIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const expandAll = () => {
    setExpandedSpanIds(new Set(spans.map((s) => s.id)));
  };

  const collapseAll = () => {
    setExpandedSpanIds(new Set());
  };

  // Compute timing bounds for waterfall latency bars
  const { minTime, totalDurationMs } = useMemo(() => {
    if (spans.length === 0) return { minTime: 0, totalDurationMs: 1 };

    let min = Infinity;
    let max = -Infinity;

    for (const s of spans) {
      const start = new Date(s.start_time).getTime();
      if (!isNaN(start)) {
        min = Math.min(min, start);
        const end = s.end_time
          ? new Date(s.end_time).getTime()
          : start + (s.latency_ms || 100);
        max = Math.max(max, end);
      }
    }

    if (min === Infinity) {
      min = spans[0] ? new Date(spans[0].start_time).getTime() : 0;
    }
    if (max <= min) max = min + 1000;

    return {
      minTime: min,
      totalDurationMs: Math.max(100, max - min),
    };
  }, [spans]);

  // Categorize spans
  const categorized = useMemo(() => {
    const agents: SpanResponse[] = [];
    const tools: SpanResponse[] = [];
    const budget: SpanResponse[] = [];
    const errors: SpanResponse[] = [];

    for (const s of spans) {
      const isError =
        s.status === "error" ||
        Boolean(s.error) ||
        (s.name && s.name.toLowerCase().includes("error")) ||
        (typeof s.error === "string" && s.error.length > 0);

      const isPermissionDenial =
        isError &&
        (s.error?.toLowerCase().includes("permission") ||
          s.error?.toLowerCase().includes("denied") ||
          s.error?.toLowerCase().includes("disallowed"));

      if (isError || isPermissionDenial) {
        errors.push(s);
      }

      if (s.span_type === "agent" || s.name.startsWith("agent:")) {
        agents.push(s);
      } else if (
        s.span_type === "tool" ||
        Boolean(s.tool_name) ||
        (!s.span_type.startsWith("orchestrator") && !s.span_type.startsWith("budget"))
      ) {
        tools.push(s);
      }

      if (
        s.span_type === "budget" ||
        s.name.startsWith("budget_") ||
        s.name.includes("budget")
      ) {
        budget.push(s);
      }
    }

    return { agents, tools, budget, errors };
  }, [spans]);

  // Filtered Spans
  const filteredSpans = useMemo(() => {
    let list: SpanResponse[] = [];
    if (selectedCategory === "all") list = spans;
    else if (selectedCategory === "agents") list = categorized.agents;
    else if (selectedCategory === "tools") list = categorized.tools;
    else if (selectedCategory === "budget") list = categorized.budget;
    else if (selectedCategory === "errors") list = categorized.errors;

    if (!searchQuery.trim()) return list;

    const query = searchQuery.toLowerCase().trim();
    return list.filter((s) => {
      const matchName = s.name.toLowerCase().includes(query);
      const matchTool = s.tool_name?.toLowerCase().includes(query);
      const matchError = s.error?.toLowerCase().includes(query);
      const matchType = s.span_type.toLowerCase().includes(query);
      return matchName || matchTool || matchError || matchType;
    });
  }, [selectedCategory, spans, categorized, searchQuery]);

  return (
    <div className="space-y-6">
      {/* Controls Bar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4 p-4 rounded-xl bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800">
        {/* Category Filter Pills */}
        <div className="flex items-center gap-1.5 overflow-x-auto text-xs pb-1 sm:pb-0">
          <button
            type="button"
            onClick={() => setSelectedCategory("all")}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
              selectedCategory === "all"
                ? "bg-indigo-600 text-white shadow-xs"
                : "bg-white dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-700 border border-zinc-200 dark:border-zinc-700"
            }`}
          >
            All Spans ({spans.length})
          </button>
          <button
            type="button"
            onClick={() => setSelectedCategory("agents")}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
              selectedCategory === "agents"
                ? "bg-purple-600 text-white shadow-xs"
                : "bg-white dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-700 border border-zinc-200 dark:border-zinc-700"
            }`}
          >
            Agents ({categorized.agents.length})
          </button>
          <button
            type="button"
            onClick={() => setSelectedCategory("tools")}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
              selectedCategory === "tools"
                ? "bg-emerald-600 text-white shadow-xs"
                : "bg-white dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-700 border border-zinc-200 dark:border-zinc-700"
            }`}
          >
            Tool Calls ({categorized.tools.length})
          </button>
          <button
            type="button"
            onClick={() => setSelectedCategory("budget")}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
              selectedCategory === "budget"
                ? "bg-amber-600 text-white shadow-xs"
                : "bg-white dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-700 border border-zinc-200 dark:border-zinc-700"
            }`}
          >
            Budget Decisions ({categorized.budget.length})
          </button>
          <button
            type="button"
            onClick={() => setSelectedCategory("errors")}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
              selectedCategory === "errors"
                ? "bg-rose-600 text-white shadow-xs"
                : "bg-white dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-700 border border-zinc-200 dark:border-zinc-700"
            }`}
          >
            Errors & Denials ({categorized.errors.length})
          </button>
        </div>

        {/* Search & Bulk Expand */}
        <div className="flex items-center gap-2">
          <div className="relative flex-1 sm:w-56">
            <input
              type="text"
              placeholder="Search spans, tools..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full px-3 py-1.5 pl-8 rounded-lg text-xs bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
            <span className="absolute left-2.5 top-2 text-zinc-400 text-xs">🔍</span>
          </div>

          <button
            type="button"
            onClick={expandAll}
            className="px-2.5 py-1.5 rounded-lg text-xs font-medium bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-700 transition-colors"
            title="Expand all tool inputs/outputs"
          >
            Expand All
          </button>
          <button
            type="button"
            onClick={collapseAll}
            className="px-2.5 py-1.5 rounded-lg text-xs font-medium bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-700 transition-colors"
            title="Collapse all spans"
          >
            Collapse All
          </button>
        </div>
      </div>

      {/* Waterfall Timeline View */}
      <div className="rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-xs overflow-hidden">
        {/* Waterfall Header / Time Axis Ruler */}
        <div className="px-6 py-3 bg-zinc-50/80 dark:bg-zinc-950/80 border-b border-zinc-200 dark:border-zinc-800 flex items-center justify-between text-[11px] font-mono text-zinc-400">
          <div className="w-2/5 font-semibold uppercase tracking-wider text-zinc-500">
            Operation / Span Details
          </div>
          <div className="w-3/5 flex items-center justify-between px-2">
            <span>0.0s</span>
            <span>+{(totalDurationMs / 2000).toFixed(1)}s</span>
            <span>+{(totalDurationMs / 1000).toFixed(1)}s ({(totalDurationMs / 1000).toFixed(2)}s total)</span>
          </div>
        </div>

        {/* Spans List */}
        {filteredSpans.length === 0 ? (
          <div className="text-center py-16 text-zinc-400 text-sm">
            No execution spans match the selected filter.
          </div>
        ) : (
          <div className="divide-y divide-zinc-100 dark:divide-zinc-800/60">
            {filteredSpans.map((span, idx) => {
              const isExpanded = expandedSpanIds.has(span.id);
              return (
                <SpanRow
                  key={span.id || idx}
                  span={span}
                  minTime={minTime}
                  totalDurationMs={totalDurationMs}
                  isExpanded={isExpanded}
                  onToggleExpand={() => toggleExpand(span.id)}
                />
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

interface SpanRowProps {
  span: SpanResponse;
  minTime: number;
  totalDurationMs: number;
  isExpanded: boolean;
  onToggleExpand: () => void;
}

function SpanRow({
  span,
  minTime,
  totalDurationMs,
  isExpanded,
  onToggleExpand,
}: SpanRowProps) {
  const [showFullInput, setShowFullInput] = useState(false);
  const [showFullOutput, setShowFullOutput] = useState(false);

  // Timing math for waterfall horizontal bar
  const startMs = new Date(span.start_time).getTime();
  const offsetMs = Math.max(0, startMs - minTime);
  const durationMs =
    span.latency_ms ??
    (span.end_time ? Math.max(10, new Date(span.end_time).getTime() - startMs) : 50);

  const leftPercent = Math.min(96, Math.max(0, (offsetMs / totalDurationMs) * 100));
  const widthPercent = Math.min(
    100 - leftPercent,
    Math.max(1.8, (durationMs / totalDurationMs) * 100)
  );

  const isError = span.status === "error" || Boolean(span.error);
  const isDenial =
    isError &&
    (span.error?.toLowerCase().includes("permission") ||
      span.error?.toLowerCase().includes("denied") ||
      span.error?.toLowerCase().includes("disallowed"));

  const isBudget =
    span.span_type === "budget" ||
    span.name.startsWith("budget_") ||
    span.name.includes("budget");

  const isAgent = span.span_type === "agent" || span.name.startsWith("agent:");

  const hasInputs = span.input_json !== null && span.input_json !== undefined;
  const hasOutputs = span.output_json !== null && span.output_json !== undefined;
  const canExpand = hasInputs || hasOutputs || Boolean(span.error) || isBudget;

  const barColor = isDenial
    ? "bg-rose-600"
    : isError
    ? "bg-red-500"
    : isBudget
    ? "bg-amber-500"
    : isAgent
    ? "bg-purple-500"
    : "bg-emerald-500";

  const typeIcon = isDenial
    ? "🚫"
    : isError
    ? "⚠️"
    : isBudget
    ? "⚖️"
    : isAgent
    ? "🤖"
    : "🔧";

  return (
    <div
      className={`transition-colors ${
        isDenial
          ? "bg-rose-50/40 dark:bg-rose-950/20"
          : isError
          ? "bg-red-50/30 dark:bg-red-950/20"
          : isExpanded
          ? "bg-zinc-50/60 dark:bg-zinc-800/20"
          : "hover:bg-zinc-50/40 dark:hover:bg-zinc-800/30"
      }`}
    >
      {/* Row Header & Latency Bar */}
      <div
        className="px-6 py-3 flex items-center justify-between cursor-pointer select-none"
        onClick={canExpand ? onToggleExpand : undefined}
      >
        {/* Left Side: Span Name & Badges */}
        <div className="w-2/5 flex items-center gap-3 pr-4 truncate">
          <span className="text-base shrink-0">{typeIcon}</span>

          <div className="truncate">
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-bold text-zinc-900 dark:text-zinc-100 truncate">
                {span.tool_name || span.name}
              </span>

              {/* Status Badge */}
              {isDenial ? (
                <span className="px-1.5 py-0.2 rounded text-[10px] font-bold uppercase tracking-wider bg-rose-100 dark:bg-rose-900/60 text-rose-700 dark:text-rose-300 shrink-0">
                  Permission Denied
                </span>
              ) : isError ? (
                <span className="px-1.5 py-0.2 rounded text-[10px] font-bold uppercase tracking-wider bg-red-100 dark:bg-red-900/60 text-red-700 dark:text-red-300 shrink-0">
                  Failed
                </span>
              ) : (
                <span className="px-1.5 py-0.2 rounded text-[10px] font-semibold uppercase tracking-wider bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 shrink-0">
                  {span.span_type}
                </span>
              )}

              {/* Cache hit */}
              {span.metadata?.cache_hit === true && (
                <span className="px-1.5 py-0.2 rounded text-[10px] font-bold uppercase tracking-wider bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 shrink-0">
                  ⚡ Cache Hit
                </span>
              )}
            </div>

            <div className="flex items-center gap-2 text-[10px] text-zinc-400 font-mono mt-0.5">
              <span>+{(offsetMs / 1000).toFixed(2)}s</span>
              <span>•</span>
              <span className="font-bold text-zinc-600 dark:text-zinc-300">
                {durationMs}ms
              </span>
              {span.parent_id && <span>(child)</span>}
            </div>
          </div>
        </div>

        {/* Right Side: Waterfall Latency Bar */}
        <div className="w-3/5 flex items-center gap-3">
          <div className="flex-1 h-5 rounded-md bg-zinc-100 dark:bg-zinc-800/80 relative overflow-hidden flex items-center">
            {/* Latency Bar */}
            <div
              className={`h-3.5 rounded-sm transition-all shadow-xs flex items-center px-1.5 text-[9px] font-mono font-bold text-white whitespace-nowrap overflow-hidden ${barColor}`}
              style={{
                marginLeft: `${leftPercent}%`,
                width: `${widthPercent}%`,
              }}
              title={`${span.name}: ${durationMs}ms`}
            >
              {widthPercent > 6 && `${durationMs}ms`}
            </div>
          </div>

          {/* Expand Chevron */}
          {canExpand && (
            <button
              type="button"
              className="text-xs text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 p-1 rounded-md shrink-0 transition-transform"
              style={{ transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)" }}
              aria-label="Expand span details"
            >
              ▼
            </button>
          )}
        </div>
      </div>

      {/* Expanded Accordion Body */}
      {isExpanded && (
        <div className="px-6 pb-4 pt-1 border-t border-zinc-100 dark:border-zinc-800/80 space-y-3 text-xs">
          {/* Error / Denial Alert Banner */}
          {span.error && (
            <div className="p-3 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900/60 text-rose-800 dark:text-rose-300 font-mono text-xs">
              <div className="font-bold mb-0.5 flex items-center gap-1.5">
                <span>⚠️</span>
                <span>{isDenial ? "Security Permission Denial" : "Span Execution Error"}</span>
              </div>
              <div className="text-[11px] whitespace-pre-wrap">{span.error}</div>
            </div>
          )}

          {/* Budget Allocation Decision Details */}
          {isBudget && (
            <div className="p-3.5 rounded-xl bg-amber-50/60 dark:bg-amber-950/30 border border-amber-200/60 dark:border-amber-900/50">
              <div className="font-semibold text-amber-900 dark:text-amber-200 text-xs mb-2 flex items-center gap-1.5">
                <span>⚖️</span>
                <span>Planner Budget Allocation Decision</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                {Boolean(span.metadata?.agent_type) && (
                  <div className="p-2 rounded-lg bg-white dark:bg-zinc-900 border border-amber-200/50 dark:border-amber-900/40">
                    <span className="text-[10px] text-zinc-400 uppercase tracking-wider block">Agent</span>
                    <span className="font-bold text-zinc-800 dark:text-zinc-200">{String(span.metadata?.agent_type)}</span>
                  </div>
                )}
                {span.metadata?.priority !== undefined && (
                  <div className="p-2 rounded-lg bg-white dark:bg-zinc-900 border border-amber-200/50 dark:border-amber-900/40">
                    <span className="text-[10px] text-zinc-400 uppercase tracking-wider block">Priority</span>
                    <span className="font-bold text-zinc-800 dark:text-zinc-200">{String(span.metadata.priority)}</span>
                  </div>
                )}
                {Boolean(span.metadata?.allocated_limits) && (
                  <div className="p-2 rounded-lg bg-white dark:bg-zinc-900 border border-amber-200/50 dark:border-amber-900/40 col-span-2">
                    <span className="text-[10px] text-zinc-400 uppercase tracking-wider block">Allocated Limits</span>
                    <pre className="font-mono text-[10px] text-zinc-700 dark:text-zinc-300 mt-0.5 overflow-x-auto">
                      {JSON.stringify(span.metadata?.allocated_limits, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Tool Inputs & Outputs Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Input JSON */}
            {hasInputs && (
              <div className="p-3.5 rounded-xl bg-zinc-50 dark:bg-zinc-950 border border-zinc-200/70 dark:border-zinc-800">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-400">
                    Inputs (Arguments)
                  </span>
                  <button
                    type="button"
                    onClick={() => setShowFullInput(!showFullInput)}
                    className="text-[10px] text-indigo-600 dark:text-indigo-400 hover:underline"
                  >
                    {showFullInput ? "Truncate" : "Show Full"}
                  </button>
                </div>
                <FormattedPayload payload={span.input_json} isExpanded={showFullInput} />
              </div>
            )}

            {/* Output JSON */}
            {hasOutputs && (
              <div className="p-3.5 rounded-xl bg-zinc-50 dark:bg-zinc-950 border border-zinc-200/70 dark:border-zinc-800">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-400">
                    Outputs (Return Payload)
                  </span>
                  <button
                    type="button"
                    onClick={() => setShowFullOutput(!showFullOutput)}
                    className="text-[10px] text-indigo-600 dark:text-indigo-400 hover:underline"
                  >
                    {showFullOutput ? "Truncate" : "Show Full"}
                  </button>
                </div>
                <FormattedPayload payload={span.output_json} isExpanded={showFullOutput} />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function FormattedPayload({
  payload,
  isExpanded,
}: {
  payload: unknown;
  isExpanded: boolean;
}) {
  const serialized = useMemo(() => {
    if (typeof payload === "string") return payload;
    try {
      return JSON.stringify(payload, null, 2);
    } catch {
      return String(payload);
    }
  }, [payload]);

  const isLong = serialized.length > 250;
  const displayText = !isExpanded && isLong ? `${serialized.slice(0, 250)}... [truncated]` : serialized;

  return (
    <pre className="p-2.5 rounded-lg bg-zinc-100 dark:bg-zinc-900 font-mono text-[11px] leading-relaxed text-zinc-800 dark:text-zinc-200 overflow-x-auto max-h-60">
      {displayText}
    </pre>
  );
}
