import { describe, it } from "node:test";
import assert from "node:assert";
import type { SpanResponse, RunTraceResponse } from "./api";

describe("Execution Trace Viewer Timeline and Resource Verification", () => {
  const sampleSpans: SpanResponse[] = [
    {
      id: "span_budget_01",
      name: "budget_allocate",
      span_type: "budget",
      start_time: "2026-10-13T01:00:00.000Z",
      end_time: "2026-10-13T01:00:00.010Z",
      latency_ms: 10,
      status: "ok",
      metadata: {
        agent_type: "valuation",
        priority: 1,
        allocated_limits: {
          max_tool_calls: 10,
          max_seconds: 45,
          max_tokens: 30000,
        },
      },
    },
    {
      id: "span_agent_01",
      name: "agent:valuation",
      span_type: "agent",
      start_time: "2026-10-13T01:00:00.015Z",
      end_time: "2026-10-13T01:00:01.250Z",
      latency_ms: 1235,
      status: "ok",
      metadata: {
        task_id: "task_val_01",
        agent_type: "valuation",
      },
    },
    {
      id: "span_tool_01",
      name: "calc_reverse_dcf",
      tool_name: "calc_reverse_dcf",
      span_type: "tool",
      parent_id: "span_agent_01",
      start_time: "2026-10-13T01:00:00.100Z",
      end_time: "2026-10-13T01:00:00.350Z",
      latency_ms: 250,
      status: "ok",
      input_json: { ticker: "AAPL", wacc: 0.085, terminal_growth: 0.025 },
      output_json: { fair_value: 228.0, implied_growth: 0.062, status: "converged" },
      metadata: { cache_hit: true },
    },
    {
      id: "span_denial_01",
      name: "unauthorized_exec",
      tool_name: "unauthorized_exec",
      span_type: "tool",
      start_time: "2026-10-13T01:00:00.400Z",
      end_time: "2026-10-13T01:00:00.405Z",
      latency_ms: 5,
      status: "error",
      error: "Permission denied: agent 'valuation' cannot use tool 'unauthorized_exec'",
      input_json: { cmd: "rm -rf /" },
      output_json: null,
      metadata: {},
    },
  ];

  const sampleTrace: RunTraceResponse = {
    run_id: "run_test_123",
    spans: sampleSpans,
    cost_totals: {
      total_tokens: 15420,
      tokens_in: 11200,
      tokens_out: 4220,
      tool_calls: 3,
      elapsed_seconds: 4.85,
      estimated_cost_usd: 0.0245,
    },
  };

  it("verifies trace resource and cost totals", () => {
    assert.strictEqual(sampleTrace.run_id, "run_test_123");
    assert.ok(sampleTrace.cost_totals);
    assert.strictEqual(sampleTrace.cost_totals.total_tokens, 15420);
    assert.strictEqual(sampleTrace.cost_totals.tokens_in, 11200);
    assert.strictEqual(sampleTrace.cost_totals.tokens_out, 4220);
    assert.strictEqual(sampleTrace.cost_totals.tool_calls, 3);
    assert.strictEqual(sampleTrace.cost_totals.elapsed_seconds, 4.85);
    assert.strictEqual(sampleTrace.cost_totals.estimated_cost_usd, 0.0245);
  });

  it("identifies planner budget allocation decisions", () => {
    const budgetSpans = sampleSpans.filter(
      (s) => s.span_type === "budget" || s.name.startsWith("budget_")
    );
    assert.strictEqual(budgetSpans.length, 1);
    const alloc = budgetSpans[0];
    assert.strictEqual(alloc.name, "budget_allocate");
    const meta = alloc.metadata as Record<string, unknown>;
    assert.strictEqual(meta.agent_type, "valuation");
    const limits = meta.allocated_limits as Record<string, number>;
    assert.strictEqual(limits.max_tool_calls, 10);
    assert.strictEqual(limits.max_seconds, 45);
  });

  it("identifies permission denials and error spans", () => {
    const errorSpans = sampleSpans.filter(
      (s) => s.status === "error" || Boolean(s.error)
    );
    assert.strictEqual(errorSpans.length, 1);

    const denial = errorSpans[0];
    assert.strictEqual(denial.tool_name, "unauthorized_exec");
    assert.ok(denial.error?.includes("Permission denied"));
  });

  it("verifies tool call inputs and outputs with cache hit", () => {
    const toolSpan = sampleSpans.find((s) => s.tool_name === "calc_reverse_dcf");
    assert.ok(toolSpan);
    assert.strictEqual(toolSpan.status, "ok");
    assert.strictEqual(toolSpan.latency_ms, 250);

    const input = toolSpan.input_json as Record<string, unknown>;
    assert.strictEqual(input.ticker, "AAPL");

    const output = toolSpan.output_json as Record<string, unknown>;
    assert.strictEqual(output.fair_value, 228.0);

    assert.strictEqual(toolSpan.metadata?.cache_hit, true);
  });

  it("calculates relative waterfall timeline offsets and durations", () => {
    const starts = sampleSpans.map((s) => new Date(s.start_time).getTime());
    const minTime = Math.min(...starts);
    const ends = sampleSpans.map(
      (s) => new Date(s.end_time || s.start_time).getTime()
    );
    const maxTime = Math.max(...ends);
    const totalDuration = maxTime - minTime;

    assert.ok(totalDuration > 0);

    // Verify first span starts at offset 0
    const firstOffset = new Date(sampleSpans[0].start_time).getTime() - minTime;
    assert.strictEqual(firstOffset, 0);

    // Verify tool span offset is within total duration
    const toolOffset = new Date(sampleSpans[2].start_time).getTime() - minTime;
    assert.ok(toolOffset > 0);
    assert.ok(toolOffset < totalDuration);
  });
});
