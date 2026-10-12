import test from "node:test";
import assert from "node:assert/strict";
import { parseSSEChunk, SSEParser } from "../sse";

test("parseSSEChunk parses a single standard SSE event", () => {
  const raw = 'event: plan_created\ndata: {"plan": ["step1", "step2"]}\n\n';
  const messages = parseSSEChunk(raw);

  assert.strictEqual(messages.length, 1);
  assert.strictEqual(messages[0].event, "plan_created");
  assert.strictEqual(messages[0].data, '{"plan": ["step1", "step2"]}');
  assert.deepStrictEqual(messages[0].parsedData, {
    plan: ["step1", "step2"],
  });
});

test("parseSSEChunk handles id, retry, and event type", () => {
  const raw =
    'id: evt-123\nevent: task_started\nretry: 5000\ndata: {"task": "sec_filings"}\n\n';
  const messages = parseSSEChunk(raw);

  assert.strictEqual(messages.length, 1);
  assert.strictEqual(messages[0].id, "evt-123");
  assert.strictEqual(messages[0].event, "task_started");
  assert.strictEqual(messages[0].retry, 5000);
  assert.deepStrictEqual(messages[0].parsedData, { task: "sec_filings" });
});

test("parseSSEChunk handles multi-line data payloads according to spec", () => {
  const raw = "data: line 1\ndata: line 2\ndata: line 3\n\n";
  const messages = parseSSEChunk(raw);

  assert.strictEqual(messages.length, 1);
  assert.strictEqual(messages[0].data, "line 1\nline 2\nline 3");
  assert.strictEqual(messages[0].parsedData, undefined);
});

test("parseSSEChunk ignores SSE comments such as keep-alive pings", () => {
  const raw =
    ': ping keep-alive\n\nevent: tool_called\ndata: {"tool": "run_python"}\n\n';
  const messages = parseSSEChunk(raw);

  assert.strictEqual(messages.length, 1);
  assert.strictEqual(messages[0].event, "tool_called");
  assert.deepStrictEqual(messages[0].parsedData, { tool: "run_python" });
});

test("parseSSEChunk handles Windows CRLF line endings", () => {
  const raw =
    'event: gap_check\r\ndata: {"gap": false}\r\n\r\nevent: report_ready\r\ndata: {"done": true}\r\n\r\n';
  const messages = parseSSEChunk(raw);

  assert.strictEqual(messages.length, 2);
  assert.strictEqual(messages[0].event, "gap_check");
  assert.strictEqual(messages[1].event, "report_ready");
});

test("SSEParser handles chunked stream feeds across message boundaries", () => {
  const parser = new SSEParser();

  // Part 1: incomplete event line
  const m1 = parser.feed("event: verifi");
  assert.strictEqual(m1.length, 0);

  // Part 2: finish event line, start data
  const m2 = parser.feed('cation_done\ndata: {"confidence": 0.');
  assert.strictEqual(m2.length, 0);

  // Part 3: complete data line and double newline boundary
  const m3 = parser.feed("95}\n\n");
  assert.strictEqual(m3.length, 1);
  assert.strictEqual(m3[0].event, "verification_done");
  assert.deepStrictEqual(m3[0].parsedData, { confidence: 0.95 });

  // Reset clears state
  parser.reset();
  const m4 = parser.feed("event: new_event\ndata: 123\n\n");
  assert.strictEqual(m4.length, 1);
  assert.strictEqual(m4[0].event, "new_event");
});
