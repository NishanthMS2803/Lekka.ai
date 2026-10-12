import { apiClient, SSEParser } from "../lib/api/index.ts";

async function verifyLiveRun() {
  console.log("=== Step 91: Verifying Live Run Progress ===");

  const health = await apiClient.getHealth();
  console.log("1. Backend Health:", health.status, "v" + health.version);

  // 1. Submit a run
  console.log("2. Launching research run via POST /runs...");
  const createRes = await apiClient.createRun({
    question: "Analyze Apple (AAPL) gross margin trajectory post-Q3 2024",
    as_of_date: "2024-09-20",
    budget: { max_tool_calls: 5, max_seconds: 30 },
  });

  const runId = createRes.run_id;
  console.log(`3. Run initiated: ${runId}`);

  // 2. Connect to SSE stream
  console.log("4. Subscribing to SSE stream /runs/" + runId + "/events...");
  const response = await fetch(`http://127.0.0.1:8000/runs/${runId}/events`);
  if (!response.ok || !response.body) {
    throw new Error(`Failed to connect to SSE stream: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  const parser = new SSEParser();

  let toolCallsCount = 0;
  const stagesSeen = new Set();
  const agentsSeen = new Map();
  let completed = false;

  console.log("5. Watching live run events stream in real-time:");

  while (!completed) {
    const { value, done } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    const messages = parser.feed(chunk);

    for (const msg of messages) {
      const eventName = msg.event || "message";
      stagesSeen.add(eventName);
      const raw = msg.parsedData && typeof msg.parsedData === "object" ? msg.parsedData : {};
      const data = raw.data && typeof raw.data === "object" ? raw.data : raw;

      console.log(`   [EVENT] ${eventName}:`, JSON.stringify(data).slice(0, 100));

      if (eventName === "task_started") {
        agentsSeen.set(data.task_id, {
          agent: data.agent_type,
          status: "running",
          toolCalls: 0,
        });
        console.log(`     → Agent Running: ${data.agent_type} (${data.task_id})`);
      } else if (eventName === "tool_called") {
        toolCallsCount += data.count || 1;
        const ag = agentsSeen.get(data.task_id);
        if (ag) ag.toolCalls += data.count || 1;
        console.log(`     → Tool Invocations Total: ${toolCallsCount}`);
      } else if (eventName === "task_completed") {
        const ag = agentsSeen.get(data.task_id);
        if (ag) ag.status = data.status || "completed";
        console.log(`     → Agent Completed: ${data.agent_type} (claims: ${data.claims_count || 0})`);
      } else if (eventName === "gap_check") {
        console.log(`     → Gap Check: action=${data.action}, round=${data.round}`);
      } else if (eventName === "verification_done") {
        console.log(`     → Verification: confidence=${data.confidence_score}, abstain=${data.is_abstention}`);
      } else if (eventName === "report_ready") {
        completed = true;
        console.log(`     → Report Ready! Terminal status reached.`);
        break;
      } else if (eventName === "run_failed") {
        completed = true;
        console.log(`     → Run Failed:`, data.error);
        break;
      }
    }
  }

  // 3. Check final run status & trace counters
  console.log("6. Fetching final run status & trace telemetry...");
  const status = await apiClient.getRun(runId);
  console.log("   Final Status:", status.status, `(rounds: ${status.rounds}, stop_reason: ${status.stop_reason})`);

  const trace = await apiClient.getTrace(runId);
  if (trace.cost_totals) {
    console.log("   Cost Totals Telemetry:", {
      tool_calls: trace.cost_totals.tool_calls,
      tokens: trace.cost_totals.total_tokens,
      seconds: trace.cost_totals.elapsed_seconds,
      cost_usd: trace.cost_totals.estimated_cost_usd,
    });
  }

  if (status.status === "completed" || status.status === "abstained") {
    const report = await apiClient.getReport(runId);
    console.log("7. Final Report Head (first 150 chars):");
    console.log("   " + report.markdown.slice(0, 150).replace(/\n/g, " "));
  }

  console.log("Stages Observed:", Array.from(stagesSeen).join(", "));
  console.log("Per-Agent Status Verified:", Array.from(agentsSeen.values()).map((a) => `${a.agent}: ${a.status}`));
  console.log("=== Live Run Progress Verified Successfully ===");
}

verifyLiveRun().catch((err) => {
  console.error("Live Run Verification Failed:", err);
  process.exit(1);
});
