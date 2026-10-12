"""Tests for EventBus, Server-Sent Events (SSE), event order, and execution trace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from lekka_ai.api.app import create_app
from lekka_ai.api.dependencies import get_llm, get_task_runner
from lekka_ai.events import EventBus, EventType, RunEvent
from lekka_ai.llm.base import LLMResponse
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.orchestrator.dispatch import TaskResult
from lekka_ai.orchestrator.planner import PlannedTaskSpec, PlanSpec
from lekka_ai.orchestrator.state import PlannedTask
from lekka_ai.orchestrator.understand import ParsedQuery
from lekka_ai.schemas.claims import Claim, ClaimType, Evidence


@pytest.mark.asyncio
async def test_event_bus_emission_and_sse_formatting() -> None:
    """EventBus emits events in order and formats compliant SSE wire payloads."""
    bus = EventBus()
    run_id = "test-run-123"

    event1 = await bus.emit(run_id, EventType.RUN_STARTED, {"question": "Test?"})
    event2 = await bus.emit(run_id, EventType.PLAN_CREATED, {"task_count": 2})

    # Validate RunEvent model and SSE protocol format
    assert event1.event == "run_started"
    assert event1.run_id == run_id
    assert event2.event == "plan_created"
    sse1 = event1.to_sse()
    assert sse1.startswith("event: run_started\ndata: ")
    assert sse1.endswith("\n\n")

    raw_json = sse1.removeprefix("event: run_started\ndata: ").removesuffix("\n\n")
    parsed1 = json.loads(raw_json)
    assert parsed1["event"] == "run_started"
    assert parsed1["run_id"] == run_id
    assert parsed1["data"]["question"] == "Test?"

    # History retrieval
    history = bus.get_history(run_id)
    assert len(history) == 2
    assert [e.event for e in history] == ["run_started", "plan_created"]

    # Historical replay via subscribe
    replayed: list[RunEvent] = []
    # Finish the run to let subscribe terminate
    await bus.emit(run_id, EventType.REPORT_READY, {"status": "completed"})

    async for ev in bus.subscribe(run_id):
        replayed.append(ev)

    assert len(replayed) == 3
    assert [e.event for e in replayed] == [
        "run_started",
        "plan_created",
        "report_ready",
    ]


@pytest.mark.asyncio
async def test_sse_streaming_endpoint_and_event_order() -> None:
    """GET /runs/{id}/events streams SSE and verifies canonical event sequence."""
    valid_claims = [
        Claim(
            claim_id="claim_market_01",
            text="Market capitalization is $3.4T.",
            type=ClaimType.FACT,
            confidence=1.0,
            produced_by="market",
            evidence=[
                Evidence(
                    source_type="fmp_api",
                    document_id="AAPL",
                    tool_call_id="tc_001",
                    raw_value={"market_cap": 3400000000000.0},
                )
            ],
        ),
        Claim(
            claim_id="claim_val_02",
            text="Fair value scenario estimates $230.00.",
            type=ClaimType.CALCULATION,
            confidence=0.9,
            derived_by="run_dcf",
            produced_by="valuation",
            evidence=[
                Evidence(
                    source_type="dcf_model",
                    document_id="AAPL",
                    tool_call_id="tc_002",
                    raw_value={"fair_value": 230.0},
                )
            ],
        ),
    ]

    fake_understand = ParsedQuery(
        intent="investment_evaluation",
        tickers=["AAPL"],
        needs_clarification=False,
    )
    fake_plan = PlanSpec(
        tasks=[
            PlannedTaskSpec(
                task_id="task_market_01",
                agent_type="market",
                instruction="Collect market cap and price",
                priority=1,
            )
        ]
    )
    gap_json = '{"action": "done", "follow_up_tasks": [], "reasons": ["wave complete"]}'
    report_markdown = (
        "# Executive Summary\nMarket capitalization is $3.4T [claim_market_01]. "
        "Fair value scenario estimates $230.00 [claim_val_02].\n"
    )

    fake_llm = FakeLLM(
        [
            LLMResponse(structured_output=fake_understand),
            LLMResponse(structured_output=fake_plan),
            LLMResponse(text=gap_json),
            LLMResponse(text=report_markdown),
        ]
    )

    async def fake_runner(task: PlannedTask, scratch_dir: Path) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            status="completed",
            claims=valid_claims,
            confidence=0.95,
            confidence_rationale="Verified with filings.",
            usage={"tool_calls": 2, "tokens_in": 120, "tokens_out": 80},
        )

    app = create_app()
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_task_runner] = lambda: fake_runner

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Start the run
        post_res = await client.post(
            "/runs",
            json={
                "question": "What is Apple's valuation?",
                "as_of_date": "2024-09-20",
                "budget": {"max_rounds": 1},
            },
        )
        assert post_res.status_code == 200
        run_id = post_res.json()["run_id"]

        # Stream SSE events
        sse_res = await client.get(f"/runs/{run_id}/events")
        assert sse_res.status_code == 200
        assert "text/event-stream" in sse_res.headers.get("content-type", "")

        raw_stream = sse_res.text
        assert len(raw_stream) > 0

        # Parse SSE messages
        messages: list[tuple[str, dict[str, Any]]] = []
        raw_blocks = [b.strip() for b in raw_stream.split("\n\n") if b.strip()]

        for block in raw_blocks:
            lines = block.splitlines()
            event_name = ""
            payload = {}
            for line in lines:
                if line.startswith("event: "):
                    event_name = line.removeprefix("event: ").strip()
                elif line.startswith("data: "):
                    data_str = line.removeprefix("data: ").strip()
                    payload = json.loads(data_str)
            assert event_name, f"Block missing event header: {block}"
            assert payload, f"Block missing data payload: {block}"
            messages.append((event_name, payload))

        event_types = [m[0] for m in messages]

        # Verify all expected event types exist in exact chronological sequence:
        # run_started -> plan_created -> task_started -> tool_called -> task_completed
        # -> gap_check -> verification_done -> report_ready
        expected_sequence = [
            "run_started",
            "plan_created",
            "task_started",
            "tool_called",
            "task_completed",
            "gap_check",
            "verification_done",
            "report_ready",
        ]
        assert event_types == expected_sequence

        # Verify payload fields for each event
        for ev_type, p in messages:
            assert p["event"] == ev_type
            assert p["run_id"] == run_id
            assert "timestamp" in p
            assert isinstance(p["data"], dict)


@pytest.mark.asyncio
async def test_sse_streaming_failure_event_order() -> None:
    """Run failure emits run_failed and halts SSE stream."""
    fake_understand = ParsedQuery(
        intent="investment_evaluation",
        tickers=["AAPL"],
        needs_clarification=False,
    )
    fake_plan = PlanSpec(
        tasks=[
            PlannedTaskSpec(
                task_id="task_fail_01",
                agent_type="market",
                instruction="Crash during execution",
                priority=1,
            )
        ]
    )

    fake_llm = FakeLLM(
        [
            LLMResponse(structured_output=fake_understand),
            LLMResponse(structured_output=fake_plan),
            LLMResponse(text="INVALID_NOT_JSON"),
        ]
    )

    async def normal_runner(task: PlannedTask, scratch_dir: Path) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            status="completed",
            claims=[],
            confidence=0.5,
            confidence_rationale="Done",
        )

    app = create_app()
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_task_runner] = lambda: normal_runner

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        post_res = await client.post(
            "/runs",
            json={"question": "Check crashing run"},
        )
        assert post_res.status_code == 200
        run_id = post_res.json()["run_id"]

        sse_res = await client.get(f"/runs/{run_id}/events")
        assert sse_res.status_code == 200

        raw_stream = sse_res.text
        raw_blocks = [b.strip() for b in raw_stream.split("\n\n") if b.strip()]
        event_types = []
        for block in raw_blocks:
            for line in block.splitlines():
                if line.startswith("event: "):
                    event_types.append(line.removeprefix("event: ").strip())

        assert "run_started" in event_types
        assert "plan_created" in event_types
        assert "task_started" in event_types
        assert event_types[-1] == "run_failed"


@pytest.mark.asyncio
async def test_trace_endpoint_returns_spans_and_cost_totals() -> None:
    """GET /runs/{id}/trace returns recorded spans and aggregated cost totals."""
    valid_claims = [
        Claim(
            claim_id="claim_trace_01",
            text="Revenue $100B.",
            type=ClaimType.FACT,
            confidence=1.0,
            produced_by="fundamental",
            evidence=[
                Evidence(
                    source_type="sec_filing",
                    document_id="AAPL",
                    tool_call_id="tc_001",
                    raw_value={"rev": 100000000000.0},
                )
            ],
        )
    ]
    fake_understand = ParsedQuery(
        intent="investment_evaluation",
        tickers=["AAPL"],
        needs_clarification=False,
    )
    fake_plan = PlanSpec(
        tasks=[
            PlannedTaskSpec(
                task_id="task_fundamental_01",
                agent_type="fundamental",
                instruction="Extract revenue",
                priority=1,
            )
        ]
    )
    gap_json = '{"action": "done", "follow_up_tasks": [], "reasons": ["wave complete"]}'
    report_markdown = "# Summary\nRevenue $100B [claim_trace_01].\n"

    fake_llm = FakeLLM(
        [
            LLMResponse(structured_output=fake_understand),
            LLMResponse(structured_output=fake_plan),
            LLMResponse(text=gap_json),
            LLMResponse(text=report_markdown),
        ]
    )

    async def fake_runner(task: PlannedTask, scratch_dir: Path) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            status="completed",
            claims=valid_claims,
            confidence=1.0,
            confidence_rationale="Grounded.",
            usage={"tokens_in": 500, "tokens_out": 250, "tool_calls": 3},
        )

    app = create_app()
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_task_runner] = lambda: fake_runner

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        post_res = await client.post(
            "/runs",
            json={"question": "AAPL Revenue"},
        )
        assert post_res.status_code == 200
        run_id = post_res.json()["run_id"]

        # Wait for run to finish
        status_res = await client.get(f"/runs/{run_id}?wait=true")
        assert status_res.status_code == 200
        assert status_res.json()["status"] == "completed"

        # Fetch trace
        trace_res = await client.get(f"/runs/{run_id}/trace")
        assert trace_res.status_code == 200
        trace_data = trace_res.json()

        assert trace_data["run_id"] == run_id
        assert "spans" in trace_data
        assert isinstance(trace_data["spans"], list)
        assert "cost_totals" in trace_data

        cost = trace_data["cost_totals"]
        assert cost["total_tokens"] >= 750
        assert cost["tokens_in"] >= 500
        assert cost["tokens_out"] >= 250
        assert cost["tool_calls"] >= 3
        assert cost["elapsed_seconds"] >= 0.0
        assert cost["estimated_cost_usd"] >= 0.0


@pytest.mark.asyncio
async def test_trace_and_events_404_for_unknown_run() -> None:
    """GET /runs/{unknown}/trace and /events return 404 Not Found."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        unknown_id = "00000000-0000-0000-0000-000000000000"
        trace_res = await client.get(f"/runs/{unknown_id}/trace")
        assert trace_res.status_code == 404

        events_res = await client.get(f"/runs/{unknown_id}/events")
        assert events_res.status_code == 404
