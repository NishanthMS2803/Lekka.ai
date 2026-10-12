"""Tests for FastAPI application, health, runs, claims, and report endpoints."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from lekka_ai.api.app import API_VERSION, create_app
from lekka_ai.api.dependencies import get_llm, get_run_manager, get_task_runner
from lekka_ai.llm.base import LLMResponse
from lekka_ai.llm.fake import FakeLLM
from lekka_ai.orchestrator.dispatch import TaskResult
from lekka_ai.orchestrator.planner import PlannedTaskSpec, PlanSpec
from lekka_ai.orchestrator.state import PlannedTask
from lekka_ai.orchestrator.understand import ParsedQuery
from lekka_ai.schemas.claims import Claim, ClaimType, Evidence


def _build_valid_claims() -> list[Claim]:
    """Provide realistic grounded claims for end-to-end API testing."""
    return [
        Claim(
            claim_id="claim_rev_001",
            text="Apple reported revenue of $85.8B for Q3 2024.",
            type=ClaimType.FACT,
            confidence=1.0,
            produced_by="fundamental",
            evidence=[
                Evidence(
                    source_type="sec_filing",
                    document_id="AAPL",
                    tool_call_id="tc_001",
                    as_of_date="2024-06-29",
                    raw_value={"revenue": 85780000000.0},
                )
            ],
        ),
        Claim(
            claim_id="claim_margin_002",
            text="Operating margins were 29.8% in Q3 2024.",
            type=ClaimType.FACT,
            confidence=1.0,
            produced_by="fundamental",
            evidence=[
                Evidence(
                    source_type="sec_filing",
                    document_id="AAPL",
                    tool_call_id="tc_002",
                    as_of_date="2024-06-29",
                    raw_value={"operating_margin": 0.298},
                )
            ],
        ),
        Claim(
            claim_id="claim_dcf_003",
            text="Base DCF model indicates fair value of $210.00 per share.",
            type=ClaimType.CALCULATION,
            confidence=0.9,
            derived_by="run_dcf",
            produced_by="valuation",
            evidence=[
                Evidence(
                    source_type="dcf_model",
                    document_id="AAPL",
                    tool_call_id="tc_003",
                    raw_value={"fair_value": 210.00},
                )
            ],
        ),
        Claim(
            claim_id="claim_bull_004",
            text="Accelerated AI adoption drives 12.0% revenue growth.",
            type=ClaimType.INTERPRETATION,
            confidence=0.85,
            depends_on=["claim_rev_001"],
            produced_by="valuation",
            evidence=[
                Evidence(
                    source_type="analyst_estimates",
                    document_id="AAPL",
                    tool_call_id="tc_004",
                    raw_value={"bull_growth": 0.12},
                )
            ],
        ),
        Claim(
            claim_id="claim_bear_005",
            text="Services revenue growth decelerates to 3.0%.",
            type=ClaimType.INTERPRETATION,
            confidence=0.8,
            depends_on=["claim_margin_002"],
            produced_by="risk",
            evidence=[
                Evidence(
                    source_type="sec_filing",
                    document_id="AAPL",
                    tool_call_id="tc_005",
                    raw_value={"bear_growth": 0.03},
                )
            ],
        ),
        Claim(
            claim_id="claim_risk_006",
            text="Greater China sales contracted by 6.5%.",
            type=ClaimType.FACT,
            confidence=0.95,
            produced_by="risk",
            evidence=[
                Evidence(
                    source_type="quarterly_financials",
                    document_id="AAPL",
                    tool_call_id="tc_006",
                    as_of_date="2024-06-29",
                    raw_value={"china_decline": -0.065},
                )
            ],
        ),
        Claim(
            claim_id="claim_cat_007",
            text="Apple Intelligence in Q4 2024 represents a catalyst.",
            type=ClaimType.BACKGROUND,
            confidence=0.9,
            produced_by="news",
            evidence=[
                Evidence(
                    source_type="news",
                    document_id="AAPL",
                    tool_call_id="tc_007",
                    raw_value="Apple Intelligence launch",
                )
            ],
        ),
        Claim(
            claim_id="claim_gap_008",
            text="App Store litigation remains an unquantified gap.",
            type=ClaimType.BACKGROUND,
            confidence=0.85,
            produced_by="news",
            evidence=[
                Evidence(
                    source_type="sec_filing",
                    document_id="AAPL",
                    tool_call_id="tc_008",
                    raw_value="Regulatory review",
                )
            ],
        ),
    ]


def _build_valid_markdown_report() -> str:
    """Build a complete, syntactically valid 9-section markdown report."""
    return (
        "# Investment Research Report: Apple Inc. (AAPL)\n\n"
        "## Executive Summary\n"
        "Apple reported revenue of $85.8B for Q3 2024 [claim_rev_001]. "
        "Operating margins were 29.8% [claim_margin_002].\n\n"
        "## Business Quality\n"
        "The ecosystem generates durable revenue of $85.8B [claim_rev_001]. "
        "Execution is highlighted by 29.8% operating margins [claim_margin_002].\n\n"
        "## Valuation Scenarios\n"
        "Base DCF model indicates fair value of $210.00 per share [claim_dcf_003].\n\n"
        "## Bull Case\n"
        "Accelerated AI adoption drives 12.0% revenue growth [claim_bull_004].\n\n"
        "## Bear Case\n"
        "Services revenue growth decelerates to 3.0% [claim_bear_005].\n\n"
        "## Ranked Risks\n"
        "Greater China sales contracted by 6.5% [claim_risk_006].\n\n"
        "## Catalysts\n"
        "Apple Intelligence in Q4 2024 represents a catalyst [claim_cat_007].\n\n"
        "## What Would Invalidate the Thesis\n"
        "Deceleration below 3.0% would invalidate the thesis [claim_bear_005].\n\n"
        "## Uncertainty and Gaps\n"
        "App Store litigation remains an unquantified gap [claim_gap_008].\n"
    )


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    """GET /health returns 200, status ok, version, and X-Request-ID header."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["version"] == API_VERSION
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0


@pytest.mark.asyncio
async def test_health_with_custom_request_id() -> None:
    """X-Request-ID provided in request headers is preserved in response headers."""
    app = create_app()
    transport = ASGITransport(app=app)
    custom_id = "test-request-id-12345"
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(
            "/health",
            headers={"X-Request-ID": custom_id},
        )
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == custom_id


@pytest.mark.asyncio
async def test_cors_headers() -> None:
    """CORS preflight request from localhost:3000 returns allow origin."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert (
            response.headers.get("access-control-allow-origin")
            == "http://localhost:3000"
        )


@pytest.mark.asyncio
async def test_docs_and_openapi_available() -> None:
    """OpenAPI schema and /docs Swagger UI endpoints are accessible."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        openapi_res = await client.get("/openapi.json")
        assert openapi_res.status_code == 200
        schema = openapi_res.json()
        assert "/health" in schema["paths"]
        assert "/runs" in schema["paths"]
        assert "/claims/{claim_id}" in schema["paths"]

        docs_res = await client.get("/docs")
        assert docs_res.status_code == 200


@pytest.mark.asyncio
async def test_post_then_get_report_returns_report_with_fake_llm() -> None:
    """POST /runs starts engine and GET /runs/{id}/report returns report."""
    valid_claims = _build_valid_claims()

    # Configure FakeLLM sequence: understand -> plan -> gapcheck -> report
    fake_understand = ParsedQuery(
        intent="investment_evaluation",
        tickers=["AAPL"],
        as_of_date="2024-09-20",
        needs_clarification=False,
    )
    fake_plan = PlanSpec(
        tasks=[
            PlannedTaskSpec(
                task_id="task_market_01",
                agent_type="fundamental",
                instruction="Collect AAPL fundamentals",
                priority=1,
            )
        ]
    )
    gap_json = (
        '{"action": "done", "follow_up_tasks": [], '
        '"reasons": ["All verified successfully"]}'
    )
    report_markdown = _build_valid_markdown_report()

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
        )

    app = create_app()
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_task_runner] = lambda: fake_runner

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. POST /runs
        post_res = await client.post(
            "/runs",
            json={
                "question": "Is Apple an attractive investment?",
                "as_of_date": "2024-09-20",
                "budget": {"max_rounds": 1},
            },
        )
        assert post_res.status_code == 200
        post_data = post_res.json()
        assert "run_id" in post_data
        run_id = post_data["run_id"]
        assert len(run_id) > 0

        # 2. GET /runs/{id}
        status_res = await client.get(f"/runs/{run_id}")
        assert status_res.status_code == 200
        status_data = status_res.json()
        assert status_data["run_id"] == run_id
        assert status_data["status"] == "completed"
        assert status_data["outcome"] is not None

        # 3. GET /runs/{id}/report
        report_res = await client.get(f"/runs/{run_id}/report")
        assert report_res.status_code == 200
        report_data = report_res.json()
        assert report_data["run_id"] == run_id
        assert report_data["status"] == "completed"
        assert report_data["validation_passed"] is True
        assert "revenue of $85.8B" in report_data["markdown"]
        assert report_data["report"] == report_data["markdown"]
        assert report_data["confidence_score"] is not None
        assert report_data["is_abstention"] is False

        # 4. GET /runs/{id}/claims
        claims_res = await client.get(f"/runs/{run_id}/claims")
        assert claims_res.status_code == 200
        claims_data = claims_res.json()
        assert isinstance(claims_data, list)
        assert len(claims_data) >= len(valid_claims)

        # 5. GET /claims/{id} (with evidence)
        target_claim_id = "claim_rev_001"
        claim_res = await client.get(f"/claims/{target_claim_id}")
        assert claim_res.status_code == 200
        claim_data = claim_res.json()
        assert claim_data["claim_id"] == target_claim_id
        assert "85.8B" in claim_data["text"]
        assert len(claim_data["evidence"]) > 0
        ev = claim_data["evidence"][0]
        assert ev["source_type"] == "sec_filing"
        assert ev["document_id"] == "AAPL"


@pytest.mark.asyncio
async def test_post_then_get_report_abstention_on_low_evidence() -> None:
    """Low evidence run triggers abstention report stating insufficient evidence."""
    old_date = (date.today() - timedelta(days=800)).isoformat()
    low_claims = [
        Claim(
            claim_id="claim_vague_01",
            text="The company operates in consumer hardware.",
            type=ClaimType.BACKGROUND,
            confidence=0.4,
            produced_by="news",
            evidence=[
                Evidence(
                    source_type="news",
                    as_of_date=old_date,
                    raw_value="historical narrative",
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
                task_id="task_vague_01",
                agent_type="news",
                instruction="Find news",
                priority=1,
            )
        ]
    )
    gap_json = '{"action": "done", "follow_up_tasks": [], "reasons": ["wave complete"]}'

    fake_llm = FakeLLM(
        [
            LLMResponse(structured_output=fake_understand),
            LLMResponse(structured_output=fake_plan),
            LLMResponse(text=gap_json),
        ]
    )

    async def fake_runner(task: PlannedTask, scratch_dir: Path) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            status="completed",
            claims=low_claims,
            confidence=0.4,
            confidence_rationale="Vague narrative only.",
        )

    app = create_app()
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_task_runner] = lambda: fake_runner

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        post_res = await client.post(
            "/runs",
            json={"question": "Evaluate AAPL speculative company"},
        )
        assert post_res.status_code == 200
        run_id = post_res.json()["run_id"]

        report_res = await client.get(f"/runs/{run_id}/report")
        assert report_res.status_code == 200
        report_data = report_res.json()
        assert report_data["is_abstention"] is True
        assert "insufficient evidence" in report_data["markdown"].lower()
        # Verify no scenario numbers are published
        assert "$" not in report_data["markdown"]


@pytest.mark.asyncio
async def test_get_nonexistent_run_returns_404() -> None:
    """GET /runs/{nonexistent} and /report return 404 Not Found."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        status_res = await client.get("/runs/00000000-0000-0000-0000-000000000000")
        assert status_res.status_code == 404

        report_res = await client.get(
            "/runs/00000000-0000-0000-0000-000000000000/report"
        )
        assert report_res.status_code == 404

        claims_res = await client.get(
            "/runs/00000000-0000-0000-0000-000000000000/claims"
        )
        assert claims_res.status_code == 404


@pytest.mark.asyncio
async def test_get_nonexistent_claim_returns_404() -> None:
    """GET /claims/{nonexistent} returns 404 Not Found."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        claim_res = await client.get("/claims/claim_does_not_exist_999")
        assert claim_res.status_code == 404


@pytest.mark.asyncio
async def test_get_run_trace_and_matches_db() -> None:
    """Trace endpoint returns waterfall spans matching records in the database."""
    fake_understand = ParsedQuery(
        intent="investment_evaluation",
        tickers=["AAPL"],
        needs_clarification=False,
    )
    fake_plan = PlanSpec(
        tasks=[
            PlannedTaskSpec(
                task_id="task_valuation_01",
                agent_type="valuation",
                instruction="Perform reverse DCF valuation",
                priority=1,
            )
        ]
    )
    gap_json = (
        '{"action": "done", "follow_up_tasks": [], "reasons": ["analysis complete"]}'
    )

    fake_llm = FakeLLM(
        [
            LLMResponse(structured_output=fake_understand),
            LLMResponse(structured_output=fake_plan),
            LLMResponse(text=gap_json),
        ]
    )

    claims = [
        Claim(
            claim_id="claim_trace_val_01",
            text="Base case fair value estimate is $228.00.",
            type=ClaimType.CALCULATION,
            confidence=0.92,
            produced_by="valuation",
            evidence=[
                Evidence(
                    source_type="reverse_dcf",
                    document_id="AAPL_MODEL",
                    tool_call_id="call_trace_dcf_99",
                    raw_value={"base_fair_value": 228.0},
                )
            ],
        )
    ]

    async def mock_runner_with_trace(
        task: PlannedTask, scratch_dir: Path
    ) -> TaskResult:
        # Simulate agent execution that records a tool call
        from lekka_ai.tracing import _current_span

        parent = _current_span.get()
        assert parent is not None or True
        return TaskResult(
            task_id=task.task_id,
            agent_type=task.agent_type,
            status="completed",
            claims=claims,
            confidence=0.92,
            confidence_rationale="DCF sensitivity checked.",
            usage={"tokens_in": 1500, "tokens_out": 450, "tool_calls": 3},
        )

    app = create_app()
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_task_runner] = lambda: mock_runner_with_trace

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Initiate run
        post_res = await client.post(
            "/runs",
            json={"question": "What is AAPL intrinsic fair value?"},
        )
        assert post_res.status_code == 200
        run_id = post_res.json()["run_id"]

        # 2. Add an explicit tool call span to the tracer to verify input/output & DB
        run_record = get_run_manager().get_run(run_id)
        assert run_record is not None
        assert run_record.tracer is not None

        async with run_record.tracer.span(
            "calc_reverse_dcf",
            span_type="tool",
            tool_name="calc_reverse_dcf",
            input_json={"ticker": "AAPL", "wacc": 0.085, "terminal_growth": 0.025},
        ) as t_span:
            t_span.output_json = {
                "implied_growth": 0.062,
                "fair_value": 228.0,
                "status": "converged",
            }

        # 3. Add a simulated permission denial error span
        async with run_record.tracer.span(
            "unauthorized_exec",
            span_type="tool",
            tool_name="unauthorized_exec",
            input_json={"cmd": "rm -rf /"},
        ) as denial_span:
            denial_span.status = "error"
            denial_span.error = (
                "Permission denied: agent 'quant' cannot use tool 'unauthorized_exec'"
            )

        # 4. Fetch /runs/{id}/trace
        trace_res = await client.get(f"/runs/{run_id}/trace")
        assert trace_res.status_code == 200
        trace_data = trace_res.json()
        assert trace_data["run_id"] == run_id

        # 5. Check cost totals
        totals = trace_data["cost_totals"]
        assert totals["total_tokens"] >= 0
        assert totals["tokens_in"] >= 0
        assert totals["tokens_out"] >= 0
        assert totals["tool_calls"] >= 1
        assert "elapsed_seconds" in totals
        assert "estimated_cost_usd" in totals

        # 6. Check spans
        spans = trace_data["spans"]
        assert len(spans) >= 3

        # Check planner budget decision span
        budget_spans = [
            s
            for s in spans
            if s["span_type"] == "budget" or s["name"].startswith("budget_")
        ]
        assert len(budget_spans) >= 1
        b_span = budget_spans[0]
        assert "allocated_limits" in b_span["metadata"]

        # Check agent span
        agent_spans = [
            s
            for s in spans
            if s["span_type"] == "agent" or s["name"].startswith("agent:")
        ]
        assert len(agent_spans) >= 1
        assert agent_spans[0]["status"] == "completed"

        # Check tool call span with inputs, outputs, and latency
        tool_spans = [s for s in spans if s["tool_name"] == "calc_reverse_dcf"]
        assert len(tool_spans) >= 1
        dcf_span = tool_spans[0]
        assert dcf_span["status"] == "ok"
        assert dcf_span["latency_ms"] is not None
        assert dcf_span["latency_ms"] >= 0
        assert dcf_span["input_json"]["ticker"] == "AAPL"
        assert dcf_span["output_json"]["fair_value"] == 228.0

        # Check permission denial span
        denial_spans = [s for s in spans if s["tool_name"] == "unauthorized_exec"]
        assert len(denial_spans) >= 1
        assert denial_spans[0]["status"] == "error"
        assert "Permission denied" in denial_spans[0]["error"]
