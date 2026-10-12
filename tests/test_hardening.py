"""Tests for Step 96 hardening and security features.

Covers:
- Request rate limiting on POST /runs
- Max question length validation
- Ticker allowlist validation
- Run concurrency limits
- Hard per-run cost cap from config
- Secret redaction in structured logs and traces
- Visible regulatory disclaimers in reports and UI
- Startup check disabling fault injection outside test/dev
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from lekka_ai.api.app import create_app
from lekka_ai.api.models import CreateRunRequest, RunReportResponse
from lekka_ai.api.ratelimit import default_rate_limiter
from lekka_ai.api.services import RunManager
from lekka_ai.config import Settings, get_settings, verify_startup_configuration
from lekka_ai.logging import redact_secrets, redact_secrets_processor
from lekka_ai.orchestrator.budget import RunBudget
from lekka_ai.orchestrator.understand import UnknownTickerError, understand_query
from lekka_ai.report.synthesize import SynthesizedReport
from lekka_ai.tracing import Span


@pytest.mark.asyncio
async def test_request_rate_limiting_on_post_runs() -> None:
    """POST /runs enforces rate limiting per client IP."""
    default_rate_limiter.reset()
    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Patch limit to 3 for testing and mock execute_run
        with (
            patch.object(get_settings(), "RATE_LIMIT_RUNS_PER_MINUTE", 3),
            patch.object(get_settings(), "RATE_LIMIT_ENABLED", True),
            patch.object(RunManager, "execute_run", new_callable=AsyncMock),
        ):
            for i in range(3):
                res = await client.post(
                    "/runs",
                    json={"question": f"Valid question {i} for AAPL?"},
                )
                assert res.status_code == 200, f"Request {i} should succeed"

            # 4th request must be rate limited with 429
            blocked_res = await client.post(
                "/runs",
                json={"question": "Rate limited request for AAPL?"},
            )
            assert blocked_res.status_code == 429
            assert "Rate limit exceeded" in blocked_res.json()["detail"]
            assert "Retry-After" in blocked_res.headers

    default_rate_limiter.reset()


def test_max_question_length_validation() -> None:
    """Questions exceeding configured MAX_QUESTION_LENGTH are rejected with 422."""
    with patch.object(get_settings(), "MAX_QUESTION_LENGTH", 50):
        # Within limit
        req = CreateRunRequest(question="What is Apple's gross margin?")
        assert len(req.question) <= 50

        # Exceeding limit
        long_q = "A" * 51
        with pytest.raises(ValueError, match="exceeds maximum allowed length"):
            CreateRunRequest(question=long_q)


def test_ticker_allowlist_validation_request() -> None:
    """CreateRunRequest validates ticker against allowlist when configured."""
    with (
        patch.object(get_settings(), "TICKER_ALLOWLIST", ["AAPL", "MSFT"]),
        patch.object(get_settings(), "TICKER_ALLOWLIST_ENABLED", True),
    ):
        # Approved ticker succeeds
        valid_req = CreateRunRequest(
            question="Analyze valuation",
            ticker="AAPL",
        )
        assert valid_req.ticker == "AAPL"

        # Disapproved ticker fails validation
        with pytest.raises(ValueError, match="not in the allowed universe"):
            CreateRunRequest(
                question="Analyze valuation",
                ticker="DISALLOWED_TICKER",
            )


@pytest.mark.asyncio
async def test_ticker_allowlist_in_query_understanding() -> None:
    """Query understanding rejects tickers outside configured allowlist."""
    from lekka_ai.llm.base import LLMResponse

    with (
        patch.object(get_settings(), "TICKER_ALLOWLIST", ["AAPL", "GOOGL"]),
        patch.object(get_settings(), "TICKER_ALLOWLIST_ENABLED", True),
    ):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = LLMResponse(
            text=(
                '{"intent": "investment_evaluation", '
                '"tickers": ["TSLA"], "company_names": ["Tesla"]}'
            )
        )

        with pytest.raises(UnknownTickerError, match="allowed ticker universe"):
            await understand_query("Analyze TSLA revenue", llm=mock_llm)


@pytest.mark.asyncio
async def test_run_concurrency_limit() -> None:
    """POST /runs rejects requests when active runs reach MAX_CONCURRENT_RUNS."""
    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        with (
            patch.object(get_settings(), "MAX_CONCURRENT_RUNS", 2),
            patch.object(RunManager, "get_active_runs_count", return_value=2),
        ):
            res = await client.post(
                "/runs",
                json={"question": "Concurrent test for AAPL?"},
            )
            assert res.status_code == 429
            assert "Maximum concurrent runs limit reached" in res.json()["detail"]


def test_hard_per_run_cost_cap_enforcement() -> None:
    """Hard per-run cost cap overrides higher requested budget values."""
    with patch.object(get_settings(), "HARD_PER_RUN_COST_CAP_USD", 1.5):
        # 1. Enforced on CreateRunRequest
        req = CreateRunRequest(
            question="High budget run",
            budget={"cost": 10.0, "max_usd": 25.0},
        )
        assert req.budget is not None
        assert req.budget["cost"] <= 1.5
        assert req.budget["max_usd"] <= 1.5

        # 2. Enforced on RunBudget
        budget = RunBudget.from_settings(cost=10.0)
        assert budget.cost == 1.5

        # 3. Exhausted when spent reaches hard cap
        budget.used_cost = 1.49
        assert not budget.is_exhausted()
        assert budget.can_allocate()

        budget.used_cost = 1.50
        assert budget.is_exhausted()
        assert not budget.can_allocate()


def test_redaction_of_secrets_in_logs_and_traces() -> None:
    """Secrets, API keys, and sensitive tokens are masked in logs and traces."""
    # 1. Redact secrets in dict keys
    payload = {
        "api_key": "AQ.Ab8RN6Jo3f61w4tCaPBu7zA-JW--xaN8nKtFSjLSN_vT1aV29g",
        "secret_token": "sk-1234567890abcdef1234567890abcdef",
        "gemini_api_key": "AQ.secret123456789012345678901234567890",
        "password": "supersecretpassword",
        "nested": {
            "auth": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "safe_param": "AAPL",
        },
    }
    redacted = redact_secrets(payload)
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["secret_token"] == "[REDACTED]"
    assert redacted["gemini_api_key"] == "[REDACTED]"
    assert redacted["password"] == "[REDACTED]"
    assert redacted["nested"]["auth"] == "[REDACTED]"
    assert redacted["nested"]["safe_param"] == "AAPL"

    # 2. Redact pattern inside raw string
    raw_str = (
        "Error calling Gemini with key "
        "AQ.Ab8RN6Jo3f61w4tCaPBu7zA-JW--xaN8nKtFSjLSN_vT1aV29g failed"
    )
    masked_str = redact_secrets(raw_str)
    assert "AQ.Ab8RN6Jo" not in masked_str
    assert "[REDACTED]" in masked_str

    # 3. Structlog processor
    event = {"event": "call_api", "api_key": "secret_key"}
    processed = redact_secrets_processor(logging.getLogger(), "info", event)
    assert processed["api_key"] == "[REDACTED]"

    # 4. Trace Span input, output, and metadata redaction
    span = Span(
        name="test_span",
        input_json={"api_key": "my_secret_key", "ticker": "AAPL"},
        output_json={"auth_token": "bearer_secret", "result": 42},
        metadata={"sec_user_agent": "secret_agent", "info": "ok"},
    )
    assert isinstance(span.input_json, dict)
    assert span.input_json["api_key"] == "[REDACTED]"
    assert span.input_json["ticker"] == "AAPL"
    assert isinstance(span.output_json, dict)
    assert span.output_json["auth_token"] == "[REDACTED]"
    assert span.output_json["result"] == 42
    assert isinstance(span.metadata, dict)
    assert span.metadata["sec_user_agent"] == "[REDACTED]"
    assert span.metadata["info"] == "ok"


def test_visible_disclaimer_in_report_and_response() -> None:
    """Synthesized reports and API models guarantee a visible disclaimer."""
    # SynthesizedReport model validator
    report = SynthesizedReport(
        markdown="# AAPL Report\n\nRevenue grew 10% [claim_1].",
        sentence_to_claims={"Revenue grew 10%": ["claim_1"]},
    )
    assert "research tool, not financial advice" in report.markdown.lower()

    # RunReportResponse model validator
    resp = RunReportResponse(
        run_id="run_123",
        markdown="# Summary Report",
    )
    assert resp.disclaimer == "Lekka AI is a research tool, not financial advice."
    assert "research tool, not financial advice" in resp.markdown.lower()


def test_startup_check_fault_injection_security() -> None:
    """Startup verification raises RuntimeError if fault injection enabled outside
    test or dev environments.
    """
    # 1. Dev/test allows fault injection
    dev_settings = Settings(
        ENV="development",
        DATABASE_URL="postgresql+psycopg://user:pass@localhost:5432/db",
        GEMINI_API_KEY="test-key",
        GEMINI_MODEL="gemini-flash",
        GEMINI_EMBEDDING_MODEL="embedding-004",
        SEC_USER_AGENT="test@lekka.ai",
        FAULT_INJECTION_ENABLED=True,
    )
    verify_startup_configuration(dev_settings)  # Should not raise

    # 2. Production with fault injection disabled is allowed
    prod_clean = Settings(
        ENV="production",
        DATABASE_URL="postgresql+psycopg://user:pass@localhost:5432/db",
        GEMINI_API_KEY="test-key",
        GEMINI_MODEL="gemini-flash",
        GEMINI_EMBEDDING_MODEL="embedding-004",
        SEC_USER_AGENT="test@lekka.ai",
        FAULT_INJECTION_ENABLED=False,
    )
    verify_startup_configuration(prod_clean)  # Should not raise

    # 3. Production with fault injection enabled MUST raise RuntimeError
    prod_bad = Settings(
        ENV="production",
        DATABASE_URL="postgresql+psycopg://user:pass@localhost:5432/db",
        GEMINI_API_KEY="test-key",
        GEMINI_MODEL="gemini-flash",
        GEMINI_EMBEDDING_MODEL="embedding-004",
        SEC_USER_AGENT="test@lekka.ai",
        FAULT_INJECTION_ENABLED=True,
    )
    with pytest.raises(RuntimeError, match="Fault injection is enabled in non-dev"):
        verify_startup_configuration(prod_bad)


@pytest.mark.asyncio
async def test_fault_plan_rejected_outside_dev() -> None:
    """POST /runs with fault_plan returns 403 Forbidden in non-dev environments."""
    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        with patch.object(get_settings(), "ENV", "production"):
            res = await client.post(
                "/runs",
                json={
                    "question": "Can I run a fault plan in production?",
                    "fault_plan": {"drop_calls": True},
                },
            )
            assert res.status_code == 403
            assert "strictly prohibited" in res.json()["detail"]
