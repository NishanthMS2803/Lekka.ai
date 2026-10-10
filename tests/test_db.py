"""Tests for database connectivity, sessions, and repository operations."""

from datetime import date

import pytest
from pydantic import BaseModel, Field
from sqlalchemy import text

from lekka_ai.db import (
    create_agent_run,
    create_run,
    create_task,
    get_engine,
    get_run,
    get_session,
    get_tool_call,
)
from lekka_ai.tools.base import Tool
from lekka_ai.tools.echo import EchoTool
from lekka_ai.tracing import Tracer, wrap_tool_run


class FailingInput(BaseModel):
    """Input payload for FailingTool."""

    value: int = Field(..., description="Integer value to test")


class FailingOutput(BaseModel):
    """Output payload for FailingTool."""

    result: int = Field(..., description="Calculation result")


class FailingTool(Tool[FailingInput, FailingOutput]):
    """Tool that deliberately raises an error for failure testing."""

    name: str = "failing_calc"
    description: str = "A tool that throws an error during calculation."
    input_model: type[FailingInput] = FailingInput
    output_model: type[FailingOutput] = FailingOutput

    async def run(self, input_data: FailingInput) -> FailingOutput:
        """Raise an exception to test error tracing."""
        raise RuntimeError(f"Division by zero in financial formula: {input_data.value}")


def is_database_reachable() -> bool:
    """Check if the database is reachable."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.mark.db
def test_session_select_one() -> None:
    """Open a session and execute SELECT 1, skipping if DB is unavailable."""
    if not is_database_reachable():
        pytest.skip("Database is unavailable")

    session_generator = get_session()
    session = next(session_generator)
    try:
        result = session.execute(text("SELECT 1")).scalar_one()
        assert result == 1
    finally:
        try:
            next(session_generator)
        except StopIteration:
            pass


@pytest.mark.db
def test_create_and_get_run_repository() -> None:
    """Insert and read back a run using repository functions."""
    if not is_database_reachable():
        pytest.skip("Database is unavailable")

    session_generator = get_session()
    session = next(session_generator)
    try:
        new_run = create_run(
            session,
            question="What is Apple's revenue for FY2024?",
            status="running",
            as_of_date=date(2024, 9, 30),
            budget_json={"max_subagents": 4, "max_tokens": 50000},
        )
        assert new_run.id is not None
        assert new_run.question == "What is Apple's revenue for FY2024?"

        fetched_run = get_run(session, new_run.id)
        assert fetched_run is not None
        assert fetched_run.id == new_run.id
        assert fetched_run.question == "What is Apple's revenue for FY2024?"
        assert fetched_run.status == "running"
        assert fetched_run.as_of_date == date(2024, 9, 30)
        assert fetched_run.budget_json == {
            "max_subagents": 4,
            "max_tokens": 50000,
        }
        assert fetched_run.created_at is not None
    finally:
        try:
            next(session_generator)
        except StopIteration:
            pass


@pytest.mark.db
@pytest.mark.asyncio
async def test_tracing_tool_call_persisted_to_db() -> None:
    """Successful tool execution persists a ToolCall record to the database."""
    if not is_database_reachable():
        pytest.skip("Database is unavailable")

    session_generator = get_session()
    session = next(session_generator)
    try:
        run = create_run(session, question="Calculate profit margins")
        task = create_task(
            session,
            run_id=run.id,
            agent_type="analyst",
            instruction="Execute tool call",
        )
        agent_run = create_agent_run(session, task_id=task.id)

        tracer = Tracer(session=session, agent_run_id=agent_run.id)
        tool = EchoTool()
        traced_tool = wrap_tool_run(tracer, tool)

        result = await traced_tool({"message": "database tool test"})
        assert result.echoed == "database tool test"

        assert len(tracer.spans) == 1
        span = tracer.spans[0]
        assert span.tool_call_id is not None

        db_call = get_tool_call(session, span.tool_call_id)
        assert db_call is not None
        assert db_call.agent_run_id == agent_run.id
        assert db_call.tool_name == "echo"
        assert db_call.input_json == {"message": "database tool test"}
        assert db_call.output_json == {"echoed": "database tool test"}
        assert db_call.error is None
        assert db_call.latency_ms is not None
        assert db_call.latency_ms >= 0
    finally:
        try:
            next(session_generator)
        except StopIteration:
            pass


@pytest.mark.db
@pytest.mark.asyncio
async def test_tracing_failing_tool_persisted_to_db() -> None:
    """Failing tool still writes an error record and latency to tool_calls table."""
    if not is_database_reachable():
        pytest.skip("Database is unavailable")

    session_generator = get_session()
    session = next(session_generator)
    try:
        run = create_run(session, question="Analyze broken ledger")
        task = create_task(
            session,
            run_id=run.id,
            agent_type="auditor",
            instruction="Run audited check",
        )
        agent_run = create_agent_run(session, task_id=task.id)

        tracer = Tracer(session=session, agent_run_id=agent_run.id)
        failing_tool = FailingTool()
        traced_tool = wrap_tool_run(tracer, failing_tool)

        with pytest.raises(RuntimeError, match="Division by zero in financial formula"):
            await traced_tool({"value": 42})

        assert len(tracer.spans) == 1
        span = tracer.spans[0]
        assert span.status == "error"
        assert span.tool_call_id is not None

        db_call = get_tool_call(session, span.tool_call_id)
        assert db_call is not None
        assert db_call.agent_run_id == agent_run.id
        assert db_call.tool_name == "failing_calc"
        assert db_call.input_json == {"value": 42}
        assert db_call.output_json is None
        assert db_call.error is not None
        assert "Division by zero in financial formula: 42" in db_call.error
        assert db_call.latency_ms is not None
        assert db_call.latency_ms >= 0
    finally:
        try:
            next(session_generator)
        except StopIteration:
            pass
