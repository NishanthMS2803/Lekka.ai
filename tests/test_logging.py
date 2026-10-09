"""Tests for structured logging and context variable binding."""

import io
import json

from lekka_ai.logging import (
    bind_agent_id,
    bind_context,
    bind_run_id,
    bind_task_id,
    clear_context,
    configure_logging,
    get_logger,
)


def test_bound_ids_appear_in_json_output() -> None:
    """Bound run_id, task_id, and agent_id appear in JSON log output."""
    stream = io.StringIO()
    configure_logging(env="production", stream=stream)
    clear_context()

    bind_context(run_id="run-001", task_id="task-002", agent_id="agent-003")
    logger = get_logger("test.prod")
    logger.info("executing subagent task")

    output = stream.getvalue().strip()
    data = json.loads(output)

    assert data["event"] == "executing subagent task"
    assert data["run_id"] == "run-001"
    assert data["task_id"] == "task-002"
    assert data["agent_id"] == "agent-003"
    assert data["level"] == "info"

    clear_context()


def test_bound_ids_appear_in_dev_output() -> None:
    """Bound trace IDs appear in dev console log output."""
    stream = io.StringIO()
    configure_logging(env="development", stream=stream)
    clear_context()

    bind_run_id("run-dev-1")
    bind_task_id("task-dev-2")
    bind_agent_id("agent-dev-3")

    logger = get_logger("test.dev")
    logger.info("dev log message")

    output = stream.getvalue()
    assert "dev log message" in output
    assert "run-dev-1" in output
    assert "task-dev-2" in output
    assert "agent-dev-3" in output

    clear_context()


def test_clear_context_removes_bound_ids() -> None:
    """clear_context ensures previous IDs do not leak into future log lines."""
    stream = io.StringIO()
    configure_logging(env="production", stream=stream)
    clear_context()

    bind_context(run_id="temp-run", task_id="temp-task", agent_id="temp-agent")
    clear_context()

    logger = get_logger("test.clear")
    logger.info("unbound event")

    output = stream.getvalue().strip()
    data = json.loads(output)

    assert data["event"] == "unbound event"
    assert "run_id" not in data
    assert "task_id" not in data
    assert "agent_id" not in data
