"""Structured logging configuration and context helpers."""

import logging
import sys
from typing import Any, TextIO

import structlog
from structlog.types import Processor


def configure_logging(
    env: str = "development",
    stream: TextIO | None = None,
) -> None:
    """Configure structured logging (JSON in production, pretty in development)."""
    target_stream: TextIO = stream if stream is not None else sys.stdout
    is_prod = env.lower() in {"production", "prod"}

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Processor
    if is_prod:
        renderer = structlog.processors.JSONRenderer()
    else:
        use_colors = hasattr(target_stream, "isatty") and target_stream.isatty()
        renderer = structlog.dev.ConsoleRenderer(colors=use_colors)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(target_stream)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


def bind_context(
    *,
    run_id: str | None = None,
    task_id: str | None = None,
    agent_id: str | None = None,
    **extra: Any,
) -> None:
    """Bind trace context variables (run_id, task_id, agent_id) to context."""
    context: dict[str, Any] = {**extra}
    if run_id is not None:
        context["run_id"] = run_id
    if task_id is not None:
        context["task_id"] = task_id
    if agent_id is not None:
        context["agent_id"] = agent_id

    structlog.contextvars.bind_contextvars(**context)


def bind_run_id(run_id: str) -> None:
    """Bind run_id to current logging context."""
    structlog.contextvars.bind_contextvars(run_id=run_id)


def bind_task_id(task_id: str) -> None:
    """Bind task_id to current logging context."""
    structlog.contextvars.bind_contextvars(task_id=task_id)


def bind_agent_id(agent_id: str) -> None:
    """Bind agent_id to current logging context."""
    structlog.contextvars.bind_contextvars(agent_id=agent_id)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
