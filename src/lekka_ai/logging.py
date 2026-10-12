"""Structured logging configuration and context helpers."""

import logging
import re
import sys
from typing import Any, TextIO

import structlog
from structlog.types import Processor

SENSITIVE_EXACT_KEYS = {
    "token",
    "auth_token",
    "access_token",
    "refresh_token",
    "api_key",
    "apikey",
    "secret",
    "password",
    "authorization",
    "auth",
    "cookie",
    "jwt",
    "private_key",
    "client_secret",
}

SENSITIVE_PATTERNS = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "private_key",
    "bearer",
    "gemini_api_key",
    "sec_user_agent",
)

SECRET_REGEXES = [
    re.compile(r"AQ\.[a-zA-Z0-9_\-]{20,}", re.IGNORECASE),
    re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
    re.compile(r"sk-[a-zA-Z0-9_\-]{20,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[a-zA-Z0-9\-._~+/]+=*", re.IGNORECASE),
]


def is_sensitive_key(k: Any) -> bool:
    """Check if dictionary key represents a credential or secret."""
    key_lower = str(k).lower().strip()
    # Token count/usage metrics should not be treated as secrets
    if "tokens" in key_lower:
        return False
    if key_lower in SENSITIVE_EXACT_KEYS:
        return True
    if any(p in key_lower for p in SENSITIVE_PATTERNS):
        return True
    if key_lower.endswith("_token") or key_lower.startswith("token_"):
        return True
    return False


def redact_secrets(val: Any) -> Any:
    """Recursively mask secrets and credentials in logs and trace payloads."""
    if isinstance(val, dict):
        redacted_dict: dict[Any, Any] = {}
        for k, v in val.items():
            if is_sensitive_key(k):
                redacted_dict[k] = "[REDACTED]"
            else:
                redacted_dict[k] = redact_secrets(v)
        return redacted_dict
    elif isinstance(val, list):
        return [redact_secrets(item) for item in val]
    elif isinstance(val, tuple):
        return tuple(redact_secrets(item) for item in val)
    elif isinstance(val, set):
        return {redact_secrets(item) for item in val}
    elif isinstance(val, str):
        res = val
        for rx in SECRET_REGEXES:
            res = rx.sub("[REDACTED]", res)
        return res
    return val


def redact_secrets_processor(
    _logger: Any,
    _method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Structlog processor to sanitize sensitive tokens and credentials."""
    return redact_secrets(event_dict)


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
        redact_secrets_processor,
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


def bind_request_id(request_id: str) -> None:
    """Bind request_id to current logging context."""
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
