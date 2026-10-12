"""Error models, retry classifiers, and repeat detection utilities."""

from __future__ import annotations

import json
from typing import Any


class ToolError(Exception):
    """Base exception for all tool execution errors in the harness."""

    retryable: bool = False

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class RetryableToolError(ToolError):
    """Transient tool failure eligible for retry with backoff."""

    retryable: bool = True

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class RepeatedToolCallError(ToolError):
    """Raised or reported when repeated identical tool calls are detected."""

    retryable: bool = False

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


def is_retryable_tool_error(exc: BaseException) -> bool:
    """Determine whether a tool execution exception is retryable.

    Inspects explicit retryable flags, standard OS/network timeout exceptions,
    and HTTP transient status codes (429, 500, 502, 503, 504).
    """
    if getattr(exc, "retryable", False) is True:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True

    try:
        import httpx

        if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            if exc.response.status_code in {429, 500, 502, 503, 504}:
                return True
    except ImportError:
        pass

    return False


def format_tool_error(exc: BaseException) -> dict[str, Any]:
    """Convert an exception into a structured error observation dictionary."""
    return {
        "error": True,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "retryable": is_retryable_tool_error(exc),
    }


def canonicalize_tool_args(args: Any) -> str:
    """Canonicalize tool input arguments into a stable JSON string for comparison."""
    if hasattr(args, "model_dump"):
        dumped = args.model_dump()
    elif isinstance(args, dict):
        dumped = args
    elif isinstance(args, (list, tuple)):
        dumped = list(args)
    else:
        return str(args)

    try:
        return json.dumps(dumped, sort_keys=True, default=str)
    except Exception:
        return str(dumped)
