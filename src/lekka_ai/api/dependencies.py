"""FastAPI dependency injection providers for LLM, runners, and storage."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy.orm import Session

from lekka_ai.api.services import RunManager, default_run_manager
from lekka_ai.config import get_settings
from lekka_ai.db.session import get_session
from lekka_ai.events import EventBus, default_event_bus
from lekka_ai.llm.base import LLMClient
from lekka_ai.llm.gemini import GeminiClient
from lekka_ai.orchestrator.engine import TaskRunnerCallable
from lekka_ai.tools.base import ToolRegistry


def get_llm() -> LLMClient:
    """Provide default Gemini LLM client configured from environment."""
    settings = get_settings()
    return GeminiClient(
        api_key=settings.GEMINI_API_KEY,
        model=settings.GEMINI_MODEL,
    )


def get_task_runner() -> TaskRunnerCallable | None:
    """Provide task runner callable (overrideable in tests)."""
    return None


def get_tool_registry() -> ToolRegistry | None:
    """Provide tool registry (overrideable in tests)."""
    return None


def get_run_manager() -> RunManager:
    """Provide the active RunManager instance."""
    return default_run_manager


def get_event_bus() -> EventBus:
    """Provide the active EventBus instance."""
    return default_event_bus


def get_optional_db_session() -> Generator[Session | None, None, None]:
    """Yield database session if reachable, otherwise None."""
    session: Session | None = None
    session_gen: Generator[Session, None, None] | None = None
    try:
        session_gen = get_session()
        session = next(session_gen)
    except Exception:
        session = None

    try:
        yield session
    finally:
        if session_gen is not None:
            try:
                next(session_gen)
            except (StopIteration, Exception):
                pass
