"""API package for Lekka AI."""

from lekka_ai.api.app import app, create_app
from lekka_ai.api.dependencies import (
    get_event_bus,
    get_llm,
    get_optional_db_session,
    get_run_manager,
    get_task_runner,
    get_tool_registry,
)
from lekka_ai.api.models import (
    ClaimResponse,
    CostTotalsResponse,
    CreateRunRequest,
    EvidenceResponse,
    RunCreateResponse,
    RunReportResponse,
    RunStatusResponse,
    RunTraceResponse,
    SpanResponse,
    claim_to_response,
)
from lekka_ai.api.routes import claims_router, runs_router
from lekka_ai.api.services import RunManager, RunRecord, default_run_manager
from lekka_ai.events import EventBus, EventType, RunEvent, default_event_bus

__all__ = [
    "ClaimResponse",
    "CostTotalsResponse",
    "CreateRunRequest",
    "EventBus",
    "EventType",
    "EvidenceResponse",
    "RunCreateResponse",
    "RunEvent",
    "RunManager",
    "RunRecord",
    "RunReportResponse",
    "RunStatusResponse",
    "RunTraceResponse",
    "SpanResponse",
    "app",
    "claim_to_response",
    "claims_router",
    "create_app",
    "default_event_bus",
    "default_run_manager",
    "get_event_bus",
    "get_llm",
    "get_optional_db_session",
    "get_run_manager",
    "get_task_runner",
    "get_tool_registry",
    "runs_router",
]
