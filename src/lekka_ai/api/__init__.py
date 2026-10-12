"""API package for Lekka AI."""

from lekka_ai.api.app import app, create_app
from lekka_ai.api.dependencies import (
    get_llm,
    get_optional_db_session,
    get_run_manager,
    get_task_runner,
    get_tool_registry,
)
from lekka_ai.api.models import (
    ClaimResponse,
    CreateRunRequest,
    EvidenceResponse,
    RunCreateResponse,
    RunReportResponse,
    RunStatusResponse,
    claim_to_response,
)
from lekka_ai.api.routes import claims_router, runs_router
from lekka_ai.api.services import RunManager, RunRecord, default_run_manager

__all__ = [
    "ClaimResponse",
    "CreateRunRequest",
    "EvidenceResponse",
    "RunCreateResponse",
    "RunManager",
    "RunRecord",
    "RunReportResponse",
    "RunStatusResponse",
    "app",
    "claim_to_response",
    "claims_router",
    "create_app",
    "default_run_manager",
    "get_llm",
    "get_optional_db_session",
    "get_run_manager",
    "get_task_runner",
    "get_tool_registry",
    "runs_router",
]
