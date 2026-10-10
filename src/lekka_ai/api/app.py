"""FastAPI application factory and API configuration."""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware.base import RequestResponseEndpoint

from lekka_ai.logging import bind_request_id, clear_context, get_logger

logger = get_logger(__name__)

API_VERSION = "0.1.0"


class HealthResponse(BaseModel):
    """Health check response payload."""

    status: str = "ok"
    version: str = API_VERSION


def create_app(cors_origins: list[str] | None = None) -> FastAPI:
    """Create and configure a FastAPI application instance."""
    app = FastAPI(
        title="Lekka AI API",
        description="Multi-agent financial research system API",
        version=API_VERSION,
    )

    # Configure CORS
    origins = cors_origins if cors_origins is not None else ["http://localhost:3000"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Logging and request-id tracing middleware
    @app.middleware("http")
    async def log_requests(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        clear_context()
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        bind_request_id(request_id)

        start_time = time.perf_counter()
        logger.info(
            "HTTP request started",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else None,
        )

        try:
            response = await call_next(request)
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            response.headers["X-Request-ID"] = request_id
            logger.info(
                "HTTP request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                latency_ms=latency_ms,
            )
            return response
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(
                "HTTP request failed",
                method=request.method,
                path=request.url.path,
                error=str(exc),
                latency_ms=latency_ms,
            )
            raise

    # Health check route
    @app.get("/health", response_model=HealthResponse)
    async def health() -> dict[str, Any]:
        """Return application health status and version."""
        return {"status": "ok", "version": API_VERSION}

    return app


app = create_app()
