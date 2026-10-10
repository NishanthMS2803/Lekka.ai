"""Tests for FastAPI application, health endpoint, middleware, and CORS."""

import pytest
from httpx import ASGITransport, AsyncClient

from lekka_ai.api.app import API_VERSION, create_app


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    """GET /health returns 200, status ok, version, and X-Request-ID header."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["version"] == API_VERSION
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0


@pytest.mark.asyncio
async def test_health_with_custom_request_id() -> None:
    """X-Request-ID provided in request headers is preserved in response headers."""
    app = create_app()
    transport = ASGITransport(app=app)
    custom_id = "test-request-id-12345"
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(
            "/health",
            headers={"X-Request-ID": custom_id},
        )
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == custom_id


@pytest.mark.asyncio
async def test_cors_headers() -> None:
    """CORS preflight request from localhost:3000 returns allow origin."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert (
            response.headers.get("access-control-allow-origin")
            == "http://localhost:3000"
        )


@pytest.mark.asyncio
async def test_docs_and_openapi_available() -> None:
    """OpenAPI schema and /docs Swagger UI endpoints are accessible."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        openapi_res = await client.get("/openapi.json")
        assert openapi_res.status_code == 200
        schema = openapi_res.json()
        assert "/health" in schema["paths"]

        docs_res = await client.get("/docs")
        assert docs_res.status_code == 200
