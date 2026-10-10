# Lekka AI

[![CI](https://github.com/NishanthMS2803/Lekka.ai/actions/workflows/ci.yml/badge.svg)](https://github.com/NishanthMS2803/Lekka.ai/actions/workflows/ci.yml)

A multi-agent financial research system designed for rigorous fundamental and quantitative equity analysis using autonomous specialist subagents, deterministic financial tools, and grounded evidence.

---

## Architecture & Purpose

Lekka AI coordinates specialist agents (fundamental analysts, quant auditors, filing researchers) to answer complex financial questions. Key design principles:
- **Zero Financial Hallucination**: Financial numbers and metrics originate exclusively from deterministic tools or official filings, never model memory.
- **Audited Tool Harness & Tracing**: Every calculation, API query, and LLM call is traced with execution latency, token usage, and status persisted to PostgreSQL.
- **Type Safety & Schema Validation**: Built with Python 3.12, Pydantic v2, SQLAlchemy 2.0, and strict Mypy checking.

---

## Prerequisites

- [Python 3.12+](https://www.python.org/)
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)
- [Docker](https://www.docker.com/) & Docker Compose

---

## Setup Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/NishanthMS2803/Lekka.ai.git
cd "Lekka AI"
make install
```

### 2. Configure Environment Variables
Copy the example environment configuration and supply your API keys:
```bash
cp .env.example .env
```
Key configuration fields in `.env`:
- `DATABASE_URL`: Connection string for PostgreSQL (e.g. `postgresql+psycopg://lekka_ai:lekka_ai@localhost:5432/lekka_ai`)
- `GEMINI_API_KEY`: API key for Google Gemini (`gemini-3.8-flash`)
- `SEC_USER_AGENT`: User-Agent header formatted as `SampleCompany AdminContact@<sample company domain>.com` for SEC EDGAR compliance

### 3. Start Database Service
Start the PostgreSQL 16 container with `pgvector` extension:
```bash
make db-up
```

### 4. Run Database Migrations
Apply Alembic migrations to set up core tables (`runs`, `tasks`, `agent_runs`, `tool_calls`):
```bash
uv run alembic upgrade head
```

### 5. Launch the API Server
Start the FastAPI development server:
```bash
make run
```
Access the interactive OpenAPI Swagger UI at `http://localhost:8000/docs`.

---

## Makefile Targets

| Command | Description |
|---|---|
| `make install` | Synchronize and install dependencies using `uv sync` |
| `make lint` | Run Ruff linter and format verification |
| `make typecheck` | Run strict Mypy type-checking across `src` and `tests` |
| `make test` | Run Pytest test suite (skips external live provider tests) |
| `make run` | Run FastAPI application on port 8000 with auto-reload |
| `make db-up` | Start background PostgreSQL + pgvector Docker container |
| `make db-down` | Stop PostgreSQL Docker container |
| `make db-logs` | Stream logs from the database container |
