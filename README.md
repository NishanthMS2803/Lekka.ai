# Lekka AI

[![CI](https://github.com/NishanthMS2803/Lekka.ai/actions/workflows/ci.yml/badge.svg)](https://github.com/NishanthMS2803/Lekka.ai/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Next.js 16](https://img.shields.io/badge/next.js-16-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multi-agent financial research system designed for rigorous fundamental and quantitative equity analysis using autonomous specialist subagents, deterministic financial tools, a shared claim ledger, and verifiable evidence provenance.

> **Disclaimer:** Lekka AI is a research tool, not financial advice.

---

## Table of Contents

- [What It Is](#what-it-is)
- [Architecture Diagram](#architecture-diagram)
- [End-to-End Execution Flow](#end-to-end-execution-flow)
- [Getting Started & Setup](#getting-started--setup)
  - [Prerequisites](#prerequisites)
  - [Option A: Local Development with uv](#option-a-local-development-with-uv)
  - [Option B: Docker Full-Stack Deployment](#option-b-docker-full-stack-deployment)
  - [Environment Configuration (`.env`)](#environment-configuration-env)
- [How to Run a Research Question](#how-to-run-a-research-question)
  - [1. Web Interface (Next.js)](#1-web-interface-nextjs)
  - [2. Interactive CLI](#2-interactive-cli)
  - [3. REST & Streaming API (cURL)](#3-rest--streaming-api-curl)
- [How to Run Evaluations](#how-to-run-evaluations)
- [Project Structure](#project-structure)
- [Core Design Decisions](#core-design-decisions)
  - [1. Sandboxed Agent Harness](#1-sandboxed-agent-harness)
  - [2. Shared Claim Ledger](#2-shared-claim-ledger)
  - [3. Claim Taxonomy](#3-claim-taxonomy)
  - [4. Hierarchical Budgets & Cost Caps](#4-hierarchical-budgets--cost-caps)
  - [5. Deterministic Confidence & Abstention](#5-deterministic-confidence--abstention)
- [Documentation & Deep Dives](#documentation--deep-dives)

---

## What It Is

Lekka AI solves the critical failure modes of applying generic Large Language Models to equity research: **hallucinated metrics, temporal lookahead bias, ungrounded mental math, and unchecked prompt injection**.

Instead of allowing an LLM to generate financial assertions from memory, Lekka AI orchestrates a team of autonomous specialist agents (Fundamental, Valuation, Market, News, and Risk) constrained by a secure execution harness. Key features include:

- **Zero Parametric Math**: Valuation outputs (DCF, Reverse DCF, multiples, scenario grids) originate exclusively from deterministic mathematical tools.
- **Audited Claim Ledger**: Every statement in a synthesized report is backed by an atomic claim linked to raw tool payloads and official SEC EDGAR accession numbers.
- **Strict Point-in-Time Filters (`as_of_date`)**: Prevents future information leakage across all filings, daily price bars, and analyst consensus estimates.
- **Deterministic Confidence & Abstention**: Computes confidence via six explicit mathematical rules. Questions lacking verifiable evidence route to a structured abstention report with zero fabricated numbers.
- **Full-Stack Observability & Trace Viewer**: End-to-end telemetry with interactive waterfall timelines, latency bars, token spend, and estimated dollar costs.

---

## Architecture Diagram

```mermaid
flowchart TD
    User([User / Web UI / CLI / API]) --> Gateway[FastAPI Gateway<br/>Rate Limiting, Concurrency Gate, Cost Cap]

    subgraph Orchestrator [Multi-Agent Orchestration Pipeline]
        Gateway --> Understand[1. Query Understanding<br/>Intent, Ticker, Ambiguity Check]
        Understand -->|Ambiguous| Clarify([Demand Clarification])
        Understand -->|Clear| Planner[2. Dynamic DAG Planner<br/>Decomposition & Budget Allocation]
        
        Planner --> Dispatcher[3. Task Dispatcher<br/>Parallel Specialist Execution]
        
        subgraph Specialists [Specialist Subagent Swarm]
            Fund[Fundamental Specialist<br/>SEC Filings, XBRL Statements]
            Val[Valuation Specialist<br/>DCF, Reverse DCF, Multiples]
            Mkt[Market Specialist<br/>Price Action, Volatility, Beta]
            News[News Specialist<br/>8-K Catalysts, Untrusted Delimiters]
            Risk[Risk Specialist<br/>Downside Theses, Debt Red Flags]
        end
        
        Dispatcher --> Fund
        Dispatcher --> Val
        Dispatcher --> Mkt
        Dispatcher --> News
        Dispatcher --> Risk
        
        Fund --> Ledger[(4. Shared Claim Ledger<br/>Atomic Facts & Calculations)]
        Val --> Ledger
        Mkt --> Ledger
        News --> Ledger
        Risk --> Ledger
        
        Ledger --> Verifier[5. Cross-Source Verifier<br/>Conflict Detection & Reconciliation]
        Verifier --> GapCheck{6. Gap Check<br/>Sufficient Evidence?}
        GapCheck -->|Gaps Found| Planner
        GapCheck -->|Complete| Confidence[7. Deterministic Confidence Scoring<br/>6 Mathematical Rules]
        
        Confidence -->|< 0.60| Abstain([Abstention Report<br/>Missing Data Audit & Zero Numbers])
        Confidence -->|>= 0.60| Synth[8. Report Synthesizer<br/>Strict Ledger Citations & Post-Check]
        Synth --> FinalReport([Final Equity Research Report])
    end

    subgraph Observability [Telemetry & Streaming Engine]
        Dispatcher -.-> SSE[Event Bus & Live SSE<br/>GET /runs/{id}/events]
        Dispatcher -.-> Trace[Trace Viewer & OpenTelemetry<br/>GET /runs/{id}/trace]
    end

    subgraph Storage [Persistence Layer]
        Postgres[(PostgreSQL 16 & pgvector<br/>Runs, Tasks, Claims, Filing Chunks)]
    end

    Orchestrator --> Storage
```

---

## End-to-End Execution Flow

1. **Submission & Ingestion**: The client sends a question and an optional `as_of_date` via the Next.js UI, CLI, or `POST /runs`.
2. **Gateway Hardening**: The request is checked against a sliding-window IP rate limiter, a 2,000-character length limit, a ticker allowlist, and an active run concurrency gate.
3. **Query Understanding**: The system classifies user intent, extracts stock tickers, and verifies temporal validity. Ambiguous queries halt immediately to demand clarification.
4. **DAG Planning & Budgeting**: The planner decomposes the query into parallel tasks and divides the per-run token and dollar budget among specialists.
5. **Sandboxed Execution**: Subagents execute authorized tools through sandboxed `ToolView` proxies. Unpermitted tools raise `PermissionDenied`. Untrusted third-party news content is enclosed in `<untrusted_data>` tags.
6. **Claim Ledger Ingestion**: Subagents submit structured claims (`fact`, `calculation`, `interpretation`, `background`) linked to underlying tool outputs.
7. **Verification & Conflict Resolution**: Cross-agent reconciliation checks claims for numeric contradictions, applying confidence penalties to discrepancies.
8. **Gap Checking**: If essential financial statements or valuation metrics are missing, the orchestrator spawns follow-up tasks.
9. **Confidence & Routing**: The confidence score is calculated. If confidence is below 0.60, an abstention report is produced. Otherwise, the report synthesizer compiles a 9-section report.
10. **Anti-Hallucination Regex Post-Check**: Every sentence in the synthesized report is checked against cited claims. If ungrounded numbers appear, automated repair triggers.
11. **Streaming & Tracing**: Live execution events stream via Server-Sent Events (SSE), and all execution spans, token usage, and dollar costs are persisted for the trace viewer.

---

## Getting Started & Setup

### Prerequisites

- **Python 3.12+**
- [**uv**](https://github.com/astral-sh/uv) (fast Python package and project manager)
- **Docker & Docker Compose** (for PostgreSQL with `pgvector`)
- **Node.js 20+** (if developing the frontend locally outside Docker)

---

### Option A: Local Development with uv

#### 1. Clone the Repository
```bash
git clone https://github.com/NishanthMS2803/Lekka.ai.git
cd "Lekka AI"
```

#### 2. Install Python Dependencies
```bash
make install
```

#### 3. Configure Environment Variables
```bash
cp .env.example .env
```
Fill in the values in `.env` (see [Environment Configuration](#environment-configuration-env) below).

#### 4. Start the Database
Start the PostgreSQL 16 container with the `pgvector` extension:
```bash
make db-up
```

#### 5. Apply Database Migrations
```bash
uv run alembic upgrade head
```

#### 6. Start the Backend API
```bash
make run
```
The FastAPI documentation will be available at `http://localhost:8000/docs`.

#### 7. Start the Frontend Web UI (in a separate terminal)
```bash
make frontend-dev
```
Open `http://localhost:3000` in your browser.

---

### Option B: Docker Full-Stack Deployment

Launch the complete stack (database, API server with automated migrations, and Next.js frontend) with Docker Compose:

```bash
# 1. Prepare environment configuration
cp .env.example .env

# 2. Build and launch all services in the background
make up
```

- **Frontend Application**: `http://localhost:3000`
- **Backend API & OpenAPI Docs**: `http://localhost:8000/docs`
- **Healthcheck**: `http://localhost:8000/health`

To stop all services:
```bash
make down
```

---

### Environment Configuration (`.env`)

Copy `.env.example` to `.env` and configure your credentials:

```ini
# Environment
ENV=development

# Database Settings
POSTGRES_USER=lekka_ai
POSTGRES_PASSWORD=lekka_ai
POSTGRES_DB=lekka_ai
DATABASE_URL=postgresql+psycopg://lekka_ai:lekka_ai@localhost:5432/lekka_ai

# LLM & Embedding Settings (Google Gemini)
GEMINI_API_KEY=<YOUR_GEMINI_API_KEY>
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=text-embedding-004
EMBEDDING_DIMENSION=768

# SEC EDGAR Compliance (Format: SampleCompany AdminContact@domain.com)
SEC_USER_AGENT=<YOUR_APP_NAME_AND_CONTACT_EMAIL>

# Optional Financial Data Providers
MARKET_DATA_API_KEY=<OPTIONAL_MARKET_API_KEY>
NEWS_PROVIDER=finnhub
NEWS_API_KEY=<OPTIONAL_FINNHUB_KEY>
ESTIMATES_API_KEY=<OPTIONAL_ESTIMATES_KEY>

# Hardening & Operational Limits
MAX_QUESTION_LENGTH=2000
MAX_CONCURRENT_RUNS=5
HARD_PER_RUN_COST_CAP_USD=2.0
RATE_LIMIT_RUNS_PER_MINUTE=30
RATE_LIMIT_ENABLED=true
TICKER_ALLOWLIST_ENABLED=false
```

---

## How to Run a Research Question

### 1. Web Interface (Next.js)

Navigate to `http://localhost:3000`:
1. Enter your research question (e.g. *"Evaluate Apple's valuation and capital allocation as of Q3 2024"*).
2. Set the point-in-time cutoff date (`as_of_date`).
3. Click **Start Research Run**.
4. Watch live agent tasks, tool calls, and streaming claims in real time.
5. Review the synthesized report with interactive claim drill-downs, scenario tables, and the execution trace.

---

### 2. Interactive CLI

Query specialist agents directly from your terminal:

```bash
# Full multi-agent evaluation
uv run lekka-ai ask "Evaluate AAPL valuation and balance sheet quality" --as-of 2024-09-20

# Target a specific specialist agent
uv run lekka-ai ask "Run DCF sensitivity and reverse DCF for AAPL" --agent valuation --as-of 2024-09-20
uv run lekka-ai ask "Analyze volatility, beta vs SPY, and price momentum for NVDA" --agent market --as-of 2024-09-20
uv run lekka-ai ask "Summarize material catalysts and 8-K filings for TSLA" --agent news --as-of 2024-09-20

# Run a baseline comparison (zero-tool raw LLM)
uv run lekka-ai ask "What was MSFT FY2024 revenue?" --variant direct
```

---

### 3. REST & Streaming API (cURL)

#### Launch a Research Run
```bash
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Evaluate AAPL valuation and business quality as of Q3 2024",
    "as_of_date": "2024-09-20",
    "budget": {"max_tokens": 50000, "max_cost_usd": 1.00}
  }'
```
Response:
```json
{
  "run_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "status": "running"
}
```

#### Stream Live Execution Events (SSE)
```bash
curl -N http://localhost:8000/runs/7c9e6679-7425-40de-944b-e07fc1f90ae7/events
```

#### Retrieve the Final Report
```bash
curl http://localhost:8000/runs/7c9e6679-7425-40de-944b-e07fc1f90ae7/report
```

#### Inspect Execution Trace & Costs
```bash
curl http://localhost:8000/runs/7c9e6679-7425-40de-944b-e07fc1f90ae7/trace
```

---

## How to Run Evaluations

Lekka AI includes a comprehensive benchmarking harness (`src/lekka_ai/eval/`) that compares system variants (V1–V5 & ablations) against curated ground-truth datasets.

### Run Selected Benchmark Datasets
```bash
# Run fact-checking evaluation across direct LLM and single-agent variants
make eval VARIANTS=direct,single DATASET=eval/datasets/facts_v0.jsonl
```

### Run the Full Benchmark Suite (Offline / Deterministic)
```bash
# Evaluates all variants across facts, adversarial, failure, and risk datasets
make eval-all
```

### Run Live Benchmarks (with Google Gemini API)
```bash
make eval-all LIVE=1
```

### Validate Benchmark Datasets
```bash
make validate-datasets
```

### View Evaluation Results
- **In Browser**: Visit `http://localhost:3000/evals` to explore the interactive variant comparison table, quality vs. cost plots, and confidence calibration curves.
- **In Markdown**: Read the auto-generated evaluation report in [`eval/results/REPORT.md`](file:///Users/nishanth/Documents/Projects/Lekka%20AI/eval/results/REPORT.md).

For deep dive details on datasets, metrics, and lookahead mitigation, see [`docs/EVALUATION.md`](file:///Users/nishanth/Documents/Projects/Lekka%20AI/docs/EVALUATION.md).

---

## Project Structure

```text
.
├── src/lekka_ai/               # Core Python application package
│   ├── api/                    # FastAPI application, routes, models, rate limiting
│   │   ├── routes/             # Endpoints: /runs, /claims, /evals, /health
│   │   ├── ratelimit.py        # Sliding-window IP rate limiter
│   │   └── models.py           # Pydantic API schemas and validators
│   ├── data/                   # Data ingestion, SEC EDGAR client, market data adapters
│   ├── db/                     # PostgreSQL models, migrations, and repository layer
│   ├── eval/                   # Benchmarking harness, graders, and variant definitions
│   ├── harness/                # Agent runtime, ToolView proxies, repair loops
│   ├── llm/                    # Gemini API integration and embeddings
│   ├── orchestrator/           # Dynamic planner, dispatcher, verifier, confidence engine
│   ├── report/                 # 9-section report synthesizer & abstention generator
│   ├── tools/                  # Deterministic valuation tools (DCF, multiples, risk)
│   ├── config.py               # Pydantic Settings and operational limits
│   ├── logging.py              # Structlog setup and secret redaction
│   └── tracing.py              # OpenTelemetry-compatible tracing and telemetry
├── agents/                     # Declarative subagent profiles and prompts
│   ├── profiles.yaml           # Agent permissions, tools, and budgets
│   └── prompts/                # Specialist system prompts (analyst, valuation, etc.)
├── docs/                       # System documentation
│   ├── ARCHITECTURE.md         # Detailed architectural specification
│   └── EVALUATION.md           # Benchmark datasets, metrics, and leakage mitigation
├── eval/                       # Benchmark data and generated results
│   ├── datasets/               # facts_v0, adversarial_v0, failures_v0, risk_recall_v0
│   └── results/                # REPORT.md, results.csv, and comparative plots
├── frontend/                   # Next.js 16 Web Application (App Router, Tailwind)
│   ├── app/                    # Pages: /, /runs/[id], /runs/[id]/trace, /evals
│   └── components/             # InteractiveReport, TraceViewer, Header, ClaimDrillDown
├── tests/                      # Pytest suite (unit, integration, hardening, secret scans)
├── docker-compose.yml          # Production multi-service container orchestration
├── Dockerfile                  # Multi-stage production backend container build
├── Makefile                    # Standard developer workflow commands
└── pyproject.toml              # Project dependencies and tool configurations
```

---

## Core Design Decisions

### 1. Sandboxed Agent Harness
Agents operate through a sandboxed `ToolView` proxy. An agent cannot call tools outside its declarative profile in `agents/profiles.yaml`. The harness features an automated repair loop that feeds Pydantic validation errors back to the model for up to 3 self-healing attempts. All third-party external text is quarantined in `<untrusted_data>` delimiters to neutralize prompt injection attacks.

### 2. Shared Claim Ledger
Instead of passing unstructured conversational text between agents, all subagents publish atomic claims to a shared ledger. The ledger validates citation integrity, ensuring every cited `tool_call_id` was legitimately executed. Cross-agent reconciliation detects conflicting values (e.g. filing vs. news discrepancies) and penalizes confidence accordingly.

### 3. Claim Taxonomy
To enforce grounding, claims are classified into four explicit types:
- **`fact`**: Verified numerical or historical data point from an official source (must cite raw tool output and SEC accession).
- **`calculation`**: Output of deterministic financial models (DCF, beta, WACC) with linked inputs.
- **`interpretation`**: Analytical deduction synthesized strictly from verified facts or calculations.
- **`background`**: General industry context (bound by length constraints to prevent fluff).

### 4. Hierarchical Budgets & Cost Caps
Research runs operate under strict hierarchical resource envelopes:
- **Global Hard Cost Cap**: Hard stop at `$2.00` per run (`HARD_PER_RUN_COST_CAP_USD`).
- **Dynamic Task Budgets**: The planner allocates token and time slices to subtasks based on query complexity.
- **Graceful Degradation**: If an agent exhausts its budget, it synthesizes partial findings from existing ledger claims rather than failing the run.

### 5. Deterministic Confidence & Abstention
Confidence is never an arbitrary LLM guess. It is deterministically calculated across six rules (verified fact ratio, conflict penalties, evidence recency, background proportion, role coverage, and budget limits). If the final score falls below **0.60**, the system abstains from issuing recommendations, details missing evidence, and enforces a strict zero-fabricated-numbers policy.

---

## Documentation & Deep Dives

- **[System Architecture Specification](docs/ARCHITECTURE.md)**: Deep dive into subagent profiles, security boundaries, DAG planning, and verification mathematics.
- **[Evaluation & Benchmarking Guide](docs/EVALUATION.md)**: Datasets, metric definitions, variant comparison, and the point-in-time lookahead mitigation architecture.

---

## Verification & Testing

Run the automated test suite:
```bash
# Run unit, integration, and security tests
make test

# Verify secret scanning across git-tracked files
uv run pytest tests/test_secret_scan.py

# Run frontend tests, linting, and build validation
make frontend-test
make frontend-lint
make frontend-typecheck
make frontend-build
```

---

## License

MIT License. See `LICENSE` for details.
