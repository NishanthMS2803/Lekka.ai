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

## Data Sources & Retrieval Engine

Lekka AI implements structured, rate-limited adapters with point-in-time (`as_of_date`) filtering to prevent future data leakage:

1. **Market Data & Prices**:
   - `YFinanceProvider` / `get_market_data`: Historical daily OHLCV bars, close prices, volume, and returns.
2. **SEC EDGAR Filings & XBRL**:
   - `SecClient`: Ticker-to-CIK mapping cached from SEC, compliant User-Agent headers, and rate limiting (<10 req/s).
   - `list_filings`: Discovers 10-K, 10-Q, and 8-K filings filtered by filing date.
   - `get_company_facts`: Parses XBRL company facts with accounting tag fallbacks (Revenues, NetIncomeLoss, OperatingIncomeLoss, Operating Cash Flow, Capex, Cash & Cash Equivalents, Debt).
   - `get_financials`: Standardized statement metrics plus derived quantities (YoY growth, operating margin, net margin, Free Cash Flow, net debt).
   - Document section parser: Cleans HTML filings and extracts sections by Item headings (e.g. Item 1, 1A, 7, 7A, 8).
   - Semantic retrieval (`search_filings`): pgvector cosine search over 800-token chunks embedded with Gemini (`text-embedding-004`), returning citation documents.
3. **News Providers**:
   - `FinnhubNewsProvider` & `TavilyNewsProvider` / `get_news`: Gathers recent company news articles, automatically labeling all third-party text as untrusted external content.
4. **Analyst Estimates & Industry Data**:
   - `FinnhubEstimatesProvider` / `get_analyst_estimates` & `get_industry_data`: Consensus price targets, ratings, peer tickers, and industry multiples, with structured graceful degradation when API keys are unconfigured.

---

## Filing Ingestion Command

To fetch, clean, split, chunk, and embed a company's latest SEC 10-K and 10-Q filings into the local `filings` and `filing_chunks` vector database:

```bash
make ingest TICKER=AAPL
```

You can specify any US ticker symbol (e.g. `TICKER=MSFT`, `TICKER=NVDA`).

---

## Quantitative Valuation & Modeling Engine

Lekka AI includes a comprehensive quantitative valuation suite accessible directly as registered agent tools:

- **DCF Valuation (`run_dcf`)**: Explicit assumptions (starting FCF or revenue + margins, WACC discount rate, Gordon Growth terminal rate, net debt, shares), returning Enterprise Value, Equity Value, fair value per share, and explicit cash flow forecast table.
- **Reverse DCF (`run_reverse_dcf`)**: Solves for the constant growth rate or margin implied by the current market price using `scipy.optimize.brentq`.
- **Multiples & Historical Percentiles (`compute_multiples`)**: Computes P/E, Forward P/E, EV/EBITDA, EV/Sales, P/FCF, PEG, and historical percentile ranks against historical distributions.
- **Scenario Table & Sensitivity Grid (`scenario_table`)**: Evaluates Bear/Base/Bull scenarios with probabilities to produce weighted fair values and upside/downside percentages, alongside 2D sensitivity grids over WACC and terminal growth rates.
- **Risk Statistics (`compute_risk_stats`)**: Historical returns, annualized volatility, max drawdown, beta vs SPY benchmark, relative strength, and moving average trends.

---

## Agent Harness & Runtime Architecture

Lekka AI features an enterprise agent execution and safety harness (`src/lekka_ai/harness/`) designed for rigorous equity research:

- **Config-Driven Profiles (`agents/profiles.yaml`)**:
  Declarative YAML configuration specifying tool permissions, system prompt paths (`agents/prompts/*.md`), default execution limits, and typed answer extension schemas.
- **Strict Tool Permissions & Isolation (`ToolView`)**:
  Agents operate through sandboxed `ToolView` proxies. Calls to unpermitted tools are blocked upfront and raise `PermissionDenied`, preventing privilege escalation and cross-specialist contamination.
- **Multi-Turn Autonomous Loop (`AgentRuntime`)**:
  Orchestrates LLM turns with configurable safety boundaries (maximum iterations, token caps, runtime budgets, repair attempt limits).
- **Self-Healing Validation Repair**:
  If a tool payload or submission fails Pydantic schema validation, the runtime feeds the exact validation error back to the LLM for automated in-flight repair (up to 3 attempts).
- **Claim Ledger & Provenance Verification**:
  Submissions require structured claims (`fact`, `calculation`, `interpretation`, `background`). The runtime strictly validates citation integrity, ensuring every cited `tool_call_id` was legitimately executed by the agent.
- **Untrusted Content Isolation**:
  External third-party text (e.g. news headlines, press releases) is automatically wrapped in `<untrusted_data>` delimiter tags with escaped closing tags, neutralizing prompt injection attacks.
- **Full Audit Tracing (`Tracer`)**:
  Every LLM generation and tool execution produces an immutable telemetry trace recorded to SQLite/PostgreSQL with inputs, outputs, errors, token counts, and latency.

---

## Specialist Agent Roster

| Agent Profile | Role & Responsibilities | Authorized Tools | Typed Extension |
|---|---|---|---|
| **`analyst`** | Generalist equity analyst performing comprehensive fundamental and quantitative research. Enforces evidence-backed claims and permits explicit insufficient evidence responses. | `get_market_data`, `get_financials`, `list_filings`, `search_filings`, `get_news`, `get_analyst_estimates`, `get_industry_data`, `run_dcf`, `run_reverse_dcf`, `compute_multiples`, `scenario_table`, `compute_risk_stats`, `echo`, `submit_answer` | Standard Submission |
| **`market`** | Technical & price action specialist. Analyzes historical price movements, annualized volatility, drawdowns, beta relative to SPY, relative strength, and moving average trend classifications. Strictly prohibited from issuing investment verdicts. | `get_market_data`, `compute_risk_stats`, `submit_answer` | `MarketAnswerExtension` (trend label, beta, volatility, moving averages) |
| **`valuation`** | Quantitative valuation specialist. Builds explicit Bear / Base / Bull scenario models, solves reverse DCF for market-implied growth expectations, and reports sensitivity grids. Zero mental math rule enforced; complex custom calculations executed via Python sandbox. | `get_market_data`, `get_financials`, `get_analyst_estimates`, `run_dcf`, `run_reverse_dcf`, `compute_multiples`, `scenario_table`, `run_python`, `submit_answer` | `ValuationAnswerExtension` (bear/base/bull values, implied growth, current price) |
| **`news`** | Corporate catalyst and developments specialist. Discovers recent news developments and SEC Form 8-K filings with dates. Classifies each catalyst across two dimensions: impact (`positive`, `negative`, `neutral`, `uncertain`) and novelty (`new` vs `already_known`). Treats all news text as untrusted. | `get_news`, `list_filings`, `search_filings`, `submit_answer` | `NewsAnswerExtension` (structured catalyst list, impact, novelty, source) |

---

## Command Line Interface (CLI)

Lekka AI includes a Typer-powered CLI (`lekka-ai ask`) for interactive research queries:

```bash
# Query the generic baseline analyst
uv run lekka-ai ask "What was Apple's revenue and operating income in FY2023?" --as-of 2024-01-15

# Query the market specialist
uv run lekka-ai ask "Evaluate price action and beta for AAPL relative to SPY" --agent market --as-of 2024-09-20

# Query the valuation specialist
uv run lekka-ai ask "Build DCF scenarios and reverse DCF for AAPL" --agent valuation --as-of 2024-09-20

# Query the news and catalyst specialist
uv run lekka-ai ask "Identify recent material catalysts and 8-K filings for AAPL" --agent news --as-of 2024-09-20

# Run direct LLM baseline (zero tools)
uv run lekka-ai ask "What was AAPL free cash flow in 2023?" --variant direct
```

---

## Evaluation & Benchmarking Harness

The evaluation harness (`src/lekka_ai/eval/`) evaluates and benchmarks model variants against curated ground-truth datasets:

```bash
# Run fact-checking benchmark across direct LLM and single agent
make eval VARIANTS=direct,single DATASET=eval/datasets/facts_v0.jsonl
```

- **Pluggable Variants**: Compare `direct` (unassisted zero-tool LLM) vs `single` (autonomous tool-calling agent runtime).
- **Programmatic Graders**: Extracts numeric values and scales across units (thousands, millions, billions) to evaluate `factual_accuracy` within configurable percentage tolerances.
- **Metric Reporting**: Outputs summary tables with accuracy percentages, token usage, tool call counts, latency, and costs saved to `eval/results/`.

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
| `make ingest TICKER=AAPL` | Ingest, parse, chunk, and embed latest 10-K & 10-Q filings |
| `make eval` | Run evaluation benchmark on dataset across specified variants |

