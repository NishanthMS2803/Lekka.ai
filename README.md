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

