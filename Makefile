.PHONY: install lint typecheck test run db-up db-down db-logs ingest eval eval-all validate-datasets frontend-dev frontend-build frontend-lint frontend-typecheck frontend-test frontend-gen-api

TICKER ?= AAPL
VARIANTS ?= direct,single
DATASET ?= eval/datasets/facts_v0.jsonl
OUTPUT_DIR ?= eval/results

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src tests

test:
	uv run pytest

run:
	uv run uvicorn lekka_ai.api.app:app --host 0.0.0.0 --port 8000 --reload

db-up:
	docker compose up -d

db-down:
	docker compose down

db-logs:
	docker compose logs -f db

ingest:
	uv run python -m lekka_ai.data.ingest --ticker $(TICKER)

eval:
	uv run python -m lekka_ai.eval.cli --dataset $(DATASET) --variants $(VARIANTS) --output-dir $(OUTPUT_DIR)

eval-all:
	uv run python -m lekka_ai.eval.eval_all $(if $(LIVE),,--mock)

validate-datasets:
	uv run python -m lekka_ai.eval.validate

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

frontend-lint:
	cd frontend && npm run lint

frontend-typecheck:
	cd frontend && npm run typecheck

frontend-test:
	cd frontend && npm run test

frontend-gen-api:
	cd frontend && npm run gen:api

