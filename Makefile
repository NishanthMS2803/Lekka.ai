.PHONY: install lint typecheck test run db-up db-down db-logs ingest eval

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

