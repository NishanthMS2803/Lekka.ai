.PHONY: install lint typecheck test run db-up db-down db-logs ingest

TICKER ?= AAPL

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

