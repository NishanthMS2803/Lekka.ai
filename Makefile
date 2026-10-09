.PHONY: install lint typecheck test db-up db-down db-logs

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src tests

test:
	uv run pytest

db-up:
	docker compose up -d

db-down:
	docker compose down

db-logs:
	docker compose logs -f db
