.PHONY: install lint typecheck test

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src tests

test:
	uv run pytest
