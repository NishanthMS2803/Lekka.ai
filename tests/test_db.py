"""Tests for database connectivity, sessions, and repository operations."""

from datetime import date

import pytest
from sqlalchemy import text

from lekka_ai.db import create_run, get_engine, get_run, get_session


def is_database_reachable() -> bool:
    """Check if the database is reachable."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.mark.db
def test_session_select_one() -> None:
    """Open a session and execute SELECT 1, skipping if DB is unavailable."""
    if not is_database_reachable():
        pytest.skip("Database is unavailable")

    session_generator = get_session()
    session = next(session_generator)
    try:
        result = session.execute(text("SELECT 1")).scalar_one()
        assert result == 1
    finally:
        try:
            next(session_generator)
        except StopIteration:
            pass


@pytest.mark.db
def test_create_and_get_run_repository() -> None:
    """Insert and read back a run using repository functions."""
    if not is_database_reachable():
        pytest.skip("Database is unavailable")

    session_generator = get_session()
    session = next(session_generator)
    try:
        new_run = create_run(
            session,
            question="What is Apple's revenue for FY2024?",
            status="running",
            as_of_date=date(2024, 9, 30),
            budget_json={"max_subagents": 4, "max_tokens": 50000},
        )
        assert new_run.id is not None
        assert new_run.question == "What is Apple's revenue for FY2024?"

        fetched_run = get_run(session, new_run.id)
        assert fetched_run is not None
        assert fetched_run.id == new_run.id
        assert fetched_run.question == "What is Apple's revenue for FY2024?"
        assert fetched_run.status == "running"
        assert fetched_run.as_of_date == date(2024, 9, 30)
        assert fetched_run.budget_json == {
            "max_subagents": 4,
            "max_tokens": 50000,
        }
        assert fetched_run.created_at is not None
    finally:
        try:
            next(session_generator)
        except StopIteration:
            pass
