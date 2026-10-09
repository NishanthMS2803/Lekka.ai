"""Tests for database connectivity and session operations."""

import pytest
from sqlalchemy import text

from lekka_ai.db import get_engine, get_session


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
