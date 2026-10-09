"""Database repository helpers for research runs."""

import uuid
from datetime import date
from typing import Any

from sqlalchemy.orm import Session

from lekka_ai.db.models import Run


def create_run(
    session: Session,
    *,
    question: str,
    status: str = "pending",
    as_of_date: date | None = None,
    budget_json: dict[str, Any] | None = None,
) -> Run:
    """Create and persist a new research run."""
    run = Run(
        question=question,
        status=status,
        as_of_date=as_of_date,
        budget_json=budget_json if budget_json is not None else {},
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def get_run(session: Session, run_id: uuid.UUID) -> Run | None:
    """Retrieve a research run by ID."""
    return session.get(Run, run_id)
