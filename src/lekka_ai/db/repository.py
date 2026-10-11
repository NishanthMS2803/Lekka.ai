"""Database repository helpers for research runs."""

from __future__ import annotations

import uuid
from datetime import date
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from lekka_ai.db.models import AgentRun, Filing, FilingChunk, Run, Task, ToolCall

if TYPE_CHECKING:
    from lekka_ai.data.chunking import FilingChunkData


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


def create_task(
    session: Session,
    *,
    run_id: uuid.UUID,
    agent_type: str,
    instruction: str,
    status: str = "pending",
    parent_task_id: uuid.UUID | None = None,
) -> Task:
    """Create and persist a sub-task for an agent."""
    task = Task(
        run_id=run_id,
        agent_type=agent_type,
        instruction=instruction,
        status=status,
        parent_task_id=parent_task_id,
    )
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


def create_agent_run(
    session: Session,
    *,
    task_id: uuid.UUID,
    iterations: int = 0,
    tokens_in: int = 0,
    tokens_out: int = 0,
    status: str = "running",
    limits_hit: list[Any] | dict[str, Any] | None = None,
) -> AgentRun:
    """Create and persist an agent execution cycle."""
    agent_run = AgentRun(
        task_id=task_id,
        iterations=iterations,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        status=status,
        limits_hit=limits_hit if limits_hit is not None else [],
    )
    session.add(agent_run)
    session.commit()
    session.refresh(agent_run)
    return agent_run


def create_tool_call(
    session: Session,
    *,
    agent_run_id: uuid.UUID,
    tool_name: str,
    input_json: dict[str, Any] | None = None,
    output_json: dict[str, Any] | list[Any] | None = None,
    error: str | None = None,
    latency_ms: int | None = None,
    cache_hit: bool = False,
) -> ToolCall:
    """Create and persist a tool call invocation record."""
    tool_call = ToolCall(
        agent_run_id=agent_run_id,
        tool_name=tool_name,
        input_json=input_json,
        output_json=output_json,
        error=error,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
    )
    session.add(tool_call)
    session.commit()
    session.refresh(tool_call)
    return tool_call


def get_tool_call(session: Session, tool_call_id: uuid.UUID) -> ToolCall | None:
    """Retrieve a tool call record by ID."""
    return session.get(ToolCall, tool_call_id)


def create_filing_record(
    session: Session,
    *,
    ticker: str,
    form: str,
    filing_date: date,
    accession: str,
    section_name: str,
    text: str,
    raw_text: str | None = None,
) -> Filing:
    """Create and persist a filing section record."""
    filing = Filing(
        ticker=ticker.strip().upper(),
        form=form.strip().upper(),
        filing_date=filing_date,
        accession=accession.strip(),
        section_name=section_name.strip(),
        text=text,
        raw_text=raw_text,
    )
    session.add(filing)
    session.commit()
    session.refresh(filing)
    return filing


def get_filing_section(
    session: Session,
    *,
    ticker: str,
    accession: str,
    section_name: str,
) -> Filing | None:
    """Retrieve a specific filing section by ticker, accession, and section name."""
    from sqlalchemy import select

    stmt = select(Filing).where(
        Filing.ticker == ticker.strip().upper(),
        Filing.accession == accession.strip(),
        Filing.section_name == section_name.strip(),
    )
    return session.execute(stmt).scalar_one_or_none()


def get_filing_sections(
    session: Session,
    *,
    ticker: str,
    accession: str,
) -> list[Filing]:
    """Retrieve all sections of a filing by ticker and accession number."""
    from sqlalchemy import select

    stmt = (
        select(Filing)
        .where(
            Filing.ticker == ticker.strip().upper(),
            Filing.accession == accession.strip(),
        )
        .order_by(Filing.section_name)
    )
    return list(session.execute(stmt).scalars().all())


def save_filing_document(
    session: Session,
    *,
    ticker: str,
    form: str,
    filing_date: date,
    accession: str,
    sections: dict[str, str],
    raw_text: str | None = None,
) -> list[Filing]:
    """Save all sections of a filing document in a single transaction."""
    records: list[Filing] = []
    clean_ticker = ticker.strip().upper()
    clean_form = form.strip().upper()
    clean_acc = accession.strip()

    from sqlalchemy import delete

    # Remove existing records for this filing accession to ensure idempotency
    session.execute(
        delete(Filing).where(
            Filing.ticker == clean_ticker,
            Filing.accession == clean_acc,
        )
    )

    for sec_name, sec_text in sections.items():
        filing = Filing(
            ticker=clean_ticker,
            form=clean_form,
            filing_date=filing_date,
            accession=clean_acc,
            section_name=sec_name.strip(),
            text=sec_text,
            raw_text=raw_text,
        )
        session.add(filing)
        records.append(filing)

    session.commit()
    for r in records:
        session.refresh(r)
    return records


def save_filing_chunks(
    session: Session,
    *,
    chunks: list[FilingChunkData],
    embeddings: list[list[float]] | None = None,
    filing_id: uuid.UUID | None = None,
) -> list[FilingChunk]:
    """Save filing chunks with optional embeddings in a single transaction."""
    if not chunks:
        return []

    if embeddings is not None and len(embeddings) != len(chunks):
        raise ValueError(
            f"Embeddings count ({len(embeddings)}) does not match "
            f"chunks count ({len(chunks)})"
        )

    records: list[FilingChunk] = []
    for idx, c in enumerate(chunks):
        emb = embeddings[idx] if embeddings is not None else None
        record = FilingChunk(
            filing_id=filing_id,
            ticker=c.ticker.strip().upper(),
            form=c.form.strip().upper(),
            filing_date=c.filing_date,
            accession=c.accession.strip(),
            section_name=c.section_name.strip(),
            chunk_index=c.chunk_index,
            text=c.text,
            token_count=c.token_count,
            embedding=emb,
        )
        session.add(record)
        records.append(record)

    session.commit()
    for r in records:
        session.refresh(r)
    return records


def get_filing_chunks(
    session: Session,
    *,
    ticker: str,
    accession: str,
    section_name: str | None = None,
) -> list[FilingChunk]:
    """Retrieve chunks for a filing, optionally filtered by section name."""
    from sqlalchemy import select

    stmt = select(FilingChunk).where(
        FilingChunk.ticker == ticker.strip().upper(),
        FilingChunk.accession == accession.strip(),
    )
    if section_name:
        stmt = stmt.where(FilingChunk.section_name == section_name.strip())

    stmt = stmt.order_by(FilingChunk.section_name, FilingChunk.chunk_index)
    return list(session.execute(stmt).scalars().all())


def find_similar_chunks(
    session: Session,
    *,
    query_embedding: list[float],
    limit: int = 5,
    ticker: str | None = None,
    section_name: str | None = None,
    sections: list[str] | None = None,
    form_types: list[str] | None = None,
    as_of_date: date | None = None,
) -> list[tuple[FilingChunk, float]]:
    """Find most similar filing chunks using cosine distance (<=>).

    Args:
        session: Active SQLAlchemy session.
        query_embedding: Embedding vector for similarity search.
        limit: Maximum number of chunks to return.
        ticker: Optional ticker symbol filter.
        section_name: Optional single section name filter.
        sections: Optional list of section names filter.
        form_types: Optional list of form types (e.g. ['10-K', '10-Q']).
        as_of_date: Optional point-in-time cutoff date.

    Returns:
        List of tuples: (FilingChunk, distance).
    """
    from sqlalchemy import select

    distance_col = FilingChunk.embedding.cosine_distance(query_embedding).label(
        "distance"
    )
    stmt = select(FilingChunk, distance_col).where(FilingChunk.embedding.isnot(None))
    if ticker:
        stmt = stmt.where(FilingChunk.ticker == ticker.strip().upper())
    if section_name:
        stmt = stmt.where(FilingChunk.section_name == section_name.strip())
    if sections:
        norm_sections = [s.strip() for s in sections if s.strip()]
        if norm_sections:
            stmt = stmt.where(FilingChunk.section_name.in_(norm_sections))
    if form_types:
        norm_forms = [f.strip().upper() for f in form_types if f.strip()]
        if norm_forms:
            stmt = stmt.where(FilingChunk.form.in_(norm_forms))
    if as_of_date:
        stmt = stmt.where(FilingChunk.filing_date <= as_of_date)

    stmt = stmt.order_by(distance_col).limit(limit)
    rows = session.execute(stmt).all()
    return [(row[0], float(row[1])) for row in rows]
