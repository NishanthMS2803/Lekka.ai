"""SQLAlchemy ORM models for core execution runs, tasks, and tool calls."""

import uuid
from datetime import date, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from lekka_ai.db.base import Base


class Run(Base):
    """Top-level research run initiated by a user or orchestrator."""

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    as_of_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    budget_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    tasks: Mapped[list["Task"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Task(Base):
    """Sub-task delegated to a specialist agent."""

    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    agent_type: Mapped[str] = mapped_column(String(100), nullable=False)
    instruction: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    parent_task_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True
    )

    run: Mapped["Run"] = relationship(back_populates="tasks")
    agent_runs: Mapped[list["AgentRun"]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )
    subtasks: Mapped[list["Task"]] = relationship(back_populates="parent_task")
    parent_task: Mapped["Task | None"] = relationship(
        back_populates="subtasks", remote_side=[id]
    )


class AgentRun(Base):
    """Execution cycle of a subagent handling a task."""

    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    iterations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_in: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="running")
    limits_hit: Mapped[list[Any] | dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, default=list
    )

    task: Mapped["Task"] = relationship(back_populates="agent_runs")
    tool_calls: Mapped[list["ToolCall"]] = relationship(
        back_populates="agent_run", cascade="all, delete-orphan"
    )


class ToolCall(Base):
    """Invocation of an external or local tool by an agent."""

    __tablename__ = "tool_calls"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    agent_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    tool_name: Mapped[str] = mapped_column(String(100), nullable=False)
    input_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, default=dict
    )
    output_json: Mapped[dict[str, Any] | list[Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    agent_run: Mapped["AgentRun"] = relationship(back_populates="tool_calls")


class ToolCache(Base):
    """Cached response for external tools and APIs with TTL policies."""

    __tablename__ = "tool_cache"

    key: Mapped[str] = mapped_column(String(512), primary_key=True)
    value_json: Mapped[dict[str, Any] | list[Any] | Any] = mapped_column(
        JSONB, nullable=False
    )
    fact_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )


class Filing(Base):
    """Parsed SEC filing document section with raw and cleaned text."""

    __tablename__ = "filings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    form: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    filing_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    accession: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    section_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class FilingChunk(Base):
    """Text chunk of an SEC filing section with embeddings for vector search."""

    __tablename__ = "filing_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filing_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("filings.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    form: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    filing_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    accession: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    section_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index(
            "ix_filing_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index(
            "ix_filing_chunks_ticker_accession_section",
            "ticker",
            "accession",
            "section_name",
        ),
    )
