"""create filing_chunks table with pgvector and hnsw index

Revision ID: d4e5f6a1b2c3
Revises: c7b2a9e34d10
Create Date: 2026-10-12 05:15:00.000000

"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a1b2c3"
down_revision: str | Sequence[str] | None = "c7b2a9e34d10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema to include filing_chunks table with pgvector and HNSW index."""
    # Ensure pgvector extension is enabled
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    op.create_table(
        "filing_chunks",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("filing_id", sa.UUID(), nullable=True),
        sa.Column("ticker", sa.String(length=20), nullable=False),
        sa.Column("form", sa.String(length=20), nullable=False),
        sa.Column("filing_date", sa.Date(), nullable=False),
        sa.Column("accession", sa.String(length=50), nullable=False),
        sa.Column("section_name", sa.String(length=50), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "token_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.Vector(768),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["filing_id"],
            ["filings.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        op.f("ix_filing_chunks_ticker"),
        "filing_chunks",
        ["ticker"],
        unique=False,
    )
    op.create_index(
        op.f("ix_filing_chunks_form"),
        "filing_chunks",
        ["form"],
        unique=False,
    )
    op.create_index(
        op.f("ix_filing_chunks_filing_date"),
        "filing_chunks",
        ["filing_date"],
        unique=False,
    )
    op.create_index(
        op.f("ix_filing_chunks_accession"),
        "filing_chunks",
        ["accession"],
        unique=False,
    )
    op.create_index(
        op.f("ix_filing_chunks_section_name"),
        "filing_chunks",
        ["section_name"],
        unique=False,
    )
    op.create_index(
        op.f("ix_filing_chunks_filing_id"),
        "filing_chunks",
        ["filing_id"],
        unique=False,
    )

    # Create HNSW vector index using cosine distance operator
    op.execute(
        "CREATE INDEX ix_filing_chunks_embedding_hnsw "
        "ON filing_chunks USING hnsw (embedding vector_cosine_ops);"
    )


def downgrade() -> None:
    """Downgrade schema removing filing_chunks table."""
    op.execute("DROP INDEX IF EXISTS ix_filing_chunks_embedding_hnsw;")
    op.drop_index(op.f("ix_filing_chunks_filing_id"), table_name="filing_chunks")
    op.drop_index(op.f("ix_filing_chunks_section_name"), table_name="filing_chunks")
    op.drop_index(op.f("ix_filing_chunks_accession"), table_name="filing_chunks")
    op.drop_index(op.f("ix_filing_chunks_filing_date"), table_name="filing_chunks")
    op.drop_index(op.f("ix_filing_chunks_form"), table_name="filing_chunks")
    op.drop_index(op.f("ix_filing_chunks_ticker"), table_name="filing_chunks")
    op.drop_table("filing_chunks")
