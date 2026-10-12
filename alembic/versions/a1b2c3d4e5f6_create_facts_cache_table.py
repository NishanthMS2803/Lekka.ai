"""create facts_cache table

Revision ID: a1b2c3d4e5f6
Revises: f6a1b2c3d4e5
Create Date: 2026-10-12 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "f6a1b2c3d4e5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema to include facts_cache table."""
    op.create_table(
        "facts_cache",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("ticker", sa.String(length=20), nullable=False),
        sa.Column("metric_key", sa.String(length=100), nullable=False),
        sa.Column(
            "value",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("source", sa.String(length=255), nullable=True),
        sa.Column("fact_type", sa.String(length=50), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=True),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claim_id", sa.String(length=100), nullable=True),
        sa.Column("period", sa.String(length=50), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_facts_cache_ticker", "facts_cache", ["ticker"], unique=False)
    op.create_index(
        "ix_facts_cache_metric_key", "facts_cache", ["metric_key"], unique=False
    )
    op.create_index(
        "ix_facts_cache_fact_type", "facts_cache", ["fact_type"], unique=False
    )
    op.create_index(
        "ix_facts_cache_expires_at", "facts_cache", ["expires_at"], unique=False
    )
    op.create_index(
        "ix_facts_cache_claim_id", "facts_cache", ["claim_id"], unique=False
    )
    op.create_index(
        "ix_facts_cache_ticker_metric",
        "facts_cache",
        ["ticker", "metric_key"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema removing facts_cache table."""
    op.drop_index("ix_facts_cache_ticker_metric", table_name="facts_cache")
    op.drop_index("ix_facts_cache_claim_id", table_name="facts_cache")
    op.drop_index("ix_facts_cache_expires_at", table_name="facts_cache")
    op.drop_index("ix_facts_cache_fact_type", table_name="facts_cache")
    op.drop_index("ix_facts_cache_metric_key", table_name="facts_cache")
    op.drop_index("ix_facts_cache_ticker", table_name="facts_cache")
    op.drop_table("facts_cache")
