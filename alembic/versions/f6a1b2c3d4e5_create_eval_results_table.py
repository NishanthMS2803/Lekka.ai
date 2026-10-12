"""create eval_results table

Revision ID: f6a1b2c3d4e5
Revises: e5f6a1b2c3d4
Create Date: 2026-10-12 06:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f6a1b2c3d4e5"
down_revision: str | Sequence[str] | None = "e5f6a1b2c3d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema to include eval_results table."""
    op.create_table(
        "eval_results",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("eval_run_id", sa.String(length=100), nullable=False),
        sa.Column("example_id", sa.String(length=100), nullable=False),
        sa.Column("variant", sa.String(length=50), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column(
            "claims",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("tool_calls", sa.Integer(), server_default="0", nullable=False),
        sa.Column("latency_ms", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("cost", sa.Float(), server_default="0.0", nullable=False),
        sa.Column(
            "metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_eval_results_eval_run_id",
        "eval_results",
        ["eval_run_id"],
        unique=False,
    )
    op.create_index(
        "ix_eval_results_example_id",
        "eval_results",
        ["example_id"],
        unique=False,
    )
    op.create_index(
        "ix_eval_results_variant",
        "eval_results",
        ["variant"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema removing eval_results table."""
    op.drop_index("ix_eval_results_variant", table_name="eval_results")
    op.drop_index("ix_eval_results_example_id", table_name="eval_results")
    op.drop_index("ix_eval_results_eval_run_id", table_name="eval_results")
    op.drop_table("eval_results")
