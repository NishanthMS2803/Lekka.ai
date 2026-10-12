"""create claims and evidence tables

Revision ID: e5f6a1b2c3d4
Revises: d4e5f6a1b2c3
Create Date: 2026-10-12 05:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "e5f6a1b2c3d4"
down_revision: str | Sequence[str] | None = "d4e5f6a1b2c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema to include claims and evidence tables."""
    op.create_table(
        "claims",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("claim_id", sa.String(length=100), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("task_id", sa.UUID(), nullable=True),
        sa.Column("agent_run_id", sa.UUID(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=False),
        sa.Column(
            "confidence",
            sa.Float(),
            server_default=sa.text("1.0"),
            nullable=False,
        ),
        sa.Column("derived_by", sa.String(length=255), nullable=True),
        sa.Column("produced_by", sa.String(length=100), nullable=True),
        sa.Column(
            "contradicted_by",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "depends_on",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["task_id"],
            ["tasks.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["agent_run_id"],
            ["agent_runs.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_claims_claim_id",
        "claims",
        ["claim_id"],
        unique=False,
    )
    op.create_index(
        "ix_claims_run_id",
        "claims",
        ["run_id"],
        unique=False,
    )
    op.create_index(
        "ix_claims_task_id",
        "claims",
        ["task_id"],
        unique=False,
    )
    op.create_index(
        "ix_claims_agent_run_id",
        "claims",
        ["agent_run_id"],
        unique=False,
    )
    op.create_index(
        "ix_claims_type",
        "claims",
        ["type"],
        unique=False,
    )

    op.create_table(
        "evidence",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("claim_id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=True),
        sa.Column("task_id", sa.UUID(), nullable=True),
        sa.Column("agent_run_id", sa.UUID(), nullable=True),
        sa.Column("tool_call_id", sa.UUID(), nullable=True),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("document_id", sa.String(length=100), nullable=True),
        sa.Column("section", sa.String(length=100), nullable=True),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("as_of_date", sa.Date(), nullable=True),
        sa.Column(
            "raw_value",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["claim_id"],
            ["claims.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["task_id"],
            ["tasks.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["agent_run_id"],
            ["agent_runs.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["tool_call_id"],
            ["tool_calls.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_evidence_claim_id",
        "evidence",
        ["claim_id"],
        unique=False,
    )
    op.create_index(
        "ix_evidence_run_id",
        "evidence",
        ["run_id"],
        unique=False,
    )
    op.create_index(
        "ix_evidence_task_id",
        "evidence",
        ["task_id"],
        unique=False,
    )
    op.create_index(
        "ix_evidence_agent_run_id",
        "evidence",
        ["agent_run_id"],
        unique=False,
    )
    op.create_index(
        "ix_evidence_tool_call_id",
        "evidence",
        ["tool_call_id"],
        unique=False,
    )
    op.create_index(
        "ix_evidence_source_type",
        "evidence",
        ["source_type"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema by dropping evidence and claims tables."""
    op.drop_index("ix_evidence_source_type", table_name="evidence")
    op.drop_index("ix_evidence_tool_call_id", table_name="evidence")
    op.drop_index("ix_evidence_agent_run_id", table_name="evidence")
    op.drop_index("ix_evidence_task_id", table_name="evidence")
    op.drop_index("ix_evidence_run_id", table_name="evidence")
    op.drop_index("ix_evidence_claim_id", table_name="evidence")
    op.drop_table("evidence")

    op.drop_index("ix_claims_type", table_name="claims")
    op.drop_index("ix_claims_agent_run_id", table_name="claims")
    op.drop_index("ix_claims_task_id", table_name="claims")
    op.drop_index("ix_claims_run_id", table_name="claims")
    op.drop_index("ix_claims_claim_id", table_name="claims")
    op.drop_table("claims")
