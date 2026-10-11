"""create filings table

Revision ID: c7b2a9e34d10
Revises: 589e8fe706ca
Create Date: 2026-10-12 05:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7b2a9e34d10"
down_revision: str | Sequence[str] | None = "589e8fe706ca"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema to include filings table."""
    op.create_table(
        "filings",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("ticker", sa.String(length=20), nullable=False),
        sa.Column("form", sa.String(length=20), nullable=False),
        sa.Column("filing_date", sa.Date(), nullable=False),
        sa.Column("accession", sa.String(length=50), nullable=False),
        sa.Column("section_name", sa.String(length=50), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_filings_ticker"),
        "filings",
        ["ticker"],
        unique=False,
    )
    op.create_index(
        op.f("ix_filings_form"),
        "filings",
        ["form"],
        unique=False,
    )
    op.create_index(
        op.f("ix_filings_filing_date"),
        "filings",
        ["filing_date"],
        unique=False,
    )
    op.create_index(
        op.f("ix_filings_accession"),
        "filings",
        ["accession"],
        unique=False,
    )
    op.create_index(
        op.f("ix_filings_section_name"),
        "filings",
        ["section_name"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_filings_section_name"), table_name="filings")
    op.drop_index(op.f("ix_filings_accession"), table_name="filings")
    op.drop_index(op.f("ix_filings_filing_date"), table_name="filings")
    op.drop_index(op.f("ix_filings_form"), table_name="filings")
    op.drop_index(op.f("ix_filings_ticker"), table_name="filings")
    op.drop_table("filings")
