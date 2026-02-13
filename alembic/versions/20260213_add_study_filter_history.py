"""add study_filter_history table

Revision ID: 20260213_filter
Revises: 62c9acaea15a
Create Date: 2026-02-13

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20260213_filter"
down_revision: Union[str, Sequence[str], None] = "62c9acaea15a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "study_filter_history",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("study_id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("filters", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["study_id"], ["studies.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_study_filter_history_study_id", "study_filter_history", ["study_id"], unique=False)
    op.create_index("ix_study_filter_history_user_id", "study_filter_history", ["user_id"], unique=False)
    op.create_index("idx_study_filter_history_study_user", "study_filter_history", ["study_id", "user_id"], unique=False)
    op.create_index("idx_study_filter_history_created", "study_filter_history", ["study_id", "created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_study_filter_history_created", table_name="study_filter_history")
    op.drop_index("idx_study_filter_history_study_user", table_name="study_filter_history")
    op.drop_index("ix_study_filter_history_user_id", table_name="study_filter_history")
    op.drop_index("ix_study_filter_history_study_id", table_name="study_filter_history")
    op.drop_table("study_filter_history")
