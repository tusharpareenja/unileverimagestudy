"""add study_task_assignments table for normalized task storage

Revision ID: 20260408_task_assignments
Revises: 20260326_study_resp_unique
Create Date: 2026-04-08

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "20260408_task_assignments"
down_revision: Union[str, Sequence[str], None] = "20260326_study_resp_unique"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "study_task_assignments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("study_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("studies.id", ondelete="CASCADE"), nullable=False),
        sa.Column("respondent_id", sa.Integer(), nullable=False),
        sa.Column("task_index", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.String(50), nullable=False),
        sa.Column("elements_shown", postgresql.JSONB(), nullable=False),
        sa.Column("elements_shown_content", postgresql.JSONB(), nullable=True),
        sa.Column("phase_type", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    
    # Create indexes for efficient queries
    op.create_index("idx_sta_study", "study_task_assignments", ["study_id"])
    op.create_index("idx_sta_study_respondent", "study_task_assignments", ["study_id", "respondent_id"])
    
    # Create unique constraint to prevent duplicate task assignments
    op.create_unique_constraint(
        "uq_study_task_assignment",
        "study_task_assignments",
        ["study_id", "respondent_id", "task_index"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_study_task_assignment", "study_task_assignments", type_="unique")
    op.drop_index("idx_sta_study_respondent", table_name="study_task_assignments")
    op.drop_index("idx_sta_study", table_name="study_task_assignments")
    op.drop_table("study_task_assignments")
