"""add done_by_id to study_responses

Revision ID: 20260517_done_by_id
Revises: 20260408_task_assignments
Create Date: 2026-05-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260517_done_by_id"
down_revision: Union[str, Sequence[str], None] = "20260408_task_assignments"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("study_responses", sa.Column("done_by_id", sa.String(length=50), nullable=True))
    op.create_index(op.f("ix_study_responses_done_by_id"), "study_responses", ["done_by_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_study_responses_done_by_id"), table_name="study_responses")
    op.drop_column("study_responses", "done_by_id")
