"""add composite index study_id panelist_id for fast participation check

Revision ID: 20260318_study_panelist
Revises: 20260213_filter
Create Date: 2026-03-18

"""
from typing import Sequence, Union

from alembic import op


revision: str = "20260318_study_panelist"
down_revision: Union[str, Sequence[str], None] = "20260213_filter"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_study_responses_study_panelist "
        "ON study_responses (study_id, panelist_id) "
        "WHERE panelist_id IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_study_responses_study_panelist")
