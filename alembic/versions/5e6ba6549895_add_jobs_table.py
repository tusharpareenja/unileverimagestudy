"""add_jobs_table

Revision ID: 5e6ba6549895
Revises: 03db9756f08e
Create Date: 2026-02-01 00:56:27.446573
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "5e6ba6549895"
down_revision: Union[str, Sequence[str], None] = "03db9756f08e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -----------------------------
    # JOBS TABLE
    # -----------------------------
    op.create_table(
        "jobs",
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("study_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING",
                "STARTED",
                "PROCESSING",
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                name="job_status_enum",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("progress", sa.Float(), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("job_id"),
    )

    # -----------------------------
    # SAFE INDEX CREATION
    # -----------------------------
    op.execute("""
    CREATE INDEX IF NOT EXISTS ix_jobs_study_id
    ON jobs (study_id)
    """)

    op.execute("""
    CREATE INDEX IF NOT EXISTS ix_jobs_user_id
    ON jobs (user_id)
    """)

    # -----------------------------
    # PANELISTS FIXES
    # -----------------------------

    # Drop old constraint safely (if it ever existed)
    op.execute("""
    ALTER TABLE panelists
    DROP CONSTRAINT IF EXISTS uq_panelist_id_creator
    """)

    # Ensure index exists
    op.execute("""
    CREATE INDEX IF NOT EXISTS ix_panelists_id
    ON panelists (id)
    """)

    # Ensure unique constraint exists (safe + idempotent)
    op.execute("""
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE conname = 'uq_panelist_creator_id'
        ) THEN
            ALTER TABLE panelists
            ADD CONSTRAINT uq_panelist_creator_id
            UNIQUE (creator_email, id);
        END IF;
    END
    $$;
    """)

    # -----------------------------
    # STUDY RESPONSES INDEX
    # -----------------------------
    op.execute("""
    CREATE INDEX IF NOT EXISTS ix_study_responses_panelist_id
    ON study_responses (panelist_id)
    """)


def downgrade() -> None:
    # -----------------------------
    # STUDY RESPONSES
    # -----------------------------
    op.execute("""
    DROP INDEX IF EXISTS ix_study_responses_panelist_id
    """)

    # -----------------------------
    # PANELISTS
    # -----------------------------
    op.execute("""
    ALTER TABLE panelists
    DROP CONSTRAINT IF EXISTS uq_panelist_creator_id
    """)

    op.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS uq_panelist_id_creator
    ON panelists (id, creator_email)
    """)

    op.execute("""
    DROP INDEX IF EXISTS ix_panelists_id
    """)

    # -----------------------------
    # JOBS
    # -----------------------------
    op.execute("""
    DROP INDEX IF EXISTS ix_jobs_user_id
    """)

    op.execute("""
    DROP INDEX IF EXISTS ix_jobs_study_id
    """)

    op.drop_table("jobs")
