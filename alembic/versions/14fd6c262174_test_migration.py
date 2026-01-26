"""test_migration

Revision ID: 14fd6c262174
Revises: 4139586e1f2c
Create Date: 2026-01-26
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "14fd6c262174"
down_revision: Union[str, Sequence[str], None] = "4139586e1f2c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add `name` column ONLY if it does not exist
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'panelists'
                AND column_name = 'name'
            ) THEN
                ALTER TABLE panelists ADD COLUMN name VARCHAR(100);
            END IF;
        END
        $$;
    """)

    # 2. Backfill name safely
    op.execute("""
        UPDATE panelists
        SET name = creator_email
        WHERE name IS NULL
    """)

    # 3. Make name NOT NULL (only if column exists)
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'panelists'
                AND column_name = 'name'
            ) THEN
                ALTER TABLE panelists
                ALTER COLUMN name SET NOT NULL;
            END IF;
        END
        $$;
    """)

    # 4. Drop index safely
    op.execute("DROP INDEX IF EXISTS idx_panelist_search")

    # 5. Recreate search index safely
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_panelist_search
        ON panelists (id, name)
    """)


def downgrade() -> None:
    # Drop composite index safely
    op.execute("DROP INDEX IF EXISTS idx_panelist_search")

    # Recreate old index safely
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_panelist_search
        ON panelists (id)
    """)

    # Drop name column safely
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'panelists'
                AND column_name = 'name'
            ) THEN
                ALTER TABLE panelists DROP COLUMN name;
            END IF;
        END
        $$;
    """)
