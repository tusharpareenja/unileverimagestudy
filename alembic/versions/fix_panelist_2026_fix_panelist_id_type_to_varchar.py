"""fix_panelist_id_type_and_constraints

Revision ID: fix_panelist_2026
Revises: ad1e068883f6
Create Date: 2026-01-26 16:47:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'fix_panelist_2026'
down_revision: Union[str, Sequence[str], None] = 'ad1e068883f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Ultra-robust upgrade to handle any inconsistent state."""
    
    # 1. Foreign Key cleanup
    op.execute("ALTER TABLE study_responses DROP CONSTRAINT IF EXISTS study_responses_panelist_id_fkey")
    
    # 2. Panelist PK cleanup
    op.execute("ALTER TABLE panelists DROP CONSTRAINT IF EXISTS panelists_pkey CASCADE")
    op.execute("DROP INDEX IF EXISTS idx_panelist_search")
    
    # 3. Add surrogate PK helper
    # Check if column exists, if not, add it
    op.execute("""
        DO $$ 
        BEGIN 
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='panelists' AND column_name='internal_id') THEN
                ALTER TABLE panelists ADD COLUMN internal_id SERIAL;
            END IF;
        END $$;
    """)
    # Ensure it's the primary key
    op.execute("ALTER TABLE panelists ADD PRIMARY KEY (internal_id)")
    
    # 4. Type conversions
    op.execute("ALTER TABLE study_responses ALTER COLUMN panelist_id TYPE VARCHAR(8) USING panelist_id::text")
    op.execute("ALTER TABLE panelists ALTER COLUMN id TYPE VARCHAR(8) USING id::text")

    # 5. Deduplication and uniqueness
    op.execute("""
        DELETE FROM panelists a USING panelists b
        WHERE a.internal_id > b.internal_id 
        AND a.id = b.id 
        AND a.creator_email = b.creator_email
    """)
    
    op.execute("ALTER TABLE panelists DROP CONSTRAINT IF EXISTS uq_panelist_creator_id")
    op.execute("ALTER TABLE panelists ADD CONSTRAINT uq_panelist_creator_id UNIQUE (creator_email, id)")
    
    op.execute("DROP INDEX IF EXISTS ix_panelists_id")
    op.execute("CREATE INDEX ix_panelists_id ON panelists (id)")


def downgrade() -> None:
    """Ultra-robust downgrade."""
    op.execute("ALTER TABLE panelists DROP CONSTRAINT IF EXISTS uq_panelist_creator_id")
    op.execute("DROP INDEX IF EXISTS ix_panelists_id")
    
    # Restore old structure if possible
    try:
        op.execute("ALTER TABLE panelists DROP COLUMN IF EXISTS internal_id")
        op.execute("ALTER TABLE panelists ADD PRIMARY KEY (id)")
    except Exception:
        pass
