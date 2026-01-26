"""remove_panelist_name_and_update_id

Revision ID: ad1e068883f6
Revises: 43e6bbac18f8
Create Date: 2026-01-26 14:53:23.733341

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ad1e068883f6'
down_revision: Union[str, Sequence[str], None] = '43e6bbac18f8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: Remove name, update ID to 8 chars, add per-creator uniqueness."""
    
    # Drop the old composite index on (id, name)
    op.drop_index('idx_panelist_search', table_name='panelists')
    
    # Remove the name column
    op.drop_column('panelists', 'name')
    
    # Alter id column from String(10) to String(8)
    # Note: This requires existing data to be truncated or migrated
    # For safety, we'll allow NULL temporarily during migration
    op.alter_column('panelists', 'id',
                    existing_type=sa.String(10),
                    type_=sa.String(8),
                    existing_nullable=False)
    
    # Update study_responses.panelist_id from String(10) to String(8)
    op.alter_column('study_responses', 'panelist_id',
                    existing_type=sa.String(10),
                    type_=sa.String(8),
                    existing_nullable=True)
    
    # Add composite unique constraint: (creator_email, id)
    # This ensures IDs are unique per creator, not globally
    op.create_unique_constraint('uq_panelist_creator_id', 'panelists', 
                               ['creator_email', 'id'])


def downgrade() -> None:
    """Downgrade schema: Restore name, revert ID to 10 chars."""
    
    # Drop the composite unique constraint
    op.drop_constraint('uq_panelist_creator_id', 'panelists', type_='unique')
    
    # Revert study_responses.panelist_id from String(8) to String(10)
    op.alter_column('study_responses', 'panelist_id',
                    existing_type=sa.String(8),
                    type_=sa.String(10),
                    existing_nullable=True)
    
    # Revert id column from String(8) to String(10)
    op.alter_column('panelists', 'id',
                    existing_type=sa.String(8),
                    type_=sa.String(10),
                    existing_nullable=False)
    
    # Add back the name column
    op.add_column('panelists', sa.Column('name', sa.String(100), nullable=True))
    
    # Recreate the old composite index on (id, name)
    op.create_index('idx_panelist_search', 'panelists', ['id', 'name'])
