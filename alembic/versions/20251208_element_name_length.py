"""increase element name length

Revision ID: 20251208_element_name_length
Revises: 20251208_add_text_study_type
Create Date: 2025-12-08 22:20:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20251208_element_name_length'
down_revision = '20251208_add_text_study_type'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Increase the length of name column in study_elements table to 1000
    op.alter_column('study_elements', 'name', 
                    type_=sa.String(1000), 
                    existing_type=sa.String(100), 
                    existing_nullable=False)


def downgrade() -> None:
    # Revert back to 100
    # Note: This might fail if there are values longer than 100 chars
    op.alter_column('study_elements', 'name', 
                    type_=sa.String(100), 
                    existing_type=sa.String(1000), 
                    existing_nullable=False)
