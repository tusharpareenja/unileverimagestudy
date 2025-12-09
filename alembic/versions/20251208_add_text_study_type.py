"""Add text to study_type_enum

Revision ID: 20251208_add_text_study_type
Revises: rev_20251128_layer_type
Create Date: 2025-12-08 18:38:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20251208_add_text_study_type'
down_revision = 'rev_20251128_layer_type'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add 'text' value to the existing study_type_enum
    # PostgreSQL requires using ALTER TYPE ... ADD VALUE
    op.execute("ALTER TYPE study_type_enum ADD VALUE IF NOT EXISTS 'text'")


def downgrade() -> None:
    # Note: PostgreSQL does not support removing enum values directly
    # You would need to recreate the enum type without 'text' and update all references
    # For safety, we'll just note this cannot be easily downgraded
    # If needed, manual intervention would be required
    pass
