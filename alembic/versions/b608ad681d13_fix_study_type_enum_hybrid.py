"""fix_study_type_enum_hybrid

Revision ID: b608ad681d13
Revises: 52bc036b4699
Create Date: 2026-01-23 13:17:28.470025

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b608ad681d13'
down_revision: Union[str, Sequence[str], None] = '52bc036b4699'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Add 'hybrid' to study_type_enum."""
    # Check if 'hybrid' already exists in the enum
    connection = op.get_bind()
    result = connection.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM pg_enum e "
            "JOIN pg_type t ON e.enumtypid = t.oid "
            "WHERE t.typname = 'study_type_enum' AND e.enumlabel = 'hybrid')"
        )
    )
    hybrid_exists = result.scalar()
    
    if not hybrid_exists:
        # ALTER TYPE ... ADD VALUE cannot run in a transaction
        # We need to use execute with special parameters
        op.execute("COMMIT")
        op.execute("ALTER TYPE study_type_enum ADD VALUE 'hybrid'")


def downgrade() -> None:
    """Downgrade schema - Remove 'hybrid' from study_type_enum."""
    # PostgreSQL doesn't support removing enum values directly
    # This would require recreating the enum type, which is complex
    # For now, we'll leave the value in place
    pass
