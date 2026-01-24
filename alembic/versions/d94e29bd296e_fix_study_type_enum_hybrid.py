"""fix_study_type_enum_hybrid

Revision ID: d94e29bd296e
Revises: b608ad681d13
Create Date: 2026-01-23 13:19:58.935486

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd94e29bd296e'
down_revision: Union[str, Sequence[str], None] = 'b608ad681d13'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
