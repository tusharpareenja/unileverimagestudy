"""per_creator_panelist_uniqueness

Revision ID: 6017f1a8e921
Revises: cb44aaf0f7d8
Create Date: 2026-01-26 15:11:03.396301

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6017f1a8e921'
down_revision: Union[str, Sequence[str], None] = 'cb44aaf0f7d8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
