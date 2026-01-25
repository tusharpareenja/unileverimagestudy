"""added_panelist_iddd

Revision ID: 356662f2d050
Revises: 544331b3a58c
Create Date: 2026-01-25 19:55:45.432284

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '356662f2d050'
down_revision: Union[str, Sequence[str], None] = '544331b3a58c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
