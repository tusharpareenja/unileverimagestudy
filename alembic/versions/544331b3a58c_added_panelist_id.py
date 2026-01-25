"""added_panelist_id

Revision ID: 544331b3a58c
Revises: f41e4f3cfca6
Create Date: 2026-01-25 19:55:39.767987

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '544331b3a58c'
down_revision: Union[str, Sequence[str], None] = 'f41e4f3cfca6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
