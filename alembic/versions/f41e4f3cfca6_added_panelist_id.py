"""added_panelist_id

Revision ID: f41e4f3cfca6
Revises: 9584902e5a2f
Create Date: 2026-01-25 19:55:20.023733

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f41e4f3cfca6'
down_revision: Union[str, Sequence[str], None] = '9584902e5a2f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
