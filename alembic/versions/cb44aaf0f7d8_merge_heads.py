"""merge_heads

Revision ID: cb44aaf0f7d8
Revises: ad1e068883f6, 43e6bbac18f8
Create Date: 2026-01-26 14:56:18.530565

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cb44aaf0f7d8'
down_revision: Union[str, Sequence[str], None] = ('ad1e068883f6', '43e6bbac18f8')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
