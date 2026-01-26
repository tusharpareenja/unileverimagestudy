"""merge_heads_before_revert

Revision ID: 9d7847fb2521
Revises: 6017f1a8e921, cb44aaf0f7d8
Create Date: 2026-01-26 15:49:07.815929

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9d7847fb2521'
down_revision: Union[str, Sequence[str], None] = ('6017f1a8e921', 'cb44aaf0f7d8')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
