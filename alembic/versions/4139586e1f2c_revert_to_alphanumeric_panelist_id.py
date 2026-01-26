"""revert_to_alphanumeric_panelist_id

Revision ID: 4139586e1f2c
Revises: 9d7847fb2521
Create Date: 2026-01-26 15:49:08.637629

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4139586e1f2c'
down_revision: Union[str, Sequence[str], None] = '9d7847fb2521'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
