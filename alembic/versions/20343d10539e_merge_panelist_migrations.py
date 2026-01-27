"""merge panelist migrations

Revision ID: 20343d10539e
Revises: 14fd6c262174, fix_panelist_2026
Create Date: 2026-01-27 15:03:07.954161

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20343d10539e'
down_revision: Union[str, Sequence[str], None] = ('14fd6c262174', 'fix_panelist_2026')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
