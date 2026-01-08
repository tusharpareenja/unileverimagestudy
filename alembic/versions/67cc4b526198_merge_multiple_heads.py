"""merge multiple heads

Revision ID: 67cc4b526198
Revises: 20251208_element_name_length, 801c27908657
Create Date: 2026-01-07 12:00:51.484149

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '67cc4b526198'
down_revision: Union[str, Sequence[str], None] = ('20251208_element_name_length', '801c27908657')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
