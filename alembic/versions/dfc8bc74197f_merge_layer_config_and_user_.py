"""merge layer config and user optimizations

Revision ID: dfc8bc74197f
Revises: rev_20251128_layer_image_config, fec53fdfc29b
Create Date: 2025-11-28 13:23:57.778627

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dfc8bc74197f'
down_revision: Union[str, Sequence[str], None] = ('rev_20251128_layer_image_config', 'fec53fdfc29b')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
