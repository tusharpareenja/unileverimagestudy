"""merge multiple heads

Revision ID: e90a63782b08
Revises: rev_20251103_transform_layers, 5f3834fe0402
Create Date: 2025-11-25 09:45:59.113197

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e90a63782b08'
down_revision: Union[str, Sequence[str], None] = ('rev_20251103_transform_layers', '5f3834fe0402')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
