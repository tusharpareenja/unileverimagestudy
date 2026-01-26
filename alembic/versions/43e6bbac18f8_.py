"""empty message

Revision ID: 43e6bbac18f8
Revises: 09d9552265f8
Create Date: 2026-01-26 13:40:47.953195

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '43e6bbac18f8'
down_revision: Union[str, Sequence[str], None] = '09d9552265f8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
