"""fix_study_type_enum_hybrid

Revision ID: 52bc036b4699
Revises: 91638b56b1ab
Create Date: 2026-01-23 13:11:35.661941

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '52bc036b4699'
down_revision: Union[str, Sequence[str], None] = '91638b56b1ab'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
