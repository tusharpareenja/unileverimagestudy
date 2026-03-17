"""add product_keys to studies

Revision ID: 20260225_product_keys
Revises: f41e4f3cfca6
Create Date: 2026-02-25

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '20260225_product_keys'
down_revision: Union[str, Sequence[str], None] = 'f41e4f3cfca6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('studies', sa.Column('product_keys', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column('studies', 'product_keys')
