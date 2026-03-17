"""add product_id to studies

Revision ID: 20260225_study_product_id
Revises: 20260225_product_keys
Create Date: 2026-02-25

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '20260225_study_product_id'
down_revision: Union[str, Sequence[str], None] = '20260225_product_keys'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('studies', sa.Column('product_id', sa.String(length=100), nullable=True))


def downgrade() -> None:
    op.drop_column('studies', 'product_id')
