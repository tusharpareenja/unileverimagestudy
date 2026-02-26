"""panelist scope by creator_domain

Revision ID: 20260225_panelist_domain
Revises: 20260225_study_product_id
Create Date: 2026-02-25

Scope panelists by company domain so all users with the same email domain
(e.g. unilever.com) share one pool of panelists.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '20260225_panelist_domain'
down_revision: Union[str, Sequence[str], None] = '20260225_study_product_id'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('panelists', sa.Column('creator_domain', sa.String(length=255), nullable=True))
    # Backfill: set creator_domain to the part after @ in creator_email (lowercase)
    op.execute("""
        UPDATE panelists
        SET creator_domain = LOWER(SUBSTRING(creator_email FROM POSITION('@' IN creator_email) + 1))
        WHERE creator_domain IS NULL AND creator_email LIKE '%@%'
    """)
    op.execute("""
        UPDATE panelists
        SET creator_domain = ''
        WHERE creator_domain IS NULL
    """)
    op.alter_column(
        'panelists',
        'creator_domain',
        existing_type=sa.String(255),
        nullable=False,
    )
    op.create_index('idx_panelist_creator_domain', 'panelists', ['creator_domain'], unique=False)
    op.drop_constraint('uq_panelist_creator_id', 'panelists', type_='unique')
    op.create_unique_constraint('uq_panelist_domain_id', 'panelists', ['creator_domain', 'id'])


def downgrade() -> None:
    op.drop_constraint('uq_panelist_domain_id', 'panelists', type_='unique')
    op.create_unique_constraint('uq_panelist_creator_id', 'panelists', ['creator_email', 'id'])
    op.drop_index('idx_panelist_creator_domain', table_name='panelists')
    op.drop_column('panelists', 'creator_domain')
