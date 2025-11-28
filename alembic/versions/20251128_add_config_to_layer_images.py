from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'rev_20251128_layer_image_config'
down_revision = 'rev_20251128_layer_type'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add config JSONB column to layer_images table
    op.add_column('layer_images', 
        sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    # Drop the config column
    op.drop_column('layer_images', 'config')
