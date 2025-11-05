from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'rev_20251103_transform_layers'
down_revision = 'b863edc7e69e'
branch_labels = None
depends_on = None


def upgrade() -> None:
	# Add nullable JSONB column 'transform' to study_layers
	op.add_column('study_layers', sa.Column('transform', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
	# Drop column 'transform'
	op.drop_column('study_layers', 'transform')
