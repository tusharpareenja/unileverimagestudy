from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'rev_20251128_layer_type'
down_revision = 'rev_20251103_transform_layers'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create enum type for layer_type
    op.execute("CREATE TYPE layer_type_enum AS ENUM ('image', 'text')")
    
    # Add layer_type column to study_layers table with default value 'image'
    op.add_column('study_layers', 
        sa.Column('layer_type', sa.Enum('image', 'text', name='layer_type_enum'), 
                  nullable=False, server_default='image'))


def downgrade() -> None:
    # Drop the layer_type column
    op.drop_column('study_layers', 'layer_type')
    
    # Drop the enum type
    op.execute("DROP TYPE layer_type_enum")
