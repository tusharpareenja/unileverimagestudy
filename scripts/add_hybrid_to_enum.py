"""
Script to manually add 'hybrid' to the study_type_enum in PostgreSQL.
This bypasses Alembic and adds the value directly.
"""
from sqlalchemy import create_engine, text
from app.core.config import settings

print(f"Connecting to database...")

try:
    # Create engine from settings
    engine = create_engine(settings.DATABASE_URL)
    
    # Get a raw connection and set autocommit
    with engine.connect() as connection:
        # Enable autocommit mode for ALTER TYPE
        connection = connection.execution_options(isolation_level="AUTOCOMMIT")
        
        # Check if 'hybrid' already exists
        result = connection.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM pg_enum e
                JOIN pg_type t ON e.enumtypid = t.oid
                WHERE t.typname = 'study_type_enum' AND e.enumlabel = 'hybrid'
            )
        """))
        
        exists = result.scalar()
        
        if exists:
            print("✓ 'hybrid' already exists in study_type_enum")
        else:
            print("Adding 'hybrid' to study_type_enum...")
            connection.execute(text("ALTER TYPE study_type_enum ADD VALUE 'hybrid'"))
            print("✓ Successfully added 'hybrid' to study_type_enum")
        
        # Verify all values
        result = connection.execute(text("""
            SELECT enumlabel FROM pg_enum e
            JOIN pg_type t ON e.enumtypid = t.oid
            WHERE t.typname = 'study_type_enum'
            ORDER BY enumlabel
        """))
        
        values = [row[0] for row in result]
        print(f"\nCurrent study_type_enum values: {', '.join(values)}")
    
    print("\n✓ Done! You can now restart your server.")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
