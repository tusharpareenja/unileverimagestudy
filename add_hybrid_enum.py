"""
Simple standalone script to add 'hybrid' to study_type_enum.
Run this with: python add_hybrid_enum.py
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment
load_dotenv()

# Connect to database
db_url = os.getenv('DATABASE_URL')
if not db_url:
    print("ERROR: DATABASE_URL not found")
    exit(1)

print("Connecting to database...")
engine = create_engine(db_url)

try:
    with engine.connect() as conn:
        # Set autocommit
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        
        # Check if exists
        exists = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM pg_enum e
                JOIN pg_type t ON e.enumtypid = t.oid  
                WHERE t.typname = 'study_type_enum' AND e.enumlabel = 'hybrid'
            )
        """)).scalar()
        
        if exists:
            print("✓ 'hybrid' already exists")
        else:
            print("Adding 'hybrid'...")
            conn.execute(text("ALTER TYPE study_type_enum ADD VALUE 'hybrid'"))
            print("✓ Added 'hybrid'")
        
        # Show all values
        result = conn.execute(text("""
            SELECT enumlabel FROM pg_enum e
            JOIN pg_type t ON e.enumtypid = t.oid
            WHERE t.typname = 'study_type_enum'
            ORDER BY enumlabel
        """))
        values = [r[0] for r in result]
        print(f"\nEnum values: {', '.join(values)}")
        print("\n✓ Done!")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
