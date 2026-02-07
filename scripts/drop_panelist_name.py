import sys
import os
from sqlalchemy import text

# Add the parent directory to sys.path to allow imports from "app"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import engine

def drop_panelist_name_column():
    """
    Drops the 'name' column from the 'panelists' table.
    """
    print("Attempting to drop 'name' column from 'panelists' table...")
    
    with engine.connect() as connection:
        # We wrap in a transaction to be safe
        with connection.begin():
            try:
                # Check if the column exists first
                check_query = text("""
                    SELECT 1 
                    FROM information_schema.columns 
                    WHERE table_name='panelists' AND column_name='name';
                """)
                exists = connection.execute(check_query).fetchone()
                
                if exists:
                    print("Column 'name' found. Dropping it now...")
                    drop_query = text("ALTER TABLE panelists DROP COLUMN name;")
                    connection.execute(drop_query)
                    print("Successfully dropped 'name' column.")
                else:
                    print("Column 'name' does not exist in 'panelists' table. No action taken.")
                    
            except Exception as e:
                print(f"Error during migration: {e}")
                raise

if __name__ == "__main__":
    drop_panelist_name_column()
