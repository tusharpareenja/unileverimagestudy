from app.db.session import engine
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text(
        "SELECT column_name, data_type, column_default "
        "FROM information_schema.columns "
        "WHERE table_name = 'studies' AND column_name = 'last_step'"
    ))
    rows = list(result)
    if rows:
        print("Column exists:", rows)
    else:
        print("Column does NOT exist!")
