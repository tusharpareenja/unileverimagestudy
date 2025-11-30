from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.db.base import Base

# Determine if we're connecting to a local database
is_local_db = "localhost" in settings.DATABASE_URL or "127.0.0.1" in settings.DATABASE_URL

# Configure connection arguments based on environment
if is_local_db:
    # Local database - no SSL required
    connect_args = {}
else:
    # Azure or remote database - require SSL with keepalives
    connect_args = {
        "sslmode": "require",
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    }

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,      # health-check connections before use
    pool_recycle=1800,       # recycle connections every 30 minutes to avoid stale SSL
    pool_size=10,
    max_overflow=20,
    connect_args=connect_args
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
