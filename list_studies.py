
import sys
import os
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

# Add current directory to path
sys.path.append(os.getcwd())

from app.db.session import engine
from app.models.study_model import Study

def list_studies():
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        studies = db.execute(select(Study.id, Study.title).limit(5)).all()
        for s in studies:
            print(f"Study ID: {s.id}, Title: {s.title}")
    finally:
        db.close()

if __name__ == "__main__":
    list_studies()
