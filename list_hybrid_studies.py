import sys
import os
sys.path.append(os.getcwd())

from app.db.session import SessionLocal
from app.models.study_model import Study
from sqlalchemy import select

def list_hybrid_studies():
    db = SessionLocal()
    try:
        stmt = select(Study).where(Study.study_type == 'hybrid')
        studies = db.execute(stmt).scalars().all()
        print(f"Found {len(studies)} hybrid studies.")
        for s in studies:
            print(f"ID: {s.id}, Title: {s.title}, Type: {s.study_type}")
    finally:
        db.close()

if __name__ == "__main__":
    list_hybrid_studies()
