
import sys
import os
from uuid import UUID
import time

# Add app to path
sys.path.append(os.getcwd())

from app.db.session import SessionLocal
from app.services.response import StudyResponseService
# Import models to ensure they are registered
from app.models.user_model import User
from app.models.study_model import Study
from app.models.response_model import StudyResponse

def profile_export():
    db = SessionLocal()
    service = StudyResponseService(db)
    # Use the study ID provided by the user
    study_id = UUID("41f0cadd-867e-4026-a4ab-1745be1f0e7a")
    
    print(f"Profiling export for study {study_id}...")
    try:
        # Iterate to trigger generation
        count = 0
        start = time.time()
        for chunk in service.generate_csv_rows_for_study_pandas(study_id):
            count += len(chunk)
        end = time.time()
        print(f"Total time: {end - start:.4f}s")
        print(f"Total bytes generated: {count}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    profile_export()
