"""Test script to verify classification questions are loaded from database"""

from app.db.session import engine
from sqlalchemy import text
from uuid import UUID

# Get a study ID that has classification questions
with engine.connect() as conn:
    # First, find a study that should have classification questions
    result = conn.execute(text(
        "SELECT study_id, COUNT(*) as count "
        "FROM study_classification_questions "
        "GROUP BY study_id "
        "LIMIT 1"
    ))
    row = result.fetchone()

    if not row:
        print("[INFO] No classification questions found in database")
        print("\nLet's check if the table exists and has any data:")
        result = conn.execute(text(
            "SELECT COUNT(*) FROM study_classification_questions"
        ))
        count = result.fetchone()[0]
        print(f"Total classification questions in database: {count}")
    else:
        study_id = row[0]
        count = row[1]
        print(f"[INFO] Found study {study_id} with {count} classification question(s)")

        # Now test the ORM relationship
        from app.db.session import SessionLocal
        from sqlalchemy.orm import selectinload
        from sqlalchemy import select
        from app.models.study_model import Study

        db = SessionLocal()
        try:
            # Load study with classification questions
            stmt = (
                select(Study)
                .options(selectinload(Study.classification_questions))
                .where(Study.id == study_id)
            )
            study = db.scalars(stmt).first()

            if study:
                print(f"\n[TEST] Study found: {study.title}")
                print(f"[TEST] Classification questions count via ORM: {len(study.classification_questions)}")

                if study.classification_questions:
                    print("[PASS] Classification questions loaded via ORM")
                    for q in study.classification_questions:
                        print(f"  - Question {q.question_id}: {q.question_text[:50]}...")
                        print(f"    Type: {q.question_type}, Required: {q.is_required}")
                else:
                    print("[FAIL] Classification questions NOT loaded via ORM (empty list)")

                # Now test schema serialization
                from app.schemas.study_schema import StudyOut
                try:
                    study_out = StudyOut.model_validate(study)
                    result = study_out.model_dump()

                    print(f"\n[TEST] Schema serialization:")
                    print(f"  classification_questions in output: {len(result.get('classification_questions', []))}")

                    if result.get('classification_questions'):
                        print("[PASS] Classification questions serialized correctly")
                        for q in result['classification_questions']:
                            print(f"  - Question {q['question_id']}: {q['question_text'][:50]}...")
                    else:
                        print("[FAIL] Classification questions empty after serialization")
                except Exception as e:
                    print(f"[ERROR] Schema validation failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[ERROR] Study {study_id} not found")
        finally:
            db.close()
