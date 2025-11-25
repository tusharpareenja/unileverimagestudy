"""Check classification questions in database"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not found in .env")
    exit(1)

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    # Check total count
    result = conn.execute(text("SELECT COUNT(*) FROM study_classification_questions"))
    total_count = result.fetchone()[0]
    print(f"Total classification questions in database: {total_count}")

    if total_count > 0:
        # Get a sample study with questions
        result = conn.execute(text("""
            SELECT
                scq.study_id,
                s.title,
                COUNT(scq.id) as question_count
            FROM study_classification_questions scq
            JOIN studies s ON s.id = scq.study_id
            GROUP BY scq.study_id, s.title
            LIMIT 1
        """))
        row = result.fetchone()

        if row:
            study_id, title, count = row
            print(f"\nStudy '{title}' (ID: {study_id}) has {count} question(s)")

            # Get the actual questions
            result = conn.execute(text("""
                SELECT question_id, question_text, question_type, is_required, answer_options
                FROM study_classification_questions
                WHERE study_id = :study_id
                ORDER BY "order"
            """), {"study_id": study_id})

            print("\nQuestions:")
            for q in result:
                print(f"  - {q.question_id}: {q.question_text}")
                print(f"    Type: {q.question_type}, Required: {q.is_required}")
                print(f"    Answer options: {q.answer_options}")
    else:
        print("\nNo classification questions found in database.")
        print("You need to create a study with classification_questions in the update payload.")
