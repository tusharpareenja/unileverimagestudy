"""Add fragrance question (Q0) to Unilever creator studies

Revision ID: 20260319_fragrance
Revises: 20260318_study_panelist
Create Date: 2026-03-19

For every study whose creator email domain is unilever.com, ensure one
classification question exists with question_id='Q0', question_text='Do you like this fragrance?',
order=0. Existing classification questions for that study get order incremented by 1.
"""
from typing import Sequence, Union
import uuid

from alembic import op
from sqlalchemy import text

revision: str = "20260319_fragrance"
down_revision: Union[str, Sequence[str], None] = "20260318_study_panelist"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

FRAGRANCE_QUESTION_ID = "Q0"
FRAGRANCE_QUESTION_TEXT = "Do you like this fragrance?"
FRAGRANCE_ANSWER_OPTIONS = '[{"id": "Y", "text": "Yes"}, {"id": "N", "text": "No"}]'


def upgrade() -> None:
    conn = op.get_bind()
    # Studies whose creator has unilever.com domain and that don't already have Q0
    rows = conn.execute(text("""
        SELECT s.id AS study_id
        FROM studies s
        JOIN users u ON u.id = s.creator_id
        WHERE LOWER(SUBSTRING(u.email FROM POSITION('@' IN u.email) + 1)) = 'unilever.com'
        AND NOT EXISTS (
            SELECT 1 FROM study_classification_questions scq
            WHERE scq.study_id = s.id AND scq.question_id = :qid
        )
    """), {"qid": FRAGRANCE_QUESTION_ID}).fetchall()

    for (study_id,) in rows:
        # Bump order of existing classification questions
        conn.execute(text("""
            UPDATE study_classification_questions
            SET "order" = "order" + 1
            WHERE study_id = :study_id
        """), {"study_id": study_id})
        # Insert fragrance question with order 0
        new_id = str(uuid.uuid4())
        conn.execute(text("""
            INSERT INTO study_classification_questions
            (id, study_id, question_id, question_text, question_type, is_required, "order", answer_options, config)
            VALUES (:id, :study_id, :question_id, :question_text, 'multiple_choice', 'Y', 0, CAST(:answer_options AS jsonb), '{"system": true}'::jsonb)
        """), {
            "id": new_id,
            "study_id": study_id,
            "question_id": FRAGRANCE_QUESTION_ID,
            "question_text": FRAGRANCE_QUESTION_TEXT,
            "answer_options": FRAGRANCE_ANSWER_OPTIONS,
        })


def downgrade() -> None:
    conn = op.get_bind()
    # Remove Q0 from studies whose creator is unilever.com and shift orders back
    rows = conn.execute(text("""
        SELECT s.id AS study_id
        FROM studies s
        JOIN users u ON u.id = s.creator_id
        WHERE LOWER(SUBSTRING(u.email FROM POSITION('@' IN u.email) + 1)) = 'unilever.com'
        AND EXISTS (
            SELECT 1 FROM study_classification_questions scq
            WHERE scq.study_id = s.id AND scq.question_id = 'Q0'
        )
    """)).fetchall()
    for (study_id,) in rows:
        conn.execute(text("""
            DELETE FROM study_classification_questions
            WHERE study_id = :study_id AND question_id = 'Q0'
        """), {"study_id": study_id})
        conn.execute(text("""
            UPDATE study_classification_questions
            SET "order" = "order" - 1
            WHERE study_id = :study_id AND "order" > 0
        """), {"study_id": study_id})
