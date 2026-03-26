"""enforce unique respondent id per study

Revision ID: 20260326_study_resp_unique
Revises: 20260225_panelist_domain, 20260319_fragrance
Create Date: 2026-03-26

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20260326_study_resp_unique"
down_revision: Union[str, Sequence[str], None] = ("20260225_panelist_domain", "20260319_fragrance")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Deduplicate existing collisions first to avoid migration failures.
    # Keep the earliest created row per (study_id, respondent_id).
    op.execute(
        """
        DELETE FROM study_responses sr
        USING (
            SELECT id
            FROM (
                SELECT
                    id,
                    ROW_NUMBER() OVER (
                        PARTITION BY study_id, respondent_id
                        ORDER BY created_at ASC NULLS LAST, id ASC
                    ) AS rn
                FROM study_responses
            ) ranked
            WHERE ranked.rn > 1
        ) dupes
        WHERE sr.id = dupes.id;
        """
    )

    op.create_unique_constraint(
        "uq_study_responses_study_respondent",
        "study_responses",
        ["study_id", "respondent_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_study_responses_study_respondent",
        "study_responses",
        type_="unique",
    )
