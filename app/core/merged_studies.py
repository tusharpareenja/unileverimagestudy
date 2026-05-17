from __future__ import annotations

from uuid import UUID

from app.core.config import settings


def _configured_pairs() -> list[dict[str, str]]:
    first_study_id = (settings.MERGE_STUDY_A or "").strip()
    second_study_id = (settings.MERGE_STUDY_B or "").strip()
    if not first_study_id or not second_study_id:
        return []
    return [{"first_study_id": first_study_id, "second_study_id": second_study_id}]


def _normalize_study_id(study_id: UUID | str | None) -> str:
    return str(study_id or "").strip().lower()


def is_merged_study(study_id: UUID | str | None) -> bool:
    normalized = _normalize_study_id(study_id)
    if not normalized:
        return False
    return any(
        normalized in {
            _normalize_study_id(pair.get("first_study_id")),
            _normalize_study_id(pair.get("second_study_id")),
        }
        for pair in _configured_pairs()
    )


def is_first_merged_study(study_id: UUID | str | None) -> bool:
    normalized = _normalize_study_id(study_id)
    return any(normalized == _normalize_study_id(pair.get("first_study_id")) for pair in _configured_pairs())


def get_second_study_id(study_id: UUID | str | None) -> str | None:
    normalized = _normalize_study_id(study_id)
    for pair in _configured_pairs():
        if normalized == _normalize_study_id(pair.get("first_study_id")):
            return pair.get("second_study_id")
    return None
