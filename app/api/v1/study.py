# app/api/v1/study.py
from __future__ import annotations

from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_active_user
from app.db.session import get_db
from app.models.user_model import User  # adjust import to your user model path
from app.schemas.study_schema import (
    StudyCreate, StudyUpdate, StudyOut, StudyListItem,
    ChangeStatusPayload, RegenerateTasksResponse, ValidateTasksResponse, StudyStatus
)
from app.services import study as study_service

router = APIRouter()

# If you have task generators, inject here. For now use placeholders.
# from app.services.task_generation_adapter import generate_grid_tasks, generate_layer_tasks



@router.post("", response_model=StudyOut, status_code=status.HTTP_201_CREATED)
def create_study_endpoint(
    payload: StudyCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    # Optional: pass BASE_URL from settings for share_url generation
    base_url_for_share: Optional[str] = None
    study = study_service.create_study(
        db=db,
        creator_id=current_user.id,
        payload=payload,
        base_url_for_share=base_url_for_share,
    )
    return study


@router.get("", response_model=List[StudyListItem])
def list_studies_endpoint(
    status_filter: Optional[StudyStatus] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    studies, _total = study_service.list_studies(
        db=db,
        owner_id=current_user.id,
        status_filter=status_filter,
        page=page,
        per_page=per_page,
    )
    return studies


@router.get("/{study_id}", response_model=StudyOut)
def get_study_endpoint(
    study_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    return study_service.get_study(db=db, study_id=study_id, owner_id=current_user.id)


@router.put("/{study_id}", response_model=StudyOut)
def update_study_endpoint(
    study_id: UUID,
    payload: StudyUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    return study_service.update_study(
        db=db, study_id=study_id, owner_id=current_user.id, payload=payload
    )


@router.delete("/{study_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_study_endpoint(
    study_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    study_service.delete_study(db=db, study_id=study_id, owner_id=current_user.id)
    return None


@router.post("/{study_id}/status", response_model=StudyOut)
def change_status_endpoint(
    study_id: UUID,
    payload: ChangeStatusPayload,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    return study_service.change_status(
        db=db, study_id=study_id, owner_id=current_user.id, new_status=payload.status
    )


@router.post("/{study_id}/regenerate-tasks", response_model=RegenerateTasksResponse)
def regenerate_tasks_endpoint(
    study_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    # Inject your generators here
    generator: Dict[str, Any] = {
        # "grid": generate_grid_tasks,
        # "layer": generate_layer_tasks,
    }
    if not generator:
        raise HTTPException(status_code=500, detail="Task generators not configured.")
    return study_service.regenerate_tasks(
        db=db, study_id=study_id, owner_id=current_user.id, generator=generator
    )


@router.get("/{study_id}/validate-tasks", response_model=ValidateTasksResponse)
def validate_tasks_endpoint(
    study_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    return study_service.validate_tasks(db=db, study_id=study_id, owner_id=current_user.id)


@router.get("/{study_id}/stats")
def study_stats_endpoint(
    study_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    study = study_service.get_study(db=db, study_id=study_id, owner_id=current_user.id)
    return {
        "total_responses": study.total_responses,
        "completed_responses": study.completed_responses,
        "abandoned_responses": study.abandoned_responses,
        "status": study.status,
        "created_at": study.created_at,
        "launched_at": study.launched_at,
        "completed_at": study.completed_at,
    }


@router.get("/{study_id}/share-url")
def share_url_endpoint(
    study_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    study = study_service.get_study(db=db, study_id=study_id, owner_id=current_user.id)
    return {"share_token": study.share_token, "share_url": study.share_url}