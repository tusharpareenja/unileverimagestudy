# app/api/v1/study.py
from __future__ import annotations

from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_active_user
from app.db.session import get_db
from app.models.user_model import User
from app.models.study_model import Study
from app.schemas.study_schema import (
    StudyCreate, StudyUpdate, StudyOut, StudyListItem,
    ChangeStatusPayload, RegenerateTasksResponse, ValidateTasksResponse, StudyStatus,
    GenerateTasksRequest, GenerateTasksResult, StudyPublicMinimal, StudyBasicDetails
)
from app.services import study as study_service
from app.services.response import StudyResponseService
from app.services.task_generation_adapter import generate_grid_tasks, generate_layer_tasks
from app.core.config import settings

router = APIRouter()


@router.post("", response_model=StudyOut, status_code=status.HTTP_201_CREATED)
def create_study_endpoint(
    payload: StudyCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    # Use settings.BASE_URL as default for share URL generation
    study = study_service.create_study(
        db=db,
        creator_id=current_user.id,
        payload=payload,
        base_url_for_share=settings.BASE_URL,
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
    rows, _total = study_service.list_studies(
        db=db,
        owner_id=current_user.id,
        status_filter=status_filter,
        page=page,
        per_page=per_page,
    )
    # Lightweight enrichment: only the counters needed for the list card
    if not rows:
        return []
    enriched: List[StudyListItem] = []
    for row in rows:
        # row is (Study, total_calc, completed_calc, abandoned_calc, avg_duration_calc)
        s = row[0]
        total_calc = int(row[1] or 0)
        completed_calc = int(row[2] or 0)
        abandoned_calc = int(row[3] or 0)
        avg_duration_calc = float(row[4] or 0)
        item = StudyListItem.model_validate(s).model_dump()
        respondents_target = int((s.audience_segmentation or {}).get("number_of_respondents") or 0)
        item.update({
            "total_responses": total_calc,
            "completed_responses": completed_calc,
            "abandoned_responses": abandoned_calc,
            "respondents_target": respondents_target,
            "respondents_completed": completed_calc,
            "average_duration": avg_duration_calc,
            "completion_rate": (completed_calc / total_calc * 100) if total_calc else 0,
            "abandonment_rate": (abandoned_calc / total_calc * 100) if total_calc else 0,
        })
        enriched.append(StudyListItem(**item))
    return enriched


@router.get("/{study_id}", response_model=StudyOut)
def get_study_endpoint(
    study_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    return study_service.get_study(db=db, study_id=study_id, owner_id=current_user.id)


@router.get("/{study_id}/basic", response_model=StudyBasicDetails)
def get_study_basic_details_endpoint(
    study_id: UUID,
    db: Session = Depends(get_db),
):
    """
    Get basic study details (no authentication required).
    Returns core study information including:
    - Study title, status, type, created date
    - Background, main question, orientation text
    - Rating scale configuration
    - Study config (audience segmentation)
    - Classification questions
    """
    details = study_service.get_study_basic_details_public(db=db, study_id=study_id)
    if not details:
        raise HTTPException(status_code=404, detail="Study not found")
    return details


@router.get("/private/{study_id}", response_model=StudyOut)
def get_study_private_endpoint(
    study_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get complete study details for the study creator only.
    This endpoint provides full access to all study information including:
    - All study metadata and configuration
    - Complete elements/layers with images
    - Classification questions
    - Tasks and response data
    - Share URL and tokens
    """
    return study_service.get_study(db=db, study_id=study_id, owner_id=current_user.id)


@router.get("/{study_id}/is-owner")
def check_study_ownership_endpoint(
    study_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Check if the logged-in user is the owner of the study and if the study is active.
    Returns ownership status and active status in a single optimized query.
    """
    from sqlalchemy import select
    
    # Single optimized query to check ownership and status
    stmt = select(Study.creator_id, Study.status).where(Study.id == study_id)
    result = db.execute(stmt).first()
    
    if not result:
        return {"is_owner": False, "is_active": False}
    
    is_owner = result.creator_id == current_user.id
    is_active = result.status == 'active'
    
    return {"is_owner": is_owner, "is_active": is_active}


@router.get("/public/{study_id}")
def get_study_public_endpoint(
    study_id: UUID,
    db: Session = Depends(get_db),
):
    """
    Get study information for public access (no authentication required).
    Only returns studies that are active and have a share_token.
    """
    minimal = study_service.get_study_public_minimal(db=db, study_id=study_id)
    if not minimal:
        raise HTTPException(
            status_code=404, 
            detail="Study not found or not publicly accessible"
        )
    # Return minimal json directly for speed
    return {
        "id": str(minimal.id),
        "title": minimal.title,
        "study_type": minimal.study_type,
        "respondents_target": minimal.respondents_target,
    }


@router.get("/public/{study_id}/details", response_model=StudyOut)
def get_study_public_details_endpoint(
    study_id: UUID,
    db: Session = Depends(get_db),
):
    """
    Get complete study details for public access (no authentication required).
    Returns full study information including elements, layers, and classification questions.
    Only returns studies that are active and have a share_token.
    """
    study = study_service.get_study_public(db=db, study_id=study_id)
    if not study:
        raise HTTPException(
            status_code=404, 
            detail="Study not found or not publicly accessible"
        )
    return study


@router.get("/share/details")
def get_study_share_details_endpoint(
    study_id: UUID,
    db: Session = Depends(get_db),
):
    """
    Public: fetch only share_url, status, title, and study_type for a study.
    """
    info = study_service.get_study_share_details(db=db, study_id=study_id)
    if not info:
        raise HTTPException(status_code=404, detail="Study not found")
    return {
        "id": str(info.get("id")),
        "title": info.get("title"),
        "study_type": info.get("study_type"),
        "status": info.get("status"),
        "share_url": info.get("share_url"),
    }


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
    generator: Dict[str, Any] = {
        "grid": generate_grid_tasks,
        "layer": generate_layer_tasks,
    }
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


@router.post("/generate-tasks", response_model=GenerateTasksResult)
def generate_tasks_from_body_endpoint(
    payload: GenerateTasksRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    # Optional persistence path if study_id is provided and owned by user
    persist_to_study: Optional[UUID] = payload.study_id
    study_row = None
    if persist_to_study is not None:
        from sqlalchemy import select
        from app.models.study_model import Study
        study_row = db.scalars(
            select(Study).where(Study.id == persist_to_study, Study.creator_id == current_user.id)
        ).first()
        if not study_row:
            raise HTTPException(status_code=404, detail="Study not found or access denied.")

    if payload.study_type == 'grid':
        if not payload.elements or len(payload.elements) == 0:
            raise HTTPException(status_code=400, detail="Grid study requires elements")
        num_elements = len(payload.elements)
        elements = []
        # Use StudyElement dataclass shape lightly: only content is used in generator for grid
        for idx, e in enumerate(payload.elements):
            class _Tmp:
                content: str = e.content
            elements.append(_Tmp())
        result = generate_grid_tasks(
            num_elements=num_elements,
            tasks_per_consumer=None,
            number_of_respondents=payload.audience_segmentation.number_of_respondents,
            exposure_tolerance_cv=payload.exposure_tolerance_cv or 1.0,
            seed=payload.seed,
            elements=elements,
        )
        if study_row is not None:
            study_row.tasks = result.get('tasks', {})
            db.commit()
        return GenerateTasksResult(tasks=result.get('tasks', {}), metadata=result.get('metadata', {}))
    elif payload.study_type == 'layer':
        if not payload.study_layers or len(payload.study_layers) == 0:
            raise HTTPException(status_code=400, detail="Layer study requires study_layers")
        # Build a minimal StudyLayer-like structure for adapter
        class _TmpImg:
            def __init__(self, name: str, url: str):
                self.name = name
                self.url = url
        class _TmpLayer:
            def __init__(self, name: str, images: list, z_index: int, order: int):
                self.name = name
                self.images = images
                self.z_index = z_index
                self.order = order
        layers = []
        for L in payload.study_layers:
            imgs = [_TmpImg(img.name, img.url) for img in L.images or []]
            layers.append(_TmpLayer(L.name, imgs, L.z_index, L.order))
        result = generate_layer_tasks(
            layers=layers,
            number_of_respondents=payload.audience_segmentation.number_of_respondents,
            exposure_tolerance_pct=payload.exposure_tolerance_pct or 2.0,
            seed=payload.seed,
        )
        if study_row is not None:
            study_row.tasks = result.get('tasks', {})
            db.commit()
        return GenerateTasksResult(tasks=result.get('tasks', {}), metadata=result.get('metadata', {}))
    else:
        raise HTTPException(status_code=400, detail="Unsupported study_type")