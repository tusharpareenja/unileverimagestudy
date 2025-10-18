# app/api/v1/study.py
from __future__ import annotations

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

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


def _generate_preview_tasks(payload: GenerateTasksRequest, number_of_respondents: int) -> Dict[str, Any]:
    """Generate a small preview of tasks for immediate display while background job runs"""
    # Generate tasks for just 1-3 respondents as a preview
    preview_respondents = min(3, number_of_respondents)
    
    if payload.study_type == 'grid':
        # Create a minimal preview for grid studies
        if not payload.elements or not payload.categories:
            return {}
        
        # Build categories data for preview
        categories_data = []
        for cat in payload.categories:
            cat_elements = [e for e in payload.elements if e.category_id == cat.category_id]
            categories_data.append({
                "category_name": cat.name,
                "elements": [
                    {
                        "element_id": str(el.element_id),
                        "name": el.name,
                        "content": el.content,
                        "alt_text": el.alt_text or el.name,
                        "element_type": el.element_type,
                    }
                    for el in cat_elements
                ]
            })
        
        # Generate preview tasks using the same algorithm but with fewer respondents
        from app.services.task_generation_core import generate_grid_tasks_v2
        try:
            result = generate_grid_tasks_v2(
                categories_data=categories_data,
                number_of_respondents=preview_respondents,
                exposure_tolerance_cv=payload.exposure_tolerance_cv or 1.0,
                seed=payload.seed,
            )
            return result.get('tasks', {})
        except Exception:
            # If preview generation fails, return empty tasks
            return {}
    
    elif payload.study_type == 'layer':
        # Create a minimal preview for layer studies
        if not payload.study_layers:
            return {}
        
        # Build layers for preview
        layers = []
        for layer in payload.study_layers:
            layer_obj = type('Layer', (), {
                'name': layer.name,
                'images': [type('Image', (), {
                    'name': img.name,
                    'url': img.url
                }) for img in layer.images or []],
                'z_index': layer.z_index,
                'order': layer.order
            })()
            layers.append(layer_obj)
        
        # Generate preview tasks
        from app.services.task_generation_adapter import generate_layer_tasks
        try:
            result = generate_layer_tasks(
                layers=layers,
                number_of_respondents=preview_respondents,
                exposure_tolerance_pct=payload.exposure_tolerance_pct or 2.0,
                seed=payload.seed,
            )
            return result.get('tasks', {})
        except Exception:
            # If preview generation fails, return empty tasks
            return {}
    
    return {}


def _ensure_study_exists(payload: GenerateTasksRequest, db: Session, current_user: User):
    """Helper function to ensure study exists, creating it if necessary"""
    from sqlalchemy import select
    from app.models.study_model import Study as StudyModel
    
    if payload.study_id is not None:
        study_row = db.scalars(
            select(StudyModel).where(StudyModel.id == payload.study_id, StudyModel.creator_id == current_user.id)
        ).first()
        if not study_row:
            raise HTTPException(status_code=404, detail="Study not found or access denied.")
        return study_row
    else:
        # Create a new draft study if minimal required fields provided
        if not payload.title or not payload.background or not payload.language or not payload.main_question or not payload.orientation_text:
            raise HTTPException(status_code=400, detail="Missing required study fields for on-the-fly creation")
        return study_service.create_study(
            db=db,
            creator_id=current_user.id,
            payload=StudyCreate(
                title=payload.title,
                background=payload.background,
                language=payload.language,
                main_question=payload.main_question,
                orientation_text=payload.orientation_text,
                study_type=payload.study_type,
                background_image_url=payload.background_image_url,
                rating_scale=payload.rating_scale,
                audience_segmentation=payload.audience_segmentation,
                categories=payload.categories,
                elements=payload.elements,
                study_layers=payload.study_layers,
                classification_questions=payload.classification_questions,
            ),
            base_url_for_share=settings.BASE_URL,
        )


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
    Handles different study statuses with appropriate messaging.
    """
    result = study_service.get_study_public_with_status_check(db=db, study_id=study_id)
    
    # Check if there's an error in the result
    if "error" in result:
        if result["error"] == "Study not found":
            raise HTTPException(
                status_code=404, 
                detail=result["message"]
            )
        elif result["error"] == "Study is paused":
            raise HTTPException(
                status_code=403, 
                detail=result["message"]
            )
        elif result["error"] == "Study is completed":
            raise HTTPException(
                status_code=410, 
                detail=result["message"]
            )
        else:
            raise HTTPException(
                status_code=403, 
                detail=result["message"]
            )
    
    # Return the study data if no errors
    return result


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
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"PUT /studies/{study_id} - User: {current_user.id}")
        logger.info(f"Payload fields: {list(payload.model_dump(exclude_none=True).keys())}")
        
        # Log specific fields that might cause validation errors
        if payload.rating_scale:
            logger.info(f"Rating scale: {payload.rating_scale.model_dump()}")
        if payload.status:
            logger.info(f"Status: {payload.status}")
        if payload.title:
            logger.info(f"Title length: {len(payload.title)}")
        
        result = study_service.update_study(
            db=db, study_id=study_id, owner_id=current_user.id, payload=payload
        )
        
        logger.info(f"Study {study_id} updated successfully")
        return result
        
    except HTTPException as e:
        logger.error(f"HTTPException in PUT /studies/{study_id}: {e.detail} (status: {e.status_code})")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in PUT /studies/{study_id}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


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
    # Check if we should use async processing for large studies
    number_of_respondents = payload.audience_segmentation.number_of_respondents if payload.audience_segmentation else 0
    
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Requested respondents: {number_of_respondents}, threshold: {settings.MAX_RESPONDENTS_FOR_SYNC}")
    logger.info(f"Using background processing: {number_of_respondents > settings.MAX_RESPONDENTS_FOR_SYNC}")
    
    if number_of_respondents > settings.MAX_RESPONDENTS_FOR_SYNC:
        # Use async background processing for large studies
        from app.services.background_task_service import background_task_service
        
        # Create study first if needed
        study_row = _ensure_study_exists(payload, db, current_user)
        
        # Log whether this is a new study or existing study
        if payload.study_id:
            logger.info(f"Regenerating tasks for existing study {payload.study_id}")
        else:
            logger.info(f"Creating new study for task generation")
        
        # Create background job
        job_id = background_task_service.create_job(
            study_id=str(study_row.id),
            user_id=str(current_user.id),
            payload=payload.model_dump()
        )
        
        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Created background job {job_id} for {number_of_respondents} respondents")
        
        # Start the job asynchronously using a background thread
        import threading
        import asyncio
        
        def run_background_job():
            """Run the background job in a separate thread with its own event loop"""
            import logging
            logger = logging.getLogger(__name__)
            
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Use a fresh DB session in this background thread
            from app.db.session import SessionLocal
            db_bg = SessionLocal()
            try:
                logger.info(f"Starting background job {job_id}")
                # Start the async job and then await the returned task to completion
                task = loop.run_until_complete(background_task_service.start_job(job_id, db_bg))
                logger.info(f"Background job {job_id} started, awaiting completion...")
                loop.run_until_complete(task)
                logger.info(f"Background job {job_id} completed successfully")
            except Exception as e:
                logger.error(f"Background job {job_id} failed: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Update job status to failed if there's an error
                job = background_task_service.get_job_status(job_id)
                if job:
                    from app.services.background_task_service import JobStatus
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                    job.message = f"Background job failed: {str(e)}"
                    job.completed_at = datetime.utcnow()
            finally:
                try:
                    db_bg.close()
                except Exception:
                    pass
                loop.close()
        
        # Start the background job in a daemon thread
        thread = threading.Thread(target=run_background_job, daemon=True)
        thread.start()
        
        # Return only job metadata; frontend will poll for status/results
        logger.info(f"Returning background job response for job {job_id}")
        return GenerateTasksResult(
            tasks={},
            metadata={
                "job_id": job_id,
                "status": "started",
                "message": f"Task generation started in background for {number_of_respondents} respondents",
                "study_id": str(study_row.id)
            }
        )
    
    # Use synchronous processing for small studies (existing behavior)
    # Upsert draft study if requested (create if no study_id provided, else load to update)
    from sqlalchemy import select
    from app.models.study_model import Study as StudyModel
    study_row = None
    if payload.study_id is not None:
        study_row = db.scalars(
            select(StudyModel).where(StudyModel.id == payload.study_id, StudyModel.creator_id == current_user.id)
        ).first()
        if not study_row:
            raise HTTPException(status_code=404, detail="Study not found or access denied.")
    else:
        # Create a new draft study if minimal required fields provided
        if not payload.title or not payload.background or not payload.language or not payload.main_question or not payload.orientation_text:
            raise HTTPException(status_code=400, detail="Missing required study fields for on-the-fly creation")
        study_row = study_service.create_study(
            db=db,
            creator_id=current_user.id,
            payload=StudyCreate(
                title=payload.title,
                background=payload.background,
                language=payload.language,
                main_question=payload.main_question,
                orientation_text=payload.orientation_text,
                study_type=payload.study_type,
                background_image_url=payload.background_image_url,
                rating_scale=payload.rating_scale,
                audience_segmentation=payload.audience_segmentation,
                categories=payload.categories,
                elements=payload.elements,
                study_layers=payload.study_layers,
                classification_questions=payload.classification_questions,
            ),
            base_url_for_share=settings.BASE_URL,
        )

    if payload.study_type == 'grid':
        if not payload.elements or len(payload.elements) == 0:
            raise HTTPException(status_code=400, detail="Grid study requires elements")
        if not payload.categories or len(payload.categories) == 0:
            raise HTTPException(status_code=400, detail="Grid study requires categories")
        
        num_elements = len(payload.elements)
        elements = []
        # Use StudyElement dataclass shape lightly: only content is used in generator for grid
        for idx, e in enumerate(payload.elements):
            class _Tmp:
                content: str = e.content
            elements.append(_Tmp())
        
        # Build categories_data from payload for v2 functions
        categories_data = []
        for cat in payload.categories:
            cat_elements = [e for e in payload.elements if e.category_id == cat.category_id]
            categories_data.append({
                "category_name": cat.name,
                "elements": [
                    {
                        "element_id": str(el.element_id),  # Convert UUID to string
                        "name": el.name,
                        "content": el.content,
                        "alt_text": el.alt_text or el.name,
                        "element_type": el.element_type,
                    }
                    for el in cat_elements
                ]
            })

        # Feasibility preflight: try planning and locking T quickly; if it fails, raise 400 with guidance
        try:
            from app.services.task_generation_core import plan_T_E_auto, preflight_lock_T, MIN_ACTIVE_PER_ROW, GRID_MAX_ACTIVE
            # Convert to category_info: {CategoryName: [CategoryName_1, ...]}
            category_info: Dict[str, List[str]] = {}
            for c in categories_data:
                cname = c["category_name"]
                category_info[cname] = [f"{cname}_{j+1}" for j in range(len(c["elements"]))]

            if any(len(v) == 0 for v in category_info.values()):
                raise HTTPException(status_code=400, detail="Each category must have at least 1 element")

            mode = "grid"
            max_active_per_row = min(GRID_MAX_ACTIVE, len(category_info))
            T, E, A_map, avg_k, A_min_used = plan_T_E_auto(category_info, mode, max_active_per_row)
            # Attempt to lock T (cheap attempts). If it raises, convert to HTTP 400 with advice
            try:
                _T_locked, _A_map_locked = preflight_lock_T(T, category_info, E, A_min_used, mode, max_active_per_row)
            except RuntimeError as e:
                # Build a simple advisory message for the frontend
                # Heuristic: suggest adding at least 1 more element to any category with only 1
                thin_cats = [cn for cn, arr in category_info.items() if len(arr) < 2]
                advice = "; ".join([
                    "Add more elements per category (≥2 recommended)",
                    "Reduce tasks per respondent (T)",
                    "Relax absence ratio"
                ])
                if thin_cats:
                    advice = f"Categories with very few elements: {', '.join(thin_cats)}. " + advice
                raise HTTPException(status_code=400, detail=f"Task generation not feasible with current elements/categories. {advice}")
        except HTTPException:
            raise
        except Exception:
            # If the quick preflight itself fails for unexpected reasons, proceed to generation as before
            pass
        
        # Use v2 function directly with categories_data
        from app.services.task_generation_core import generate_grid_tasks_v2
        result = generate_grid_tasks_v2(
            categories_data=categories_data,
            number_of_respondents=payload.audience_segmentation.number_of_respondents,
            exposure_tolerance_cv=payload.exposure_tolerance_cv or 1.0,
            seed=payload.seed,
        )
        # Persist tasks to draft study
        study_row.tasks = result.get('tasks', {})
        db.commit()
        return GenerateTasksResult(tasks=result.get('tasks', {}), metadata={**result.get('metadata', {}), "study_id": str(study_row.id)})
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

        # Feasibility preflight for layer mode
        try:
            from app.services.task_generation_core import plan_T_E_auto, preflight_lock_T
            # Convert to category_info: {LayerName: [LayerName_1, ...]}
            category_info: Dict[str, List[str]] = {}
            for L in payload.study_layers:
                count = len(L.images or [])
                category_info[L.name] = [f"{L.name}_{i+1}" for i in range(count)]
            if any(len(v) == 0 for v in category_info.values()):
                raise HTTPException(status_code=400, detail="Each layer must have at least 1 image")
            mode = "layout"
            max_active_per_row = len(category_info)
            T, E, A_map, avg_k, A_min_used = plan_T_E_auto(category_info, mode, max_active_per_row)
            try:
                _T_locked, _A_map_locked = preflight_lock_T(T, category_info, E, A_min_used, mode, max_active_per_row)
            except RuntimeError as e:
                thin_layers = [cn for cn, arr in category_info.items() if len(arr) < 2]
                advice = "; ".join([
                    "Add more images per layer (≥2 recommended)",
                    "Reduce tasks per respondent (T)",
                    "Relax absence ratio"
                ])
                if thin_layers:
                    advice = f"Layers with very few images: {', '.join(thin_layers)}. " + advice
                raise HTTPException(status_code=400, detail=f"Task generation not feasible with current layers/images. {advice}")
        except HTTPException:
            raise
        except Exception:
            # On unexpected preflight errors, continue to generation
            pass
        result = generate_layer_tasks(
            layers=layers,
            number_of_respondents=payload.audience_segmentation.number_of_respondents,
            exposure_tolerance_pct=payload.exposure_tolerance_pct or 2.0,
            seed=payload.seed,
        )
        # Persist tasks to draft study
        study_row.tasks = result.get('tasks', {})
        db.commit()
        return GenerateTasksResult(tasks=result.get('tasks', {}), metadata={**result.get('metadata', {}), "study_id": str(study_row.id)})
    else:
        raise HTTPException(status_code=400, detail="Unsupported study_type")


@router.put("/{study_id}/launch", response_model=StudyOut)
def update_and_launch_study_endpoint(
    study_id: UUID,
    payload: StudyUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Fast path: update basic study details (if provided) and set study to active in one call.
    Does not regenerate tasks; intended for non-task-affecting edits (e.g., title, text).
    """
    # Apply updates if any
    if any([
        payload.title is not None,
        payload.background is not None,
        payload.language is not None,
        payload.main_question is not None,
        payload.orientation_text is not None,
        payload.background_image_url is not None,
        payload.rating_scale is not None,
        payload.classification_questions is not None,
    ]):
        study_service.update_study(
            db=db,
            study_id=study_id,
            owner_id=current_user.id,
            payload=payload,
        )

    # Allow launching without task validation

    # Set status to active (also updates share_url and timestamps)
    study = study_service.change_status(
        db=db,
        study_id=study_id,
        owner_id=current_user.id,
        new_status='active',
    )
    return study


@router.get("/generate-tasks/status/{job_id}")
def get_task_generation_status(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Get the status of a task generation job"""
    from app.services.background_task_service import background_task_service
    
    job = background_task_service.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check if user owns this job
    if job.user_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "job_id": job.job_id,
        "study_id": job.study_id,
        "status": job.status.value,
        "progress": job.progress,
        "message": job.message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error": job.error,
        "result": job.result
    }


@router.post("/generate-tasks/cancel/{job_id}")
def cancel_task_generation(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Cancel a running task generation job"""
    from app.services.background_task_service import background_task_service
    
    job = background_task_service.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check if user owns this job
    if job.user_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    success = background_task_service.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    
    return {"message": "Job cancelled successfully"}


@router.get("/generate-tasks/jobs")
def get_user_task_generation_jobs(
    current_user: User = Depends(get_current_active_user),
):
    """Get all task generation jobs for the current user"""
    from app.services.background_task_service import background_task_service
    
    jobs = background_task_service.get_user_jobs(str(current_user.id))
    
    return [
        {
            "job_id": job.job_id,
            "study_id": job.study_id,
            "status": job.status.value,
            "progress": job.progress,
            "message": job.message,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error": job.error
        }
        for job in jobs
    ]


@router.get("/generate-tasks/result/{job_id}")
def get_task_generation_result(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get the full generated tasks once the background job is complete"""
    from app.services.background_task_service import background_task_service
    
    job = background_task_service.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check if user owns this job
    if job.user_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Check if job is completed
    from app.services.background_task_service import JobStatus
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed yet. Current status: {job.status.value}"
        )
    
    # Get the study with the generated tasks
    study = study_service.get_study(db=db, study_id=job.study_id, owner_id=current_user.id)
    
    return {
        "job_id": job.job_id,
        "study_id": job.study_id,
        "status": job.status.value,
        "tasks": study.tasks,
        "metadata": {
            "total_respondents": len(study.tasks) if study.tasks else 0,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "message": "Task generation completed successfully"
        }
    }