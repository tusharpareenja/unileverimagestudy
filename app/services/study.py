# app/services/study.py
from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime

from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, func, desc, and_, or_
from fastapi import HTTPException, status
import logging

from app.models.study_model import Study, StudyElement, StudyLayer, LayerImage, StudyClassificationQuestion, StudyCategory, StudyMember
from app.models.response_model import StudyResponse
from app.schemas.study_schema import (
    StudyCreate, StudyUpdate, StudyOut, StudyListItem,
    StudyStatus, StudyType, RegenerateTasksResponse, ValidateTasksResponse,
    StudyPublicMinimal
)
from app.services.task_generation_adapter import generate_grid_tasks, generate_layer_tasks
from app.services.cloudinary_service import upload_base64, delete_public_id
from app.core.config import settings

logger = logging.getLogger(__name__)

# If you have an adapter wrapping your existing IPED task generation algorithms,
# import it here. The adapter should provide two callables with these signatures:
# - generate_grid_tasks(num_elements, tasks_per_consumer, number_of_respondents, exposure_tolerance_cv, seed, elements) -> dict
# - generate_layer_tasks(layers, number_of_respondents, exposure_tolerance_pct, seed) -> dict
#
# from app.services.task_generation_adapter import generate_grid_tasks, generate_layer_tasks

# ---------- Helpers ----------

def _ensure_study_type_constraints(payload: StudyCreate) -> None:
    if payload.study_type == 'grid':
        if not payload.elements or len(payload.elements) < 1:
            raise HTTPException(status_code=400, detail="Grid study requires non-empty elements.")
    elif payload.study_type == 'layer':
        if not payload.study_layers or len(payload.study_layers) < 1:
            raise HTTPException(status_code=400, detail="Layer study requires at least one layer with images.")
    elif payload.study_type == 'text':
        if not payload.elements or len(payload.elements) < 1:
            raise HTTPException(status_code=400, detail="Text study requires non-empty elements (statements).")
    elif payload.study_type == 'hybrid':
        if not payload.elements or len(payload.elements) < 1:
            raise HTTPException(status_code=400, detail="Hybrid study requires non-empty elements.")
        if not payload.phase_order or len(payload.phase_order) < 1:
            raise HTTPException(status_code=400, detail="Hybrid study requires phase_order (e.g., ['grid', 'text'] or ['mix']).")
    else:
        raise HTTPException(status_code=400, detail="Unsupported study_type. Must be 'grid', 'layer', 'text', or 'hybrid'.")

def _validate_rating_scale(rating_scale: Dict[str, Any]) -> None:
    logger.info(f"Validating rating scale: {rating_scale}")
    
    required = ['min_value', 'max_value', 'min_label', 'max_label']
    for k in required:
        if k not in rating_scale:
            logger.error(f"rating_scale missing '{k}'")
            raise HTTPException(status_code=400, detail=f"rating_scale missing '{k}'")
    
    if rating_scale['max_value'] not in [5, 7, 9]:
        logger.error(f"rating_scale.max_value {rating_scale['max_value']} not in [5, 7, 9]")
        raise HTTPException(status_code=400, detail="rating_scale.max_value must be one of 5, 7, 9.")
    
    if not (1 <= int(rating_scale['min_value']) <= 9):
        logger.error(f"rating_scale.min_value {rating_scale['min_value']} not in range 1-9")
        raise HTTPException(status_code=400, detail="rating_scale.min_value must be between 1 and 9 inclusive.")
    
    if int(rating_scale['min_value']) > int(rating_scale['max_value']):
        logger.error(f"rating_scale.min_value {rating_scale['min_value']} > max_value {rating_scale['max_value']}")
        raise HTTPException(status_code=400, detail="rating_scale.min_value cannot exceed max_value.")
    
    logger.info("Rating scale validation passed")

def _generate_share_token() -> str:
    return uuid4().hex

def _build_share_url(base_url: Optional[str], study_id: str) -> str:
    # Use provided base_url or fall back to settings
    url = base_url or settings.BASE_URL
    return f"{url.rstrip('/')}/participate/{study_id}"


def _maybe_upload_data_url_to_cloudinary(url_or_data: Optional[str]) -> tuple[str, Optional[str]]:
    """If value is a data URL, upload to Cloudinary and return (secure_url, public_id).
    Otherwise return (original_value, None).
    """
    if not url_or_data:
        return "", None
    val = url_or_data.strip()
    if val.startswith("data:image"):
        secure_url, public_id = upload_base64(val, folder="studies")
        return secure_url, public_id
    return val, None

def _load_owned_study(db: Session, study_id: UUID, owner_id: UUID, for_update: bool = False) -> Study:
    stmt = (
        select(Study)
        .options(
            selectinload(Study.categories),
            selectinload(Study.elements),
            selectinload(Study.layers).selectinload(StudyLayer.images),
            selectinload(Study.classification_questions),
        )
        .where(Study.id == study_id, Study.creator_id == owner_id)
    )
    if for_update:
        stmt = stmt.with_for_update()
    study = db.scalars(stmt).first()

    if not study:
        # Check if the user is a member (Editor or Viewer)
        stmt_member = (
            select(Study)
            .join(StudyMember, StudyMember.study_id == Study.id)
            .options(
                selectinload(Study.categories),
                selectinload(Study.elements),
                selectinload(Study.layers).selectinload(StudyLayer.images),
                selectinload(Study.classification_questions),
            )
            .where(Study.id == study_id, StudyMember.user_id == owner_id)
        )
        study = db.scalars(stmt_member).first()
        
    if not study:
        raise HTTPException(status_code=404, detail="Study not found or access denied.")
    
    # If for_update is True, we only allow Creator or Admin/Editor role
    if for_update:
        if study.creator_id != owner_id:
            # Check role
            stmt_role = select(StudyMember.role).where(StudyMember.study_id == study_id, StudyMember.user_id == owner_id)
            role = db.scalar(stmt_role)
            if role not in ('admin', 'editor'):
                raise HTTPException(status_code=403, detail="You do not have permission to edit this study.")
                
    return study

def _load_owned_study_minimal(db: Session, study_id: UUID, owner_id: UUID, for_update: bool = False) -> Study:
    """Optimized study loading for operations that don't need related data (like launch)."""
    stmt = (
        select(Study)
        .where(Study.id == study_id, Study.creator_id == owner_id)
    )
    if for_update:
        stmt = stmt.with_for_update()
    study = db.scalars(stmt).first()

    if not study:
        # Check if user is a member
        stmt_member = (
            select(Study)
            .join(StudyMember, StudyMember.study_id == Study.id)
            .where(Study.id == study_id, StudyMember.user_id == owner_id)
        )
        study = db.scalars(stmt_member).first()

    if not study:
        raise HTTPException(status_code=404, detail="Study not found or access denied.")
    return study

def _load_owned_study_launch_only(db: Session, study_id: UUID, owner_id: UUID) -> Study:
    """Ultra-minimal loading for launch-only operations - only loads essential fields."""
    stmt = (
        select(Study.id, Study.title, Study.status, Study.launched_at, Study.creator_id)
        .where(Study.id == study_id, Study.creator_id == owner_id)
        .with_for_update()
    )
    result = db.execute(stmt).first()
    
    if not result:
        # Check if user is a member
        stmt_member = (
            select(Study.id, Study.title, Study.status, Study.launched_at, Study.creator_id)
            .join(StudyMember, StudyMember.study_id == Study.id)
            .where(Study.id == study_id, StudyMember.user_id == owner_id)
            .with_for_update()
        )
        result = db.execute(stmt_member).first()

    if not result:
        raise HTTPException(status_code=404, detail="Study not found or access denied.")
    
    # Create minimal study object with only needed fields
    study = Study()
    study.id = result.id
    study.title = result.title
    study.status = result.status
    study.launched_at = result.launched_at
    study.creator_id = result.creator_id
    return study

# ---------- CRUD ----------

def create_study(
    db: Session,
    creator_id: UUID,
    payload: StudyCreate,
    *,
    base_url_for_share: Optional[str] = None
) -> Study:
    _ensure_study_type_constraints(payload)
    _validate_rating_scale(payload.rating_scale.model_dump())

    share_token = _generate_share_token()
    
    # Persist aspect_ratio inside audience_segmentation JSON to avoid migrations
    audience_segmentation_json = payload.audience_segmentation.model_dump(exclude_none=True)
    # Prefer top-level aspect_ratio; else accept inside segmentation
    ar_top = getattr(payload, 'aspect_ratio', None)
    ar_in_seg = audience_segmentation_json.get('aspect_ratio') if isinstance(audience_segmentation_json, dict) else None
    try:
        if ar_top:
            audience_segmentation_json['aspect_ratio'] = ar_top
        elif ar_in_seg:
            # already present; keep as-is
            pass
    except Exception:
        pass

    study = Study(
        id=uuid4(),
        title=payload.title,
        background=payload.background,
        language=payload.language,
        main_question=payload.main_question,
        orientation_text=payload.orientation_text,
        study_type=payload.study_type,  # enum compatible
        background_image_url=payload.background_image_url,
        rating_scale=payload.rating_scale.model_dump(),
        audience_segmentation=audience_segmentation_json,
        tasks=None,
        creator_id=creator_id,
        status='draft',
        share_token=share_token,
        share_url=None,  # Will be set after study.id is available
        total_responses=0,
        completed_responses=0,
        abandoned_responses=0,
        phase_order=payload.phase_order,
        toggle_shuffle=payload.toggle_shuffle,
        last_step=payload.last_step or 1,
    )
    db.add(study)
    # UUID is already assigned above; avoid early flush to reduce DB round-trips
    study.share_url = _build_share_url(base_url_for_share, str(study.id))
    # Ensure parent row exists before inserting FK children in bulk
    db.flush()

    # Children (optimize by avoiding intermediate flushes and using bulk saves)
    with db.no_autoflush:
        if payload.study_type in ('grid', 'text', 'hybrid') and payload.elements:
            # Create categories first if provided
            category_map = {}  # category_id -> category_uuid
            if payload.categories:
                new_categories: List[StudyCategory] = []
                seen_category_ids = set()
                for cat in payload.categories:
                    if cat.category_id in seen_category_ids:
                        raise HTTPException(status_code=409, detail=f"Duplicate category_id: {cat.category_id}")
                    seen_category_ids.add(cat.category_id)
                    category_uuid = uuid4()
                    category_map[cat.category_id] = category_uuid
                    new_categories.append(StudyCategory(
                        id=category_uuid,
                        study_id=study.id,
                        category_id=cat.category_id,  # Now using UUID directly
                        name=cat.name,
                        order=cat.order,
                        phase_type=cat.phase_type
                    ))
                if new_categories:
                    db.bulk_save_objects(new_categories)
                    db.flush()  # Ensure categories exist before linking elements
            
            # Ensure element_id uniqueness within the study
            seen_ids = set()
            new_elements: List[StudyElement] = []
            for elem in payload.elements:
                if elem.element_id in seen_ids:
                    raise HTTPException(status_code=409, detail=f"Duplicate element_id: {elem.element_id}")
                seen_ids.add(elem.element_id)
                
                # Validate category_id exists
                if elem.category_id not in category_map:
                    raise HTTPException(status_code=400, detail=f"Category {elem.category_id} not found in provided categories")
                
                content_url, public_id = _maybe_upload_data_url_to_cloudinary(elem.content)
                new_elements.append(StudyElement(
                    id=uuid4(),
                    study_id=study.id,
                    category_id=category_map[elem.category_id],
                    element_id=elem.element_id,
                    name=elem.name,
                    description=elem.description,
                    element_type=elem.element_type,
                    content=content_url or elem.content,
                    alt_text=elem.alt_text,
                    cloudinary_public_id=public_id
                ))
            if new_elements:
                db.bulk_save_objects(new_elements)

        if payload.study_type == 'layer' and payload.study_layers:
            # Each layer with images
            seen_layer_ids = set()
            new_layers: List[StudyLayer] = []
            for layer in payload.study_layers:
                if layer.layer_id in seen_layer_ids:
                    raise HTTPException(status_code=409, detail=f"Duplicate layer_id: {layer.layer_id}")
                seen_layer_ids.add(layer.layer_id)
                new_layers.append(StudyLayer(
                    id=uuid4(),
                    study_id=study.id,
                    layer_id=layer.layer_id,
                    name=layer.name,
                    description=layer.description,
                    layer_type=layer.layer_type,
                    z_index=layer.z_index,
                    order=layer.order,
                    transform=layer.transform.model_dump() if getattr(layer, 'transform', None) else None
                ))
            if new_layers:
                db.bulk_save_objects(new_layers)
                # Persist layers before inserting their images to satisfy FKs
                db.flush()

            # Build images for all layers without per-layer flushes (ids are preassigned)
            new_images: List[LayerImage] = []
            for layer_row, layer in zip(new_layers, payload.study_layers or []):
                seen_image_ids = set()
                for img in layer.images or []:
                    if img.image_id in seen_image_ids:
                        raise HTTPException(status_code=409, detail=f"Duplicate image_id in layer {layer.layer_id}: {img.image_id}")
                    seen_image_ids.add(img.image_id)
                    url, public_id = _maybe_upload_data_url_to_cloudinary(img.url)
                    new_images.append(LayerImage(
                        id=uuid4(),
                        layer_id=layer_row.id,
                        image_id=img.image_id,
                        name=img.name,
                        url=url or img.url,
                        alt_text=img.alt_text,
                        order=img.order,
                        cloudinary_public_id=public_id,
                        config=img.config
                    ))
            if new_images:
                db.bulk_save_objects(new_images)

    # Handle classification questions (for both grid and layer studies)
    if payload.classification_questions:
        seen_question_ids = set()
        new_questions: List[StudyClassificationQuestion] = []
        for question in payload.classification_questions:
            if question.question_id in seen_question_ids:
                raise HTTPException(status_code=409, detail=f"Duplicate question_id: {question.question_id}")
            seen_question_ids.add(question.question_id)
            answer_options_json = [option.model_dump() for option in question.answer_options] if question.answer_options else None
            new_questions.append(StudyClassificationQuestion(
                id=uuid4(),
                study_id=study.id,
                question_id=question.question_id,
                question_text=question.question_text,
                question_type=question.question_type,
                is_required='Y' if question.is_required else 'N',
                order=question.order,
                answer_options=answer_options_json,
                config=question.config
            ))
        if new_questions:
            db.bulk_save_objects(new_questions)

    db.commit()
    db.refresh(study)
    return study

def create_study_minimal(
    db: Session,
    creator_id: UUID,
    title: str,
    background: str,
    language: str = 'en',
    last_step: int = 1
) -> UUID:
    """
    Ultra-fast study creation with only essential fields.
    Creates a draft study with minimal database operations.
    Returns only the study ID for maximum performance.

    Optimizations:
    - Single database INSERT with no SELECT
    - No ORM object creation overhead
    - No relationship loading
    - Returns UUID directly (no model conversion)
    """
    from sqlalchemy import insert

    share_token = _generate_share_token()
    study_id = uuid4()

    # Use raw SQL insert for maximum speed - bypasses ORM overhead
    stmt = insert(Study).values(
        id=study_id,
        title=title,
        background=background,
        language=language,
        main_question="",  # Will be filled in later
        orientation_text="",  # Will be filled in later
        study_type='grid',  # Default to grid, can be updated later
        rating_scale={
            "min_value": 1,
            "max_value": 5,
            "min_label": "",
            "max_label": ""
        },
        audience_segmentation={},
        creator_id=creator_id,
        status='draft',
        share_token=share_token,
        share_url=_build_share_url(settings.BASE_URL, str(study_id)),
        toggle_shuffle=False,
        last_step=last_step,
    )

    db.execute(stmt)
    db.commit()

    # Return UUID directly - no ORM object needed
    return study_id

def get_study(db: Session, study_id: UUID, owner_id: UUID) -> Study:
    return _load_owned_study(db, study_id, owner_id, for_update=False)

def get_study_exists(db: Session, study_id: UUID, owner_id: UUID) -> bool:
    """Lightweight check if study exists and is owned by user."""
    stmt = select(Study.id).where(Study.id == study_id, Study.creator_id == owner_id)
    return db.scalars(stmt).first() is not None

def check_study_access(db: Session, study_id: UUID, user_id: UUID) -> None:
    """
    Verify if a user has access to a study (Creator or Member).
    Raises HTTPException if access is denied.
    Does NOT load full study data.
    """
    # Check if creator
    stmt_creator = select(Study.creator_id).where(Study.id == study_id)
    creator_id = db.scalar(stmt_creator)
    
    if not creator_id:
        raise HTTPException(status_code=404, detail="Study not found")
        
    if creator_id == user_id:
        return

    # Check if member
    stmt_member = select(StudyMember.id).where(
        StudyMember.study_id == study_id,
        StudyMember.user_id == user_id
    )
    if not db.scalar(stmt_member):
        raise HTTPException(status_code=403, detail="Access denied")

def get_study_public(db: Session, study_id: UUID) -> Optional[Study]:
    """
    Get study information for public access (no authentication required).
    Only returns studies that are active and have a share_token.
    """
    stmt = (
        select(Study)
        .options(
            selectinload(Study.elements),
            selectinload(Study.layers).selectinload(StudyLayer.images),
            selectinload(Study.classification_questions),
        )
        .where(
            Study.id == study_id,
            Study.status == 'active',
            Study.share_token.isnot(None),
            Study.share_token != ''
        )
    )
    return db.scalars(stmt).first()

def get_study_public_minimal(db: Session, study_id: UUID) -> Optional[StudyPublicMinimal]:
    """Lightweight public fetch: only title, type, respondents target."""
    row = db.execute(
        select(
            Study.id,
            Study.title,
            Study.study_type,
            Study.audience_segmentation,
            Study.status,
            Study.share_token,
            Study.orientation_text,
            Study.language,
        ).where(Study.id == study_id)
    ).first()
    if not row:
        return None
    if row.status != 'active' or not row.share_token:
        return None
    respondents_target = 0
    try:
        seg = row.audience_segmentation or {}
        respondents_target = int(seg.get('number_of_respondents') or 0)
    except Exception:
        respondents_target = 0
    
    # Get number of tasks per respondent from study.tasks
    tasks_per_respondent = 0
    try:
        # Load the study to access tasks
        study = db.get(Study, study_id)
        if study and study.tasks:
            if isinstance(study.tasks, dict) and study.tasks:
                # Get tasks for the first respondent to determine tasks per respondent
                first_respondent_key = next(iter(study.tasks.keys()))
                first_respondent_tasks = study.tasks[first_respondent_key]
                if isinstance(first_respondent_tasks, list):
                    tasks_per_respondent = len(first_respondent_tasks)
                else:
                    tasks_per_respondent = 1 if first_respondent_tasks else 0
            elif isinstance(study.tasks, list):
                # If tasks is a flat list, we can't determine per-respondent count
                tasks_per_respondent = 0
    except Exception:
        tasks_per_respondent = 0
    
    return StudyPublicMinimal(
        id=row.id,
        title=row.title,
        study_type=row.study_type,
        respondents_target=respondents_target,
        tasks_per_respondent=tasks_per_respondent,
        status=row.status,
        orientation_text=row.orientation_text,
        language=row.language,
    )

def get_study_public_with_status_check(db: Session, study_id: UUID) -> Dict[str, Any]:
    """Get study with status checking and appropriate messaging."""
    from app.models.user_model import User as UserModel
    row = db.execute(
        select(
            Study.id,
            Study.title,
            Study.study_type,
            Study.audience_segmentation,
            Study.status,
            Study.share_token,
            Study.orientation_text,
            Study.language,
            UserModel.email.label("creator_email")
        ).join(UserModel, UserModel.id == Study.creator_id)
        .where(Study.id == study_id)
    ).first()
    
    if not row:
        return {
            "error": "Study not found",
            "message": "The study you are looking for does not exist."
        }
    
    # Handle different statuses
    if row.status in ['draft', 'paused']:
        return {
            "error": "Study is paused",
            "message": "This study is currently paused. Please ask the owner to activate the study.",
            "status": row.status
        }
    
    if row.status == 'completed':
        return {
            "error": "Study is completed",
            "message": "This study has been completed.",
            "status": row.status
        }
    
    if row.status != 'active' or not row.share_token:
        return {
            "error": "Study not accessible",
            "message": "This study is not currently accessible.",
            "status": row.status
        }
    
    # Study is active and accessible
    respondents_target = 0
    try:
        seg = row.audience_segmentation or {}
        respondents_target = int(seg.get('number_of_respondents') or 0)
    except Exception:
        respondents_target = 0
    
    # Get number of tasks per respondent from study.tasks
    tasks_per_respondent = 0
    try:
        # Load the study to access tasks
        study = db.get(Study, study_id)
        if study and study.tasks:
            if isinstance(study.tasks, dict) and study.tasks:
                # Get tasks for the first respondent to determine tasks per respondent
                first_respondent_key = next(iter(study.tasks.keys()))
                first_respondent_tasks = study.tasks[first_respondent_key]
                if isinstance(first_respondent_tasks, list):
                    tasks_per_respondent = len(first_respondent_tasks)
                else:
                    tasks_per_respondent = 1 if first_respondent_tasks else 0
            elif isinstance(study.tasks, list):
                # If tasks is a flat list, we can't determine per-respondent count
                tasks_per_respondent = 0
    except Exception:
        tasks_per_respondent = 0
    
    return {
        "id": str(row.id),
        "title": row.title,
        "study_type": row.study_type,
        "respondents_target": respondents_target,
        "tasks_per_respondent": tasks_per_respondent,
        "status": row.status,
        "orientation_text": row.orientation_text,
        "language": row.language,
        "creator_email": row.creator_email,
    }

def get_study_share_details(db: Session, study_id: UUID) -> Optional[Dict[str, Any]]:
    """Lightweight fetch of share URL and basic status info for a study."""
    row = db.execute(
        select(
            Study.id,
            Study.title,
            Study.study_type,
            Study.status,
            Study.share_url,
        ).where(Study.id == study_id)
    ).first()
    if not row:
        return None
    return {
        "id": row.id,
        "title": row.title,
        "study_type": row.study_type,
        "status": row.status,
        "share_url": row.share_url,
    }

def list_studies(
    db: Session,
    owner_id: UUID,
    *,
    status_filter: Optional[StudyStatus] = None,
    page: int = 1,
    per_page: int = 10
) -> Tuple[List[Study], int]:
    # Build correlated subqueries for counters so results always match analytics
    total_subq = (
        select(func.count(StudyResponse.id))
        .where(StudyResponse.study_id == Study.id)
        .correlate(Study)
        .scalar_subquery()
    )
    completed_subq = (
        select(func.count(StudyResponse.id))
        .where(and_(StudyResponse.study_id == Study.id, StudyResponse.is_completed == True))
        .correlate(Study)
        .scalar_subquery()
    )
    abandoned_subq = (
        select(func.count(StudyResponse.id))
        .where(and_(StudyResponse.study_id == Study.id, StudyResponse.is_abandoned == True))
        .correlate(Study)
        .scalar_subquery()
    )
    avg_duration_subq = (
        select(func.avg(StudyResponse.total_study_duration))
        .where(and_(StudyResponse.study_id == Study.id, StudyResponse.is_completed == True))
        .correlate(Study)
        .scalar_subquery()
    )

    base_stmt = (
        select(
            Study,
            total_subq.label("total_responses_calc"),
            completed_subq.label("completed_responses_calc"),
            abandoned_subq.label("abandoned_responses_calc"),
            avg_duration_subq.label("avg_duration_calc"),
        )
        .outerjoin(StudyMember, and_(StudyMember.study_id == Study.id, StudyMember.user_id == owner_id))
        .where(
            or_(
                Study.creator_id == owner_id,
                StudyMember.user_id == owner_id
            )
        )
    )
    count_stmt = (
        select(func.count(Study.id.distinct()))
        .select_from(Study)
        .outerjoin(StudyMember, and_(StudyMember.study_id == Study.id, StudyMember.user_id == owner_id))
        .where(
            or_(
                Study.creator_id == owner_id,
                StudyMember.user_id == owner_id
            )
        )
    )
    if status_filter:
        base_stmt = base_stmt.where(Study.status == status_filter)
        count_stmt = count_stmt.where(Study.status == status_filter)

    total = db.scalar(count_stmt) or 0
    rows = db.execute(
        base_stmt
        .order_by(desc(Study.created_at))
        .offset((page - 1) * per_page)
        .limit(per_page)
    ).all()
    # Return list of rows containing (Study, derived counters)
    return rows, int(total)

def update_study(
    db: Session,
    study_id: UUID,
    owner_id: UUID,
    payload: StudyUpdate
) -> Study:
    logger.info(f"update_study called for study {study_id} by user {owner_id}")
    logger.info(f"Payload: {payload.model_dump(exclude_none=True)}")

    study = _load_owned_study(db, study_id, owner_id, for_update=True)
    # Allow editing even when active, but only for the original creator once it's active.
    # Editors can edit draft studies, but not active/launched studies.
    if study.status == 'active' and study.creator_id != owner_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="the study has already been launched by the admin please go to home page"
        )

    # Handle status change if provided (align with change_status rules)
    if payload.status is not None:
        logger.info(f"Updating status to: {payload.status}")
        if payload.status not in ['draft', 'active', 'paused', 'completed']:
            logger.error(f"Invalid status: {payload.status}")
            raise HTTPException(status_code=400, detail="Invalid status.")
        # transitions timestamps
        if payload.status == 'active' and (study.launched_at is None):
            study.launched_at = datetime.utcnow()
        if payload.status == 'completed':
            study.completed_at = datetime.utcnow()
        study.status = payload.status
        # Allow status changes without task validation

    # Apply scalar updates
    if payload.title is not None:
        study.title = payload.title
    if payload.background is not None:
        study.background = payload.background
    if payload.language is not None:
        study.language = payload.language
    if payload.main_question is not None:
        study.main_question = payload.main_question
    if payload.orientation_text is not None:
        study.orientation_text = payload.orientation_text
    if payload.study_type is not None:
        study.study_type = payload.study_type
    if payload.background_image_url is not None:
        study.background_image_url = payload.background_image_url
    if payload.last_step is not None:
        # Only update last_step if the new value is greater (forward progress only)
        current_step = getattr(study, 'last_step', 1) or 1
        if payload.last_step > current_step:
            study.last_step = payload.last_step
    if payload.rating_scale is not None:
        _validate_rating_scale(payload.rating_scale.model_dump())
        study.rating_scale = payload.rating_scale.model_dump()
    if payload.audience_segmentation is not None:
        # Merge with existing to preserve aspect_ratio when omitted
        existing_seg = study.audience_segmentation or {}
        incoming_seg = payload.audience_segmentation.model_dump(exclude_none=True)
        merged_seg = {**existing_seg, **incoming_seg}
        # If top-level aspect_ratio is provided, override; otherwise preserve existing
        ar_top = getattr(payload, 'aspect_ratio', None)
        if ar_top:
            merged_seg['aspect_ratio'] = ar_top
        elif 'aspect_ratio' not in incoming_seg and 'aspect_ratio' in existing_seg:
            # Preserve existing aspect_ratio if not in incoming payload
            merged_seg['aspect_ratio'] = existing_seg['aspect_ratio']
        study.audience_segmentation = merged_seg

    if payload.phase_order is not None:
        study.phase_order = payload.phase_order

    if payload.toggle_shuffle is not None:
        study.toggle_shuffle = payload.toggle_shuffle

    # Replace children collections if provided
    if payload.elements is not None:
        # Only valid for grid, text, and hybrid
        if study.study_type not in ('grid', 'text', 'hybrid'):
            raise HTTPException(status_code=400, detail="elements can only be set for grid, text, and hybrid studies.")

        # Handle categories first - they must be provided when updating elements
        category_map = {}  # category_id -> category_uuid
        if hasattr(payload, 'categories') and payload.categories:
            # Delete old categories (cascade will handle elements)
            db.query(StudyCategory).filter(StudyCategory.study_id == study.id).delete()
            db.flush()

            # Create new categories
            seen_category_ids = set()
            for cat in payload.categories:
                if cat.category_id in seen_category_ids:
                    raise HTTPException(status_code=409, detail=f"Duplicate category_id: {cat.category_id}")
                seen_category_ids.add(cat.category_id)
                category_uuid = uuid4()
                category_map[cat.category_id] = category_uuid
                db.add(StudyCategory(
                    id=category_uuid,
                    study_id=study.id,
                    category_id=cat.category_id,
                    name=cat.name,
                    order=cat.order,
                    phase_type=cat.phase_type
                ))
            db.flush()
        else:
            # If categories not provided in payload, load existing categories
            existing_categories = db.scalars(
                select(StudyCategory).where(StudyCategory.study_id == study.id)
            ).all()
            for cat in existing_categories:
                category_map[cat.category_id] = cat.id

        # Clear and re-add elements
        # Delete old assets best-effort
        old_elements_seq = db.scalars(
            select(StudyElement).where(StudyElement.study_id == study.id)
        ).all()
        old_elements: List[StudyElement] = list(old_elements_seq)
        for oe in old_elements:
            if oe.cloudinary_public_id is not None and oe.cloudinary_public_id != "":
                delete_public_id(str(oe.cloudinary_public_id))
        db.query(StudyElement).filter(StudyElement.study_id == study.id).delete()
        db.flush()

        seen_ids = set()
        for elem in payload.elements:
            if elem.element_id in seen_ids:
                raise HTTPException(status_code=409, detail=f"Duplicate element_id: {elem.element_id}")
            seen_ids.add(elem.element_id)

            # Validate category_id exists
            if elem.category_id not in category_map:
                raise HTTPException(status_code=400, detail=f"Category {elem.category_id} not found. Please provide categories when updating elements.")

            content_url, public_id = _maybe_upload_data_url_to_cloudinary(elem.content)
            db.add(StudyElement(
                id=uuid4(),
                study_id=study.id,
                category_id=category_map[elem.category_id],
                element_id=elem.element_id,
                name=elem.name,
                description=elem.description,
                element_type=elem.element_type,
                content=content_url or elem.content,
                alt_text=elem.alt_text,
                cloudinary_public_id=public_id
            ))

    if payload.study_layers is not None:
        # Only valid for layer
        if study.study_type != 'layer':
            raise HTTPException(status_code=400, detail="study_layers can only be set for layer studies.")
        # Clear and re-add (delete assets best-effort)
        old_layer_images_seq = db.scalars(
            select(LayerImage).join(StudyLayer, StudyLayer.id == LayerImage.layer_id).where(StudyLayer.study_id == study.id)
        ).all()
        old_layer_images: List[LayerImage] = list(old_layer_images_seq)
        for oi in old_layer_images:
            if oi.cloudinary_public_id is not None and oi.cloudinary_public_id != "":
                delete_public_id(str(oi.cloudinary_public_id))
        subq = db.query(StudyLayer.id).filter(StudyLayer.study_id == study.id).subquery()
        db.query(LayerImage).filter(LayerImage.layer_id.in_(select(subq.c.id))).delete(synchronize_session=False)
        db.query(StudyLayer).filter(StudyLayer.study_id == study.id).delete()
        db.flush()

        seen_layer_ids = set()
        for layer in payload.study_layers:
            if layer.layer_id in seen_layer_ids:
                raise HTTPException(status_code=409, detail=f"Duplicate layer_id: {layer.layer_id}")
            seen_layer_ids.add(layer.layer_id)

            layer_row = StudyLayer(
                id=uuid4(),
                study_id=study.id,
                layer_id=layer.layer_id,
                name=layer.name,
                description=layer.description,
                layer_type=layer.layer_type,
                z_index=layer.z_index,
                order=layer.order,
                transform=layer.transform.model_dump() if getattr(layer, 'transform', None) else None
            )
            db.add(layer_row)
            db.flush()

            seen_image_ids = set()
            for img in layer.images or []:
                if img.image_id in seen_image_ids:
                    raise HTTPException(status_code=409, detail=f"Duplicate image_id in layer {layer.layer_id}: {img.image_id}")
                seen_image_ids.add(img.image_id)
                db.add(LayerImage(
                    id=uuid4(),
                    layer_id=layer_row.id,
                    image_id=img.image_id,
                    name=img.name,
                    url=img.url,
                    alt_text=img.alt_text,
                    order=img.order,
                    config=img.config
                ))

    # Handle classification questions updates
    if payload.classification_questions is not None:
        # Clear existing classification questions
        db.query(StudyClassificationQuestion).filter(StudyClassificationQuestion.study_id == study.id).delete()
        db.flush()
        
        # Add new classification questions
        seen_question_ids = set()
        for question in payload.classification_questions:
            if question.question_id in seen_question_ids:
                raise HTTPException(status_code=409, detail=f"Duplicate question_id: {question.question_id}")
            seen_question_ids.add(question.question_id)
            
            # Convert answer options to JSONB format
            answer_options_json = None
            if question.answer_options:
                answer_options_json = [option.model_dump() for option in question.answer_options]
            
            db.add(StudyClassificationQuestion(
                id=uuid4(),
                study_id=study.id,
                question_id=question.question_id,
                question_text=question.question_text,
                question_type=question.question_type,
                is_required='Y' if question.is_required else 'N',
                order=question.order,
                answer_options=answer_options_json,
                config=question.config
            ))

    db.commit()
    db.refresh(study)
    return study

def update_study_fast(
    db: Session,
    study_id: UUID,
    owner_id: UUID,
    payload: StudyUpdate
) -> Study:
    """Optimized version of update_study that uses minimal loading for simple updates."""
    logger.info(f"update_study_fast called for study {study_id} by user {owner_id}")
    logger.info(f"Payload: {payload.model_dump(exclude_none=True)}")
    
    # Check if we need full loading (for classification_questions updates)
    needs_full_loading = payload.classification_questions is not None
    
    if needs_full_loading:
        study = _load_owned_study(db, study_id, owner_id, for_update=True)
    else:
        study = _load_owned_study_minimal(db, study_id, owner_id, for_update=True)
    
    # Prevent members from editing active/launched studies
    if study.status == 'active' and study.creator_id != owner_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="the study has already been launched by the admin please go to home page"
        )

    # Handle status change if provided (align with change_status rules)
    if payload.status is not None:
        logger.info(f"Updating status to: {payload.status}")
        if payload.status not in ['draft', 'active', 'paused', 'completed']:
            logger.error(f"Invalid status: {payload.status}")
            raise HTTPException(status_code=400, detail="Invalid status.")
        # transitions timestamps
        if payload.status == 'active' and (study.launched_at is None):
            study.launched_at = datetime.utcnow()
        if payload.status == 'completed':
            study.completed_at = datetime.utcnow()
        study.status = payload.status
        # Allow status changes without task validation

    # Apply scalar updates
    if payload.title is not None:
        study.title = payload.title
    if payload.background is not None:
        study.background = payload.background
    if payload.language is not None:
        study.language = payload.language
    if payload.main_question is not None:
        study.main_question = payload.main_question
    if payload.orientation_text is not None:
        study.orientation_text = payload.orientation_text
    if payload.study_type is not None:
        study.study_type = payload.study_type
    if payload.background_image_url is not None:
        study.background_image_url = payload.background_image_url
    if payload.last_step is not None:
        # Only update last_step if the new value is greater (forward progress only)
        current_step = getattr(study, 'last_step', 1) or 1
        if payload.last_step > current_step:
            study.last_step = payload.last_step
    if payload.rating_scale is not None:
        _validate_rating_scale(payload.rating_scale.model_dump())
        study.rating_scale = payload.rating_scale.model_dump()
    if payload.audience_segmentation is not None:
        # Merge with existing to avoid dropping keys like aspect_ratio when omitted
        existing_seg = study.audience_segmentation or {}
        incoming_seg = payload.audience_segmentation.model_dump(exclude_none=True)
        merged_seg = {**existing_seg, **incoming_seg}
        # If top-level aspect_ratio is provided, override; otherwise preserve existing
        ar_top = getattr(payload, 'aspect_ratio', None)
        if ar_top:
            merged_seg['aspect_ratio'] = ar_top
        study.audience_segmentation = merged_seg

    if payload.phase_order is not None:
        study.phase_order = payload.phase_order

    # For fast updates, we only handle simple scalar fields
    # Complex operations (elements, layers, classification_questions) fall back to full loading
    if payload.elements is not None or payload.study_layers is not None:
        # Fall back to full update_study for complex operations
        return update_study(db, study_id, owner_id, payload)

    # Handle classification questions updates (requires full loading)
    if payload.classification_questions is not None:
        # Fall back to full update_study for classification questions
        return update_study(db, study_id, owner_id, payload)

    db.commit()
    db.refresh(study)
    return study

def update_and_launch_study_fast(
    db: Session,
    study_id: UUID,
    owner_id: UUID,
    payload: StudyUpdate
) -> Study:
    """Ultra-optimized function that combines update and launch in a single transaction."""
    # Load study with minimal data for maximum performance
    study = _load_owned_study_minimal(db, study_id, owner_id, for_update=True)
    
    # Prevent members from editing active/launched studies
    if study.status == 'active' and study.creator_id != owner_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="the study has already been launched by the admin please go to home page"
        )
    
    # Apply scalar updates
    if payload.title is not None:
        study.title = payload.title
    if payload.background is not None:
        study.background = payload.background
    if payload.language is not None:
        study.language = payload.language
    if payload.main_question is not None:
        study.main_question = payload.main_question
    if payload.orientation_text is not None:
        study.orientation_text = payload.orientation_text
    if payload.background_image_url is not None:
        study.background_image_url = payload.background_image_url
    if payload.rating_scale is not None:
        _validate_rating_scale(payload.rating_scale.model_dump())
        study.rating_scale = payload.rating_scale.model_dump()
    if payload.audience_segmentation is not None:
        # Merge to preserve existing keys (e.g., aspect_ratio) when omitted in update
        existing_seg = study.audience_segmentation or {}
        incoming_seg = payload.audience_segmentation.model_dump(exclude_none=True)
        merged_seg = {**existing_seg, **incoming_seg}
        # If top-level aspect_ratio is provided, override; otherwise preserve existing
        ar_top = getattr(payload, 'aspect_ratio', None)
        if ar_top:
            merged_seg['aspect_ratio'] = ar_top
        elif 'aspect_ratio' not in incoming_seg and 'aspect_ratio' in existing_seg:
            # Explicitly preserve existing aspect_ratio if not in incoming payload
            merged_seg['aspect_ratio'] = existing_seg['aspect_ratio']
        study.audience_segmentation = merged_seg
    
    if payload.phase_order is not None:
        study.phase_order = payload.phase_order

    # Handle complex updates that require full loading
    if payload.elements is not None or payload.study_layers is not None or payload.classification_questions is not None:
        # Fall back to full update_study for complex operations
        updated_study = update_study(db, study_id, owner_id, payload)
        # Then launch the study
        return change_status_fast(db, study_id, owner_id, 'active')

    # Set status to active (launch the study)
    if study.launched_at is None:
        study.launched_at = datetime.utcnow()
        # Update share_url when study is launched (becomes active)
        study.share_url = _build_share_url(None, str(study.id))
    
    study.status = 'active'
    
    # Single commit for both update and launch
    db.commit()
    db.refresh(study)
    return study

def launch_study_ultra_fast(
    db: Session,
    study_id: UUID,
    owner_id: UUID
) -> Study:
    """Ultra-fast launch-only function - no updates, just launch."""
    # Use direct SQL UPDATE for maximum speed
    from sqlalchemy import update
    from datetime import datetime
    
    now = datetime.utcnow()
    stmt = (
        update(Study)
        .where(Study.id == study_id, Study.creator_id == owner_id)
        .values(
            status='active',
            launched_at=now,
            share_url=_build_share_url(None, str(study_id)),
            updated_at=now
        )
        .returning(Study.id, Study.title, Study.status, Study.launched_at, Study.share_url, Study.updated_at)
    )
    
    result = db.execute(stmt).first()
    if not result:
        raise HTTPException(status_code=404, detail="Study not found or access denied.")
    
    # Create minimal study object for response
    study = Study()
    study.id = result.id
    study.title = result.title
    study.status = result.status
    study.launched_at = result.launched_at
    study.share_url = result.share_url
    study.updated_at = result.updated_at
    
    db.commit()
    return study

def delete_study(db: Session, study_id: UUID, owner_id: UUID) -> None:
    study = _load_owned_study(db, study_id, owner_id, for_update=True)
    if study.status == 'active':
        raise HTTPException(status_code=400, detail="Cannot delete active studies.")
    db.delete(study)
    db.commit()

def change_status(
    db: Session,
    study_id: UUID,
    owner_id: UUID,
    new_status: StudyStatus
) -> Study:
    study = _load_owned_study(db, study_id, owner_id, for_update=True)

    if new_status not in ['draft', 'active', 'paused', 'completed']:
        raise HTTPException(status_code=400, detail="Invalid status.")

    # transitions timestamps and share_url update
    if new_status == 'active' and (study.launched_at is None):
        study.launched_at = datetime.utcnow()
        # Update share_url when study is launched (becomes active)
        study.share_url = _build_share_url(None, str(study.id))
    if new_status == 'completed':
        study.completed_at = datetime.utcnow()

    study.status = new_status
    db.commit()
    db.refresh(study)
    return study

def change_status_fast(
    db: Session,
    study_id: UUID,
    owner_id: UUID,
    new_status: StudyStatus
) -> Study:
    """Optimized version of change_status that avoids loading related data."""
    study = _load_owned_study_minimal(db, study_id, owner_id, for_update=True)

    if new_status not in ['draft', 'active', 'paused', 'completed']:
        raise HTTPException(status_code=400, detail="Invalid status.")

    # transitions timestamps and share_url update
    if new_status == 'active' and (study.launched_at is None):
        study.launched_at = datetime.utcnow()
        # Update share_url when study is launched (becomes active)
        study.share_url = _build_share_url(None, str(study.id))
    if new_status == 'completed':
        study.completed_at = datetime.utcnow()

    study.status = new_status
    db.commit()
    db.refresh(study)
    return study

# ---------- Tasks (Regenerate / Validate) ----------

def regenerate_tasks(
    db: Session,
    study_id: UUID,
    owner_id: UUID,
    *,
    generator: Optional[Dict[str, Any]] = None
) -> RegenerateTasksResponse:
    # Optimized study loading - only load what we need
    study = db.scalars(
        select(Study)
        .where(Study.id == study_id, Study.creator_id == owner_id)
        .with_for_update()
    ).first()

    if not study:
        # Check if user is a member
        stmt_member = (
            select(Study)
            .join(StudyMember, StudyMember.study_id == Study.id)
            .where(Study.id == study_id, StudyMember.user_id == owner_id)
            .with_for_update()
        )
        study = db.scalars(stmt_member).first()

    if not study:
        raise HTTPException(status_code=404, detail="Study not found or access denied.")
    
    # Allow task regeneration for all study statuses
    # Note: Regenerating tasks for active studies may affect ongoing participants

    # Default to adapter-backed generators if none provided
    generator = generator or {"grid": generate_grid_tasks, "layer": generate_layer_tasks}

    if 'grid' not in generator or 'layer' not in generator:
        raise HTTPException(status_code=500, detail="Task generator not configured.")

    # Cache audience values to avoid repeated dict lookups
    audience = study.audience_segmentation or {}
    number_of_respondents = audience.get('number_of_respondents')
    computed_total: int = 0
    if study.study_type in ('grid', 'text'):
        # Cache constants to avoid repeated lookups
        exposure_tolerance_cv = 1.0
        seed = None

        # Optimized element loading - load all elements but with better query
        elements = db.scalars(
            select(StudyElement)
            .where(StudyElement.study_id == study.id)
            .order_by(StudyElement.element_id.asc())
        ).all()
        if not elements:
            raise HTTPException(status_code=400, detail="Elements missing for grid study.")
        num_elements = len(elements)

        logger.debug(
            "Regenerate grid: num_elements=%s respondents=%s elements_loaded=%s",
            num_elements, number_of_respondents, len(elements)
        )

        result = generator['grid'](
            num_elements=num_elements,
            tasks_per_consumer=None,
            number_of_respondents=number_of_respondents,
            exposure_tolerance_cv=exposure_tolerance_cv,
            seed=seed,
            elements=elements,
            db=db,
            study_id=str(study.id)
        )
        study.tasks = result.get('tasks', {})
        
        # Optimized total calculation - cache metadata
        meta = result.get('metadata', {})
        tpc = meta.get('tasks_per_consumer')
        try:
            computed = int(tpc or 0) * int(number_of_respondents or 0)
        except Exception:
            computed = 0
        # Fallback: count from generated tasks if available
        if not computed and isinstance(study.tasks, dict):
            try:
                computed = sum(len(v or []) for v in study.tasks.values())
            except Exception:
                computed = 0
        logger.debug("Regenerate grid computed total_tasks=%s (tpc=%s, nresp=%s, tasks_keys=%s)",
                     computed, tpc, number_of_respondents, len(study.tasks) if isinstance(study.tasks, dict) else 'n/a')
        computed_total = computed

    elif study.study_type == 'layer':
        # Cache constants to avoid repeated lookups
        exposure_tolerance_pct = 2.0
        seed = None

        # Optimized layer loading - load layers with images efficiently
        layers = db.scalars(
            select(StudyLayer)
            .options(selectinload(StudyLayer.images))
            .where(StudyLayer.study_id == study.id)
            .order_by(StudyLayer.order.asc())
        ).all()
        if not layers:
            raise HTTPException(status_code=400, detail="Layer configuration missing.")

        logger.debug(
            "Regenerate layer: respondents=%s exposure_pct=%s layers_loaded=%s",
            number_of_respondents, exposure_tolerance_pct, len(layers)
        )
        result = generator['layer'](
            layers=layers,
            number_of_respondents=number_of_respondents,
            exposure_tolerance_pct=exposure_tolerance_pct,
            seed=seed
        )
        # Attach optional background image URL to metadata for client rendering
        try:
            if isinstance(result, dict):
                meta_tmp = result.get('metadata') or {}
                if isinstance(meta_tmp, dict):
                    meta_tmp['background_image_url'] = getattr(study, 'background_image_url', None)
                    result['metadata'] = meta_tmp
        except Exception:
            pass
        study.tasks = result.get('tasks', {})
        
        # Optimized total calculation - cache metadata
        meta = result.get('metadata', {})
        tpc = meta.get('tasks_per_consumer')
        try:
            computed = int(tpc or 0) * int(number_of_respondents or 0)
        except Exception:
            computed = 0
        if not computed and isinstance(study.tasks, dict):
            try:
                computed = sum(len(v or []) for v in study.tasks.values())
            except Exception:
                computed = 0
        logger.debug("Regenerate layer computed total_tasks=%s (tpc=%s, nresp=%s, tasks_keys=%s)",
                     computed, tpc, number_of_respondents, len(study.tasks) if isinstance(study.tasks, dict) else 'n/a')
        computed_total = computed

    elif study.study_type == 'hybrid':
        phase_order = study.phase_order or []
        if not phase_order:
            raise HTTPException(status_code=400, detail="Hybrid study requires phase_order.")
        
        combined_tasks = {}
        combined_metadata = {"phases": []}
        
        # Load all categories and elements once
        categories = db.scalars(
            select(StudyCategory).where(StudyCategory.study_id == study.id)
        ).all()
        elements = db.scalars(
            select(StudyElement).where(StudyElement.study_id == study.id)
        ).all()

        is_mix = "mix" in phase_order
        # If mix, we need to gather tasks from ALL present phase types, ignoring the specific order list (except to find types)
        # Actually, "mix" might be the ONLY item, or mixed with others? 
        # Requirement: "if user choses mix than task will be generated randomly means there will be no bound grid will appear first or tet any one one can ppear at any time"
        # So if "mix" is selected, we assume ALL configured phases should be generated and mixed.
        # We can detect relevant phases by looking at the categories' phase_type.
        
        target_phases = []
        if is_mix:
            # unique phase types from categories
            target_phases = list(set(c.phase_type for c in categories if c.phase_type))
            # If for some reason categories don't have phase_type appropriately set, fallback? 
            # Assuming StudyCategory.phase_type is reliable.
        else:
            target_phases = phase_order

        # Temporary storage for mix shuffling
        tasks_per_respondent_mix = {} # resp_id -> list of tasks

        for phase_type in target_phases:
            # Generate tasks for this phase
            phase_categories = [c for c in categories if c.phase_type == phase_type]
            if not phase_categories:
                continue
                
            cat_data = []
            for cat in phase_categories:
                cat_elements = [e for e in elements if e.category_id == cat.id]
                cat_data.append({
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
            
            from app.services.task_generation_core import generate_grid_tasks_v2
            phase_result = generate_grid_tasks_v2(
                categories_data=cat_data,
                number_of_respondents=number_of_respondents,
                exposure_tolerance_cv=exposure_tolerance_cv,
                seed=seed
            )
            
            phase_tasks = phase_result.get('tasks', {})
            
            if is_mix:
                # Accumulate for shuffling later
                for resp_id, t_list in phase_tasks.items():
                    if resp_id not in tasks_per_respondent_mix:
                        tasks_per_respondent_mix[resp_id] = []
                    
                    for t in t_list:
                        # Tag with phase type but don't set final index yet
                        t['phase_type'] = phase_type
                        tasks_per_respondent_mix[resp_id].append(t)
            else:
                # Sequential appending
                for resp_id, t_list in phase_tasks.items():
                    if resp_id not in combined_tasks:
                        combined_tasks[resp_id] = []
                    
                    offset = len(combined_tasks[resp_id])
                    for t in t_list:
                        t['task_index'] = int(t.get('task_index', 0)) + offset
                        t['phase_type'] = phase_type
                    combined_tasks[resp_id].extend(t_list)
            
            combined_metadata['phases'].append({
                "phase_type": phase_type,
                "metadata": phase_result.get('metadata', {})
            })
            
        if is_mix:
            import random
            # Shuffle and assign indices
            combined_tasks = {}
            for resp_id, t_list in tasks_per_respondent_mix.items():
                random.shuffle(t_list)
                # Re-index
                for idx, t in enumerate(t_list):
                    t['task_index'] = idx
                combined_tasks[resp_id] = t_list

        study.tasks = combined_tasks
        # Simple count for computed_total
        computed_total = sum(len(v) for v in combined_tasks.values()) if combined_tasks else 0

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported study type: {study.study_type}. Must be 'grid', 'layer', 'text', or 'hybrid'.")

    # Optimized commit and response generation
    try:
        db.commit()
        # Cache logging to avoid repeated string operations
        log_msg = f"Regenerate completed: study_id={study_id} total_tasks={computed_total}"
        logger.info(log_msg)
        print(f"[regenerate] study_id={study_id} total_tasks={computed_total}")
    except Exception as e:
        logger.error(f"Failed to commit regenerate tasks: {e}")
        raise HTTPException(status_code=500, detail="Failed to save regenerated tasks")
    
    # Cache metadata to avoid repeated lookups
    response_meta = locals().get('meta', {}) if 'meta' in locals() else {}
    return RegenerateTasksResponse(
        success=True,
        message="Task matrix regenerated successfully",
        total_tasks=int(computed_total or 0),
        metadata=response_meta if isinstance(response_meta, dict) else {}
    )

def validate_tasks(
    db: Session,
    study_id: UUID,
    owner_id: UUID
) -> ValidateTasksResponse:
    study = _load_owned_study(db, study_id, owner_id, for_update=False)
    if not study.tasks:
        return ValidateTasksResponse(validation_passed=False, issues=["No tasks generated"], totals={})

    issues: List[str] = []
    # Simplified validation without IPED parameters
    totals: Dict[str, Any] = {}

    if study.study_type in ('grid', 'text'):
        # Basic structural checks
        for respondent, task_list in study.tasks.items():
            for task in task_list:
                elements_shown = task.get("elements_shown", {})
                active_count = sum(1 for k, v in elements_shown.items() if not k.endswith('_content') and int(v or 0) == 1)

        totals['respondents'] = len(study.tasks)

    elif study.study_type == 'layer':
        # Check structure presence only (semantic validation of layer/image mapping can be extended)
        for respondent, task_list in study.tasks.items():
            for task in task_list:
                if "elements_shown" not in task and "elements_shown_content" not in task:
                    issues.append(f"Task {task.get('task_id')} missing elements_shown/_content.")

        totals['respondents'] = len(study.tasks)

    else:
        issues.append(f"Unsupported study type {study.study_type}")

    return ValidateTasksResponse(validation_passed=len(issues) == 0, issues=issues, totals=totals)


def get_study_basic_details(db: Session, study_id: UUID, owner_id: UUID) -> Optional[Dict[str, Any]]:
    """
    Get basic study details for authenticated users.
    Returns core study information without heavy data like tasks, elements, or layers.
    """
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    
    # Optimized query to get basic study details with classification questions
    study = db.scalars(
        select(Study)
        .options(selectinload(Study.classification_questions))
        .where(Study.id == study_id, Study.creator_id == owner_id)
    ).first()
    
    if not study:
        return None
    
    # Build study config from audience segmentation
    study_config = {}
    if study.audience_segmentation:
        study_config.update(study.audience_segmentation)
    
    return {
        "id": study.id,
        "title": study.title,
        "status": study.status,
        "study_type": study.study_type,
        "created_at": study.created_at,
        "background": study.background,
        "main_question": study.main_question,
        "orientation_text": study.orientation_text,
        "rating_scale": study.rating_scale,
        "study_config": study_config,
        "classification_questions": study.classification_questions,
        "toggle_shuffle": study.toggle_shuffle
    }


def get_study_basic_details_public(db: Session, study_id: UUID) -> Optional[Dict[str, Any]]:
    """
    Get basic study details for public access (no authentication required).
    Ultra-fast query with minimal data loading using raw SQL for maximum speed.
    Includes element_count: number of images (grid/layer) or statements (text).
    """
    from sqlalchemy import text
    
    # Ultra-fast raw SQL query - only essential fields
    query = text("""
        SELECT id, title, status, study_type, created_at, background, 
               main_question, orientation_text, rating_scale, iped_parameters, language, toggle_shuffle
        FROM studies 
        WHERE id = :study_id
    """)
    
    result = db.execute(query, {"study_id": study_id}).first()
    
    if not result:
        return None
    
    # Build study config from audience segmentation (stored as iped_parameters)
    study_config = {}
    if result.iped_parameters:
        study_config.update(result.iped_parameters)
    
    # Load classification questions separately with a fast query
    classification_query = text("""
        SELECT id, question_id, question_text, question_type, is_required, 
               "order", answer_options, config
        FROM study_classification_questions 
        WHERE study_id = :study_id
        ORDER BY "order"
    """)
    
    classification_results = db.execute(classification_query, {"study_id": study_id}).fetchall()
    classification_questions = []
    for row in classification_results:
        classification_questions.append({
            "id": row.id,
            "question_id": row.question_id,
            "question_text": row.question_text,
            "question_type": row.question_type,
            "is_required": row.is_required,
            "order": row.order,
            "answer_options": row.answer_options,
            "config": row.config
        })
    
    # Count elements based on study type
    element_count = 0
    study_type = result.study_type
    
    if study_type == 'grid' or study_type == 'text':
        # For grid and text studies, count elements (images for grid, statements for text)
        element_count_query = text("""
            SELECT COUNT(*) as count
            FROM study_elements
            WHERE study_id = :study_id
        """)
        count_result = db.execute(element_count_query, {"study_id": study_id}).first()
        element_count = count_result.count if count_result else 0
        
    elif study_type == 'layer':
        # For layer studies, count total images across all layers
        layer_image_count_query = text("""
            SELECT COUNT(li.id) as count
            FROM layer_images li
            INNER JOIN study_layers sl ON li.layer_id = sl.id
            WHERE sl.study_id = :study_id
        """)
        count_result = db.execute(layer_image_count_query, {"study_id": study_id}).first()
        element_count = count_result.count if count_result else 0
    
    return {
        "id": result.id,
        "title": result.title,
        "status": result.status,
        "study_type": result.study_type,
        "created_at": result.created_at,
        "background": result.background,
        "main_question": result.main_question,
        "orientation_text": result.orientation_text,
        "rating_scale": result.rating_scale,
        "language": result.language,
        "study_config": study_config,
        "classification_questions": classification_questions,
        "element_count": element_count,
        "toggle_shuffle": result.toggle_shuffle
    }