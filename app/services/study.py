# app/services/study.py
from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime

from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, func, desc, and_
from fastapi import HTTPException, status
import logging

from app.models.study_model import Study, StudyElement, StudyLayer, LayerImage, StudyClassificationQuestion, StudyCategory
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
        # num_elements inferred from elements; tasks_per_consumer will auto-pick
    elif payload.study_type == 'layer':
        if not payload.study_layers or len(payload.study_layers) < 1:
            raise HTTPException(status_code=400, detail="Layer study requires at least one layer with images.")
    else:
        raise HTTPException(status_code=400, detail="Unsupported study_type. Must be 'grid' or 'layer'.")

def _validate_rating_scale(rating_scale: Dict[str, Any]) -> None:
    required = ['min_value', 'max_value', 'min_label', 'max_label']
    for k in required:
        if k not in rating_scale:
            raise HTTPException(status_code=400, detail=f"rating_scale missing '{k}'")
    if rating_scale['max_value'] not in [5, 7, 9]:
        raise HTTPException(status_code=400, detail="rating_scale.max_value must be one of 5, 7, 9.")
    if not (1 <= int(rating_scale['min_value']) <= 9):
        raise HTTPException(status_code=400, detail="rating_scale.min_value must be between 1 and 9 inclusive.")
    if int(rating_scale['min_value']) > int(rating_scale['max_value']):
        raise HTTPException(status_code=400, detail="rating_scale.min_value cannot exceed max_value.")

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
        raise HTTPException(status_code=404, detail="Study not found or access denied.")
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
        audience_segmentation=payload.audience_segmentation.model_dump(exclude_none=True),
        tasks=None,
        creator_id=creator_id,
        status='draft',
        share_token=share_token,
        share_url=None,  # Will be set after study.id is available
        total_responses=0,
        completed_responses=0,
        abandoned_responses=0,
    )
    db.add(study)
    # UUID is already assigned above; avoid early flush to reduce DB round-trips
    study.share_url = _build_share_url(base_url_for_share, str(study.id))
    # Ensure parent row exists before inserting FK children in bulk
    db.flush()

    # Children (optimize by avoiding intermediate flushes and using bulk saves)
    with db.no_autoflush:
        if payload.study_type == 'grid' and payload.elements:
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
                        order=cat.order
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
                    z_index=layer.z_index,
                    order=layer.order
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
                        cloudinary_public_id=public_id
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

def get_study(db: Session, study_id: UUID, owner_id: UUID) -> Study:
    return _load_owned_study(db, study_id, owner_id, for_update=False)

def get_study_exists(db: Session, study_id: UUID, owner_id: UUID) -> bool:
    """Lightweight check if study exists and is owned by user."""
    stmt = select(Study.id).where(Study.id == study_id, Study.creator_id == owner_id)
    return db.scalars(stmt).first() is not None

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
    return StudyPublicMinimal(
        id=row.id,
        title=row.title,
        study_type=row.study_type,
        respondents_target=respondents_target,
        status=row.status,
        orientation_text=row.orientation_text,
        language=row.language,
    )

def get_study_public_with_status_check(db: Session, study_id: UUID) -> Dict[str, Any]:
    """Get study with status checking and appropriate messaging."""
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
    
    return {
        "id": str(row.id),
        "title": row.title,
        "study_type": row.study_type,
        "respondents_target": respondents_target,
        "status": row.status,
        "orientation_text": row.orientation_text,
        "language": row.language,
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
        .where(Study.creator_id == owner_id)
    )
    count_stmt = select(func.count()).select_from(Study).where(Study.creator_id == owner_id)
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
    study = _load_owned_study(db, study_id, owner_id, for_update=True)
    # Allow editing even when active (per new requirement)

    # Handle status change if provided (align with change_status rules)
    if payload.status is not None:
        if payload.status not in ['draft', 'active', 'paused', 'completed']:
            raise HTTPException(status_code=400, detail="Invalid status.")
        # transitions timestamps
        if payload.status == 'active' and (study.launched_at is None):
            study.launched_at = datetime.utcnow()
        if payload.status == 'completed':
            study.completed_at = datetime.utcnow()
        study.status = payload.status
        # Generate tasks on activation only if tasks are missing
        if payload.status == 'active' and not study.tasks:
            try:
                from app.services.study import regenerate_tasks
                regenerate_tasks(
                    db=db,
                    study_id=study.id,
                    owner_id=owner_id,
                )
            except Exception as ex:
                # Non-fatal: keep study active even if task generation fails
                logger.error("Task generation on activation failed: %s", ex)

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
        study.audience_segmentation = payload.audience_segmentation.model_dump(exclude_none=True)

    # Replace children collections if provided
    if payload.elements is not None:
        # Only valid for grid
        if study.study_type != 'grid':
            raise HTTPException(status_code=400, detail="elements can only be set for grid studies.")
        # Clear and re-add
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
            content_url, public_id = _maybe_upload_data_url_to_cloudinary(elem.content)
            db.add(StudyElement(
                id=uuid4(),
                study_id=study.id,
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
                z_index=layer.z_index,
                order=layer.order
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
                    order=img.order
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
    if study.study_type == 'grid':
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
        # total stored only in response; audience_segmentation remains unchanged

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported study type: {study.study_type}")

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

    if study.study_type == 'grid':
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
        "classification_questions": study.classification_questions
    }


def get_study_basic_details_public(db: Session, study_id: UUID) -> Optional[Dict[str, Any]]:
    """
    Get basic study details for public access (no authentication required).
    Ultra-fast query with minimal data loading using raw SQL for maximum speed.
    """
    from sqlalchemy import text
    
    # Ultra-fast raw SQL query - only essential fields
    query = text("""
        SELECT id, title, status, study_type, created_at, background, 
               main_question, orientation_text, rating_scale, iped_parameters, language
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
        "classification_questions": classification_questions
    }