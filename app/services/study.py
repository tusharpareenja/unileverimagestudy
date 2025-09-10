# app/services/study.py
from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime

from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, func, desc
from fastapi import HTTPException, status
import logging

from app.models.study_model import Study, StudyElement, StudyLayer, LayerImage
from app.schemas.study_schema import (
    StudyCreate, StudyUpdate, StudyOut, StudyListItem,
    StudyStatus, StudyType, RegenerateTasksResponse, ValidateTasksResponse
)
from app.services.task_generation_adapter import generate_grid_tasks, generate_layer_tasks
from app.services.cloudinary_service import upload_base64, delete_public_id

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

def _build_share_url(base_url: Optional[str], share_token: str) -> Optional[str]:
    if not base_url:
        return None
    return f"{base_url.rstrip('/')}/participate/{share_token}"


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
    share_url = _build_share_url(base_url_for_share, share_token)

    study = Study(
        id=uuid4(),
        title=payload.title,
        background=payload.background,
        language=payload.language,
        main_question=payload.main_question,
        orientation_text=payload.orientation_text,
        study_type=payload.study_type,  # enum compatible
        rating_scale=payload.rating_scale.model_dump(),
        audience_segmentation=payload.audience_segmentation.model_dump(exclude_none=True),
        tasks=None,
        creator_id=creator_id,
        status='draft',
        share_token=share_token,
        share_url=share_url,
        total_responses=0,
        completed_responses=0,
        abandoned_responses=0,
    )
    db.add(study)
    db.flush()  # ensure study.id is available

    # Children
    if payload.study_type == 'grid' and payload.elements:
        # Ensure element_id uniqueness within the study
        seen_ids = set()
        order_counter = 1
        for elem in payload.elements:
            if elem.element_id in seen_ids:
                raise HTTPException(status_code=409, detail=f"Duplicate element_id: {elem.element_id}")
            seen_ids.add(elem.element_id)
            # If frontend provided secure_url/public_id in content as JSON, you could parse it.
            # For now, if content is a data URL, upload and replace.
            content_url, public_id = _maybe_upload_data_url_to_cloudinary(elem.content)
            db.add(StudyElement(
                id=uuid4(),
                study_id=study.id,
                element_id=elem.element_id,
                name=elem.name,
                description=elem.description,
                element_type=elem.element_type,  # enum
                content=content_url or elem.content,
                alt_text=elem.alt_text,
                cloudinary_public_id=public_id
            ))
            order_counter += 1

    if payload.study_type == 'layer' and payload.study_layers:
        # Each layer with images
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
                url, public_id = _maybe_upload_data_url_to_cloudinary(img.url)
                db.add(LayerImage(
                    id=uuid4(),
                    layer_id=layer_row.id,
                    image_id=img.image_id,
                    name=img.name,
                    url=url or img.url,
                    alt_text=img.alt_text,
                    order=img.order,
                    cloudinary_public_id=public_id
                ))

    db.commit()
    db.refresh(study)
    return study

def get_study(db: Session, study_id: UUID, owner_id: UUID) -> Study:
    return _load_owned_study(db, study_id, owner_id, for_update=False)

def list_studies(
    db: Session,
    owner_id: UUID,
    *,
    status_filter: Optional[StudyStatus] = None,
    page: int = 1,
    per_page: int = 10
) -> Tuple[List[Study], int]:
    base_stmt = select(Study).where(Study.creator_id == owner_id)
    count_stmt = select(func.count()).select_from(Study).where(Study.creator_id == owner_id)
    if status_filter:
        base_stmt = base_stmt.where(Study.status == status_filter)
        count_stmt = count_stmt.where(Study.status == status_filter)

    total = db.scalar(count_stmt) or 0
    seq_items = db.scalars(
        base_stmt
        .order_by(desc(Study.created_at))
        .offset((page - 1) * per_page)
        .limit(per_page)
    ).all()
    items: List[Study] = list(seq_items)
    return items, int(total)

def update_study(
    db: Session,
    study_id: UUID,
    owner_id: UUID,
    payload: StudyUpdate
) -> Study:
    study = _load_owned_study(db, study_id, owner_id, for_update=True)
    if study.status == 'active':
        raise HTTPException(status_code=400, detail="Cannot edit active studies.")

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

    # transitions timestamps
    if new_status == 'active' and (study.launched_at is None):
        study.launched_at = datetime.utcnow()
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
    study = _load_owned_study(db, study_id, owner_id, for_update=True)
    if study.status == 'active':
        raise HTTPException(status_code=400, detail="Cannot regenerate tasks for active studies.")

    # Default to adapter-backed generators if none provided
    generator = generator or {"grid": generate_grid_tasks, "layer": generate_layer_tasks}

    if 'grid' not in generator or 'layer' not in generator:
        raise HTTPException(status_code=500, detail="Task generator not configured.")

    audience = study.audience_segmentation or {}
    computed_total: int = 0
    if study.study_type == 'grid':
        number_of_respondents = audience.get('number_of_respondents')
        exposure_tolerance_cv = 1.0
        seed = None

        # Load elements (ordered by element_id lexical)
        elements: List[StudyElement] = db.scalars(
            select(StudyElement).where(StudyElement.study_id == study.id).order_by(StudyElement.element_id.asc())
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
            elements=elements
        )
        study.tasks = result.get('tasks', {})
        # Update total_tasks from metadata or computed fallback
        meta = result.get('metadata', {})
        tpc = meta.get('tasks_per_consumer')
        nresp = number_of_respondents
        try:
            computed = int(tpc or 0) * int(nresp or 0)
        except Exception:
            computed = 0
        # Fallback: count from generated tasks if available
        if not computed and isinstance(study.tasks, dict):
            try:
                computed = sum(len(v or []) for v in study.tasks.values())
            except Exception:
                computed = 0
        logger.debug("Regenerate grid computed total_tasks=%s (tpc=%s, nresp=%s, tasks_keys=%s)",
                     computed, tpc, nresp, len(study.tasks) if isinstance(study.tasks, dict) else 'n/a')
        computed_total = computed
        # total stored only in response; audience_segmentation remains unchanged

    elif study.study_type == 'layer':
        number_of_respondents = audience.get('number_of_respondents')
        exposure_tolerance_pct = 2.0
        seed = None

        # Collect layers with images in a simple list for generator
        layers = db.scalars(
            select(StudyLayer).options(selectinload(StudyLayer.images)).where(StudyLayer.study_id == study.id).order_by(StudyLayer.order.asc())
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
        study.tasks = result.get('tasks', {})
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

    db.commit()
    # Ensure something appears in logs even if logging config is minimal
    try:
        logger.info("Regenerate completed: study_id=%s total_tasks=%s", study_id, computed_total)
        print(f"[regenerate] study_id={study_id} total_tasks={computed_total}")
    except Exception:
        pass
    # Attach metadata from last generation if available
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