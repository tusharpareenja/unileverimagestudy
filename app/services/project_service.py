# app/services/project_service.py
from __future__ import annotations

from typing import List, Optional, Tuple, Any

from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, func, text, or_, and_
from fastapi import HTTPException, status
from uuid import UUID
from datetime import datetime

from app.models.project_model import Project
from app.models.study_model import Study, StudyLayer
from app.models.response_model import StudyResponse
from app.schemas.project_schema import (
    ProjectCreate, ProjectUpdate, ProjectOut, ProjectListItem,
    ValidateProductRequest, ValidateProductResponse,
)


def create_project(
    db: Session,
    creator_id: UUID,
    payload: ProjectCreate
) -> Project:
    """
    Create a new project (ultra-fast, <100ms).
    
    Optimizations:
    - Minimal fields (name, description, creator_id)
    - Single INSERT statement
    - No relationship loading
    - Returns project directly
    """
    project = Project(
        name=payload.name,
        description=payload.description,
        creator_id=creator_id
    )
    
    db.add(project)
    db.commit()
    db.refresh(project)
    
    return project


def get_projects_for_user(
    db: Session,
    user_id: UUID,
    page: int = 1,
    per_page: int = 50
) -> List[ProjectOut]:
    """
    Fetch all projects for a user (owned + shared) with study counts (optimized for <200ms).
    
    Optimizations:
    - Single query with LEFT JOIN and COUNT
    - Includes both owned projects and shared projects
    - No relationship loading (lazy="selectin" disabled for this query)
    - Pagination support
    - Returns list of dicts directly
    """
    from app.models.project_model import ProjectMember
    
    offset = (page - 1) * per_page
    
    from sqlalchemy import case, String, and_
    
    # Optimized query: fetch projects (owned or shared) with study count in one go
    # Use CASE to determine role: 'admin' if creator, else use role from ProjectMember
    query = (
        select(
            Project.id,
            Project.name,
            Project.description,
            Project.creator_id,
            Project.created_at,
            Project.updated_at,
            func.count(Study.id).label('study_count'),
            case(
                (Project.creator_id == user_id, 'admin'),
                else_=func.max(func.cast(ProjectMember.role, String))
            ).label('role')
        )
        .outerjoin(Study, Study.project_id == Project.id)
        .outerjoin(ProjectMember, and_(ProjectMember.project_id == Project.id, ProjectMember.user_id == user_id))
        .where(
            or_(
                Project.creator_id == user_id,
                ProjectMember.user_id == user_id
            )
        )
        .group_by(Project.id, Project.name, Project.description, Project.creator_id, Project.created_at, Project.updated_at)
        .order_by(Project.created_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    
    result = db.execute(query).all()
    
    # Convert to list of ProjectOut
    projects = []
    for row in result:
        projects.append(ProjectOut(
            id=row.id,
            name=row.name,
            description=row.description,
            creator_id=row.creator_id,
            role=row.role or 'viewer', # Fallback if somehow null
            created_at=row.created_at,
            updated_at=row.updated_at,
            study_count=int(row.study_count or 0)
        ))
    
    return projects


def get_project_studies(
    db: Session,
    project_id: UUID,
    user_id: UUID,
    page: int = 1,
    per_page: int = 50
) -> List[dict]:
    """
    Fetch all studies for a specific project (optimized for <200ms).
    Allows access for both project owner and project members.
    Uses live counts from study_responses so total_responses and completed_responses
    are always accurate (not the Study model's cached counters).
    """
    # Verify the project exists and user has access (owner or member)
    get_project(db, project_id, user_id)

    offset = (page - 1) * per_page

    # Correlated subqueries for live response counts (same approach as list_studies)
    total_subq = (
        select(func.count(StudyResponse.id))
        .where(StudyResponse.study_id == Study.id)
        .correlate(Study)
        .scalar_subquery()
    )
    completed_subq = (
        select(func.count(StudyResponse.id))
        .where(and_(
            StudyResponse.study_id == Study.id,
            StudyResponse.is_completed == True,
        ))
        .correlate(Study)
        .scalar_subquery()
    )

    query = (
        select(
            Study.id,
            Study.title,
            Study.study_type,
            Study.status,
            Study.created_at,
            Study.updated_at,
            total_subq.label("total_responses_calc"),
            completed_subq.label("completed_responses_calc"),
        )
        .where(Study.project_id == project_id)
        .order_by(Study.created_at.desc())
        .offset(offset)
        .limit(per_page)
    )

    result = db.execute(query).all()

    studies = []
    for row in result:
        studies.append({
            "id": str(row.id),
            "title": row.title,
            "study_type": row.study_type,
            "status": row.status,
            "created_at": row.created_at.isoformat(),
            "updated_at": row.updated_at.isoformat(),
            "total_responses": int(row.total_responses_calc or 0),
            "completed_responses": int(row.completed_responses_calc or 0),
        })
    return studies


def get_project_studies_for_export(
    db: Session,
    project_id: UUID,
    user_id: UUID,
) -> List[Study]:
    """
    Fetch all studies for a project ordered by product_id (asc, nulls last).
    Verifies project access (owner or member). Returns full Study objects with
    categories, elements, layers.images, classification_questions loaded for building study_data.
    """
    get_project(db, project_id, user_id)
    stmt = (
        select(Study)
        .options(
            selectinload(Study.categories),
            selectinload(Study.elements),
            selectinload(Study.layers).selectinload(StudyLayer.images),
            selectinload(Study.classification_questions),
        )
        .where(Study.project_id == project_id)
        .order_by(Study.product_id.asc().nulls_last())
    )
    return list(db.scalars(stmt).all())


def get_project(
    db: Session,
    project_id: UUID,
    user_id: UUID
) -> Project:
    """
    Get a single project by ID (with access check for owner or member).
    """
    from app.models.project_model import ProjectMember
    
    # Check if user is creator
    project = db.scalar(
        select(Project)
        .where(Project.id == project_id, Project.creator_id == user_id)
    )
    
    if not project:
        # Check if user is a member
        stmt_member = (
            select(Project)
            .join(ProjectMember, ProjectMember.project_id == Project.id)
            .where(Project.id == project_id, ProjectMember.user_id == user_id)
        )
        project = db.scalars(stmt_member).first()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or access denied"
        )
    
    return project


def _key_signature(keys: Optional[List[Any]]) -> Optional[Tuple[Tuple[str, ...], Tuple[float, ...]]]:
    """Build canonical signature (names_sorted, percentages_sorted) for key combination comparison."""
    if not keys:
        return None
    pairs = []
    for k in keys:
        if isinstance(k, dict):
            name = k.get("name")
            if not name:
                continue
            pairs.append((name, float(k.get("percentage", 0))))
        else:
            name = getattr(k, "name", None)
            pct = getattr(k, "percentage", 0)
            if name is not None:
                pairs.append((name, float(pct)))
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[0])
    return (tuple(p[0] for p in pairs), tuple(p[1] for p in pairs))


def validate_product_uniqueness(
    db: Session,
    project_id: UUID,
    user_id: UUID,
    payload: ValidateProductRequest,
) -> ValidateProductResponse:
    """
    Check that product_id and key percentage combination are unique within the project.
    Single lightweight query (id, product_id, product_keys only); response in double-digit ms.
    """
    get_project(db, project_id, user_id)

    product_id_taken = False
    key_combination_taken = False

    check_product_id = payload.product_id is not None and (payload.product_id or "").strip() != ""
    request_sig = _key_signature(payload.product_keys) if payload.product_keys else None
    check_keys = request_sig is not None

    if not check_product_id and not check_keys:
        return ValidateProductResponse(valid=True)

    query = (
        select(Study.id, Study.product_id, Study.product_keys)
        .where(Study.project_id == project_id)
    )
    rows = db.execute(query).all()

    for row in rows:
        if payload.study_id and row.id == payload.study_id:
            continue
        if check_product_id and row.product_id and (row.product_id or "").strip() == (payload.product_id or "").strip():
            product_id_taken = True
        if check_keys and row.product_keys:
            study_sig = _key_signature(row.product_keys)
            if study_sig and study_sig == request_sig:
                key_combination_taken = True
        if product_id_taken and key_combination_taken:
            break

    return ValidateProductResponse(
        valid=not (product_id_taken or key_combination_taken),
        product_id_taken=product_id_taken,
        key_combination_taken=key_combination_taken,
    )


def update_project(
    db: Session,
    project_id: UUID,
    user_id: UUID,
    payload: ProjectUpdate
) -> Project:
    """
    Update a project (owner only).
    """
    project = get_project(db, project_id, user_id)
    
    # Only creator can update project settings
    if project.creator_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the project creator can update project settings"
        )
    
    if payload.name is not None:
        project.name = payload.name
    if payload.description is not None:
        project.description = payload.description
    
    db.commit()
    db.refresh(project)
    
    return project


def delete_project(
    db: Session,
    project_id: UUID,
    user_id: UUID
) -> None:
    """
    Delete a project (owner only).
    """
    project = get_project(db, project_id, user_id)
    
    # Only creator can delete project
    if project.creator_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the project creator can delete the project"
        )
    
    db.delete(project)
    db.commit()
