# app/services/project_service.py
from __future__ import annotations

from typing import List, Optional
from uuid import UUID
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import select, func, text, or_
from fastapi import HTTPException, status

from app.models.project_model import Project
from app.models.study_model import Study
from app.schemas.project_schema import ProjectCreate, ProjectUpdate, ProjectOut, ProjectListItem


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
    """
    # Verify the project exists and user has access (owner or member)
    get_project(db, project_id, user_id)
    
    offset = (page - 1) * per_page
    
    # Optimized query: fetch only essential study fields
    query = (
        select(
            Study.id,
            Study.title,
            Study.study_type,
            Study.status,
            Study.created_at,
            Study.updated_at,
            Study.total_responses,
            Study.completed_responses
        )
        .where(Study.project_id == project_id)
        .order_by(Study.created_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    
    result = db.execute(query).all()
    
    # Convert to list of dicts
    studies = []
    for row in result:
        studies.append({
            'id': str(row.id),
            'title': row.title,
            'study_type': row.study_type,
            'status': row.status,
            'created_at': row.created_at.isoformat(),
            'updated_at': row.updated_at.isoformat(),
            'total_responses': row.total_responses,
            'completed_responses': row.completed_responses
        })
    
    return studies


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
