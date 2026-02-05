# app/api/v1/project.py
from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_active_user
from app.db.session import get_db
from app.models.user_model import User
from app.schemas.project_schema import (
    ProjectCreate, ProjectUpdate, ProjectOut, ProjectListItem,
    ProjectMemberInvite, ProjectMemberOut, ProjectMemberUpdate
)
from app.services import project_service
from app.services.project_member_service import project_member_service

router = APIRouter()


@router.post("/{project_id}/members/invite", response_model=ProjectMemberOut)
def invite_project_member_endpoint(
    project_id: UUID,
    payload: ProjectMemberInvite,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Invite a user to a project by email.
    User will automatically get access to all studies in the project.
    """
    member = project_member_service.invite_member(
        db=db,
        project_id=project_id,
        inviter=current_user,
        payload=payload
    )
    
    # Enrich with user details if available
    member_dict = {
        "id": member.id,
        "project_id": member.project_id,
        "user_id": member.user_id,
        "email": member.invited_email,
        "role": member.role,
        "created_at": member.created_at,
        "updated_at": member.updated_at,
        "name": None,
        "is_registered": False
    }
    
    if hasattr(member, 'user') and member.user:
        member_dict["name"] = member.user.name
        member_dict["is_registered"] = True
    
    return ProjectMemberOut(**member_dict)


@router.get("/{project_id}/members", response_model=List[ProjectMemberOut])
def list_project_members_endpoint(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    List all members of a project, including the project creator.
    """
    # Ensure current user has access to this project (owner or member)
    project_service.get_project(db=db, project_id=project_id, user_id=current_user.id)
    
    members = project_member_service.list_members(db=db, project_id=project_id)
    
    # Enrich members with user details
    enriched_members = []
    for member in members:
        member_dict = {
            "id": member.id,
            "project_id": member.project_id,
            "user_id": member.user_id,
            "email": member.invited_email,
            "role": member.role,
            "created_at": member.created_at,
            "updated_at": member.updated_at,
            "name": None,
            "is_registered": False
        }
        
        if hasattr(member, 'user') and member.user:
            member_dict["name"] = member.user.name
            member_dict["is_registered"] = True
        
        enriched_members.append(member_dict)
    
    return enriched_members


@router.patch("/{project_id}/members/{member_id}", response_model=ProjectMemberOut)
def update_project_member_endpoint(
    project_id: UUID,
    member_id: UUID,
    payload: ProjectMemberUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Update a project member's role.
    Role change will automatically sync to all studies in the project.
    """
    member = project_member_service.update_member_role(
        db=db,
        project_id=project_id,
        member_id=member_id,
        current_user=current_user,
        payload=payload
    )
    
    return ProjectMemberOut(
        id=member.id,
        project_id=member.project_id,
        user_id=member.user_id,
        email=member.invited_email,
        role=member.role,
        created_at=member.created_at,
        updated_at=member.updated_at
    )


@router.delete("/{project_id}/members/{member_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_project_member_endpoint(
    project_id: UUID,
    member_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Remove a member from the project.
    Member will also be removed from all studies in the project.
    """
    project_member_service.remove_member(
        db=db,
        project_id=project_id,
        member_id=member_id,
        current_user=current_user
    )
    return None


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def create_project_endpoint(
    payload: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a new project (requires: name, optional: description).
    Ultra-fast endpoint optimized for <100ms response time.
    """
    project = project_service.create_project(
        db=db,
        creator_id=current_user.id,
        payload=payload
    )
    
    # Return with study count (0 for new projects)
    return ProjectOut(
        id=project.id,
        name=project.name,
        description=project.description,
        creator_id=project.creator_id,
        created_at=project.created_at,
        updated_at=project.updated_at,
        study_count=0
    )


@router.get("", response_model=List[ProjectOut])
def get_projects_endpoint(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Fetch all projects for the authenticated user.
    Optimized for <200ms response time with study counts.
    """
    projects = project_service.get_projects_for_user(
        db=db,
        user_id=current_user.id,
        page=page,
        per_page=per_page
    )
    return projects


@router.get("/{project_id}", response_model=ProjectOut)
def get_project_endpoint(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get a single project by ID.
    """
    from sqlalchemy import select, func
    from app.models.study_model import Study
    from app.models.project_model import Project
    
    project = project_service.get_project(
        db=db,
        project_id=project_id,
        user_id=current_user.id
    )
    
    # Get study count for this project
    study_count = db.scalar(
        select(func.count(Study.id)).where(Study.project_id == project_id)
    ) or 0
    
    return ProjectOut(
        id=project.id,
        name=project.name,
        description=project.description,
        creator_id=project.creator_id,
        created_at=project.created_at,
        updated_at=project.updated_at,
        study_count=int(study_count)
    )


@router.get("/{project_id}/studies")
def get_project_studies_endpoint(
    project_id: UUID,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Fetch all studies for a specific project.
    Optimized for <200ms response time.
    """
    studies = project_service.get_project_studies(
        db=db,
        project_id=project_id,
        user_id=current_user.id,
        page=page,
        per_page=per_page
    )
    return studies


@router.put("/{project_id}", response_model=ProjectOut)
def update_project_endpoint(
    project_id: UUID,
    payload: ProjectUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Update a project.
    """
    from sqlalchemy import select, func
    from app.models.study_model import Study
    
    project = project_service.update_project(
        db=db,
        project_id=project_id,
        user_id=current_user.id,
        payload=payload
    )
    
    # Get study count for this project
    study_count = db.scalar(
        select(func.count(Study.id)).where(Study.project_id == project_id)
    ) or 0
    
    return ProjectOut(
        id=project.id,
        name=project.name,
        description=project.description,
        creator_id=project.creator_id,
        created_at=project.created_at,
        updated_at=project.updated_at,
        study_count=int(study_count)
    )


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project_endpoint(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Delete a project (studies will have their project_id set to NULL).
    """
    project_service.delete_project(
        db=db,
        project_id=project_id,
        user_id=current_user.id
    )
    return None
