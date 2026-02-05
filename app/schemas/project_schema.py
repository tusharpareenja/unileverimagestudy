# app/schemas/project_schema.py
from __future__ import annotations
from typing import Optional
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    """Schema for creating a new project"""
    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(None, description="Project description")


class ProjectUpdate(BaseModel):
    """Schema for updating a project"""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(None, description="Project description")


class ProjectOut(BaseModel):
    """Schema for project response"""
    id: UUID
    name: str
    description: Optional[str] = None
    creator_id: UUID
    role: str = Field('admin', description="User's role in the project (admin, editor, viewer)")
    created_at: datetime
    updated_at: datetime
    study_count: int = Field(0, description="Number of studies in this project")

    class Config:
        from_attributes = True


class ProjectListItem(BaseModel):
    """Lightweight schema for listing projects"""
    id: UUID
    name: str
    description: Optional[str] = None
    role: str = Field('admin', description="User's role in the project (admin, editor, viewer)")
    created_at: datetime
    updated_at: datetime
    study_count: int = Field(0, description="Number of studies in this project")

    class Config:
        from_attributes = True


class ProjectMemberInvite(BaseModel):
    """Schema for inviting a user to a project"""
    email: str = Field(..., description="Email address of the user to invite")
    role: str = Field(..., description="Role: 'editor' or 'viewer'")

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "role": "editor"
            }
        }


class ProjectMemberOut(BaseModel):
    """Schema for project member output"""
    id: UUID
    project_id: UUID
    user_id: Optional[UUID] = None
    email: str
    role: str
    created_at: datetime
    updated_at: datetime
    name: Optional[str] = None
    is_registered: bool = False

    class Config:
        from_attributes = True


class ProjectMemberUpdate(BaseModel):
    """Schema for updating a project member's role"""
    role: str = Field(..., description="New role: 'editor' or 'viewer'")
