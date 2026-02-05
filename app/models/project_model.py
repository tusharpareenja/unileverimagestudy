# app/models/project_model.py
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Index, UniqueConstraint, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.base import Base
import uuid


# Define project role enum (same as study roles: viewer, editor)
project_role_enum = Enum('editor', 'viewer', name='project_role_enum')


class Project(Base):
    """
    Project model for organizing studies.
    A project can contain multiple studies.
    """
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Basic info
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Ownership
    creator_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relations
    creator = relationship("User", back_populates="projects", lazy="selectin", passive_deletes=True)
    studies = relationship("Study", back_populates="project", lazy="selectin", passive_deletes=True)
    members = relationship("ProjectMember", back_populates="project", cascade="all, delete-orphan", lazy="selectin")
    
    __table_args__ = (
        Index('idx_projects_creator_id_created_at', 'creator_id', 'created_at'),
    )


class ProjectMember(Base):
    """
    Join table for project sharing with roles (similar to StudyMember).
    When a user is added to a project, they automatically get access to all studies in that project.
    """
    __tablename__ = "project_members"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id', ondelete='CASCADE'), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=True, index=True)
    
    # Role: editor or viewer (same permissions as study roles)
    role = Column(project_role_enum, nullable=False, server_default='viewer')
    
    # Invited email (handles users who don't have an account yet)
    invited_email = Column(String(255), nullable=False, index=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relations
    project = relationship("Project", back_populates="members", lazy="selectin")
    user = relationship("User", backref="shared_projects", lazy="selectin")

    __table_args__ = (
        UniqueConstraint('project_id', 'invited_email', name='uq_project_members_project_email'),
        Index('idx_project_members_project_id_role', 'project_id', 'role'),
        Index('idx_project_members_user_id', 'user_id'),
    )
