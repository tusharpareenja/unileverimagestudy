from sqlalchemy import Column, String, Boolean, DateTime, Index, UniqueConstraint, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid
from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # User identification fields
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)

    # Authentication
    password_hash = Column(String(255), nullable=False)

    # Contact information
    phone = Column(String(20), nullable=True, index=True)

    # Personal information
    date_of_birth = Column(DateTime, nullable=True)

    # Status fields
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Password reset fields
    password_reset_token = Column(String(255), nullable=True, index=True)
    password_reset_expires = Column(DateTime(timezone=True), nullable=True)

    # Relations
    studies = relationship(
        "Study",
        back_populates="creator",
        lazy="selectin",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # Indexes for better query performance
    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),
        Index('idx_user_username_active', 'username', 'is_active'),
        Index('idx_user_created_at', 'created_at'),
        Index('idx_user_last_login', 'last_login'),
        UniqueConstraint('username', name='uq_user_username'),
        UniqueConstraint('email', name='uq_user_email'),
        CheckConstraint('length(username) >= 3', name='ck_user_username_length'),
        CheckConstraint('length(email) >= 5', name='ck_user_email_length'),
        CheckConstraint('length(name) >= 2', name='ck_user_name_length'),
    )

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"

