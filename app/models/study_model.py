# app/models/study.py
from sqlalchemy import (
    Column, String, Integer, DateTime, Enum, ForeignKey, Index, UniqueConstraint, Text
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func, expression
from sqlalchemy.orm import relationship
from app.db.base import Base
import uuid

study_type_enum = Enum('grid', 'layer', name='study_type_enum')
study_status_enum = Enum('draft', 'active', 'paused', 'completed', name='study_status_enum')
element_type_enum = Enum('image', 'text', name='element_type_enum')

class Study(Base):
    __tablename__ = "studies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # Basic info
    title = Column(String(255), nullable=False)
    background = Column(Text, nullable=False)
    language = Column(String(10), nullable=False, server_default='en')
    main_question = Column(Text, nullable=False)
    orientation_text = Column(Text, nullable=False)
    study_type = Column(study_type_enum, nullable=False)

    # JSON configs
    rating_scale = Column(JSONB, nullable=False)         # {min_value, max_value in {5,7,9}, labels...}
    iped_parameters = Column(JSONB, nullable=False)      # grid/layer-specific fields; validate in schemas
    tasks = Column(JSONB, nullable=True)                 # generated task matrix

    # Ownership / meta
    creator_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    status = Column(study_status_enum, nullable=False, server_default='draft')
    share_token = Column(String(255), nullable=False, unique=True, index=True)
    share_url = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    launched_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Counters
    total_responses = Column(Integer, nullable=False, server_default=expression.text('0'))
    completed_responses = Column(Integer, nullable=False, server_default=expression.text('0'))
    abandoned_responses = Column(Integer, nullable=False, server_default=expression.text('0'))

    # Relations
    creator = relationship("User", back_populates="studies", lazy="selectin", passive_deletes=True)
    elements = relationship("StudyElement", back_populates="study", cascade="all, delete-orphan", lazy="selectin")
    layers = relationship("StudyLayer", back_populates="study", cascade="all, delete-orphan", lazy="selectin")
    study_responses = relationship("StudyResponse", back_populates="study", cascade="all, delete-orphan", lazy="selectin")

    __table_args__ = (
        UniqueConstraint('share_token', name='uq_studies_share_token'),
        Index('idx_studies_creator_status_created_at', 'creator_id', 'status', 'created_at'),
        Index('idx_studies_status_created_at', 'status', 'created_at'),
        Index('idx_studies_study_type', 'study_type'),
    )

class StudyElement(Base):
    """
    For grid studies (flat elements E1..En).
    Keep element_id like E1, E2, ... for parity with legacy tasks.
    """
    __tablename__ = "study_elements"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id = Column(UUID(as_uuid=True), ForeignKey('studies.id', ondelete='CASCADE'), nullable=False, index=True)

    element_id = Column(String(10), nullable=False)  # E1, E2, ...
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    element_type = Column(element_type_enum, nullable=False)  # image|text
    content = Column(Text, nullable=False)                    # URL or text content
    alt_text = Column(String(200), nullable=True)

    study = relationship("Study", back_populates="elements", lazy="selectin")

    __table_args__ = (
        UniqueConstraint('study_id', 'element_id', name='uq_study_elements_element_id'),
        Index('idx_study_elements_study_id_element_id', 'study_id', 'element_id'),
    )

class StudyLayer(Base):
    """
    For layer studies: each layer (category) containing images.
    Use layer_id for parity with legacy (string id).
    """
    __tablename__ = "study_layers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id = Column(UUID(as_uuid=True), ForeignKey('studies.id', ondelete='CASCADE'), nullable=False, index=True)

    layer_id = Column(String(100), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    z_index = Column(Integer, nullable=False)
    order = Column(Integer, nullable=False)

    study = relationship("Study", back_populates="layers", lazy="selectin")
    images = relationship("LayerImage", back_populates="layer", cascade="all, delete-orphan", lazy="selectin")

    __table_args__ = (
        UniqueConstraint('study_id', 'layer_id', name='uq_study_layers_layer_id'),
        Index('idx_study_layers_study_id_layer_id', 'study_id', 'layer_id'),
    )

class LayerImage(Base):
    __tablename__ = "layer_images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    layer_id = Column(UUID(as_uuid=True), ForeignKey('study_layers.id', ondelete='CASCADE'), nullable=False, index=True)

    image_id = Column(String(100), nullable=False)  # string id; keep parity with legacy
    name = Column(String(100), nullable=False)
    url = Column(Text, nullable=False)
    alt_text = Column(String(200), nullable=True)
    order = Column(Integer, nullable=False)

    layer = relationship("StudyLayer", back_populates="images", lazy="selectin")

    __table_args__ = (
        UniqueConstraint('layer_id', 'image_id', name='uq_layer_images_image_id'),
        Index('idx_layer_images_layer_id_image_id', 'layer_id', 'image_id'),
    )