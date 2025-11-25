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
    # Optional global background image URL for layer/grid rendering
    background_image_url = Column(Text, nullable=True)

    # JSON configs
    rating_scale = Column(JSONB, nullable=False)         # {min_value, max_value in {5,7,9}, labels...}
    audience_segmentation = Column('iped_parameters', JSONB, nullable=False)  # renamed logically; same column
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

    # Study creation progress tracking
    last_step = Column(Integer, nullable=False, server_default=expression.text('1'), default=1)

    # Background job tracking
    jobid = Column(String(255), nullable=True, index=True)

    # Relations
    creator = relationship("User", back_populates="studies", lazy="selectin", passive_deletes=True)
    # Grid categories for grouping elements
    categories = relationship("StudyCategory", back_populates="study", cascade="all, delete-orphan", lazy="selectin")
    elements = relationship("StudyElement", back_populates="study", cascade="all, delete-orphan", lazy="selectin")
    layers = relationship("StudyLayer", back_populates="study", cascade="all, delete-orphan", lazy="selectin")
    classification_questions = relationship("StudyClassificationQuestion", back_populates="study", cascade="all, delete-orphan", lazy="selectin")
    study_responses = relationship("StudyResponse", back_populates="study", cascade="all, delete-orphan", lazy="selectin")

    __table_args__ = (
        UniqueConstraint('share_token', name='uq_studies_share_token'),
        Index('idx_studies_creator_status_created_at', 'creator_id', 'status', 'created_at'),
        Index('idx_studies_status_created_at', 'status', 'created_at'),
        Index('idx_studies_study_type', 'study_type'),
    )

class StudyCategory(Base):
    """
    Grid categories for grouping elements, e.g., 'a', 'b', 'c' with order.
    """
    __tablename__ = "study_categories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id = Column(UUID(as_uuid=True), ForeignKey('studies.id', ondelete='CASCADE'), nullable=False, index=True)

    category_id = Column(UUID(as_uuid=True), nullable=False)  # UUID instead of string
    name = Column(String(100), nullable=False)
    order = Column(Integer, nullable=False, default=0)

    study = relationship("Study", back_populates="categories", lazy="selectin")
    elements = relationship("StudyElement", back_populates="category", cascade="all, delete-orphan", lazy="selectin")

    __table_args__ = (
        UniqueConstraint('study_id', 'category_id', name='uq_study_categories_category_id'),
        Index('idx_study_categories_study_id_category_id', 'study_id', 'category_id'),
        Index('idx_study_categories_study_id_order', 'study_id', 'order'),
    )

class StudyElement(Base):
    """
    Grid elements now belong to a category.
    """
    __tablename__ = "study_elements"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id = Column(UUID(as_uuid=True), ForeignKey('studies.id', ondelete='CASCADE'), nullable=False, index=True)

    # Category foreign key
    category_id = Column(UUID(as_uuid=True), ForeignKey('study_categories.id', ondelete='CASCADE'), nullable=False, index=True)

    element_id = Column(UUID(as_uuid=True), nullable=False)  # UUID for element
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    element_type = Column(element_type_enum, nullable=False)  # image|text
    content = Column(Text, nullable=False)                    # URL or text content
    cloudinary_public_id = Column(Text, nullable=True)
    alt_text = Column(String(200), nullable=True)

    study = relationship("Study", back_populates="elements", lazy="selectin")
    category = relationship("StudyCategory", back_populates="elements", lazy="selectin")

    __table_args__ = (
        UniqueConstraint('study_id', 'element_id', name='uq_study_elements_element_id'),
        Index('idx_study_elements_study_id_element_id', 'study_id', 'element_id'),
        Index('idx_study_elements_category_id', 'category_id'),
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
    # Layer-level transform percentages {x,y,width,height}
    transform = Column(JSONB, nullable=True)

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
    cloudinary_public_id = Column(Text, nullable=True)
    alt_text = Column(String(200), nullable=True)
    order = Column(Integer, nullable=False)

    layer = relationship("StudyLayer", back_populates="images", lazy="selectin")

    __table_args__ = (
        UniqueConstraint('layer_id', 'image_id', name='uq_layer_images_image_id'),
        Index('idx_layer_images_layer_id_image_id', 'layer_id', 'image_id'),
    )

class StudyClassificationQuestion(Base):
    """Classification questions that belong to a study."""
    __tablename__ = "study_classification_questions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id = Column(UUID(as_uuid=True), ForeignKey('studies.id', ondelete='CASCADE'), nullable=False, index=True)

    question_id = Column(String(10), nullable=False)  # Q1, Q2, ...
    question_text = Column(Text, nullable=False)
    question_type = Column(String(20), nullable=False, default='multiple_choice')  # multiple_choice, text, rating, etc.
    is_required = Column(String(1), nullable=False, default='Y')  # Y/N
    order = Column(Integer, nullable=False, default=1)
    
    # Answer options (for multiple choice questions)
    answer_options = Column(JSONB, nullable=True)  # [{"id": "A", "text": "Option 1"}, ...]
    
    # Additional configuration
    config = Column(JSONB, nullable=True)  # Additional question-specific config

    study = relationship("Study", back_populates="classification_questions", lazy="selectin")

    __table_args__ = (
        UniqueConstraint('study_id', 'question_id', name='uq_study_classification_questions_question_id'),
        Index('idx_study_classification_questions_study_id_order', 'study_id', 'order'),
    )