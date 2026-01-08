from __future__ import annotations

from typing import List, Optional, Dict, Any, Literal
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, field_validator, AliasChoices
from typing import Any, Dict, Optional

StudyType = Literal['grid', 'layer', 'text']
StudyStatus = Literal['draft', 'active', 'paused', 'completed']
ElementType = Literal['image', 'text']
LayerType = Literal['image', 'text']
StudyRole = Literal['admin', 'editor', 'viewer']

# ---------- Nested value objects ----------

class RatingScale(BaseModel):
    min_value: int = Field(ge=1, le=9)
    max_value: int = Field(..., description="One of 5,7,9")
    min_label: str
    max_label: str
    middle_label: Optional[str] = None

class AudienceSegmentation(BaseModel):
    number_of_respondents: Optional[int] = Field(None, ge=1)
    country: Optional[str] = None
    gender_distribution: Optional[Dict[str, float]] = None
    age_distribution: Optional[Dict[str, float]] = None
    # Allow aspect ratio to be sent inside segmentation as well
    aspect_ratio: Optional[str] = None

class StudyElementIn(BaseModel):
    element_id: UUID  # UUID instead of string
    name: str = Field(..., max_length=1000)
    description: Optional[str] = None
    element_type: ElementType
    content: str
    alt_text: Optional[str] = None
    # Link to category (required for grid elements) - UUID
    category_id: UUID

class StudyElementOut(StudyElementIn):
    id: UUID
    # Optional denormalized relation for convenience when reading
    category: Optional["StudyCategoryOut"] = None
    model_config = ConfigDict(from_attributes=True)

class StudyCategoryIn(BaseModel):
    category_id: UUID  # UUID instead of string
    name: str = Field(..., max_length=100)
    order: int = 0

class StudyCategoryOut(StudyCategoryIn):
    id: UUID
    model_config = ConfigDict(from_attributes=True)

class Transform(BaseModel):
    x: float = Field(ge=0.0, le=100.0)
    y: float = Field(ge=0.0, le=100.0)
    width: float = Field(gt=0.0, le=100.0)
    height: float = Field(gt=0.0, le=100.0)

    @field_validator('x', 'y', 'width', 'height')
    @classmethod
    def _clamp_percentage(cls, v: float) -> float:
        # Clamp to [0, 100] while preserving > 0 for width/height via Field constraints
        if v < 0.0:
            return 0.0
        if v > 100.0:
            return 100.0
        return float(v)

class LayerImageIn(BaseModel):
    image_id: str = Field(..., max_length=100)
    name: str = Field(..., max_length=100)
    url: str
    alt_text: Optional[str] = None
    order: int
    config: Optional[Dict[str, Any]] = None

class LayerImageOut(LayerImageIn):
    id: UUID
    config: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(from_attributes=True)

class StudyLayerIn(BaseModel):
    layer_id: str = Field(..., max_length=100)
    name: str = Field(..., max_length=100)
    description: Optional[str] = None
    layer_type: LayerType = Field(default='image')
    z_index: int
    order: int
    # Layer-level transform: percentages 0-100 relative to container
    transform: Optional[Transform] = None
    images: List[LayerImageIn] = Field(default_factory=list)

class StudyLayerOut(BaseModel):
    id: UUID
    layer_id: str = Field(..., max_length=100)
    name: str = Field(..., max_length=100)
    description: Optional[str] = None
    layer_type: LayerType = Field(default='image')
    z_index: int
    order: int
    transform: Optional[Transform] = None
    images: List[LayerImageOut] = Field(default_factory=list)
    model_config = ConfigDict(from_attributes=True)

class AnswerOption(BaseModel):
    """Answer option for multiple choice questions"""
    id: str = Field(..., max_length=10)  # A, B, C, etc.
    text: str = Field(..., max_length=200)
    order: Optional[int] = None

class StudyClassificationQuestionIn(BaseModel):
    question_id: str = Field(..., max_length=10)  # Q1, Q2, ...
    question_text: str = Field(..., max_length=500)
    question_type: str = Field(default='multiple_choice', max_length=20)  # multiple_choice, text, rating, etc.
    is_required: bool = Field(default=True)
    order: int = Field(default=1, ge=1)
    answer_options: Optional[List[AnswerOption]] = None  # For multiple choice questions
    config: Optional[Dict[str, Any]] = None  # Additional question-specific config

class StudyClassificationQuestionOut(StudyClassificationQuestionIn):
    id: UUID
    model_config = ConfigDict(from_attributes=True)

    @field_validator('is_required', mode='before')
    @classmethod
    def convert_is_required(cls, v: Any) -> bool:
        """Convert database 'Y'/'N' string to boolean"""
        if isinstance(v, str):
            return v.upper() == 'Y'
        return bool(v)

# ---------- Study payloads ----------

class StudyBase(BaseModel):
    title: str = Field(..., max_length=255)
    background: str
    language: str = Field(default='en', max_length=10)
    main_question: str
    orientation_text: str
    study_type: StudyType
    # Optional global background image URL to render behind all tasks
    background_image_url: Optional[str] = None
    # Optional aspect ratio for layer studies, e.g., "3:4", "4:3"
    aspect_ratio: Optional[str] = Field(default=None, description="Aspect ratio string like 3:4 or 4:3")
    rating_scale: RatingScale
    audience_segmentation: AudienceSegmentation
    last_step: Optional[int] = Field(default=1, ge=1)

class StudyCreateMinimal(BaseModel):
    """Minimal schema for fast study creation - only essential fields"""
    title: str = Field(..., max_length=255)
    background: str
    language: str = Field(default='en', max_length=10)
    last_step: Optional[int] = Field(default=1, ge=1)

class StudyCreate(StudyBase):
    # For grid studies, provide elements; for layer studies, provide study_layers
    categories: Optional[List[StudyCategoryIn]] = None
    elements: Optional[List[StudyElementIn]] = None
    study_layers: Optional[List[StudyLayerIn]] = None
    classification_questions: Optional[List[StudyClassificationQuestionIn]] = None

class StudyUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=255)
    background: Optional[str] = None
    language: Optional[str] = Field(None, max_length=10)
    main_question: Optional[str] = None
    orientation_text: Optional[str] = None
    study_type: Optional[StudyType] = None
    background_image_url: Optional[str] = None
    rating_scale: Optional[RatingScale] = None
    audience_segmentation: Optional[AudienceSegmentation] = None
    # Optional aspect ratio for layer studies, e.g., "3:4", "4:3"
    aspect_ratio: Optional[str] = Field(default=None, description="Aspect ratio string like 3:4 or 4:3")
    last_step: Optional[int] = Field(default=None, ge=1)
    # Replacing full collections (server should decide whether allowed while active)
    categories: Optional[List[StudyCategoryIn]] = None
    elements: Optional[List[StudyElementIn]] = None
    study_layers: Optional[List[StudyLayerIn]] = None
    classification_questions: Optional[List[StudyClassificationQuestionIn]] = None
    status: Optional[StudyStatus] = None

# ---------- Study Sharing schemas ----------

class StudyMemberBase(BaseModel):
    email: str = Field(..., validation_alias=AliasChoices("email", "invited_email"))
    role: StudyRole
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

class StudyMemberInvite(StudyMemberBase):
    pass

class StudyMemberOut(StudyMemberBase):
    id: UUID
    user_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime
    
    # Optional nested user details if available
    name: Optional[str] = None
    is_registered: bool = False

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

class StudyMemberUpdate(BaseModel):
    role: StudyRole

# ---------- Read models ----------

class StudyListItem(BaseModel):
    id: UUID
    title: str
    study_type: StudyType
    status: StudyStatus
    created_at: datetime
    last_step: int | None = None
    jobid: Optional[str] = None
    total_responses: int
    completed_responses: int
    abandoned_responses: int
    completion_rate: float | int = 0
    average_duration: float | int = 0
    abandonment_rate: float | int = 0
    respondents_target: int | None = 0
    respondents_completed: int | None = 0
    model_config = ConfigDict(from_attributes=True)


class StudyPublicMinimal(BaseModel):
    id: UUID
    title: str
    study_type: str
    respondents_target: int
    tasks_per_respondent: int
    status: str
    orientation_text: str
    language: str

    model_config = ConfigDict(from_attributes=False)

class StudyBasicDetails(BaseModel):
    """Basic study details for authenticated users - includes core study info without heavy data"""
    id: UUID
    title: str
    status: StudyStatus
    study_type: StudyType
    created_at: datetime
    background: str
    main_question: str
    orientation_text: str
    rating_scale: RatingScale
    jobid: Optional[str] = None
    study_config: Optional[Dict[str, Any]] = None  # For additional configuration
    classification_questions: Optional[List[StudyClassificationQuestionOut]] = None
    element_count: Optional[int] = None  # Number of images (grid/layer) or statements (text)

    model_config = ConfigDict(from_attributes=True)

class StudyOut(BaseModel):
    id: UUID
    title: str
    background: str
    language: str
    main_question: str
    orientation_text: str
    study_type: StudyType
    user_role: Optional[str] = None
    background_image_url: Optional[str] = None
    aspect_ratio: Optional[str] = None
    rating_scale: RatingScale
    audience_segmentation: AudienceSegmentation

    # Children
    categories: Optional[List[StudyCategoryOut]] = None
    elements: Optional[List[StudyElementOut]] = None
    study_layers: Optional[List[StudyLayerOut]] = Field(None, validation_alias="layers")
    classification_questions: Optional[List[StudyClassificationQuestionOut]] = None

    # Tasks and meta
    tasks: Optional[Dict[str, List[Dict[str, Any]]]] = None
    creator_id: UUID
    status: StudyStatus
    share_token: str
    share_url: Optional[str] = None
    jobid: Optional[str] = None

    created_at: datetime
    updated_at: datetime
    launched_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    total_responses: int
    completed_responses: int
    abandoned_responses: int

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

class StudyLaunchOut(BaseModel):
    """Lightweight response model for study launch operations - excludes heavy fields like tasks"""
    id: UUID
    title: str
    status: StudyStatus
    share_url: Optional[str] = None
    launched_at: Optional[datetime] = None
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class StudyCreateMinimalResponse(BaseModel):
    """Ultra-lightweight response for minimal study creation - only returns study ID"""
    id: UUID
    model_config = ConfigDict(from_attributes=True)

# ---------- Small helper responses ----------

class ChangeStatusPayload(BaseModel):
    status: StudyStatus

class RegenerateTasksResponse(BaseModel):
    success: bool
    message: str
    total_tasks: int
    metadata: Optional[Dict[str, Any]] = None

class ValidateTasksResponse(BaseModel):
    validation_passed: bool
    issues: List[str] = []
    totals: Dict[str, Any] = {}

class GenerateTasksRequest(BaseModel):
    study_id: Optional[UUID] = None
    last_step: Optional[int] = None
    study_type: StudyType
    audience_segmentation: AudienceSegmentation
    # Optional full study fields to create/update a draft study on the fly
    title: Optional[str] = None
    background: Optional[str] = None
    language: Optional[str] = None
    main_question: Optional[str] = None
    orientation_text: Optional[str] = None
    background_image_url: Optional[str] = None
    aspect_ratio: Optional[str] = None
    rating_scale: Optional[RatingScale] = None
    classification_questions: Optional[List[StudyClassificationQuestionIn]] = None
    categories: Optional[List[StudyCategoryIn]] = None
    elements: Optional[List[StudyElementIn]] = None
    study_layers: Optional[List[StudyLayerIn]] = None
    exposure_tolerance_cv: Optional[float] = None
    exposure_tolerance_pct: Optional[float] = None
    seed: Optional[int] = None

class GenerateTasksResult(BaseModel):
    last_step: Optional[int] = None
    tasks: Dict[str, List[Dict[str, Any]]]
    metadata: Dict[str, Any]


# Rebuild models to resolve forward references
def rebuild_models():
    """Rebuild all models to resolve forward references"""
    # Get all model classes defined in this module
    import sys
    current_module = sys.modules[__name__]
    
    for name in dir(current_module):
        obj = getattr(current_module, name)
        if (isinstance(obj, type) and 
            issubclass(obj, BaseModel) and 
            obj != BaseModel and
            obj.__module__ == __name__):
            try:
                obj.model_rebuild()
            except Exception as e:
                print(f"Warning: Could not rebuild {name}: {e}")

# Call rebuild to resolve forward references
rebuild_models()

