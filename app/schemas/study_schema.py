from __future__ import annotations

from typing import List, Optional, Dict, Any, Literal
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

StudyType = Literal['grid', 'layer']
StudyStatus = Literal['draft', 'active', 'paused', 'completed']
ElementType = Literal['image', 'text']

# ---------- Nested value objects ----------

class RatingScale(BaseModel):
    min_value: int = Field(ge=1, le=9)
    max_value: int = Field(..., description="One of 5,7,9")
    min_label: str
    max_label: str
    middle_label: Optional[str] = None

class AudienceSegmentation(BaseModel):
    number_of_respondents: int = Field(..., ge=1)
    country: Optional[str] = None
    gender_distribution: Optional[Dict[str, float]] = None
    age_distribution: Optional[Dict[str, float]] = None

class StudyElementIn(BaseModel):
    element_id: str = Field(..., max_length=10)  # e.g., E1
    name: str = Field(..., max_length=100)
    description: Optional[str] = None
    element_type: ElementType
    content: str
    alt_text: Optional[str] = None

class StudyElementOut(StudyElementIn):
    id: UUID
    model_config = ConfigDict(from_attributes=True)

class LayerImageIn(BaseModel):
    image_id: str = Field(..., max_length=100)
    name: str = Field(..., max_length=100)
    url: str
    alt_text: Optional[str] = None
    order: int

class LayerImageOut(LayerImageIn):
    id: UUID
    model_config = ConfigDict(from_attributes=True)

class StudyLayerIn(BaseModel):
    layer_id: str = Field(..., max_length=100)
    name: str = Field(..., max_length=100)
    description: Optional[str] = None
    z_index: int
    order: int
    images: List[LayerImageIn] = Field(default_factory=list)

class StudyLayerOut(BaseModel):
    id: UUID
    layer_id: str = Field(..., max_length=100)
    name: str = Field(..., max_length=100)
    description: Optional[str] = None
    z_index: int
    order: int
    images: List[LayerImageOut] = Field(default_factory=list)
    model_config = ConfigDict(from_attributes=True)

# ---------- Study payloads ----------

class StudyBase(BaseModel):
    title: str = Field(..., max_length=255)
    background: str
    language: str = Field(default='en', max_length=10)
    main_question: str
    orientation_text: str
    study_type: StudyType
    rating_scale: RatingScale
    audience_segmentation: AudienceSegmentation

class StudyCreate(StudyBase):
    # For grid studies, provide elements; for layer studies, provide study_layers
    elements: Optional[List[StudyElementIn]] = None
    study_layers: Optional[List[StudyLayerIn]] = None

class StudyUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=255)
    background: Optional[str] = None
    language: Optional[str] = Field(None, max_length=10)
    main_question: Optional[str] = None
    orientation_text: Optional[str] = None
    rating_scale: Optional[RatingScale] = None
    audience_segmentation: Optional[AudienceSegmentation] = None
    # Replacing full collections (server should decide whether allowed while active)
    elements: Optional[List[StudyElementIn]] = None
    study_layers: Optional[List[StudyLayerIn]] = None
    status: Optional[StudyStatus] = None

# ---------- Read models ----------

class StudyListItem(BaseModel):
    id: UUID
    title: str
    study_type: StudyType
    status: StudyStatus
    created_at: datetime
    total_responses: int
    completed_responses: int
    abandoned_responses: int
    model_config = ConfigDict(from_attributes=True)

class StudyOut(BaseModel):
    id: UUID
    title: str
    background: str
    language: str
    main_question: str
    orientation_text: str
    study_type: StudyType
    rating_scale: RatingScale
    audience_segmentation: AudienceSegmentation

    # Children
    elements: Optional[List[StudyElementOut]] = None
    study_layers: Optional[List[StudyLayerOut]] = None

    # Tasks and meta
    tasks: Optional[Dict[str, List[Dict[str, Any]]]] = None
    creator_id: UUID
    status: StudyStatus
    share_token: str
    share_url: Optional[str] = None

    created_at: datetime
    updated_at: datetime
    launched_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    total_responses: int
    completed_responses: int
    abandoned_responses: int

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
    study_type: StudyType
    audience_segmentation: AudienceSegmentation
    elements: Optional[List[StudyElementIn]] = None
    study_layers: Optional[List[StudyLayerIn]] = None
    exposure_tolerance_cv: Optional[float] = None
    exposure_tolerance_pct: Optional[float] = None
    seed: Optional[int] = None

class GenerateTasksResult(BaseModel):
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

