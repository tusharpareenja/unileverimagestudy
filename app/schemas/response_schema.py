# app/schemas/response.py
from __future__ import annotations

from typing import List, Optional, Dict, Any, Literal
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

# ---------- Element Interaction Schemas ----------

class ElementInteractionBase(BaseModel):
    element_id: str = Field(..., max_length=10)
    view_time_seconds: float = Field(..., ge=0.0)
    hover_count: int = Field(default=0, ge=0)
    click_count: int = Field(default=0, ge=0)
    first_view_time: Optional[datetime] = None
    last_view_time: Optional[datetime] = None

class ElementInteractionCreate(ElementInteractionBase):
    pass

class ElementInteractionUpdate(BaseModel):
    element_id: Optional[str] = Field(None, max_length=10)
    view_time_seconds: Optional[float] = Field(None, ge=0.0)
    hover_count: Optional[int] = Field(None, ge=0)
    click_count: Optional[int] = Field(None, ge=0)
    first_view_time: Optional[datetime] = None
    last_view_time: Optional[datetime] = None

class ElementInteractionOut(ElementInteractionBase):
    id: UUID
    study_response_id: UUID
    task_session_id: Optional[UUID] = None
    
    model_config = ConfigDict(from_attributes=True)

# ---------- Completed Task Schemas ----------

class CompletedTaskBase(BaseModel):
    task_id: str = Field(..., max_length=20)
    respondent_id: int = Field(..., ge=0)
    task_index: int = Field(..., ge=0)
    elements_shown_in_task: Dict[str, Any]
    elements_shown_content: Optional[Dict[str, Any]] = None
    task_type: Optional[str] = Field(None, max_length=50)
    task_context: Optional[Dict[str, Any]] = None
    task_start_time: datetime
    task_completion_time: datetime
    task_duration_seconds: float = Field(..., ge=0.0)
    rating_given: int = Field(..., ge=1, le=9)
    rating_timestamp: datetime
    elements_shown: Optional[Dict[str, Any]] = None
    layers_shown_in_task: Optional[Dict[str, Any]] = None

class CompletedTaskCreate(CompletedTaskBase):
    pass

class CompletedTaskUpdate(BaseModel):
    task_id: Optional[str] = Field(None, max_length=20)
    respondent_id: Optional[int] = Field(None, ge=0)
    task_index: Optional[int] = Field(None, ge=0)
    elements_shown_in_task: Optional[Dict[str, Any]] = None
    elements_shown_content: Optional[Dict[str, Any]] = None
    task_type: Optional[str] = Field(None, max_length=50)
    task_context: Optional[Dict[str, Any]] = None
    task_start_time: Optional[datetime] = None
    task_completion_time: Optional[datetime] = None
    task_duration_seconds: Optional[float] = Field(None, ge=0.0)
    rating_given: Optional[int] = Field(None, ge=1, le=9)
    rating_timestamp: Optional[datetime] = None
    elements_shown: Optional[Dict[str, Any]] = None
    layers_shown_in_task: Optional[Dict[str, Any]] = None

class CompletedTaskOut(CompletedTaskBase):
    id: UUID
    study_response_id: UUID
    
    model_config = ConfigDict(from_attributes=True)

# ---------- Classification Answer Schemas ----------

class ClassificationAnswerBase(BaseModel):
    question_id: str = Field(..., max_length=10)
    question_text: str = Field(..., max_length=500)
    answer: str = Field(..., max_length=1000)
    answer_timestamp: datetime
    time_spent_seconds: float = Field(default=0.0, ge=0.0)

class ClassificationAnswerCreate(ClassificationAnswerBase):
    pass

class ClassificationAnswerUpdate(BaseModel):
    question_id: Optional[str] = Field(None, max_length=10)
    question_text: Optional[str] = Field(None, max_length=500)
    answer: Optional[str] = Field(None, max_length=1000)
    answer_timestamp: Optional[datetime] = None
    time_spent_seconds: Optional[float] = Field(None, ge=0.0)

class ClassificationAnswerOut(ClassificationAnswerBase):
    id: UUID
    study_response_id: UUID
    
    model_config = ConfigDict(from_attributes=True)

# ---------- Task Session Schemas ----------

class TaskSessionBase(BaseModel):
    session_id: str = Field(..., max_length=50)
    task_id: str = Field(..., max_length=20)
    classification_page_time: float = Field(default=0.0, ge=0.0)
    orientation_page_time: float = Field(default=0.0, ge=0.0)
    individual_task_page_times: Optional[List[float]] = None
    page_transitions: Optional[List[Dict[str, Any]]] = None
    is_completed: bool = Field(default=False)
    abandonment_timestamp: Optional[datetime] = None
    abandonment_reason: Optional[str] = Field(None, max_length=200)
    recovery_attempts: int = Field(default=0, ge=0)
    browser_performance: Optional[Dict[str, Any]] = None
    page_load_times: Optional[List[float]] = None
    device_info: Optional[Dict[str, Any]] = None
    screen_resolution: Optional[str] = Field(None, max_length=20)

class TaskSessionCreate(TaskSessionBase):
    pass

class TaskSessionUpdate(BaseModel):
    session_id: Optional[str] = Field(None, max_length=50)
    task_id: Optional[str] = Field(None, max_length=20)
    classification_page_time: Optional[float] = Field(None, ge=0.0)
    orientation_page_time: Optional[float] = Field(None, ge=0.0)
    individual_task_page_times: Optional[List[float]] = None
    page_transitions: Optional[List[Dict[str, Any]]] = None
    is_completed: Optional[bool] = None
    abandonment_timestamp: Optional[datetime] = None
    abandonment_reason: Optional[str] = Field(None, max_length=200)
    recovery_attempts: Optional[int] = Field(None, ge=0)
    browser_performance: Optional[Dict[str, Any]] = None
    page_load_times: Optional[List[float]] = None
    device_info: Optional[Dict[str, Any]] = None
    screen_resolution: Optional[str] = Field(None, max_length=20)

class TaskSessionOut(TaskSessionBase):
    id: UUID
    study_response_id: UUID
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

# ---------- Study Response Schemas ----------

class StudyResponseBase(BaseModel):
    session_id: str = Field(..., max_length=50)
    respondent_id: int = Field(..., ge=0)
    current_task_index: int = Field(default=0, ge=0)
    completed_tasks_count: int = Field(default=0, ge=0)
    total_tasks_assigned: int = Field(..., ge=1)
    session_start_time: datetime
    session_end_time: Optional[datetime] = None
    is_completed: bool = Field(default=False)
    personal_info: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = Field(None, max_length=45)
    user_agent: Optional[str] = Field(None, max_length=500)
    browser_info: Optional[Dict[str, Any]] = None
    completion_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    total_study_duration: float = Field(default=0.0, ge=0.0)
    last_activity: Optional[datetime] = None
    is_abandoned: bool = Field(default=True)
    abandonment_timestamp: Optional[datetime] = None
    abandonment_reason: Optional[str] = Field(None, max_length=200)

class StudyResponseCreate(BaseModel):
    # Optional during creation; generated by server
    session_id: Optional[str] = None
    respondent_id: Optional[int] = None
    study_id: UUID
    current_task_index: int = Field(default=0, ge=0)
    completed_tasks_count: int = Field(default=0, ge=0)
    total_tasks_assigned: int = Field(..., ge=1)
    session_start_time: datetime
    session_end_time: Optional[datetime] = None
    is_completed: bool = Field(default=False)
    personal_info: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = Field(None, max_length=45)
    user_agent: Optional[str] = Field(None, max_length=500)
    browser_info: Optional[Dict[str, Any]] = None
    completion_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    total_study_duration: float = Field(default=0.0, ge=0.0)
    last_activity: Optional[datetime] = None
    is_abandoned: bool = Field(default=True)
    abandonment_timestamp: Optional[datetime] = None
    abandonment_reason: Optional[str] = Field(None, max_length=200)

class StudyResponseUpdate(BaseModel):
    session_id: Optional[str] = Field(None, max_length=50)
    respondent_id: Optional[int] = Field(None, ge=0)
    current_task_index: Optional[int] = Field(None, ge=0)
    completed_tasks_count: Optional[int] = Field(None, ge=0)
    total_tasks_assigned: Optional[int] = Field(None, ge=1)
    session_start_time: Optional[datetime] = None
    session_end_time: Optional[datetime] = None
    is_completed: Optional[bool] = None
    personal_info: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = Field(None, max_length=45)
    user_agent: Optional[str] = Field(None, max_length=500)
    browser_info: Optional[Dict[str, Any]] = None
    completion_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    total_study_duration: Optional[float] = Field(None, ge=0.0)
    last_activity: Optional[datetime] = None
    is_abandoned: Optional[bool] = None
    abandonment_timestamp: Optional[datetime] = None
    abandonment_reason: Optional[str] = Field(None, max_length=200)

class StudyResponseOut(StudyResponseBase):
    id: UUID
    study_id: UUID
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

# ---------- Detailed Response Schemas (with relationships) ----------

class StudyResponseDetail(StudyResponseOut):
    """Study response with all related data"""
    completed_tasks: List[CompletedTaskOut] = []
    classification_answers: List[ClassificationAnswerOut] = []
    element_interactions: List[ElementInteractionOut] = []
    task_sessions: List[TaskSessionOut] = []
    study_layers: Optional[List[Dict[str, Any]]] = None
    background_image_url: Optional[str] = None

class StudyResponseListItem(BaseModel):
    """Lightweight response for listing"""
    id: UUID
    session_id: str
    respondent_id: int
    personal_info: Optional[Dict[str, Any]] = None
    is_completed: bool
    is_abandoned: bool
    completion_percentage: Optional[float] = None
    total_study_duration: Optional[float] = None
    session_start_time: datetime
    session_end_time: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)

# ---------- User Details Schemas ----------

class UserDetails(BaseModel):
    """Structured user details for study participation"""
    date_of_birth: Optional[str] = Field(None, description="Date of birth in YYYY-MM-DD format")
    gender: Optional[str] = Field(None, description="Gender (male, female, other, prefer_not_to_say)")
    age_range: Optional[str] = Field(None, description="Age range (18-24, 25-34, 35-44, 45-54, 55-64, 65+)")
    education_level: Optional[str] = Field(None, description="Education level")
    occupation: Optional[str] = Field(None, description="Occupation or job title")
    location: Optional[str] = Field(None, description="Country or region")
    additional_info: Optional[Dict[str, Any]] = Field(None, description="Any additional custom fields")

# ---------- Request/Response Schemas for API ----------

class StartStudyRequest(BaseModel):
    """Request to start a new study session"""
    study_id: UUID
    personal_info: Optional[Dict[str, Any]] = None
    user_details: Optional[UserDetails] = None

class StartStudyResponse(BaseModel):
    """Response when starting a study"""
    session_id: str
    respondent_id: int
    total_tasks_assigned: int
    study_info: Dict[str, Any]  # Basic study details

class SubmitTaskRequest(BaseModel):
    """Request to submit a completed task"""
    task_id: str
    rating_given: int = Field(..., ge=1, le=9)
    task_duration_seconds: float = Field(..., ge=0.0)
    element_interactions: Optional[List[ElementInteractionCreate]] = None

class SubmitTaskResponse(BaseModel):
    """Response after submitting a task"""
    success: bool
    next_task_index: Optional[int] = None
    is_study_complete: bool = False
    completion_percentage: float

class BulkSubmitTaskItem(BaseModel):
    """Single task item for bulk submission"""
    task_id: str
    rating_given: int = Field(..., ge=1, le=9)
    task_duration_seconds: float = Field(..., ge=0.0)
    elements_shown_in_task: Optional[Dict[str, Any]] = None
    elements_shown_content: Optional[Dict[str, Any]] = None
    element_interactions: Optional[List[ElementInteractionCreate]] = None

class BulkSubmitTasksRequest(BaseModel):
    """Bulk submit multiple completed tasks in order"""
    tasks: List[BulkSubmitTaskItem]

class BulkSubmitTasksResponse(BaseModel):
    """Response after bulk task submission"""
    success: bool
    submitted_count: int
    next_task_index: Optional[int] = None
    is_study_complete: bool = False
    completion_percentage: float

class SubmitClassificationRequest(BaseModel):
    """Request to submit classification answers"""
    answers: List[ClassificationAnswerCreate]

class SubmitClassificationResponse(BaseModel):
    """Response after submitting classification"""
    success: bool
    message: str

class AbandonStudyRequest(BaseModel):
    """Request to mark study as abandoned"""
    reason: str = Field(..., max_length=200)

class AbandonStudyResponse(BaseModel):
    """Response after abandoning study"""
    success: bool
    message: str

class UpdateUserDetailsRequest(BaseModel):
    """Request to update user details for a study session"""
    user_details: UserDetails

# ---------- Analytics Schemas ----------

class StudyAnalytics(BaseModel):
    """Analytics data for a study"""
    total_responses: int
    completed_responses: int
    abandoned_responses: int
    in_progress_responses: int
    completion_rate: float
    average_duration: float
    abandonment_rate: float
    element_heatmap: Dict[str, Any]
    timing_distributions: Dict[str, Any]

class ResponseAnalytics(BaseModel):
    """Analytics for a specific response"""
    response_id: UUID
    session_id: str
    completion_percentage: float
    total_duration: float
    task_analytics: List[Dict[str, Any]]
    element_interactions: List[ElementInteractionOut]
    abandonment_analysis: Optional[Dict[str, Any]] = None


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