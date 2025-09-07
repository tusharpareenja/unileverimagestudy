# app/api/v1/response.py
from __future__ import annotations

from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_active_user
from app.db.session import get_db
from app.models.user_model import User
from app.schemas.response_schema import (
    StudyResponseOut, StudyResponseDetail, StudyResponseListItem,
    StartStudyRequest, StartStudyResponse, SubmitTaskRequest, SubmitTaskResponse,
    SubmitClassificationRequest, SubmitClassificationResponse,
    AbandonStudyRequest, AbandonStudyResponse,
    StudyAnalytics, ResponseAnalytics, CompletedTaskOut,
    ClassificationAnswerOut, ElementInteractionOut, TaskSessionOut, TaskSessionCreate,
    ElementInteractionCreate, CompletedTaskCreate, ClassificationAnswerCreate, StudyResponseCreate
)
from app.services.response import StudyResponseService, TaskSessionService

router = APIRouter()

# ---------- Study Participation Endpoints (Public) ----------

@router.post("/start-study", response_model=StartStudyResponse)
async def start_study(
    request: StartStudyRequest,
    http_request: Request,
    db: Session = Depends(get_db)
):
    """
    Start a new study session for a participant.
    This endpoint is public and doesn't require authentication.
    """
    # Extract client information
    ip_address = http_request.client.host if http_request.client else None
    user_agent = http_request.headers.get("user-agent")
    
    service = StudyResponseService(db)
    return service.start_study(request, ip_address, user_agent)

@router.post("/submit-task", response_model=SubmitTaskResponse)
async def submit_task(
    session_id: str,
    request: SubmitTaskRequest,
    db: Session = Depends(get_db)
):
    """
    Submit a completed task for a study session.
    """
    service = StudyResponseService(db)
    return service.submit_task(session_id, request)

@router.post("/submit-classification", response_model=SubmitClassificationResponse)
async def submit_classification(
    session_id: str,
    request: SubmitClassificationRequest,
    db: Session = Depends(get_db)
):
    """
    Submit classification answers for a study session.
    """
    service = StudyResponseService(db)
    success = service.submit_classification(session_id, request)
    
    return SubmitClassificationResponse(
        success=success,
        message="Classification answers submitted successfully" if success else "Failed to submit answers"
    )

@router.post("/abandon-study", response_model=AbandonStudyResponse)
async def abandon_study(
    session_id: str,
    request: AbandonStudyRequest,
    db: Session = Depends(get_db)
):
    """
    Mark a study session as abandoned.
    """
    service = StudyResponseService(db)
    success = service.abandon_study(session_id, request)
    
    return AbandonStudyResponse(
        success=success,
        message="Study marked as abandoned" if success else "Failed to abandon study"
    )

@router.get("/session/{session_id}", response_model=StudyResponseOut)
async def get_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    Get study session details by session ID.
    """
    service = StudyResponseService(db)
    response = service.get_response_by_session(session_id)
    
    if not response:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    
    return response

# ---------- Study Response Management (Authenticated) ----------

@router.get("/", response_model=List[StudyResponseListItem])
async def list_responses(
    study_id: Optional[UUID] = Query(None, description="Filter by study ID"),
    is_completed: Optional[bool] = Query(None, description="Filter by completion status"),
    is_abandoned: Optional[bool] = Query(None, description="Filter by abandonment status"),
    limit: int = Query(100, ge=1, le=1000, description="Number of responses to return"),
    offset: int = Query(0, ge=0, description="Number of responses to skip"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List study responses with optional filtering.
    Only returns responses for studies owned by the current user.
    """
    service = StudyResponseService(db)
    
    if study_id:
        # Verify user owns the study
        from app.services.study import StudyService
        study_service = StudyService(db)
        study = study_service.get_study(study_id)
        
        if not study or study.creator_id != current_user.id:
            raise HTTPException(
                status_code=403,
                detail="Access denied to this study"
            )
        
        responses = service.get_responses_by_study(study_id, limit, offset)
    else:
        # Get all studies owned by user and their responses
        from app.services.study import StudyService
        study_service = StudyService(db)
        user_studies = study_service.get_studies_by_user(current_user.id)
        study_ids = [study.id for study in user_studies]
        
        if not study_ids:
            return []
        
        # Get responses for all user's studies
        responses = []
        for study_id in study_ids:
            study_responses = service.get_responses_by_study(study_id, limit, offset)
            responses.extend(study_responses)
    
    # Apply additional filters
    if is_completed is not None:
        responses = [r for r in responses if r.is_completed == is_completed]
    
    if is_abandoned is not None:
        responses = [r for r in responses if r.is_abandoned == is_abandoned]
    
    return responses

@router.get("/{response_id}", response_model=StudyResponseDetail)
async def get_response(
    response_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific study response.
    """
    service = StudyResponseService(db)
    response = service.get_response(response_id)
    
    if not response:
        raise HTTPException(
            status_code=404,
            detail="Response not found"
        )
    
    # Verify user owns the study
    from app.services.study import StudyService
    study_service = StudyService(db)
    study = study_service.get_study(response.study_id)
    
    if not study or study.creator_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Access denied to this response"
        )
    
    return response

@router.delete("/{response_id}")
async def delete_response(
    response_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete a study response.
    """
    service = StudyResponseService(db)
    response = service.get_response(response_id)
    
    if not response:
        raise HTTPException(
            status_code=404,
            detail="Response not found"
        )
    
    # Verify user owns the study
    from app.services.study import StudyService
    study_service = StudyService(db)
    study = study_service.get_study(response.study_id)
    
    if not study or study.creator_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Access denied to this response"
        )
    
    success = service.delete_response(response_id)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to delete response"
        )
    
    return {"message": "Response deleted successfully"}

# ---------- Analytics Endpoints ----------

@router.get("/analytics/study/{study_id}", response_model=StudyAnalytics)
async def get_study_analytics(
    study_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get analytics data for a study.
    """
    # Verify user owns the study
    from app.services.study import StudyService
    study_service = StudyService(db)
    study = study_service.get_study(study_id)
    
    if not study or study.creator_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Access denied to this study"
        )
    
    service = StudyResponseService(db)
    return service.get_study_analytics(study_id)

@router.get("/analytics/response/{response_id}", response_model=ResponseAnalytics)
async def get_response_analytics(
    response_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed analytics for a specific response.
    """
    service = StudyResponseService(db)
    response = service.get_response(response_id)
    
    if not response:
        raise HTTPException(
            status_code=404,
            detail="Response not found"
        )
    
    # Verify user owns the study
    from app.services.study import StudyService
    study_service = StudyService(db)
    study = study_service.get_study(response.study_id)
    
    if not study or study.creator_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Access denied to this response"
        )
    
    analytics = service.get_response_analytics(response_id)
    if not analytics:
        raise HTTPException(
            status_code=404,
            detail="Analytics not found"
        )
    
    return analytics

# ---------- Task Session Endpoints ----------

@router.post("/task-sessions/", response_model=TaskSessionOut)
async def create_task_session(
    session_data: TaskSessionCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new task session.
    """
    service = TaskSessionService(db)
    return service.create_task_session(session_data)

@router.get("/task-sessions/{session_id}/{task_id}", response_model=TaskSessionOut)
async def get_task_session(
    session_id: str,
    task_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get a task session by session ID and task ID.
    """
    service = TaskSessionService(db)
    task_session = service.get_task_session(session_id, task_id)
    
    if not task_session:
        raise HTTPException(
            status_code=404,
            detail="Task session not found"
        )
    
    return task_session

@router.post("/task-sessions/{session_id}/{task_id}/page-transition")
async def add_page_transition(
    session_id: str,
    task_id: str,
    page_name: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Add a page transition to a task session.
    """
    service = TaskSessionService(db)
    success = service.add_page_transition(session_id, task_id, page_name)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Task session not found"
        )
    
    return {"message": "Page transition added successfully"}

@router.post("/task-sessions/{session_id}/{task_id}/element-interaction")
async def add_element_interaction(
    session_id: str,
    task_id: str,
    interaction_data: ElementInteractionCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Add an element interaction to a task session.
    """
    service = TaskSessionService(db)
    success = service.add_element_interaction(session_id, task_id, interaction_data)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Task session not found"
        )
    
    return {"message": "Element interaction added successfully"}

# ---------- Export Endpoints ----------

@router.get("/export/study/{study_id}/responses")
async def export_study_responses(
    study_id: UUID,
    format: str = Query("csv", regex="^(csv|json)$", description="Export format"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Export all responses for a study in CSV or JSON format.
    """
    # Verify user owns the study
    from app.services.study import StudyService
    study_service = StudyService(db)
    study = study_service.get_study(study_id)
    
    if not study or study.creator_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Access denied to this study"
        )
    
    service = StudyResponseService(db)
    responses = service.get_responses_by_study(study_id, limit=10000)  # Large limit for export
    
    if format == "csv":
        # Generate CSV export
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "Response ID", "Session ID", "Respondent ID", "Is Completed", 
            "Is Abandoned", "Completion Percentage", "Total Duration",
            "Session Start", "Session End", "Last Activity"
        ])
        
        # Write data
        for response in responses:
            writer.writerow([
                str(response.id),
                response.session_id,
                response.respondent_id,
                response.is_completed,
                response.is_abandoned,
                response.completion_percentage,
                response.total_study_duration,
                response.session_start_time.isoformat() if response.session_start_time else "",
                response.session_end_time.isoformat() if response.session_end_time else "",
                response.last_activity.isoformat() if response.last_activity else ""
            ])
        
        content = output.getvalue()
        output.close()
        
        return {
            "content": content,
            "filename": f"study_{study_id}_responses.csv",
            "content_type": "text/csv"
        }
    
    else:  # JSON format
        return {
            "study_id": str(study_id),
            "study_title": study.title,
            "export_timestamp": datetime.utcnow().isoformat(),
            "total_responses": len(responses),
            "responses": [StudyResponseOut.model_validate(response) for response in responses]
        }

@router.get("/export/response/{response_id}/detailed")
async def export_response_detailed(
    response_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Export detailed data for a specific response.
    """
    service = StudyResponseService(db)
    response = service.get_response(response_id)
    
    if not response:
        raise HTTPException(
            status_code=404,
            detail="Response not found"
        )
    
    # Verify user owns the study
    from app.services.study import StudyService
    study_service = StudyService(db)
    study = study_service.get_study(response.study_id)
    
    if not study or study.creator_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Access denied to this response"
        )
    
    # Get detailed response data
    detailed_response = service.get_response(response_id)
    
    return {
        "response": StudyResponseDetail.model_validate(detailed_response),
        "analytics": service.get_response_analytics(response_id),
        "export_timestamp": datetime.utcnow().isoformat()
    }