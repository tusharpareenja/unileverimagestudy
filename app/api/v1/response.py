# app/api/v1/response.py
from __future__ import annotations

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from fastapi.responses import StreamingResponse
import json
import asyncio
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.dependencies import get_current_active_user
from app.db.session import get_db
from app.models.user_model import User
from app.schemas.response_schema import (
    StudyResponseOut, StudyResponseDetail, StudyResponseListItem,
    StartStudyRequest, StartStudyResponse, SubmitTaskRequest, SubmitTaskResponse,
    BulkSubmitTasksRequest, BulkSubmitTasksResponse,
    SubmitClassificationRequest, SubmitClassificationResponse,
    AbandonStudyRequest, AbandonStudyResponse, UpdateUserDetailsRequest,
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

@router.post("/submit-tasks-bulk", response_model=BulkSubmitTasksResponse)
async def submit_tasks_bulk(
    session_id: str,
    request: BulkSubmitTasksRequest,
    db: Session = Depends(get_db)
):
    """
    Submit multiple completed tasks for a study session in one request.
    Tasks are applied in the order provided; progress and completion are updated accordingly.
    """
    service = StudyResponseService(db)
    return service.submit_tasks_bulk(session_id, request)

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

@router.get("/session/{session_id}", response_model=StudyResponseDetail)
async def get_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    Get study session details by session ID.
    """
    service = StudyResponseService(db)
    response = service.get_response_detail_by_session(session_id)
    
    if not response:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    
    # Get study layer information with z-index for layer studies
    study_layers = []
    if response.study_id:
        from app.models.study_model import StudyLayer, LayerImage
        from sqlalchemy.orm import selectinload
        
        # Get layers with images and z-index information
        layers = db.execute(
            select(StudyLayer)
            .options(selectinload(StudyLayer.images))
            .where(StudyLayer.study_id == response.study_id)
            .order_by(StudyLayer.order)
        ).scalars().all()
        
        for layer in layers:
            layer_images = []
            for image in layer.images:
                layer_images.append({
                    "id": str(image.id),
                    "name": image.name,
                    "url": image.url,
                    "alt_text": image.alt_text,
                    "order": image.order,
                    "z_index": layer.z_index  # Use layer's z_index for all images in this layer
                })
            
            study_layers.append({
                "id": str(layer.id),
                "name": layer.name,
                "order": layer.order,
                "z_index": layer.z_index,
                "images": layer_images
            })
    
    # Add layer information to response
    response_dict = StudyResponseDetail.model_validate(response).model_dump()
    response_dict["study_layers"] = study_layers
    # Add optional study background image url
    try:
        if response and getattr(response, "study_id", None):
            from app.models.study_model import Study as StudyModel
            study_row = db.execute(
                select(StudyModel).where(StudyModel.id == response.study_id)
            ).scalar_one_or_none()
            if study_row is not None:
                response_dict["background_image_url"] = getattr(study_row, "background_image_url", None)
    except Exception:
        response_dict["background_image_url"] = None

    # Enrich completed tasks with elements_shown_content from study.tasks if missing
    try:
        if response_dict.get("completed_tasks") and response_dict.get("study_id"):
            from app.models.study_model import Study as StudyModel
            study_row = db.execute(
                select(StudyModel).where(StudyModel.id == response.study_id)
            ).scalar_one_or_none()
            if study_row and isinstance(study_row.tasks, dict):
                # Respondent keys in generated tasks are 0-based; sessions may be 1-based.
                resp_id = response_dict.get("respondent_id")
                keys_to_try = [str(resp_id), str(max(0, (resp_id or 0) - 1))]
                respondent_tasks = []
                for k in keys_to_try:
                    respondent_tasks = study_row.tasks.get(k)
                    if respondent_tasks:
                        break
                index_to_content = {
                    int(t.get("task_index")): t.get("elements_shown_content")
                    for t in (respondent_tasks or []) if isinstance(t, dict)
                }
                for ct in response_dict.get("completed_tasks", []):
                    if ct.get("elements_shown_content") is None:
                        task_index = ct.get("task_index")
                        if task_index in index_to_content:
                            ct["elements_shown_content"] = index_to_content[task_index]
    except Exception:
        # Non-fatal enrichment
        pass
    
    # Map classification answer codes to human-readable labels using study configuration
    try:
        if response and getattr(response, "study_id", None):
            from app.models.study_model import StudyClassificationQuestion
            from app.models.study_model import Study as StudyModel
            # Build options map per question
            questions = db.execute(
                select(StudyClassificationQuestion)
                .where(StudyClassificationQuestion.study_id == response.study_id)
                .order_by(StudyClassificationQuestion.order)
            ).scalars().all()
            qid_to_options: Dict[str, Dict[str, str]] = {}
            for q in questions:
                options_map: Dict[str, str] = {}
                if isinstance(q.answer_options, list):
                    for opt in q.answer_options:
                        if not isinstance(opt, dict):
                            continue
                        text = opt.get("text") or opt.get("label") or opt.get("name")
                        if text is None:
                            continue
                        for key_name in ("id", "value", "code", "label"):
                            if key_name in opt and opt[key_name] is not None:
                                options_map[str(opt[key_name])] = text
                if options_map:
                    qid_to_options[q.question_id] = options_map

            # Convert ORM -> Pydantic dict
            resp_out = StudyResponseDetail.model_validate(response).model_dump()
            # Add layer information to the response
            resp_out["study_layers"] = study_layers
            # Ensure background_image_url is included (may be None)
            try:
                study_row2 = db.execute(
                    select(StudyModel).where(StudyModel.id == response.study_id)
                ).scalar_one_or_none()
                resp_out["background_image_url"] = getattr(study_row2, "background_image_url", None) if study_row2 else None
            except Exception:
                resp_out["background_image_url"] = None
            # Transform answers
            for ans in resp_out.get("classification_answers", []) or []:
                qid = ans.get("question_id")
                raw = ans.get("answer")
                mapped = raw
                try:
                    options_map = qid_to_options.get(qid)
                    if isinstance(raw, str) and options_map:
                        left_code = raw.split(' - ', 1)[0]
                        right_part = raw.split(' - ', 1)[1].strip() if ' - ' in raw else None
                        if raw in options_map:
                            mapped = options_map[raw]
                        elif left_code in options_map:
                            mapped = options_map[left_code]
                        elif right_part and right_part in options_map:
                            mapped = options_map[right_part]
                        else:
                            parts = raw.split(' - ', 1)
                            if len(parts) == 2 and parts[1].strip():
                                mapped = parts[1].strip()
                    elif isinstance(raw, str):
                        parts = raw.split(' - ', 1)
                        if len(parts) == 2 and parts[1].strip():
                            mapped = parts[1].strip()
                except Exception:
                    mapped = raw
                ans["answer"] = mapped
            return resp_out
    except Exception:
        # Fallback to raw response if mapping fails
        pass

    return response_dict

@router.put("/session/{session_id}/user-details")
async def update_user_details(
    session_id: str,
    request: UpdateUserDetailsRequest,
    db: Session = Depends(get_db)
):
    """
    Update user details for a study session.
    This endpoint is public and doesn't require authentication.
    """
    service = StudyResponseService(db)
    user_details_dict = request.user_details.model_dump(exclude_none=True)
    success = service.update_user_details(session_id, user_details_dict)
    
    return {
        "success": success,
        "message": "User details updated successfully" if success else "Failed to update user details"
    }

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
        # Optimized: lightweight ownership check
        from app.services import study as study_service
        if not study_service.get_study_exists(db, study_id, current_user.id):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this study"
            )
        
        # Use SQL-level filtering for better performance
        responses = service.get_responses_by_study_filtered(
            study_id, limit, offset, is_completed, is_abandoned
        )
    else:
        # Get all studies owned by user and their responses
        from app.services import study as study_service
        user_studies, _ = study_service.list_studies(db, current_user.id)
        study_ids = [study.id for study in user_studies]
        
        if not study_ids:
            return []
        
        # Get responses for all user's studies with SQL-level filtering
        responses = []
        for study_id in study_ids:
            study_responses = service.get_responses_by_study_filtered(
                study_id, limit, offset, is_completed, is_abandoned
            )
            responses.extend(study_responses)
    
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
    from app.services import study as study_service
    study = study_service.get_study(db=db, study_id=response.study_id, owner_id=current_user.id)
    
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
    from app.services import study as study_service
    study = study_service.get_study(db=db, study_id=response.study_id, owner_id=current_user.id)
    
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
    Get analytics data for a study - optimized for fast performance with rate limiting.
    """
    import time
    start_time = time.time()
    
    # Fast ownership verification - only check creator_id, don't load full study
    from sqlalchemy import select
    from app.models.study_model import Study
    
    ownership_check = select(Study.creator_id).where(Study.id == study_id)
    result = db.execute(ownership_check).first()
    
    if not result or result.creator_id != current_user.id:
        raise HTTPException(
            status_code=404,
            detail="Study not found or access denied"
        )
    
    # No rate limiting - allow real-time updates
    
    service = StudyResponseService(db)
    analytics = service.get_study_analytics(study_id)
    
    end_time = time.time()
    print(f"Analytics query took: {(end_time - start_time)*1000:.2f}ms")
    
    return analytics


@router.get("/analytics/study/{study_id}/stream")
async def stream_study_analytics(
    study_id: UUID,
    interval_seconds: int = Query(10, ge=5, le=60),  # Increased minimum for scalability
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Server-Sent Events (SSE) stream of study analytics.
    Emits an event every `interval_seconds` with the same payload shape
    as /analytics/study/{study_id}.
    Optimized for scalability with 100+ users.
    """
    # Ownership check (fast)
    from sqlalchemy import select
    from app.models.study_model import Study
    ownership_check = select(Study.creator_id).where(Study.id == study_id)
    result = db.execute(ownership_check).first()
    if not result or result.creator_id != current_user.id:
        raise HTTPException(
            status_code=404,
            detail="Study not found or access denied"
        )

    service = StudyResponseService(db)

    async def event_generator():
        last_payload: str | None = None
        request_count = 0
        
        while True:
            # Smart caching: Use cache for most requests, refresh occasionally
            if request_count % 2 == 0:  # Refresh every 2nd request (every 20 seconds with 10s interval)
                analytics = service.get_study_analytics(study_id)
            else:
                # Use cached data for faster response
                analytics = service._get_cached_analytics(study_id)
                if not analytics:
                    analytics = service.get_study_analytics(study_id)
            
            payload = json.dumps(analytics.model_dump())
            # Only send when changed to reduce client work
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            
            request_count += 1
            await asyncio.sleep(interval_seconds)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

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
    from app.services import study as study_service
    study = study_service.get_study(db=db, study_id=response.study_id, owner_id=current_user.id)
    
    analytics = service.get_response_analytics(response_id)
    if not analytics:
        raise HTTPException(
            status_code=404,
            detail="Analytics not found"
        )
    
    return analytics


@router.post("/check-abandoned-sessions")
async def check_abandoned_sessions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Manually trigger check for sessions that should be marked as abandoned (2+ hours inactive).
    This endpoint can be called periodically or by a background task.
    """
    service = StudyResponseService(db)
    count = service.check_and_mark_abandoned_sessions()
    
    return {
        "message": f"Checked for abandoned sessions",
        "sessions_marked_abandoned": count,
        "timestamp": datetime.utcnow().isoformat()
    }

# ---------- Task Session Endpoints ----------

@router.post("/task-sessions/", response_model=TaskSessionOut)
async def create_task_session(
    session_data: TaskSessionCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new task session. Public, no authentication required.
    """
    service = TaskSessionService(db)
    return service.create_task_session(session_data)

@router.get("/task-sessions/{session_id}/{task_id}", response_model=TaskSessionOut)
async def get_task_session(
    session_id: str,
    task_id: str,
    db: Session = Depends(get_db)
):
    """
    Get a task session by session ID and task ID. Public, no authentication required.
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
    db: Session = Depends(get_db)
):
    """
    Add a page transition to a task session. Public, no authentication required.
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
    db: Session = Depends(get_db)
):
    """
    Add an element interaction to a task session. Public, no authentication required.
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
    from app.services import study as study_service
    study = study_service.get_study(db=db, study_id=study_id, owner_id=current_user.id)
    
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
    from app.services import study as study_service
    study = study_service.get_study(db=db, study_id=response.study_id, owner_id=current_user.id)
    
    # Get detailed response data
    detailed_response = service.get_response(response_id)
    
    return {
        "response": StudyResponseDetail.model_validate(detailed_response),
        "analytics": service.get_response_analytics(response_id),
        "export_timestamp": datetime.utcnow().isoformat()
    }


@router.get("/export/study/{study_id}/flattened-csv")
async def export_study_flattened_csv(
    study_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Export a flattened CSV where each row is a respondent-task with columns:
    Panelist(session_id), QQ* classification answers, Gender, Age, Task, Layer_* visibility flags, Rating, ResponseTime.
    """
    # Verify user owns the study
    from app.services import study as study_service
    study = study_service.get_study(db=db, study_id=study_id, owner_id=current_user.id)

    service = StudyResponseService(db)

    def csv_generator():
        import csv
        import io
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        batch_size = 5000
        counter = 0
        for row in service.generate_csv_rows_for_study_optimized(study_id):
            writer.writerow(row)
            counter += 1
            if counter % batch_size == 0:
                data = buffer.getvalue()
                if data:
                    yield data
                buffer.seek(0)
                buffer.truncate(0)
        # flush remainder
        data = buffer.getvalue()
        if data:
            yield data

    filename = f"study_{study_id}_flattened_export.csv"
    headers = {
        "Content-Disposition": f"attachment; filename={filename}"
    }
    return StreamingResponse(csv_generator(), media_type="text/csv", headers=headers)

@router.get("/respondent/{respondent_id}/study/{study_id}/info")
async def get_respondent_study_info(
    respondent_id: int,
    study_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get study information for a specific respondent including classification questions 
    and tasks assigned to that respondent (tasks[respondent_id]).
    This endpoint is public and doesn't require authentication.
    """
    service = StudyResponseService(db)
    
    # Get study details
    from app.services import study as study_service
    study = study_service.get_study_basic_details_public(db, study_id)
    if not study:
        raise HTTPException(
            status_code=404,
            detail="Study not found"
        )
    
    # Get classification questions for this study
    from app.models.study_model import StudyClassificationQuestion
    classification_questions = db.execute(
        select(StudyClassificationQuestion)
        .where(StudyClassificationQuestion.study_id == study_id)
        .order_by(StudyClassificationQuestion.order)
    ).scalars().all()
    
    # Get tasks assigned to this specific respondent (tasks[respondent_id])
    respondent_tasks = service.get_respondent_tasks(study_id, respondent_id)

    # Build lightweight metadata: tasks_per_consumer, respondents target, background image url
    from app.models.study_model import Study as StudyModel
    meta_row = db.execute(
        select(StudyModel.background_image_url, StudyModel.audience_segmentation)
        .where(StudyModel.id == study_id)
    ).first()
    background_image_url = None
    respondents_target = 0
    if meta_row:
        background_image_url = meta_row.background_image_url
        try:
            seg = meta_row.audience_segmentation or {}
            respondents_target = int(seg.get('number_of_respondents') or 0)
        except Exception:
            respondents_target = 0
    tasks_per_consumer = len(respondent_tasks or [])

    return {
        "respondent_id": respondent_id,
        "study_id": str(study_id),
        "study_info": {
            "id": str(study["id"]),
            "title": study["title"],
            "study_type": study["study_type"],
            "main_question": study["main_question"],
            "orientation_text": study["orientation_text"],
            "rating_scale": study["rating_scale"],
            "language": study["language"]
        },
        "metadata": {
            "tasks_per_consumer": tasks_per_consumer,
            "number_of_respondents": respondents_target,
            "background_image_url": background_image_url,
        },
        "classification_questions": [
            {
                "question_id": q.question_id,
                "question_text": q.question_text,
                "question_type": q.question_type,
                "answer_options": q.answer_options,
                "order": q.order,
                "is_required": q.is_required
            }
            for q in classification_questions
        ],
        "assigned_tasks": respondent_tasks
    }