# app/services/response.py
from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID, uuid4
from datetime import datetime, timedelta

from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, func, desc, and_, or_
from fastapi import HTTPException, status

from app.models.response_model import (
    StudyResponse, CompletedTask, ClassificationAnswer, 
    ElementInteraction, TaskSession
)
from app.models.study_model import Study
from app.schemas.response_schema import (
    StudyResponseCreate, StudyResponseUpdate, StudyResponseOut,
    CompletedTaskCreate, ClassificationAnswerCreate,
    ElementInteractionCreate, TaskSessionCreate,
    StartStudyRequest, StartStudyResponse, SubmitTaskRequest,
    SubmitTaskResponse, SubmitClassificationRequest,
    AbandonStudyRequest, StudyAnalytics, ResponseAnalytics
)

# ---------- Study Response Service ----------

class StudyResponseService:
    """Service for managing study responses and participant sessions."""
    
    def __init__(self, db: Session):
        self.db = db
    
    # ---------- Core CRUD Operations ----------
    
    def create_response(self, response_data: StudyResponseCreate) -> StudyResponse:
        """Create a new study response session."""
        # Validate study exists and is active
        study = self.db.get(Study, response_data.study_id)
        if not study:
            raise HTTPException(
                status_code=404, 
                detail="Study not found"
            )
        
        if study.status != 'active':
            raise HTTPException(
                status_code=400,
                detail="Study is not active"
            )
        
        # Generate unique session ID
        session_id = f"session_{uuid4().hex[:16]}"
        
        # Get next available respondent ID
        respondent_id = self._get_next_respondent_id(response_data.study_id)
        
        # Derive total tasks assigned for this respondent
        total_tasks_assigned = 0
        try:
            if isinstance(study.tasks, list):
                # Flat list of tasks
                total_tasks_assigned = len(study.tasks)
            elif isinstance(study.tasks, dict):
                # Either per-respondent list OR index-keyed dict ("0","1",...)
                per_resp = study.tasks.get(str(respondent_id))
                if isinstance(per_resp, list):
                    total_tasks_assigned = len(per_resp)
                else:
                    # If values are task dicts keyed by numeric strings, count keys
                    # e.g., {"0": {...}, "1": {...}, ...}
                    total_tasks_assigned = len(study.tasks)
        except Exception:
            total_tasks_assigned = 0

        # Create response
        response = StudyResponse(
            study_id=response_data.study_id,
            session_id=session_id,
            respondent_id=respondent_id,
            total_tasks_assigned=total_tasks_assigned,
            session_start_time=response_data.session_start_time or datetime.utcnow(),
            personal_info=response_data.personal_info,
            ip_address=response_data.ip_address,
            user_agent=response_data.user_agent,
            browser_info=response_data.browser_info,
            last_activity=datetime.utcnow()
        )
        
        self.db.add(response)
        self.db.commit()
        self.db.refresh(response)
        
        # Update study response counters
        self._update_study_counters(response_data.study_id)
        
        return response
    
    def get_response(self, response_id: UUID) -> Optional[StudyResponse]:
        """Get a study response by ID."""
        return self.db.get(StudyResponse, response_id)
    
    def get_response_by_session(self, session_id: str) -> Optional[StudyResponse]:
        """Get a study response by session ID."""
        stmt = select(StudyResponse).where(StudyResponse.session_id == session_id)
        return self.db.execute(stmt).scalar_one_or_none()
    
    def get_responses_by_study(self, study_id: UUID, limit: int = 100, offset: int = 0) -> List[StudyResponse]:
        """Get all responses for a study."""
        stmt = (
            select(StudyResponse)
            .where(StudyResponse.study_id == study_id)
            .order_by(desc(StudyResponse.created_at))
            .limit(limit)
            .offset(offset)
        )
        return list(self.db.execute(stmt).scalars().all())
    
    def update_response(self, response_id: UUID, update_data: StudyResponseUpdate) -> Optional[StudyResponse]:
        """Update a study response."""
        response = self.get_response(response_id)
        if not response:
            return None
        
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(response, field, value)
        
        response.updated_at = datetime.utcnow()
        response.last_activity = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(response)
        return response
    
    def delete_response(self, response_id: UUID) -> bool:
        """Delete a study response."""
        response = self.get_response(response_id)
        if not response:
            return False
        
        self.db.delete(response)
        self.db.commit()
        return True
    
    # ---------- Study Participation Flow ----------
    
    def start_study(self, request: StartStudyRequest, ip_address: str = None, user_agent: str = None) -> StartStudyResponse:
        """Start a new study session for a participant."""
        # Validate study exists and is active
        study = self.db.get(Study, request.study_id)
        if not study:
            raise HTTPException(status_code=404, detail="Study not found")
        
        if study.status != 'active':
            raise HTTPException(status_code=400, detail="Study is not active")
        
        # Check if study has tasks
        if not study.tasks or len(study.tasks) == 0:
            raise HTTPException(status_code=400, detail="Study has no tasks")
        
        # Merge personal_info and user_details
        combined_personal_info = {}
        if request.personal_info:
            combined_personal_info.update(request.personal_info)
        if request.user_details:
            # Convert user_details to dict and merge
            user_details_dict = request.user_details.model_dump(exclude_none=True)
            combined_personal_info.update(user_details_dict)
        
        # Create response session
        response_data = StudyResponseCreate(
            study_id=request.study_id,
            session_start_time=datetime.utcnow(),
            personal_info=combined_personal_info if combined_personal_info else None,
            ip_address=ip_address,
            user_agent=user_agent,
            total_tasks_assigned=len(study.tasks)
        )
        
        response = self.create_response(response_data)
        
        return StartStudyResponse(
            session_id=response.session_id,
            respondent_id=response.respondent_id,
            total_tasks_assigned=response.total_tasks_assigned,
            study_info={
                "id": str(study.id),
                "title": study.title,
                "study_type": study.study_type,
                "main_question": study.main_question,
                "orientation_text": study.orientation_text,
                "rating_scale": study.rating_scale
            }
        )
    
    def submit_task(self, session_id: str, request: SubmitTaskRequest) -> SubmitTaskResponse:
        """Submit a completed task."""
        response = self.get_response_by_session(session_id)
        if not response:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if response.is_completed:
            raise HTTPException(status_code=400, detail="Study already completed")
        
        # Create completed task record
        task_data = CompletedTaskCreate(
            task_id=request.task_id,
            respondent_id=response.respondent_id,
            task_index=response.current_task_index,
            elements_shown_in_task={},  # Will be populated from study.tasks below
            task_start_time=datetime.utcnow() - timedelta(seconds=request.task_duration_seconds),
            task_completion_time=datetime.utcnow(),
            task_duration_seconds=request.task_duration_seconds,
            rating_given=request.rating_given,
            rating_timestamp=datetime.utcnow()
        )
        
        completed_task = CompletedTask(**task_data.model_dump())
        completed_task.study_response_id = response.id
        
        # Populate elements_shown/task metadata from study.tasks
        study: Study = self.db.get(Study, response.study_id)
        try:
            elements_shown_in_task: Dict[str, Any] = {}
            elements_shown_content: Optional[Dict[str, Any]] = None
            task_context: Optional[Dict[str, Any]] = None
            
            if study and isinstance(study.tasks, dict):
                respondent_key = str(response.respondent_id)
                tasks_for_respondent = study.tasks.get(respondent_key)
                if isinstance(tasks_for_respondent, list):
                    if 0 <= response.current_task_index < len(tasks_for_respondent):
                        task_def = tasks_for_respondent[response.current_task_index] or {}
                    else:
                        task_def = {}
                else:
                    # Index-keyed dict format: tasks_for_respondent is None, tasks are at top level as {"0": {...}, ...}
                    task_def = study.tasks.get(str(response.current_task_index), {}) or {}

                # Attempt multiple possible field names
                elements_shown_in_task = (
                    task_def.get("elements_shown_in_task")
                    or task_def.get("elements_shown")
                    or {}
                )
                elements_shown_content = task_def.get("elements_shown_content")
                task_context = {
                    "main_question": study.main_question,
                    "orientation_text": study.orientation_text,
                    "rating_scale": study.rating_scale,
                    "generated_task_id": task_def.get("task_id"),
                }
            
            completed_task.elements_shown_in_task = elements_shown_in_task or {}
            if elements_shown_content is not None:
                completed_task.elements_shown_content = elements_shown_content
            # Helpful duplicates for backward compatibility
            if not getattr(completed_task, "elements_shown", None):
                completed_task.elements_shown = elements_shown_in_task or None
            if study and study.study_type == 'layer' and not getattr(completed_task, "layers_shown_in_task", None):
                completed_task.layers_shown_in_task = elements_shown_content or None
            
            # Set task metadata
            if study:
                completed_task.task_type = study.study_type
            if task_context:
                completed_task.task_context = task_context
        except Exception:
            # Best-effort population; proceed even if tasks structure is unexpected
            pass
        
        self.db.add(completed_task)
        
        # Update response progress
        response.completed_tasks_count += 1
        response.current_task_index += 1
        response.completion_percentage = (response.completed_tasks_count / response.total_tasks_assigned) * 100.0
        response.last_activity = datetime.utcnow()
        
        # Add element interactions if provided
        if request.element_interactions:
            for interaction_data in request.element_interactions:
                interaction = ElementInteraction(**interaction_data.model_dump())
                interaction.study_response_id = response.id
                self.db.add(interaction)
        
        # Check if study is complete (after this submission)
        is_complete = response.current_task_index >= (response.total_tasks_assigned or 0)
        if is_complete:
            self._mark_response_completed(response)
        
        self.db.commit()
        self.db.refresh(response)
        
        return SubmitTaskResponse(
            success=True,
            next_task_index=response.current_task_index if not is_complete else None,
            is_study_complete=is_complete,
            completion_percentage=response.completion_percentage
        )
    
    def submit_classification(self, session_id: str, request: SubmitClassificationRequest) -> bool:
        """Submit classification answers."""
        response = self.get_response_by_session(session_id)
        if not response:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Add classification answers
        for answer_data in request.answers:
            answer = ClassificationAnswer(**answer_data.model_dump())
            answer.study_response_id = response.id
            self.db.add(answer)
        
        response.last_activity = datetime.utcnow()
        self.db.commit()
        
        return True
    
    def abandon_study(self, session_id: str, request: AbandonStudyRequest) -> bool:
        """Mark a study response as abandoned."""
        response = self.get_response_by_session(session_id)
        if not response:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if response.is_completed:
            raise HTTPException(status_code=400, detail="Study already completed")
        
        response.is_abandoned = True
        response.is_completed = False
        response.abandonment_timestamp = datetime.utcnow()
        response.abandonment_reason = request.reason
        response.last_activity = datetime.utcnow()
        
        self.db.commit()
        
        # Update study counters
        self._update_study_counters(response.study_id)
        
        return True
    
    def update_user_details(self, session_id: str, user_details: Dict[str, Any]) -> bool:
        """Update user details for a study session."""
        response = self.get_response_by_session(session_id)
        if not response:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if response.is_completed:
            raise HTTPException(status_code=400, detail="Cannot update details for completed study")
        
        # Merge with existing personal_info
        existing_info = response.personal_info or {}
        existing_info.update(user_details)
        
        response.personal_info = existing_info
        response.last_activity = datetime.utcnow()
        
        self.db.commit()
        return True
    
    # ---------- Analytics and Reporting ----------
    
    def get_study_analytics(self, study_id: UUID) -> StudyAnalytics:
        """Get analytics data for a study."""
        # Get response counts
        total_responses = self.db.execute(
            select(func.count(StudyResponse.id))
            .where(StudyResponse.study_id == study_id)
        ).scalar()
        
        completed_responses = self.db.execute(
            select(func.count(StudyResponse.id))
            .where(and_(
                StudyResponse.study_id == study_id,
                StudyResponse.is_completed == True
            ))
        ).scalar()
        
        abandoned_responses = self.db.execute(
            select(func.count(StudyResponse.id))
            .where(and_(
                StudyResponse.study_id == study_id,
                StudyResponse.is_abandoned == True
            ))
        ).scalar()
        
        # Calculate rates
        completion_rate = (completed_responses / total_responses * 100) if total_responses > 0 else 0
        abandonment_rate = (abandoned_responses / total_responses * 100) if total_responses > 0 else 0
        
        # Get average duration
        avg_duration = self.db.execute(
            select(func.avg(StudyResponse.total_study_duration))
            .where(and_(
                StudyResponse.study_id == study_id,
                StudyResponse.is_completed == True
            ))
        ).scalar() or 0
        
        # Get element heatmap (simplified)
        element_heatmap = self._get_element_heatmap(study_id)
        
        # Get timing distributions
        timing_distributions = self._get_timing_distributions(study_id)
        
        return StudyAnalytics(
            total_responses=total_responses,
            completed_responses=completed_responses,
            abandoned_responses=abandoned_responses,
            completion_rate=completion_rate,
            average_duration=avg_duration,
            abandonment_rate=abandonment_rate,
            element_heatmap=element_heatmap,
            timing_distributions=timing_distributions
        )
    
    def get_response_analytics(self, response_id: UUID) -> Optional[ResponseAnalytics]:
        """Get detailed analytics for a specific response."""
        response = self.get_response(response_id)
        if not response:
            return None
        
        # Get completed tasks with interactions
        completed_tasks = self.db.execute(
            select(CompletedTask)
            .where(CompletedTask.study_response_id == response_id)
            .order_by(CompletedTask.task_index)
        ).scalars().all()
        
        # Get element interactions
        element_interactions = self.db.execute(
            select(ElementInteraction)
            .where(ElementInteraction.study_response_id == response_id)
        ).scalars().all()
        
        # Build task analytics
        task_analytics = []
        for task in completed_tasks:
            task_analytics.append({
                "task_id": task.task_id,
                "task_index": task.task_index,
                "rating_given": task.rating_given,
                "duration_seconds": task.task_duration_seconds,
                "completion_time": task.task_completion_time.isoformat()
            })
        
        # Abandonment analysis
        abandonment_analysis = None
        if response.is_abandoned:
            abandonment_analysis = {
                "timestamp": response.abandonment_timestamp.isoformat() if response.abandonment_timestamp else None,
                "reason": response.abandonment_reason,
                "completion_percentage": response.completion_percentage,
                "tasks_completed": response.completed_tasks_count
            }
        
        return ResponseAnalytics(
            response_id=response.id,
            session_id=response.session_id,
            completion_percentage=response.completion_percentage,
            total_duration=response.total_study_duration,
            task_analytics=task_analytics,
            element_interactions=[ElementInteractionOut.model_validate(interaction) for interaction in element_interactions],
            abandonment_analysis=abandonment_analysis
        )
    
    # ---------- Helper Methods ----------
    
    def _get_next_respondent_id(self, study_id: UUID) -> int:
        """Get the next available respondent ID for a study."""
        max_respondent = self.db.execute(
            select(func.max(StudyResponse.respondent_id))
            .where(StudyResponse.study_id == study_id)
        ).scalar()
        
        return (max_respondent or 0) + 1
    
    def _mark_response_completed(self, response: StudyResponse) -> None:
        """Mark a response as completed, handling timezone-aware timestamps."""
        from datetime import timezone
        response.is_completed = True
        response.is_abandoned = False
        end_time = datetime.now(timezone.utc)
        response.session_end_time = end_time
        response.completion_percentage = 100.0
        
        if response.session_start_time:
            start_time = response.session_start_time
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            response.total_study_duration = (end_time - start_time).total_seconds()
    
    def _update_study_counters(self, study_id: UUID) -> None:
        """Update study response counters."""
        study = self.db.get(Study, study_id)
        if not study:
            return
        
        # Count responses
        total_responses = self.db.execute(
            select(func.count(StudyResponse.id))
            .where(StudyResponse.study_id == study_id)
        ).scalar()
        
        completed_responses = self.db.execute(
            select(func.count(StudyResponse.id))
            .where(and_(
                StudyResponse.study_id == study_id,
                StudyResponse.is_completed == True
            ))
        ).scalar()
        
        abandoned_responses = self.db.execute(
            select(func.count(StudyResponse.id))
            .where(and_(
                StudyResponse.study_id == study_id,
                StudyResponse.is_abandoned == True
            ))
        ).scalar()
        
        # Update study
        study.total_responses = total_responses
        study.completed_responses = completed_responses
        study.abandoned_responses = abandoned_responses
        
        self.db.commit()
    
    def _get_element_heatmap(self, study_id: UUID) -> Dict[str, Any]:
        """Get element interaction heatmap data."""
        # Simplified implementation - you can expand this
        interactions = self.db.execute(
            select(ElementInteraction)
            .join(StudyResponse)
            .where(StudyResponse.study_id == study_id)
        ).scalars().all()
        
        heatmap = {}
        for interaction in interactions:
            element_id = interaction.element_id
            if element_id not in heatmap:
                heatmap[element_id] = {
                    "total_view_time": 0.0,
                    "total_hovers": 0,
                    "total_clicks": 0,
                    "interaction_count": 0
                }
            
            heatmap[element_id]["total_view_time"] += interaction.view_time_seconds
            heatmap[element_id]["total_hovers"] += interaction.hover_count
            heatmap[element_id]["total_clicks"] += interaction.click_count
            heatmap[element_id]["interaction_count"] += 1
        
        return heatmap
    
    def _get_timing_distributions(self, study_id: UUID) -> Dict[str, Any]:
        """Get timing distribution data."""
        # Get task durations
        durations = self.db.execute(
            select(CompletedTask.task_duration_seconds)
            .join(StudyResponse)
            .where(StudyResponse.study_id == study_id)
        ).scalars().all()
        
        if not durations:
            return {"task_durations": [], "average_duration": 0}
        
        return {
            "task_durations": list(durations),
            "average_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations)
        }

# ---------- Task Session Service ----------

class TaskSessionService:
    """Service for managing individual task sessions."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_task_session(self, session_data: TaskSessionCreate) -> TaskSession:
        """Create a new task session."""
        # Resolve study_response_id from the provided session_id
        response = self.db.execute(
            select(StudyResponse).where(StudyResponse.session_id == session_data.session_id)
        ).scalar_one_or_none()
        if not response:
            raise HTTPException(status_code=404, detail="Session not found")

        task_session = TaskSession(**session_data.model_dump())
        task_session.study_response_id = response.id

        self.db.add(task_session)
        self.db.commit()
        self.db.refresh(task_session)
        return task_session
    
    def get_task_session(self, session_id: str, task_id: str) -> Optional[TaskSession]:
        """Get a task session by session ID and task ID."""
        stmt = select(TaskSession).where(
            and_(
                TaskSession.session_id == session_id,
                TaskSession.task_id == task_id
            )
        )
        return self.db.execute(stmt).scalar_one_or_none()
    
    def update_task_session(self, session_id: str, task_id: str, update_data: Dict[str, Any]) -> Optional[TaskSession]:
        """Update a task session."""
        task_session = self.get_task_session(session_id, task_id)
        if not task_session:
            return None
        
        for field, value in update_data.items():
            if hasattr(task_session, field):
                setattr(task_session, field, value)
        
        task_session.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(task_session)
        return task_session
    
    def add_page_transition(self, session_id: str, task_id: str, page_name: str) -> bool:
        """Add a page transition to a task session."""
        task_session = self.get_task_session(session_id, task_id)
        if not task_session:
            return False
        
        if not task_session.page_transitions:
            task_session.page_transitions = []
        
        transition = {
            "page": page_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        task_session.page_transitions.append(transition)
        task_session.updated_at = datetime.utcnow()
        
        self.db.commit()
        return True
    
    def add_element_interaction(self, session_id: str, task_id: str, interaction_data: ElementInteractionCreate) -> bool:
        """Add an element interaction to a task session."""
        task_session = self.get_task_session(session_id, task_id)
        if not task_session:
            return False
        
        interaction = ElementInteraction(**interaction_data.model_dump())
        interaction.task_session_id = task_session.id
        
        self.db.add(interaction)
        self.db.commit()
        return True