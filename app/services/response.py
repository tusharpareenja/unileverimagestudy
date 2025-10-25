# app/services/response.py
from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple, Iterable
from uuid import UUID, uuid4
from datetime import datetime, timedelta
import time

from sqlalchemy.orm import Session, selectinload, load_only
from sqlalchemy import select, func, desc, and_, or_
from fastapi import HTTPException, status

from app.models.response_model import (
    StudyResponse, CompletedTask, ClassificationAnswer, 
    ElementInteraction, TaskSession
)
from app.models.study_model import Study, StudyClassificationQuestion, StudyElement
from app.schemas.response_schema import (
    StudyResponseCreate, StudyResponseUpdate, StudyResponseOut,
    CompletedTaskCreate, ClassificationAnswerCreate,
    ElementInteractionCreate, TaskSessionCreate,
    StartStudyRequest, StartStudyResponse, SubmitTaskRequest,
    SubmitTaskResponse, SubmitClassificationRequest, BulkSubmitTasksRequest, BulkSubmitTasksResponse,
    AbandonStudyRequest, StudyAnalytics, ResponseAnalytics
)

# ---------- Study Response Service ----------

class StudyResponseService:
    """Service for managing study responses and participant sessions."""
    
    def __init__(self, db: Session):
        self.db = db
        # Lightweight in-process cache for analytics to accelerate repeated GETs
        # Keyed by study_id; values are (expires_at_epoch_seconds, StudyAnalytics)
        if not hasattr(StudyResponseService, "_analytics_cache"):
            StudyResponseService._analytics_cache: Dict[str, Tuple[float, Any]] = {}
        if not hasattr(StudyResponseService, "_analytics_ttl_seconds"):
            StudyResponseService._analytics_ttl_seconds: int = 15
    
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
            last_activity=datetime.utcnow(),
            status='in_progress',  # Set status as in_progress when session starts
            is_abandoned=False,    # Set to False initially, will be True after 15 minutes if no activity
            is_completed=False     # Set to False initially
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
    
    def get_response_by_respondent_id(self, respondent_id: int) -> Optional[StudyResponse]:
        """Get a study response by respondent ID."""
        stmt = select(StudyResponse).where(StudyResponse.respondent_id == respondent_id)
        return self.db.execute(stmt).scalar_one_or_none()
    
    def get_response_by_respondent_and_study(self, respondent_id: int, study_id: UUID) -> Optional[StudyResponse]:
        """Get a study response by respondent ID and study ID."""
        stmt = select(StudyResponse).where(
            and_(
                StudyResponse.respondent_id == respondent_id,
                StudyResponse.study_id == study_id
            )
        )
        return self.db.execute(stmt).scalar_one_or_none()
    
    def get_respondent_ids_for_study(self, study_id: UUID) -> List[int]:
        """Get all respondent IDs for a specific study."""
        stmt = select(StudyResponse.respondent_id).where(StudyResponse.study_id == study_id)
        results = self.db.execute(stmt).scalars().all()
        return list(results)
    
    def get_respondent_tasks(self, study_id: UUID, respondent_id: int) -> List[Dict[str, Any]]:
        """Get tasks assigned to a specific respondent."""
        # Get the study to access tasks
        study = self.db.get(Study, study_id)
        if not study or not study.tasks:
            return []
        
        # Handle different task structures
        if isinstance(study.tasks, list):
            # Flat list of tasks - return all tasks
            return study.tasks
        elif isinstance(study.tasks, dict):
            # Per-respondent tasks or index-keyed tasks
            respondent_key = str(respondent_id)
            if respondent_key in study.tasks:
                tasks = study.tasks[respondent_key]
                if isinstance(tasks, list):
                    return tasks
                else:
                    return [tasks] if tasks else []
            else:
                # Fallback: return tasks keyed by numeric strings (0, 1, 2, ...)
                numeric_tasks = []
                for key, value in study.tasks.items():
                    if key.isdigit():
                        numeric_tasks.append(value)
                return numeric_tasks
        
        return []
    
    def get_response_detail_by_session(self, session_id: str) -> Optional[StudyResponse]:
        """Get a study response by session ID with related details, optimized for speed."""
        stmt = (
            select(StudyResponse)
            .where(StudyResponse.session_id == session_id)
            .options(
                load_only(
                    StudyResponse.id,
                    StudyResponse.study_id,
                    StudyResponse.session_id,
                    StudyResponse.respondent_id,
                    StudyResponse.current_task_index,
                    StudyResponse.completed_tasks_count,
                    StudyResponse.total_tasks_assigned,
                    StudyResponse.session_start_time,
                    StudyResponse.session_end_time,
                    StudyResponse.is_completed,
                    StudyResponse.personal_info,
                    StudyResponse.ip_address,
                    StudyResponse.user_agent,
                    StudyResponse.browser_info,
                    StudyResponse.completion_percentage,
                    StudyResponse.total_study_duration,
                    StudyResponse.last_activity,
                    StudyResponse.is_abandoned,
                    StudyResponse.abandonment_timestamp,
                    StudyResponse.abandonment_reason,
                    StudyResponse.created_at,
                    StudyResponse.updated_at,
                ),
                selectinload(StudyResponse.completed_tasks).load_only(
                    CompletedTask.id,
                    CompletedTask.study_response_id,
                    CompletedTask.task_id,
                    CompletedTask.respondent_id,
                    CompletedTask.task_index,
                    CompletedTask.elements_shown_in_task,
                    CompletedTask.elements_shown_content,
                    CompletedTask.layers_shown_in_task,
                    CompletedTask.task_type,
                    CompletedTask.task_context,
                    CompletedTask.task_start_time,
                    CompletedTask.task_completion_time,
                    CompletedTask.task_duration_seconds,
                    CompletedTask.rating_given,
                    CompletedTask.rating_timestamp,
                ),
                selectinload(StudyResponse.classification_answers).load_only(
                    ClassificationAnswer.id,
                    ClassificationAnswer.study_response_id,
                    ClassificationAnswer.question_id,
                    ClassificationAnswer.question_text,
                    ClassificationAnswer.answer,
                    ClassificationAnswer.answer_timestamp,
                    ClassificationAnswer.time_spent_seconds,
                ),
                selectinload(StudyResponse.element_interactions).load_only(
                    ElementInteraction.id,
                    ElementInteraction.study_response_id,
                    ElementInteraction.task_session_id,
                    ElementInteraction.element_id,
                    ElementInteraction.view_time_seconds,
                    ElementInteraction.hover_count,
                    ElementInteraction.click_count,
                    ElementInteraction.first_view_time,
                    ElementInteraction.last_view_time,
                ),
                selectinload(StudyResponse.task_sessions).load_only(
                    TaskSession.id,
                    TaskSession.study_response_id,
                    TaskSession.session_id,
                    TaskSession.task_id,
                    TaskSession.classification_page_time,
                    TaskSession.orientation_page_time,
                    TaskSession.individual_task_page_times,
                    TaskSession.page_transitions,
                    TaskSession.is_completed,
                    TaskSession.abandonment_timestamp,
                    TaskSession.abandonment_reason,
                    TaskSession.recovery_attempts,
                    TaskSession.browser_performance,
                    TaskSession.page_load_times,
                    TaskSession.device_info,
                    TaskSession.screen_resolution,
                    TaskSession.created_at,
                    TaskSession.updated_at,
                ),
            )
        )
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

    def get_responses_by_study_filtered(
        self,
        study_id: UUID,
        limit: int = 100,
        offset: int = 0,
        is_completed: Optional[bool] = None,
        is_abandoned: Optional[bool] = None,
    ) -> List[StudyResponse]:
        """Get responses for a study with optional SQL-level completion/abandon filters."""
        conditions = [StudyResponse.study_id == study_id]
        if is_completed is not None:
            conditions.append(StudyResponse.is_completed == is_completed)
        if is_abandoned is not None:
            conditions.append(StudyResponse.is_abandoned == is_abandoned)

        stmt = (
            select(StudyResponse)
            .where(and_(*conditions))
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
        """Ultra-fast start study - minimal DB operations for instant response."""
        # Single query to get only essential study data
        study_row = self.db.execute(
            select(Study.status, Study.title, Study.study_type, Study.main_question, Study.orientation_text, Study.rating_scale, Study.tasks)
            .where(Study.id == request.study_id)
        ).first()
        
        if not study_row:
            raise HTTPException(status_code=404, detail="Study not found")
        
        if study_row.status != 'active':
            raise HTTPException(status_code=400, detail="Study is not active")
        
        if not study_row.tasks or len(study_row.tasks) == 0:
            raise HTTPException(status_code=400, detail="Study has no tasks")
        
        # Generate IDs - session_id without DB query, respondent_id with proper sequential logic
        from uuid import uuid4 as _uuid4
        session_id = f"session_{_uuid4().hex[:16]}"
        respondent_id = self._get_next_respondent_id(request.study_id)
        
        # Fast task count calculation - get tasks for this specific respondent
        total_tasks_assigned = self._calculate_respondent_task_count(study_row.tasks, respondent_id)
        
        # Minimal personal info merge
        combined_personal_info = None
        if request.personal_info:
            combined_personal_info = request.personal_info
        elif request.user_details:
            combined_personal_info = request.user_details.model_dump(exclude_none=True)
        
        now_utc = datetime.utcnow()
        
        # Create response with minimal fields
        new_response = StudyResponse(
            study_id=request.study_id,
            session_id=session_id,
            respondent_id=respondent_id,
            total_tasks_assigned=total_tasks_assigned,
            session_start_time=now_utc,
            personal_info=combined_personal_info,
            ip_address=ip_address,
            user_agent=user_agent,
            last_activity=now_utc,
            status='in_progress',
            is_abandoned=False,
            is_completed=False
        )
        
        # Single database operation
        self.db.add(new_response)
        self.db.commit()
        
        # Return immediately without refresh or counter updates
        return StartStudyResponse(
            session_id=session_id,
            respondent_id=respondent_id,
            total_tasks_assigned=total_tasks_assigned,
            study_info={
                "id": str(request.study_id),
                "title": study_row.title,
                "study_type": study_row.study_type,
                "main_question": study_row.main_question,
                "orientation_text": study_row.orientation_text,
                "rating_scale": study_row.rating_scale
            }
        )
    
    def submit_task(self, session_id: str, request: SubmitTaskRequest) -> SubmitTaskResponse:
        """Submit a completed task."""
        response = self.get_response_by_session(session_id)
        if not response:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if response.is_completed:
            raise HTTPException(status_code=400, detail="Study already completed")
        
        # Create completed task record (fast path - no heavy enrichment)
        now_utc = datetime.utcnow()
        # Accept visibility payload from either elements_shown_in_task or elements_shown
        try:
            raw_visibility = (
                getattr(request, 'elements_shown_in_task', None)
                or getattr(request, 'elements_shown', None)
                or {}
            )
        except Exception:
            raw_visibility = {}

        # Build a name-keyed visibility map for grid studies
        name_visibility: Dict[str, Any] = {}
        study_row: Optional[Study] = self.db.get(Study, response.study_id)
        if study_row and str(study_row.study_type) == 'grid':
            # Load element id->name map in E-number order
            elements = self.db.execute(
                select(StudyElement).where(StudyElement.study_id == study_row.id)
            ).scalars().all()
            id_to_name: Dict[str, str] = {str(e.element_id).upper(): e.name for e in elements}

            # If client didn't send anything, try fallback from study.tasks
            payload = raw_visibility
            if not isinstance(payload, dict) or len(payload) == 0:
                try:
                    if isinstance(study_row.tasks, dict):
                        rk = str(response.respondent_id)
                        tasks_for_r = study_row.tasks.get(rk)
                        if isinstance(tasks_for_r, list) and 0 <= response.current_task_index < len(tasks_for_r):
                            task_def = tasks_for_r[response.current_task_index] or {}
                        else:
                            task_def = study_row.tasks.get(str(response.current_task_index), {}) or {}
                        payload = task_def.get('elements_shown_in_task') or task_def.get('elements_shown') or {}
                except Exception:
                    payload = {}

            # If payload is a JSON string, parse
            if isinstance(payload, str):
                try:
                    import json
                    payload = json.loads(payload)
                except Exception:
                    payload = {}

            # Normalize to name->0/1
            if isinstance(payload, dict):
                for eid, name in id_to_name.items():
                    val = payload.get(eid)
                    if val is None:
                        # Try content-based or by name key
                        val = payload.get(f"{eid}_content") or payload.get(name) or payload.get(f"{name}_content")
                        if isinstance(val, str):
                            present = 1 if val.strip() else 0
                        else:
                            try:
                                present = int(val) if val in (0,1) else (1 if val else 0)
                            except Exception:
                                present = 0
                    else:
                        try:
                            present = int(val) if val in (0,1) else (1 if val else 0)
                        except Exception:
                            present = 0
                    name_visibility[name.replace(' ', '_')] = 1 if present == 1 else 0
        elif study_row and str(study_row.study_type) == 'layer':
            # For layer studies, include z-index information in elements_shown_in_task
            from app.models.study_model import StudyLayer, LayerImage
            
            # Get layer structure with z-index information
            layers = self.db.execute(
                select(StudyLayer)
                .where(StudyLayer.study_id == study_row.id)
                .order_by(StudyLayer.order)
            ).scalars().all()
            
            # Create layer name to z-index mapping
            layer_name_to_z_index: Dict[str, int] = {}
            for layer in layers:
                layer_name_to_z_index[layer.name] = layer.z_index
            
            # Process the raw visibility data to include z-index
            if isinstance(raw_visibility, dict):
                for key, value in raw_visibility.items():
                    if isinstance(key, str):
                        # Extract layer name from key (e.g., "background_1" -> "background")
                        import re
                        match = re.search(r"^(.+?)_(\d+)$", key)
                        if match:
                            layer_name = match.group(1)
                            image_num = match.group(2)
                            
                            # Get z-index for this layer
                            z_index = layer_name_to_z_index.get(layer_name, 0)
                            
                            # Create enhanced visibility data with z-index
                            if isinstance(value, dict):
                                # If already a dict, add z_index to it
                                enhanced_value = value.copy()
                                enhanced_value["z_index"] = z_index
                                name_visibility[key] = enhanced_value
                            else:
                                # If simple value, create dict with visibility and z_index
                                name_visibility[key] = {
                                    "visible": 1 if value else 0,
                                    "z_index": z_index
                                }
                        else:
                            # Fallback for keys that don't match pattern
                            name_visibility[key] = value
            else:
                # If not a dict, keep as-is
                name_visibility = raw_visibility if isinstance(raw_visibility, dict) else {}
        else:
            # Non-grid or no study: keep as-is if dict, else empty
            name_visibility = raw_visibility if isinstance(raw_visibility, dict) else {}
        task_data = CompletedTaskCreate(
            task_id=request.task_id,
            respondent_id=response.respondent_id,
            task_index=response.current_task_index,
            elements_shown_in_task=name_visibility,
            task_start_time=now_utc - timedelta(seconds=request.task_duration_seconds),
            task_completion_time=now_utc,
            task_duration_seconds=request.task_duration_seconds,
            rating_given=request.rating_given,
            rating_timestamp=now_utc
        )
        
        completed_task = CompletedTask(**task_data.model_dump())
        completed_task.study_response_id = response.id
        
        # For layer studies, persist which layers/images were shown for this task
        if study_row and str(study_row.study_type) == 'layer':
            layer_payload: Any = None
            # Try request fields first if present in future clients
            try:
                layer_payload = getattr(request, 'layers_shown_in_task', None) or getattr(request, 'elements_shown_content', None)
            except Exception:
                layer_payload = None
            # Fallback to task definition from study.tasks
            if layer_payload is None or layer_payload == {}:
                try:
                    task_def = None
                    if isinstance(study_row.tasks, dict):
                        rk = str(response.respondent_id)
                        tasks_for_r = study_row.tasks.get(rk)
                        if isinstance(tasks_for_r, list) and 0 <= response.current_task_index - 1 < len(tasks_for_r):
                            task_def = tasks_for_r[response.current_task_index - 1] or {}
                        else:
                            task_def = study_row.tasks.get(str(response.current_task_index - 1), {}) or {}
                    if isinstance(task_def, dict):
                        layer_payload = task_def.get('layers_shown_in_task') or task_def.get('elements_shown_content')
                except Exception:
                    layer_payload = None
            if layer_payload is not None:
                # Prefer saving to layers_shown_in_task, keep elements_shown_content for backward compatibility
                if completed_task.layers_shown_in_task is None:
                    completed_task.layers_shown_in_task = layer_payload
                if completed_task.elements_shown_content is None:
                    completed_task.elements_shown_content = layer_payload
        # For grid studies, if elements_shown_content is missing, hydrate from generated tasks
        if study_row and str(study_row.study_type) == 'grid':
            try:
                if completed_task.elements_shown_content is None and isinstance(study_row.tasks, dict):
                    # Resolve respondent and task index from task_id when possible
                    rid_from_task = None
                    idx_from_task = None
                    try:
                        parts = str(request.task_id).split('_')
                        if len(parts) == 2:
                            rid_from_task = int(parts[0])
                            idx_from_task = int(parts[1])
                    except Exception:
                        rid_from_task = None
                        idx_from_task = None

                    # Prefer exact respondent key; try 0-based then 1-based
                    candidate_keys = []
                    if rid_from_task is not None:
                        candidate_keys.extend([str(rid_from_task), str(max(0, rid_from_task - 1))])
                    candidate_keys.extend([str(response.respondent_id), str(max(0, (response.respondent_id or 0) - 1))])
                    tasks_for_r = None
                    for k in candidate_keys:
                        val = (study_row.tasks or {}).get(k)
                        if isinstance(val, list):
                            tasks_for_r = val
                            break
                    if isinstance(tasks_for_r, list):
                        # Use index from task_id when present; else current_task_index (1-based -> 0-based)
                        idx = idx_from_task if isinstance(idx_from_task, int) else max(0, (response.current_task_index or 1) - 1)
                        if 0 <= idx < len(tasks_for_r):
                            tdef = tasks_for_r[idx]
                            if isinstance(tdef, dict):
                                esc = tdef.get('elements_shown_content')
                                if esc:
                                    # Store name, url, visible per key; normalize key to CategoryName_ImageName
                                    out_map = {}
                                    vis_map = completed_task.elements_shown_in_task or {}
                                    for k, v in esc.items():
                                        if isinstance(v, dict):
                                            # Derive category name from key like "Category 1_4"
                                            cat_name = k.rsplit('_', 1)[0] if isinstance(k, str) and '_' in k else ''
                                            elem_name = v.get('name')
                                            new_key = f"{cat_name}_{elem_name}" if cat_name and elem_name else k
                                            out_map[new_key] = {
                                                "name": elem_name,
                                                "url": v.get('content'),
                                                "visible": 1 if vis_map.get(k) or vis_map.get(new_key) else 0,
                                            }
                                        else:
                                            out_map[k] = None
                                    completed_task.elements_shown_content = out_map
            except Exception:
                pass
        
        self.db.add(completed_task)
        
        # Update response progress (no extra reads)
        response.completed_tasks_count = (response.completed_tasks_count or 0) + 1
        response.current_task_index = (response.current_task_index or 0) + 1
        try:
            total_assigned = int(response.total_tasks_assigned or 0)
            done = int(response.completed_tasks_count or 0)
            response.completion_percentage = (done / total_assigned * 100.0) if total_assigned > 0 else 0.0
        except Exception:
            response.completion_percentage = 0.0
        response.last_activity = now_utc

        # Bulk insert element interactions if provided
        if request.element_interactions:
            interactions = []
            for interaction_data in request.element_interactions:
                interaction = ElementInteraction(**interaction_data.model_dump())
                interaction.study_response_id = response.id
                interactions.append(interaction)
            if interactions:
                self.db.bulk_save_objects(interactions)
        
        # Mark complete if done
        is_complete = response.current_task_index >= (response.total_tasks_assigned or 0)
        if is_complete:
            self._mark_response_completed(response)
            # Check if study should be auto-completed after this response completion
            self._check_and_complete_studies()
        
        # Single commit at end
        self.db.commit()
        
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

    def submit_tasks_bulk(self, session_id: str, request: BulkSubmitTasksRequest) -> BulkSubmitTasksResponse:
        """Submit multiple completed tasks in a single transaction (optimized for large payloads)."""
        response = self.get_response_by_session(session_id)
        if not response:
            raise HTTPException(status_code=404, detail="Session not found")
        if response.is_completed:
            raise HTTPException(status_code=400, detail="Study already completed")

        now_utc = datetime.utcnow()
        study_row: Optional[Study] = self.db.get(Study, response.study_id)

        # Preload element name->url once for grid enrichment
        name_to_url: Dict[str, str] = {}
        if study_row and str(study_row.study_type) == 'grid':
            elems = self.db.execute(
                select(StudyElement.name, StudyElement.content).where(StudyElement.study_id == response.study_id)
            ).all()
            name_to_url = {str(n).replace(' ', '_'): c for (n, c) in elems}

        tasks_to_insert: List[CompletedTask] = []
        interactions_to_insert: List[ElementInteraction] = []
        submitted = 0

        for item in request.tasks or []:
            # Build task model
            elements_shown_in_task = item.elements_shown_in_task or {}
            
            # For layer studies, enhance elements_shown_in_task with z-index information
            if study_row and str(study_row.study_type) == 'layer' and isinstance(elements_shown_in_task, dict):
                from app.models.study_model import StudyLayer, LayerImage
                
                # Get layer structure with z-index information
                layers = self.db.execute(
                    select(StudyLayer)
                    .where(StudyLayer.study_id == study_row.id)
                    .order_by(StudyLayer.order)
                ).scalars().all()
                
                # Create layer name to z-index mapping
                layer_name_to_z_index: Dict[str, int] = {}
                for layer in layers:
                    layer_name_to_z_index[layer.name] = layer.z_index
                
                # Enhance elements_shown_in_task with z-index information
                enhanced_elements_shown = {}
                for key, value in elements_shown_in_task.items():
                    if isinstance(key, str):
                        # Extract layer name from key (e.g., "background_1" -> "background")
                        import re
                        match = re.search(r"^(.+?)_(\d+)$", key)
                        if match:
                            layer_name = match.group(1)
                            image_num = match.group(2)
                            
                            # Get z-index for this layer
                            z_index = layer_name_to_z_index.get(layer_name, 0)
                            
                            # Create enhanced visibility data with z-index
                            if isinstance(value, dict):
                                # If already a dict, add z_index to it
                                enhanced_value = value.copy()
                                enhanced_value["z_index"] = z_index
                                enhanced_elements_shown[key] = enhanced_value
                            else:
                                # If simple value, create dict with visibility and z_index
                                enhanced_elements_shown[key] = {
                                    "visible": 1 if value else 0,
                                    "z_index": z_index
                                }
                        else:
                            # Fallback for keys that don't match pattern
                            enhanced_elements_shown[key] = value
                    else:
                        enhanced_elements_shown[key] = value
                
                elements_shown_in_task = enhanced_elements_shown
            
            task_model = CompletedTask(
                task_id=item.task_id,
                respondent_id=response.respondent_id,
                task_index=response.current_task_index,
                elements_shown_in_task=elements_shown_in_task,
                elements_shown_content=item.elements_shown_content or None,
                task_start_time=now_utc - timedelta(seconds=item.task_duration_seconds),
                task_completion_time=now_utc,
                task_duration_seconds=item.task_duration_seconds,
                rating_given=item.rating_given,
                rating_timestamp=now_utc,
                task_type=str(study_row.study_type) if study_row and study_row.study_type else None
            )
            task_model.study_response_id = response.id

            # Grid enrichment for content map if missing
            if study_row and str(study_row.study_type) == 'grid' and task_model.elements_shown_in_task and not task_model.elements_shown_content:
                enriched = None
                # Try: resolve from Study.tasks by task_id
                try:
                    rid = None
                    tidx = None
                    parts = str(item.task_id).split('_')
                    if len(parts) == 2:
                        rid = int(parts[0])
                        tidx = int(parts[1])
                    # Candidate respondent keys: exact, zero-based variant, session respondent ids
                    candidate_keys = [
                        str(rid), str(max(0, (rid or 0) - 1)),
                        str(response.respondent_id), str(max(0, (response.respondent_id or 0) - 1))
                    ]
                    tasks_for_r = None
                    for key in candidate_keys:
                        val = (study_row.tasks or {}).get(key)
                        if isinstance(val, list):
                            tasks_for_r = val
                            break
                    if isinstance(tasks_for_r, list) and isinstance(tidx, int) and 0 <= tidx < len(tasks_for_r):
                        tdef = tasks_for_r[tidx]
                        if isinstance(tdef, dict):
                            esc = tdef.get('elements_shown_content')
                            if esc:
                                out_map: Dict[str, Any] = {}
                                vis_map = task_model.elements_shown_in_task or {}
                                for k, v in esc.items():
                                    if isinstance(v, dict):
                                        # Normalize key to CategoryName_ImageName using category from k (CategoryName_Index)
                                        cat_name = k.rsplit('_', 1)[0] if isinstance(k, str) and '_' in k else ''
                                        elem_name = v.get('name')
                                        new_key = f"{cat_name}_{elem_name}" if cat_name and elem_name else k
                                        out_map[new_key] = {
                                            "name": elem_name,
                                            "url": v.get('content'),
                                            "visible": 1 if vis_map.get(k) or vis_map.get(new_key) else 0,
                                        }
                                    else:
                                        out_map[k] = None
                                enriched = out_map
                except Exception:
                    enriched = None
                # Fallback: use preloaded element name->url map, but we don't know element names from index; store minimal
                if not enriched and name_to_url:
                    out_map2: Dict[str, Any] = {}
                    for k, v in (task_model.elements_shown_in_task or {}).items():
                        try:
                            shown = int(v) == 1
                        except Exception:
                            shown = bool(v)
                        if shown:
                            url = name_to_url.get(str(k)) or name_to_url.get(str(k).replace(' ', '_'))
                            out_map2[str(k)] = {
                                "name": None,
                                "url": url,
                                "visible": 1,
                            }
                        else:
                            out_map2[str(k)] = {
                                "name": None,
                                "url": None,
                                "visible": 0,
                            }
                    enriched = out_map2
                if enriched:
                    task_model.elements_shown_content = enriched

            tasks_to_insert.append(task_model)

            # Queue interactions
            if item.element_interactions:
                for interaction_data in item.element_interactions:
                    interaction = ElementInteraction(**interaction_data.model_dump())
                    interaction.study_response_id = response.id
                    interactions_to_insert.append(interaction)

            # Update progress counters in-memory
            response.completed_tasks_count = (response.completed_tasks_count or 0) + 1
            response.current_task_index = (response.current_task_index or 0) + 1
            submitted += 1

            # Stop early if completed
            if response.current_task_index >= (response.total_tasks_assigned or 0):
                self._mark_response_completed(response)
                # Check if study should be auto-completed after this response completion
                self._check_and_complete_studies()
                break

        # Bulk insert
        if tasks_to_insert:
            self.db.bulk_save_objects(tasks_to_insert)
        if interactions_to_insert:
            self.db.bulk_save_objects(interactions_to_insert)

        # Finalize response progress
        total_assigned = int(response.total_tasks_assigned or 0)
        done = int(response.completed_tasks_count or 0)
        response.completion_percentage = (done / total_assigned * 100.0) if total_assigned > 0 else 0.0
        response.last_activity = now_utc

        self.db.commit()

        is_complete = bool(response.is_completed)
        return BulkSubmitTasksResponse(
            success=True,
            submitted_count=submitted,
            next_task_index=None if is_complete else response.current_task_index,
            is_study_complete=is_complete,
            completion_percentage=float(response.completion_percentage or 0.0)
        )

    # ---------- CSV Export ----------

    def generate_csv_rows_for_study(self, study_id: UUID) -> Iterable[List[Any]]:
        """
        Yield CSV rows for all responses in a study.

        Columns produced (order):
        - Panelist (session_id)
        - QQ1..QQN (classification answers in question order if present)
        - Gender
        - Age
        - Task (task index starting at 1)
        - Dynamic Layer_* columns (Layer_{taskIndex}_{layerIndex}) with 0/1 visibility
        - Rating
        - ResponseTime (seconds)

        Notes:
        - Age is derived from personal_info.dob or personal_info.date_of_birth (YYYY-MM-DD).
        - Gender comes from personal_info.gender if available.
        - Layer visibility is inferred from CompletedTask.elements_shown_content or layers_shown_in_task.
        """
        # Preload minimal study and responses
        study: Optional[Study] = self.db.get(Study, study_id)
        if not study:
            raise HTTPException(status_code=404, detail="Study not found")

        # Fetch responses (single query)
        responses: List[StudyResponse] = list(self.db.execute(
            select(StudyResponse.id, StudyResponse.session_id, StudyResponse.personal_info)
            .where(StudyResponse.study_id == study_id)
        ).all())
        # Early return if no responses
        if not responses:
            yield ["Panelist"]  # minimal header to keep CSV valid
            return
        # Index positions for lightweight tuples
        RESP_ID_IDX = 0
        RESP_SESSION_IDX = 1
        RESP_PI_IDX = 2
        response_ids = [r[RESP_ID_IDX] for r in responses]

        # Build question ordering from study configuration (preferred over discovered answers)
        # Map question_id -> column header text (use question_text) and keep answer option maps for code->text
        question_id_to_col: Dict[str, str] = {}
        question_id_to_options: Dict[str, Dict[str, str]] = {}
        questions = self.db.execute(
            select(StudyClassificationQuestion)
            .where(StudyClassificationQuestion.study_id == study_id)
            .order_by(StudyClassificationQuestion.order)
        ).scalars().all()
        next_q_num = 1
        for q in questions:
            # Use full question text; fallback to QQn
            question_id_to_col[q.question_id] = (q.question_text or f"QQ{next_q_num}")
            next_q_num += 1
            options_map: Dict[str, str] = {}
            if isinstance(q.answer_options, list):
                for opt in q.answer_options:
                    if not isinstance(opt, dict):
                        continue
                    text = opt.get("text") or opt.get("label") or opt.get("name")
                    if text is None:
                        continue
                    # Support multiple keys that might appear in stored answers
                    for key_name in ("id", "value", "code", "label"):
                        if key_name in opt and opt[key_name] is not None:
                            options_map[str(opt[key_name])] = text
            if options_map:
                question_id_to_options[q.question_id] = options_map

        # If there are answers for questions not in config, append them in discovery order (batched)
        if not question_id_to_col and response_ids:
            discovered = self.db.execute(
                select(ClassificationAnswer.question_id)
                .where(ClassificationAnswer.study_response_id.in_(response_ids))
            ).all()
            for (qid,) in discovered:
                if qid not in question_id_to_col:
                    question_id_to_col[qid] = f"QQ{next_q_num}"
                    next_q_num += 1

        # Build consistent ordered headers for all layer slots using layer name + image name
        import re
        def parse_slot(key: str) -> tuple[str, int]:
            # returns (layer_label, image_index)
            m = re.search(r"^(.*?)[ _-]?(\d+)$", key)
            if m:
                return (m.group(1).strip(), int(m.group(2)))
            return (key.strip(), 0)

        # Accumulate first-seen names per slot and per layer
        slot_to_image_name: Dict[tuple[str, int], str] = {}
        layer_to_display: Dict[str, str] = {}
        layer_to_max_image: Dict[str, int] = {}

        # Preload all tasks for the study and group by response (single query)
        tasks_seq = self.db.execute(
            select(CompletedTask)
            .where(CompletedTask.study_response_id.in_(response_ids))
            .order_by(CompletedTask.study_response_id, CompletedTask.task_index)
            ).scalars().all()
        from collections import defaultdict
        tasks_by_response: Dict[UUID, List[CompletedTask]] = defaultdict(list)
        for t in tasks_seq:
            tasks_by_response[t.study_response_id].append(t)

        for resp in responses:
            tasks = tasks_by_response.get(resp[RESP_ID_IDX], [])
            for t in tasks:
                data = t.layers_shown_in_task or t.elements_shown_content or {}
                if not isinstance(data, dict):
                    continue
                for k, v in data.items():
                    if not isinstance(k, str):
                        continue
                    layer_label, image_idx = parse_slot(k)
                    # Prefer explicit layer_name when present
                    if isinstance(v, dict):
                        lbl = (v.get("layer_name") or layer_label or "").strip()
                        if lbl:
                            layer_to_display[layer_label] = lbl
                        img = (v.get("name") or v.get("alt_text") or "").strip()
                        if img and (layer_label, image_idx) not in slot_to_image_name:
                            slot_to_image_name[(layer_label, image_idx)] = img
                    # track max index for each layer
                    current_max = layer_to_max_image.get(layer_label, 0)
                    if image_idx > current_max:
                        layer_to_max_image[layer_label] = image_idx

        # Order layers by numeric index found inside their label; fallback to alpha
        def layer_order_key(label: str) -> tuple[int, str]:
            m = re.search(r"(\d+)", label)
            return (int(m.group(1)) if m else 9999, label)

        ordered_layers = sorted(layer_to_max_image.keys(), key=layer_order_key)

        # Build final layer columns list in order - use actual layer names
        layer_headers: List[str] = []
        layer_slots: List[tuple[str, int]] = []
        
        # Get layer structure from database to create proper ordering
        from app.models.study_model import StudyLayer, LayerImage
        
        # Get layers ordered by their order field
        layers = self.db.execute(
            select(StudyLayer)
            .where(StudyLayer.study_id == study_id)
            .order_by(StudyLayer.order)
        ).scalars().all()
        
        # Create ordered list of layer names based on database order
        ordered_layer_names = [layer.name for layer in layers]
        
        # Process layers in database order
        for layer_name in ordered_layer_names:
            if layer_name in layer_to_max_image:
                max_idx = layer_to_max_image[layer_name]
                for idx in range(1, max_idx + 1):
                    # Use actual layer name with image index
                    header = f"{layer_name}_{idx}"
                    layer_headers.append(header)
                    layer_slots.append((layer_name, idx))

        # Build header (for grid: CategoryName_ImageName columns)
        header: List[str] = ["Panelist"]
        header.extend([question_id_to_col[qid] for qid in question_id_to_col])
        header.extend(["Gender", "Age", "Task"])
        # If study is grid, add CategoryName_ImageName columns; else add layer columns
        study_obj: Optional[Study] = self.db.get(Study, study_id)
        is_grid = bool(study_obj and str(study_obj.study_type) == 'grid')
        grid_columns: List[tuple[str, str, int]] = []  # (cat_name, element_name, one_based_index_within_cat)
        if is_grid:
            # Load categories with elements to build deterministic headers
            from app.models.study_model import StudyCategory, StudyElement
            categories = self.db.execute(
                select(StudyCategory)
                .where(StudyCategory.study_id == study_id)
                .order_by(StudyCategory.order)
            ).scalars().all()
            for cat in categories:
                elements = self.db.execute(
                    select(StudyElement)
                    .where(StudyElement.study_id == study_id, StudyElement.category_id == cat.id)
                    .order_by(StudyElement.element_id)
                ).scalars().all()
                for idx, el in enumerate(elements, start=1):
                    grid_columns.append((cat.name, el.name, idx))
            # Headers as CategoryName_ImageName
            header.extend([f"{c}_{e}" for (c, e, _) in grid_columns])
        else:
            # Layer columns: all slots per layer in numeric order with attached image names
            header.extend(layer_headers)
        header.extend(["Rating", "ResponseTime"])

        # Yield header
        yield header

        # Helper to compute age
        def compute_age(personal_info: Optional[Dict[str, Any]]) -> Optional[int]:
            if not personal_info:
                return None
            dob_str = personal_info.get("dob") or personal_info.get("date_of_birth")
            if not dob_str:
                return personal_info.get("age")
            try:
                from datetime import date
                year, month, day = [int(x) for x in str(dob_str).split("-")[:3]]
                born = date(year, month, day)
                today = date.today()
                return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
            except Exception:
                return None

        # Emit rows
        # Preload all answers and group by response (single query)
        answers_seq = self.db.execute(
            select(ClassificationAnswer)
            .where(ClassificationAnswer.study_response_id.in_(response_ids))
        ).scalars().all()
        answers_by_response: Dict[UUID, List[ClassificationAnswer]] = defaultdict(list)
        for a in answers_seq:
            answers_by_response[a.study_response_id].append(a)

        for resp in responses:
            # Build map for classification answers for this response
            answers = answers_by_response.get(resp[RESP_ID_IDX], [])
            answer_map: Dict[Any, Any] = {}
            for a in answers:
                options_map = question_id_to_options.get(a.question_id)
                answer_map[a.question_id] = self._format_classification_answer(a, options_map)

            gender = None
            pi = resp[RESP_PI_IDX]
            if isinstance(pi, dict):
                gender = pi.get("gender") or pi.get("Gender")
            age_val = compute_age(pi if isinstance(pi, dict) else None)

            # Fetch tasks ordered by task_index (preloaded)
            tasks = tasks_by_response.get(resp[RESP_ID_IDX], [])

            for t in tasks:
                row: List[Any] = []
                row.append(resp[RESP_SESSION_IDX])
                # Classification answers in header order
                for qid in question_id_to_col.keys():
                    row.append(answer_map.get(qid))
                row.append(gender)
                row.append(age_val)
                row.append(t.task_index + 1)

                if is_grid:
                    # Grid: presence determined by elements_shown_in_task keys like "CategoryName_Index"
                    grid_data = t.elements_shown_in_task or t.elements_shown or {}
                    presence_map: Dict[str, int] = {}
                    if isinstance(grid_data, dict):
                        for k, v in grid_data.items():
                            if not isinstance(k, str) or k.endswith('_content'):
                                continue
                            try:
                                val = int(v) if v in (0, 1) else (1 if v else 0)
                            except Exception:
                                val = 1 if v else 0
                            presence_map[k.strip()] = val
                            presence_map[k.strip().replace(' ', '_')] = val
                            presence_map[k.strip().lower()] = val
                    content_map = t.elements_shown_content if isinstance(t.elements_shown_content, dict) else {}
                    # Emit per header using the category + index mapping; fallback checks content_map by CategoryName_Index
                    for (cat_name, el_name, one_idx) in grid_columns:
                        candidates = [
                            f"{cat_name}_{one_idx}",
                            f"{cat_name.replace(' ', '_')}_{one_idx}",
                            f"{cat_name.lower()}_{one_idx}",
                        ]
                        present = 0
                        for cand in candidates:
                            if cand in presence_map:
                                present = presence_map[cand]
                                break
                            if cand.lower() in presence_map:
                                present = presence_map[cand.lower()]
                                break
                        if present == 0 and content_map:
                            # Fallback: content_map keys are usually CategoryName_Index
                            for cand in candidates:
                                v = content_map.get(cand)
                                if isinstance(v, dict) and (v.get('url') or v.get('name')):
                                    present = 1
                                    break
                        row.append(present)
                else:
                    # Layer: visibility per ordered slots
                    layer_data = t.layers_shown_in_task or t.elements_shown_content or {}
                    for base_label, idx in layer_slots:
                        value = 0
                        if isinstance(layer_data, dict):
                            # Find matching key regardless of spacing/hyphen style
                            candidates = [
                                f"{base_label} {idx}", f"{base_label}_{idx}", f"{base_label}-{idx}",
                                f"{base_label.strip()} {idx}", f"{base_label.strip()}_{idx}", f"{base_label.strip()}-{idx}"
                            ]
                            for cand in candidates:
                                if cand in layer_data and layer_data[cand]:
                                    value = 1
                                    break
                        row.append(value)

                row.append(float(t.rating_given))
                row.append(float(t.task_duration_seconds or 0.0))

                yield row

    def _extract_num_layers(self, task: CompletedTask) -> int:
        """Infer number of layers from task content."""
        data = task.elements_shown_content or task.layers_shown_in_task or {}
        try:
            # Expecting a dict like {"layers": [ {...}, {...} ]} or just a list
            if isinstance(data, dict) and isinstance(data.get("layers"), list):
                return len(data["layers"]) or 0
            if isinstance(data, list):
                return len(data) or 0
        except Exception:
            pass
        return 0

    def _extract_layer_visibility_map(self, task: CompletedTask) -> Dict[Tuple[int, int], bool]:
        """
        Build a map of (taskIndex, layerIndex) -> visible bool for a single task.
        We mark all layers present in the content for this task as visible (True).
        """
        visibility: Dict[Tuple[int, int], bool] = {}
        data = task.elements_shown_content or task.layers_shown_in_task or {}
        try:
            layers = None
            if isinstance(data, dict) and isinstance(data.get("layers"), list):
                layers = data.get("layers")
            elif isinstance(data, list):
                layers = data
            if isinstance(layers, list):
                for idx, _layer in enumerate(layers, start=1):
                    visibility[(task.task_index, idx)] = True
        except Exception:
            pass
        return visibility

    def _format_classification_answer(self, answer: ClassificationAnswer, options_map: Optional[Dict[str, str]] = None) -> Any:
        """
        Return a user-friendly classification answer for CSV output.
        If the stored answer contains an internal code like "9de47e73 - Option 1",
        we strip the code and keep only the human label after the hyphen.
        """
        try:
            raw = answer.answer
            if isinstance(raw, str):
                # If options map provided, try to map code -> text first
                if options_map:
                    # Accept exact code or left part of "code - label"
                    left_code = raw.split(' - ', 1)[0]
                    right_part = raw.split(' - ', 1)[1].strip() if ' - ' in raw else None
                    if raw in options_map:
                        return options_map[raw]
                    if left_code in options_map:
                        return options_map[left_code]
                    if right_part and right_part in options_map:
                        return options_map[right_part]
                # Common pattern: "<code> - <label>"; take the part after the first ' - '
                parts = raw.split(' - ', 1)
                if len(parts) == 2 and parts[1].strip():
                    return parts[1].strip()
                return raw.strip()
            return raw
        except Exception:
            return getattr(answer, 'answer', None)

    # ---------- Optimized CSV Export (bulk queries, no N+1) ----------
    def generate_csv_rows_for_study_optimized(self, study_id: UUID) -> Iterable[List[Any]]:
        """High-performance CSV flattener for a study.
        Minimizes queries by bulk-loading responses, tasks, and answers.
        """
        from sqlalchemy import select, func

        # Basic study and questions
        study: Optional[Study] = self.db.get(Study, study_id)
        if not study:
            raise HTTPException(status_code=404, detail="Study not found")

        # Load responses for this study
        responses: List[StudyResponse] = self.db.execute(
            select(StudyResponse).where(StudyResponse.study_id == study_id)
        ).scalars().all()
        if not responses:
            # Header only
            yield ["Panelist", "Gender", "Age", "Task", "Rating", "ResponseTime"]
            return

        response_ids = [r.id for r in responses]

        # Bulk load tasks and answers
        all_tasks: List[CompletedTask] = self.db.execute(
            select(CompletedTask)
            .where(CompletedTask.study_response_id.in_(response_ids))
        ).scalars().all()

        all_answers: List[ClassificationAnswer] = self.db.execute(
            select(ClassificationAnswer)
            .where(ClassificationAnswer.study_response_id.in_(response_ids))
        ).scalars().all()

        # Group by response
        tasks_by_response: Dict[UUID, List[CompletedTask]] = {}
        for t in all_tasks:
            tasks_by_response.setdefault(t.study_response_id, []).append(t)
        for lst in tasks_by_response.values():
            lst.sort(key=lambda x: (x.task_index or 0))

        answers_by_response: Dict[UUID, List[ClassificationAnswer]] = {}
        for a in all_answers:
            answers_by_response.setdefault(a.study_response_id, []).append(a)

        # Questions and option maps
        question_id_to_col: Dict[str, str] = {}
        question_id_to_options: Dict[str, Dict[str, str]] = {}
        questions = self.db.execute(
            select(StudyClassificationQuestion)
            .where(StudyClassificationQuestion.study_id == study_id)
            .order_by(StudyClassificationQuestion.order)
        ).scalars().all()
        next_q_num = 1
        for q in questions:
            question_id_to_col[q.question_id] = (q.question_text or f"QQ{next_q_num}")
            next_q_num += 1
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
                question_id_to_options[q.question_id] = options_map

        # Discover layer keys across all tasks (only for layer studies). For grid we add per-element headers elsewhere.
        layer_keys_set: set[str] = set()
        synthetic_layer_len = 0
        if str(study.study_type) == 'layer':
            for t in all_tasks:
                data = t.layers_shown_in_task or t.elements_shown_content
                if isinstance(data, dict):
                    for k in data.keys():
                        if isinstance(k, str):
                            layer_keys_set.add(k)
                elif isinstance(data, list):
                    try:
                        synthetic_layer_len = max(synthetic_layer_len, len(data))
                    except Exception:
                        pass

        def sort_key(layer_key: str) -> tuple[int, int]:
            import re
            m2 = re.search(r"(\d+)[ _-](\d+)$", layer_key)
            if m2:
                return (int(m2.group(1)), int(m2.group(2)))
            m1 = re.search(r"(\d+)$", layer_key)
            if m1:
                return (int(m1.group(1)), 0)
            return (9999, 9999)

        # Sort layer keys by database layer order instead of alphabetical
        layer_keys: List[str] = []
        if layer_keys_set:
            # Get layer structure from database to create proper ordering
            from app.models.study_model import StudyLayer, LayerImage
            
            # Get layers ordered by their order field
            layers = self.db.execute(
                select(StudyLayer)
                .where(StudyLayer.study_id == study_id)
                .order_by(StudyLayer.order)
            ).scalars().all()
            
            # Create ordered list of layer names based on database order
            ordered_layer_names = [layer.name for layer in layers]
            
            # Sort layer keys by database order
            layer_keys = []
            for layer_name in ordered_layer_names:
                # Find all keys that start with this layer name
                matching_keys = [key for key in layer_keys_set if key.startswith(layer_name + "_")]
                # Sort by image number
                matching_keys.sort(key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0)
                layer_keys.extend(matching_keys)
            
            # Add any remaining keys that don't match layer names
            remaining_keys = [key for key in layer_keys_set if key not in layer_keys]
            layer_keys.extend(remaining_keys)
        # If no explicit dict keys found but list-based structure exists, synthesize generic keys Layer_1..N
        if not layer_keys and synthetic_layer_len > 0:
            layer_keys = [f"Layer_{i}" for i in range(1, synthetic_layer_len + 1)]

        # Friendly headers for layer keys - use actual layer names
        layer_key_to_header: Dict[str, str] = {}
        if layer_keys and layer_keys_set:
            # Get layer structure from database to create proper ordering
            from app.models.study_model import StudyLayer, LayerImage
            
            # Get layers ordered by their order field
            layers = self.db.execute(
                select(StudyLayer)
                .where(StudyLayer.study_id == study_id)
                .order_by(StudyLayer.order)
            ).scalars().all()
            
            # Create mapping from layer names to their database order
            layer_name_to_order: Dict[str, int] = {}
            for layer_order, layer in enumerate(layers, 1):
                layer_name_to_order[layer.name] = layer_order
            
            # Map discovered keys to proper layer names
            for key in layer_keys:
                # Extract layer name from key (e.g., "background_1" -> "background")
                import re
                match = re.search(r"^(.+?)_(\d+)$", key)
                if match:
                    layer_name = match.group(1)
                    image_num = match.group(2)
                    
                    # Use the actual layer name with image number
                    layer_key_to_header[key] = f"{layer_name}_{image_num}"
                else:
                    # Fallback to original key
                    layer_key_to_header[key] = key

        # Grid headers (CategoryName_ImageName by category order and element_id)
        grid_columns_opt: List[tuple[str, str, int]] = []  # (cat_name, element_name, one_based_index)
        if str(study.study_type) == 'grid':
            from app.models.study_model import StudyCategory, StudyElement
            categories = self.db.execute(
                select(StudyCategory)
                .where(StudyCategory.study_id == study_id)
                .order_by(StudyCategory.order)
            ).scalars().all()
            for cat in categories:
                elements = self.db.execute(
                    select(StudyElement)
                    .where(StudyElement.study_id == study_id, StudyElement.category_id == cat.id)
                    .order_by(StudyElement.element_id)
                ).scalars().all()
                for idx, el in enumerate(elements, start=1):
                    grid_columns_opt.append((cat.name, el.name, idx))

        # Build header
        header: List[str] = ["Panelist"]
        header.extend([question_id_to_col[qid] for qid in question_id_to_col])
        header.extend(["Gender", "Age", "Task"])
        if grid_columns_opt:
            # Sanitize headers minimally for CSV (replace commas with underscores)
            def sanitize_header(s: str) -> str:
                return s.replace(',', '_').replace('\n', ' ').replace('\r', ' ').strip()
            header.extend([sanitize_header(f"{c}_{e}") for (c, e, _) in grid_columns_opt])
        elif layer_keys:
            header.extend([layer_key_to_header.get(k, k.replace(' ', '_')) for k in layer_keys])
        header.extend(["Rating", "ResponseTime"])
        yield header

        # Helper: age
        def compute_age(personal_info: Optional[Dict[str, Any]]) -> Optional[int]:
            if not personal_info:
                return None
            dob_str = personal_info.get("dob") or personal_info.get("date_of_birth")
            if not dob_str:
                return personal_info.get("age")
            try:
                from datetime import date
                year, month, day = [int(x) for x in str(dob_str).split("-")[:3]]
                born = date(year, month, day)
                today = date.today()
                return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
            except Exception:
                return None

        # Emit rows
        for resp in responses:
            answer_list = answers_by_response.get(resp.id, [])
            answer_map: Dict[Any, Any] = {}
            for a in answer_list:
                options_map = question_id_to_options.get(a.question_id)
                answer_map[a.question_id] = self._format_classification_answer(a, options_map)

            gender = resp.personal_info.get("gender") if resp.personal_info else None
            age_val = compute_age(resp.personal_info)

            for t in tasks_by_response.get(resp.id, []):
                row: List[Any] = []
                row.append(resp.session_id)
                for qid in question_id_to_col.keys():
                    row.append(answer_map.get(qid))
                row.append(gender)
                row.append(age_val)
                row.append((t.task_index or 0) + 1)

                if grid_columns_opt:
                    # Use elements_shown_in_task CategoryName_Index keys; fallback to elements_shown_content
                    presence_raw = t.elements_shown_in_task or t.elements_shown or {}
                    presence_map: Dict[str, int] = {}
                    if isinstance(presence_raw, dict):
                        for k, v in presence_raw.items():
                            if isinstance(k, str):
                                try:
                                    presence_map[k.strip()] = int(v) if v in (0, 1) else (1 if v else 0)
                                except Exception:
                                    presence_map[k.strip()] = 1 if v else 0
                                # also store normalized spacings
                                presence_map[k.strip().replace(' ', '_')] = presence_map[k.strip()]
                                presence_map[k.strip().lower()] = presence_map[k.strip()]
                    content_map = t.elements_shown_content if isinstance(t.elements_shown_content, dict) else {}
                    for (cat_name, el_name, one_idx) in grid_columns_opt:
                        # primary: CategoryName_Index
                        candidates = [
                            f"{cat_name}_{one_idx}",
                            f"{cat_name.replace(' ', '_')}_{one_idx}",
                            f"{cat_name.lower()}_{one_idx}",
                        ]
                        present = 0
                        for cand in candidates:
                            if cand in presence_map:
                                present = presence_map[cand]
                                break
                            if cand.lower() in presence_map:
                                present = presence_map[cand.lower()]
                                break
                        if present == 0 and content_map:
                            # fallback: CategoryName_ImageName present in content_map
                            header_key = f"{cat_name}_{el_name}"
                            v = content_map.get(header_key) or content_map.get(header_key.replace(' ', '_'))
                            if isinstance(v, dict) and (v.get('url') or v.get('name')):
                                present = 1
                        row.append(present)
                elif layer_keys:
                    layer_data = t.layers_shown_in_task or t.elements_shown_in_task or t.elements_shown_content or {}
                    for key in layer_keys:
                        value = 0
                        if isinstance(layer_data, dict):
                            v = layer_data.get(key)
                            if isinstance(v, dict):
                                vis = v.get("visible")
                                if vis is None:
                                    vis = True if len(v.keys()) > 0 else False
                                value = 1 if bool(vis) else 0
                            else:
                                try:
                                    value = 1 if int(v) != 0 else 0
                                except Exception:
                                    value = 1 if bool(v) else 0
                        elif isinstance(layer_data, list) and key.startswith("Layer_"):
                            try:
                                idx = int(key.split("_", 1)[1]) - 1
                                if 0 <= idx < len(layer_data):
                                    item = layer_data[idx]
                                    if isinstance(item, dict):
                                        vis = item.get("visible")
                                        if vis is None:
                                            vis = True if len(item.keys()) > 0 else False
                                        value = 1 if bool(vis) else 0
                                    else:
                                        value = 1 if bool(item) else 0
                            except Exception:
                                value = 0
                        row.append(value)

                row.append(float(t.rating_given))
                row.append(float(t.task_duration_seconds or 0.0))

                yield row
    
    def abandon_study(self, session_id: str, request: AbandonStudyRequest) -> bool:
        """Mark a study response as abandoned."""
        response = self.get_response_by_session(session_id)
        if not response:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if response.is_completed:
            raise HTTPException(status_code=400, detail="Study already completed")
        
        response.is_abandoned = True
        response.is_completed = False
        response.status = 'abandoned'  # Set status as abandoned
        response.abandonment_timestamp = datetime.utcnow()
        response.abandonment_reason = request.reason
        response.last_activity = datetime.utcnow()
        
        self.db.commit()
        
        # Update study counters
        self._update_study_counters(response.study_id)
        
        # Check if study should be auto-completed
        self._check_and_complete_studies()
        
        # Invalidate analytics cache since data changed
        self.invalidate_analytics_cache(response.study_id)
        
        return True
    
    def check_and_mark_abandoned_sessions(self) -> int:
        """Check for sessions that have been inactive for 15+ minutes and mark them as abandoned."""
        from datetime import timedelta
        
        # Find sessions that are in_progress but haven't been active for 15+ minutes
        cutoff_time = datetime.utcnow() - timedelta(minutes=15)
        
        abandoned_sessions = self.db.execute(
            select(StudyResponse)
            .where(
                and_(
                    StudyResponse.status == 'in_progress',
                    StudyResponse.last_activity < cutoff_time,
                    StudyResponse.is_completed == False
                )
            )
        ).scalars().all()
        
        count = 0
        for session in abandoned_sessions:
            session.is_abandoned = True
            session.is_completed = False
            session.status = 'abandoned'
            session.abandonment_timestamp = datetime.utcnow()
            session.abandonment_reason = 'Inactive for 15+ minutes'
            count += 1
        
        if count > 0:
            self.db.commit()
            # Update study counters for affected studies
            study_ids = set(session.study_id for session in abandoned_sessions)
            for study_id in study_ids:
                self._update_study_counters(study_id)
        
        # Check for study completion after marking abandoned sessions
        self._check_and_complete_studies()
        
        return count
    
    def _check_and_complete_studies(self):
        """Check if studies should be automatically marked as completed."""
        import logging
        from app.models.study_model import Study
        from app.services.study import change_status_fast
        
        logger = logging.getLogger(__name__)
        
        # Find active studies that might be ready for completion
        active_studies = self.db.execute(
            select(Study)
            .where(Study.status == 'active')
        ).scalars().all()
        
        for study in active_studies:
            # Get current response counts for this study
            total_responses = self.db.execute(
                select(func.count(StudyResponse.id))
                .where(StudyResponse.study_id == study.id)
            ).scalar() or 0
            
            completed_responses = self.db.execute(
                select(func.count(StudyResponse.id))
                .where(
                    and_(
                        StudyResponse.study_id == study.id,
                        StudyResponse.is_completed == True
                    )
                )
            ).scalar() or 0
            
            abandoned_responses = self.db.execute(
                select(func.count(StudyResponse.id))
                .where(
                    and_(
                        StudyResponse.study_id == study.id,
                        StudyResponse.is_abandoned == True
                    )
                )
            ).scalar() or 0
            
            # Check if all expected respondents have either completed or abandoned
            expected_respondents = study.audience_segmentation.get('number_of_respondents', 0) if study.audience_segmentation else 0
            
            # Debug logging
            logger.info(f"Study {study.id} completion check:")
            logger.info(f"  - Expected respondents: {expected_respondents}")
            logger.info(f"  - Total responses: {total_responses}")
            logger.info(f"  - Completed: {completed_responses}")
            logger.info(f"  - Abandoned: {abandoned_responses}")
            logger.info(f"  - Audience segmentation: {study.audience_segmentation}")
            
            # Check if study should be completed
            should_complete = False
            completion_reason = ""
            
            if expected_respondents > 0:
                # Study has explicit expected respondents
                if (completed_responses + abandoned_responses) >= expected_respondents:
                    should_complete = True
                    completion_reason = f"all {expected_respondents} expected respondents completed/abandoned"
                else:
                    logger.info(f"Study {study.id} needs {expected_respondents - (completed_responses + abandoned_responses)} more respondents to complete")
            else:
                # No expected respondents set, check if all current respondents have completed/abandoned
                if total_responses > 0 and (completed_responses + abandoned_responses) >= total_responses:
                    should_complete = True
                    completion_reason = f"all {total_responses} current respondents completed/abandoned"
                else:
                    logger.info(f"Study {study.id} has {total_responses} total responses, {completed_responses} completed, {abandoned_responses} abandoned")
            
            if should_complete:
                logger.info(f"Study {study.id} auto-completion triggered: {completion_reason}")
                
                # Mark study as completed
                try:
                    change_status_fast(self.db, study.id, study.creator_id, 'completed')
                    logger.info(f"Study {study.id} automatically marked as completed")
                except Exception as e:
                    logger.error(f"Failed to auto-complete study {study.id}: {e}")
    
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
        """Get analytics data for a study - ultra-optimized with raw SQL for maximum speed."""
        # Caching disabled: always compute fresh analytics
        now = time.time()
        
        # Ultra-fast raw SQL query for all statistics
        from sqlalchemy import text
        
        stats_query = text("""
            SELECT 
                COUNT(*) as total_responses,
                COUNT(*) FILTER (WHERE is_completed = true) as completed_responses,
                COUNT(*) FILTER (WHERE is_abandoned = true) as abandoned_responses,
                COUNT(*) FILTER (WHERE status = 'in_progress') as in_progress_responses,
                AVG(total_study_duration) FILTER (WHERE is_completed = true) as avg_duration
            FROM study_responses 
            WHERE study_id = :study_id
        """)
        
        result = self.db.execute(stats_query, {"study_id": study_id}).first()
        
        total_responses = result.total_responses or 0
        completed_responses = result.completed_responses or 0
        abandoned_responses = result.abandoned_responses or 0
        in_progress_responses = result.in_progress_responses or 0
        avg_duration = float(result.avg_duration or 0)
        
        # Calculate rates
        completion_rate = (completed_responses / total_responses * 100) if total_responses > 0 else 0
        abandonment_rate = (abandoned_responses / total_responses * 100) if total_responses > 0 else 0
        
        # Ultra-fast analytics - return minimal data if no responses
        if total_responses == 0:
            return StudyAnalytics(
                total_responses=0,
                completed_responses=0,
                abandoned_responses=0,
                in_progress_responses=0,
                completion_rate=0,
                average_duration=0,
                abandonment_rate=0,
                element_heatmap={},
                timing_distributions={"task_durations": [], "average_duration": 0, "min_duration": 0, "max_duration": 0, "total_tasks": 0}
            )
        
        # Get minimal element heatmap (raw SQL)
        element_heatmap = self._get_element_heatmap_ultra_fast(study_id)
        
        # Get minimal timing distributions (raw SQL)
        timing_distributions = self._get_timing_distributions_ultra_fast(study_id)
        
        analytics = StudyAnalytics(
            total_responses=total_responses,
            completed_responses=completed_responses,
            abandoned_responses=abandoned_responses,
            in_progress_responses=in_progress_responses,
            completion_rate=completion_rate,
            average_duration=avg_duration,
            abandonment_rate=abandonment_rate,
            element_heatmap=element_heatmap,
            timing_distributions=timing_distributions
        )
        return analytics
    
    def _get_cached_analytics(self, study_id: UUID) -> Optional[StudyAnalytics]:
        """Caching disabled; always recompute."""
        return None
    
    def invalidate_analytics_cache(self, study_id: UUID) -> None:
        """Caching disabled; nothing to invalidate."""
        return None
    
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
        """Get the next available respondent ID for a study, starting from 1."""
        # Count existing responses for this study to get the next sequential ID
        existing_count = self.db.execute(
            select(func.count(StudyResponse.id))
            .where(StudyResponse.study_id == study_id)
        ).scalar()
        
        return existing_count + 1
    
    def _calculate_respondent_task_count(self, tasks: Any, respondent_id: int) -> int:
        """Calculate the number of tasks assigned to a specific respondent."""
        if not tasks:
            return 0
        
        if isinstance(tasks, list):
            # Flat list of tasks - all respondents get all tasks
            return len(tasks)
        elif isinstance(tasks, dict):
            # Per-respondent tasks or index-keyed tasks
            respondent_key = str(respondent_id)
            if respondent_key in tasks:
                respondent_tasks = tasks[respondent_key]
                if isinstance(respondent_tasks, list):
                    return len(respondent_tasks)
                else:
                    return 1 if respondent_tasks else 0
            else:
                # Fallback: count all numeric keys (0, 1, 2, ...)
                numeric_count = 0
                for key, value in tasks.items():
                    if key.isdigit() and value:
                        numeric_count += 1
                return numeric_count
        
        return 0
    
    def _mark_response_completed(self, response: StudyResponse) -> None:
        """Mark a response as completed, handling timezone-aware timestamps."""
        from datetime import timezone
        response.is_completed = True
        response.is_abandoned = False
        response.status = 'completed'  # Set status as completed
        end_time = datetime.now(timezone.utc)
        response.session_end_time = end_time
        response.completion_percentage = 100.0
        response.last_activity = end_time  # Update last activity
        
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
    
    def _get_element_heatmap_ultra_fast(self, study_id: UUID) -> Dict[str, Any]:
        """Get element interaction heatmap data - ultra-fast with raw SQL and minimal data."""
        from sqlalchemy import text
        
        # Ultra-fast raw SQL with minimal data
        heatmap_query = text("""
            SELECT 
                ei.element_id,
                SUM(ei.view_time_seconds) as total_view_time,
                SUM(ei.hover_count) as total_hovers,
                SUM(ei.click_count) as total_clicks,
                COUNT(ei.id) as interaction_count
            FROM element_interactions ei
            JOIN study_responses sr ON ei.study_response_id = sr.id
            WHERE sr.study_id = :study_id
            GROUP BY ei.element_id
            LIMIT 20
        """)
        
        results = self.db.execute(heatmap_query, {"study_id": study_id}).fetchall()
        
        heatmap = {}
        for result in results:
            heatmap[result.element_id] = {
                "total_view_time": float(result.total_view_time or 0),
                "total_hovers": int(result.total_hovers or 0),
                "total_clicks": int(result.total_clicks or 0),
                "interaction_count": int(result.interaction_count or 0)
            }
        
        return heatmap
    
    def _get_timing_distributions_ultra_fast(self, study_id: UUID) -> Dict[str, Any]:
        """Get timing distribution data - ultra-fast with raw SQL and minimal sampling."""
        from sqlalchemy import text
        
        # Ultra-fast raw SQL for timing statistics
        timing_query = text("""
            SELECT 
                AVG(ct.task_duration_seconds) as avg_duration,
                MIN(ct.task_duration_seconds) as min_duration,
                MAX(ct.task_duration_seconds) as max_duration,
                COUNT(ct.id) as total_tasks
            FROM completed_tasks ct
            JOIN study_responses sr ON ct.study_response_id = sr.id
            WHERE sr.study_id = :study_id
        """)
        
        result = self.db.execute(timing_query, {"study_id": study_id}).first()
        
        if not result or result.total_tasks == 0:
            return {"task_durations": [], "average_duration": 0, "min_duration": 0, "max_duration": 0, "total_tasks": 0}
        
        # Get minimal sample for distribution (ultra-small limit for speed)
        sample_query = text("""
            SELECT ct.task_duration_seconds
            FROM completed_tasks ct
            JOIN study_responses sr ON ct.study_response_id = sr.id
            WHERE sr.study_id = :study_id
            ORDER BY ct.task_completion_time DESC
            LIMIT 20
        """)
        
        sample_durations = [row[0] for row in self.db.execute(sample_query, {"study_id": study_id}).fetchall()]
        
        return {
            "task_durations": sample_durations,
            "average_duration": float(result.avg_duration or 0),
            "min_duration": float(result.min_duration or 0),
            "max_duration": float(result.max_duration or 0),
            "total_tasks": int(result.total_tasks or 0)
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