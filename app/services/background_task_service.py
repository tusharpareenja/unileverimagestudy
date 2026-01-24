# app/services/background_task_service.py
import asyncio
import uuid
from uuid import UUID
import time
import psutil
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from sqlalchemy.orm import Session
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    PENDING = "pending"
    STARTED = "started"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskGenerationJob:
    job_id: str
    study_id: str
    user_id: str
    payload: Dict[str, Any]
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class BackgroundTaskService:
    def __init__(self):
        self.jobs: Dict[str, TaskGenerationJob] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start_cleanup_task(self):
        """Start background cleanup task to remove old completed jobs"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_old_jobs())
    
    async def _cleanup_old_jobs(self):
        """Remove jobs older than 24 hours"""
        while True:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                jobs_to_remove = [
                    job_id for job_id, job in self.jobs.items()
                    if job.created_at < cutoff_time and job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
                ]
                for job_id in jobs_to_remove:
                    del self.jobs[job_id]
                    if job_id in self.running_tasks:
                        del self.running_tasks[job_id]
                logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(3600)  # Run every hour
    
    def create_job(self, study_id: str, user_id: str, payload: Dict[str, Any]) -> str:
        """Create a new task generation job"""
        job_id = str(uuid.uuid4())
        job = TaskGenerationJob(
            job_id=job_id,
            study_id=study_id,
            user_id=user_id,
            payload=payload,
            status=JobStatus.PENDING,
            created_at=datetime.utcnow()
        )
        self.jobs[job_id] = job
        logger.info(f"Created job {job_id} for study {study_id}")
        return job_id
    
    async def start_job(self, job_id: str, db: Session):
        """Start processing a job in the background"""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs[job_id]
        if job.status != JobStatus.PENDING:
            raise ValueError(f"Job {job_id} is not in pending status")
        
        # Start the background task
        task = asyncio.create_task(self._process_job(job_id, db))
        self.running_tasks[job_id] = task
        job.status = JobStatus.STARTED
        job.started_at = datetime.utcnow()
        job.message = "Task generation started"
        
        logger.info(f"Started job {job_id}")
        return task
    
    async def _process_job(self, job_id: str, db: Session):
        """Process a task generation job"""
        try:
            job = self.jobs[job_id]
            job.status = JobStatus.PROCESSING
            job.message = "Generating tasks..."
            
            # Memory check removed - Azure will handle memory limits automatically
            # if not self._check_memory_usage():
            #     raise RuntimeError("Insufficient memory for task generation")
            
            # Import here to avoid circular imports
            from app.services.task_generation_adapter import generate_grid_tasks, generate_layer_tasks
            
            # Process based on study type
            if job.payload.get('study_type') in ('grid', 'text'):
                result = await self._generate_grid_tasks_async(job, db)
            elif job.payload.get('study_type') == 'layer':
                result = await self._generate_layer_tasks_async(job, db)
            elif job.payload.get('study_type') == 'hybrid':
                result = await self._generate_hybrid_tasks_async(job, db)
            else:
                raise ValueError(f"Unsupported study type: {job.payload.get('study_type')}. Must be 'grid', 'layer', 'text', or 'hybrid'.")
            
            # Save results
            await self._save_results(job, result, db)
            
            # Auto-launch study when tasks are generated successfully
            await self._launch_study(job, db)
            
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress = 100.0
            job.message = "Task generation completed successfully"
            job.result = result
            
            logger.info(f"Job {job_id} completed successfully")
            
        except RuntimeError as e:
            # Handle specific preflight errors
            if "Preflight failed" in str(e) or "consider more elements/categories" in str(e):
                job = self.jobs[job_id]
                job.status = JobStatus.FAILED
                job.completed_at = datetime.utcnow()
                job.error = str(e)
                job.message = f"Study configuration error: {str(e)}. Please add more elements/categories or adjust study parameters."
                logger.error(f"Job {job_id} failed due to study configuration: {e}")
            else:
                # Re-raise other RuntimeErrors
                raise
        except Exception as e:
            job = self.jobs[job_id]
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error = str(e)
            job.message = f"Task generation failed: {str(e)}"
            logger.error(f"Job {job_id} failed: {e}")
            import traceback
            logger.error(f"Job {job_id} traceback: {traceback.format_exc()}")
        finally:
            # Clean up running task
            if job_id in self.running_tasks:
                del self.running_tasks[job_id]
    
    async def _generate_grid_tasks_async(self, job: TaskGenerationJob, db: Session):
        """Generate grid tasks asynchronously with progress tracking"""
        return await self._generate_tasks_for_phase(job, db, phase_type=None)

    async def _generate_tasks_for_phase(self, job: TaskGenerationJob, db: Session, phase_type: Optional[str] = None, progress_range: tuple = (20.0, 90.0)):
        """Helper to generate tasks for a specific phase (or all if phase_type is None)"""
        from app.services.task_generation_core import generate_grid_tasks_v2
        
        # Build categories data
        categories_data = []
        payload_categories = job.payload.get('categories') or []
        payload_elements = job.payload.get('elements') or []
        
        # Filter categories by phase_type if provided
        filtered_categories = payload_categories
        if phase_type:
            filtered_categories = [c for c in payload_categories if c.get('phase_type') == phase_type]
            
        if not filtered_categories:
            logger.warning(f"No categories found for phase_type={phase_type}")
            return {"tasks": {}, "metadata": {}}

        for cat in filtered_categories:
            cat_elements = [e for e in payload_elements if e.get('category_id') == cat.get('category_id')]
            categories_data.append({
                "category_name": cat.get('name'),
                "elements": [
                    {
                        "element_id": str(el.get('element_id')),
                        "name": el.get('name'),
                        "content": el.get('content'),
                        "alt_text": el.get('alt_text') or el.get('name'),
                        "element_type": el.get('element_type'),
                    }
                    for el in cat_elements
                ]
            })
        
        # Prepare progress callback
        p_start, p_end = progress_range
        def on_progress(done: int, N: int):
            try:
                frac = max(0.0, min(1.0, done / max(1, N)))
                job.progress = p_start + ((p_end - p_start) * frac)
                phase_msg = f" ({phase_type})" if phase_type else ""
                job.message = f"Building respondents{phase_msg} {done}/{N}..."
            except Exception:
                pass
        
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: generate_grid_tasks_v2(
                        categories_data=categories_data,
                        number_of_respondents=job.payload.get('audience_segmentation', {}).get('number_of_respondents', 0),
                        exposure_tolerance_cv=job.payload.get('exposure_tolerance_cv', 1.0),
                        seed=job.payload.get('seed'),
                        progress_callback=on_progress
                    )
                ),
                timeout=settings.TASK_GENERATION_TIMEOUT
            )
        except RuntimeError as e:
            if "Preflight failed" in str(e) or "consider more elements/categories" in str(e):
                logger.error(f"Preflight error in generation phase {phase_type}: {e}")
                raise RuntimeError(f"Study configuration error in {phase_type} phase: {str(e)}.")
            else:
                raise
        
        return result

    async def _generate_hybrid_tasks_async(self, job: TaskGenerationJob, db: Session):
        """Generate tasks for hybrid studies by combining multiple phases"""
        phase_order = job.payload.get('phase_order', [])
        if not phase_order:
            raise ValueError("Hybrid study requires phase_order")
            
        job.progress = 10.0
        job.message = "Initializing hybrid task generation..."
        
        combined_tasks = {}
        combined_metadata = {"phases": []}
        
        is_mix = "mix" in phase_order
        payload_categories = job.payload.get('categories') or []
        
        target_phases = []
        if is_mix:
            target_phases = list(set(c.get('phase_type') for i, c in enumerate(payload_categories) if c.get('phase_type')))
        else:
            target_phases = phase_order

        num_phases = len(target_phases)
        tasks_per_respondent_mix = {}

        for i, phase_type in enumerate(target_phases):
            # Calculate progress range for this phase
            p_start = 10.0 + (i * (80.0 / num_phases))
            p_end = 10.0 + ((i + 1) * (80.0 / num_phases))
            
            job.message = f"Starting generation for phase: {phase_type}..."
            phase_result = await self._generate_tasks_for_phase(job, db, phase_type=phase_type, progress_range=(p_start, p_end))
            
            # Merge results
            phase_tasks = phase_result.get('tasks', {})
            
            if is_mix:
                for resp_id, t_list in phase_tasks.items():
                    if resp_id not in tasks_per_respondent_mix:
                        tasks_per_respondent_mix[resp_id] = []
                    for t in t_list:
                        t['phase_type'] = phase_type
                        tasks_per_respondent_mix[resp_id].append(t)
            else:
                for resp_id, tasks in phase_tasks.items():
                    if resp_id not in combined_tasks:
                        combined_tasks[resp_id] = []
                    
                    # Update task indexes to be continuous
                    offset = len(combined_tasks[resp_id])
                    for t in tasks:
                        t['task_index'] = int(t.get('task_index', 0)) + offset
                        t['phase_type'] = phase_type
                    
                    combined_tasks[resp_id].extend(tasks)
            
            combined_metadata['phases'].append({
                "phase_type": phase_type,
                "metadata": phase_result.get('metadata', {})
            })
            
        if is_mix:
            import random
            job.message = "Randomizing mixed tasks..."
            for resp_id, t_list in tasks_per_respondent_mix.items():
                random.shuffle(t_list)
                for idx, t in enumerate(t_list):
                    t['task_index'] = idx
                combined_tasks[resp_id] = t_list

        job.progress = 90.0
        job.message = "Combining all phases..."
        
        return {
            "tasks": combined_tasks,
            "metadata": combined_metadata
        }
    
    async def _generate_layer_tasks_async(self, job: TaskGenerationJob, db: Session):
        """Generate layer tasks asynchronously with progress tracking"""
        from app.services.task_generation_adapter import generate_layer_tasks
        
        # Update progress
        job.progress = 10.0
        job.message = "Planning layer task generation..."
        
        # Build layers data
        layers = []
        for layer in job.payload.get('study_layers', []):
            layer_obj = type('Layer', (), {
                'name': layer.get('name', ''),
                'images': [type('Image', (), {
                    'name': img.get('name', ''),
                    'url': img.get('url', '')
                }) for img in layer.get('images', [])],
                'z_index': layer.get('z_index', 0),  # Default to 0 if not provided
                'order': layer.get('order', 0)      # Default to 0 if not provided
            })()
            layers.append(layer_obj)
        
        job.progress = 20.0
        job.message = "Starting layer task generation..."
        
        # Prepare progress callback
        def on_progress(done: int, N: int):
            try:
                frac = max(0.0, min(1.0, done / max(1, N)))
                job.progress = 20.0 + (70.0 * frac)
                job.message = f"Building respondents {done}/{N}..."
            except Exception:
                pass
        
        # Generate tasks with timeout (run in executor)
        loop = asyncio.get_event_loop()
        from app.services.task_generation_core import generate_layer_tasks_v2
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: generate_layer_tasks_v2(
                        layers_data=[{
                            "name": l.name, 
                            "z_index": getattr(l, 'z_index', 0),
                            "order": getattr(l, 'order', 0),
                            "images": [{"name": i.name, "url": i.url} for i in l.images]
                        } for l in layers],
                        number_of_respondents=job.payload.get('audience_segmentation', {}).get('number_of_respondents', 0),
                        exposure_tolerance_pct=job.payload.get('exposure_tolerance_pct', 2.0),
                        seed=job.payload.get('seed'),
                        progress_callback=on_progress
                    )
                ),
                timeout=settings.TASK_GENERATION_TIMEOUT
            )
        except RuntimeError as e:
            if "Preflight failed" in str(e) or "consider more elements/categories" in str(e):
                logger.error(f"Preflight error in layer generation: {e}")
                raise RuntimeError(f"Study configuration error: {str(e)}. Please add more elements/categories or adjust study parameters.")
            else:
                raise
        
        job.progress = 90.0
        job.message = "Saving generated tasks..."
        
        # Attach background image URL to metadata if available
        if isinstance(result, dict):
            try:
                # Get the study to access background_image_url
                from app.models.study_model import Study
                study = db.query(Study).filter(Study.id == job.study_id).first()
                if study and hasattr(study, 'background_image_url'):
                    bg_url = getattr(study, 'background_image_url', None)
                    if bg_url is not None:
                        meta = result.get('metadata', {})
                        if isinstance(meta, dict):
                            meta['background_image_url'] = bg_url
                            result['metadata'] = meta
            except Exception as e:
                logger.warning(f"Could not attach background image URL: {e}")
        
        return result
    
    async def _save_results(self, job: TaskGenerationJob, result: Dict[str, Any], db: Session):
        """Save generated tasks to the study"""
        from app.models.study_model import Study
        
        # Get the study
        study = db.query(Study).filter(
            Study.id == job.study_id,
            Study.creator_id == job.user_id
        ).first()
        
        if not study:
            raise ValueError(f"Study {job.study_id} not found or access denied")
        
        # Save tasks
        study.tasks = result.get('tasks', {})
        
        # Save last_step if provided in payload
        last_step_val = job.payload.get('last_step')
        if last_step_val is not None and isinstance(last_step_val, int):
            current_step = getattr(study, 'last_step', 1) or 1
            if last_step_val > current_step:
                setattr(study, 'last_step', last_step_val)
        
        db.commit()
        db.refresh(study)
        
        # Store the updated last_step in job result for response
        if job.result is None:
            job.result = {}
        job.result['last_step'] = getattr(study, 'last_step', None)
        
        logger.info(f"Saved tasks for study {job.study_id}, last_step={study.last_step}")
    
    async def _launch_study(self, job: TaskGenerationJob, db: Session):
        """Launch study by setting status to active and applying any study updates"""
        try:
            from app.services.study import change_status, update_study
            from app.schemas.study_schema import StudyUpdate
            
            # Check if payload contains study update information
            study_updates = {}
            if job.payload.get('title'):
                study_updates['title'] = job.payload.get('title')
            if job.payload.get('background'):
                study_updates['background'] = job.payload.get('background')
            if job.payload.get('language'):
                study_updates['language'] = job.payload.get('language')
            if job.payload.get('main_question'):
                study_updates['main_question'] = job.payload.get('main_question')
            if job.payload.get('orientation_text'):
                study_updates['orientation_text'] = job.payload.get('orientation_text')
            if job.payload.get('background_image_url'):
                study_updates['background_image_url'] = job.payload.get('background_image_url')
            if job.payload.get('rating_scale'):
                study_updates['rating_scale'] = job.payload.get('rating_scale')
            if job.payload.get('audience_segmentation'):
                study_updates['audience_segmentation'] = job.payload.get('audience_segmentation')
            if job.payload.get('phase_order'):
                study_updates['phase_order'] = job.payload.get('phase_order')
            
            # Apply study updates if any
            if study_updates:
                logger.info(f"Applying study updates: {list(study_updates.keys())}")
                update_payload = StudyUpdate(**study_updates)
                update_study(
                    db=db,
                    study_id=UUID(job.study_id),
                    owner_id=UUID(job.user_id),
                    payload=update_payload
                )
                logger.info(f"Study {job.study_id} updated successfully")
            
            # Keep study in draft status (don't auto-launch)
            logger.info(f"Study {job.study_id} updated and ready (draft status)")
            
            # Update job result to include study_id
            if job.result is None:
                job.result = {}
            job.result['study_id'] = job.study_id
            job.result['study_status'] = 'draft'
            
        except Exception as e:
            logger.error(f"Failed to launch study {job.study_id}: {e}")
            # Don't fail the job if launch fails, just log the error
    
    # Memory check method removed - Azure handles memory limits automatically
    
    def get_job_status(self, job_id: str) -> Optional[TaskGenerationJob]:
        """Get the status of a job"""
        return self.jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False
        
        # Cancel the running task
        if job_id in self.running_tasks:
            self.running_tasks[job_id].cancel()
            del self.running_tasks[job_id]
        
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        job.message = "Job cancelled by user"
        
        logger.info(f"Cancelled job {job_id}")
        return True
    
    def get_user_jobs(self, user_id: str) -> list[TaskGenerationJob]:
        """Get all jobs for a user"""
        return [job for job in self.jobs.values() if job.user_id == user_id]

# Global instance
background_task_service = BackgroundTaskService()
