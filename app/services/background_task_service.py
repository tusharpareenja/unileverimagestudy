# app/services/background_task_service.py
import asyncio
import uuid
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.core.config import settings
from app.db.session import SessionLocal
from app.models.job_model import Job, JobStatus


logger = logging.getLogger(__name__)

# Re-export JobStatus / TaskGenerationJob-like structure for backward compatibility type hinting if needed
# But better to use the model directly.
# For existing type hints in other files:
# We might need a small adapter or just update the types.
# The previous `TaskGenerationJob` was a dataclass. The new `Job` is an SQLAlchemy model.
# They are interface-compatible mostly (attributes access).

class BackgroundTaskService:
    def __init__(self):
        # We still need to track running asyncio tasks in memory to cancel them if needed.
        # But the STATUS of the job is in the DB.
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start_cleanup_task(self):
        """Start background cleanup task to remove old completed jobs"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_old_jobs())
    
    async def _cleanup_old_jobs(self):
        """Remove jobs older than 24 hours from DB"""
        while True:
            try:
                db = SessionLocal()
                try:
                    cutoff_time = datetime.utcnow() - timedelta(hours=24)
                    # Find old jobs
                    old_jobs = db.query(Job).filter(
                        Job.created_at < cutoff_time,
                        Job.status.in_([JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED])
                    ).all()
                    
                    count = 0
                    for job in old_jobs:
                        db.delete(job)
                        if job.job_id in self.running_tasks:
                            del self.running_tasks[job.job_id]
                        count += 1
                    
                    db.commit()
                    if count > 0:
                        logger.info(f"Cleaned up {count} old jobs from DB")
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
            
            await asyncio.sleep(3600)  # Run every hour
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Helper to convert objects to JSON serializable formats"""
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        if hasattr(obj, 'isoformat'): # datetime
            return obj.isoformat()
        return obj

    def create_job(self, study_id: str, user_id: str, payload: Dict[str, Any]) -> str:
        """Create a new task generation job in DB"""
        db = SessionLocal()
        try:
            job_id = str(uuid.uuid4())
            
            # Sanitize payload for JSON storage
            safe_payload = self._make_json_serializable(payload)
            
            new_job = Job(
                job_id=job_id,
                study_id=study_id,
                user_id=user_id,
                status=JobStatus.PENDING,
                progress=0.0,
                message="Initialized",
                result={"payload": safe_payload} # Store sanitized payload
            )
            db.add(new_job)
            db.commit()
            logger.info(f"Created job {job_id} for study {study_id}")
            return job_id
        except Exception as e:
            logger.error(f"Error creating job: {e}")
            db.rollback()
            raise e
        finally:
            db.close()
    
    async def start_job(self, job_id: str, db: Session):
        """Start processing a job in the background"""
        # We need to run this on the main loop or similar? 
        # This function identifies the job and spawns the processing task.
        
        # Check DB for job
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        if job.status != JobStatus.PENDING:
            # If it's already started by another worker?
            if job.status in [JobStatus.STARTED, JobStatus.PROCESSING]:
                logger.warning(f"Job {job_id} already started")
                return None
            # If failed/completed, etc.
            raise ValueError(f"Job {job_id} is not in pending status")
        
        # Update status to STARTED
        job.status = JobStatus.STARTED
        job.started_at = datetime.utcnow()
        job.message = "Task generation started"
        db.commit()
        
        # Retrieve payload from result stash
        payload = job.result.get('payload') if job.result else {}
        
        # Start the background task
        # We pass the payload explicitly to _process_job
        task = asyncio.create_task(self._process_job(job_id, payload, db))
        self.running_tasks[job_id] = task
        
        logger.info(f"Started job {job_id}")
        return task
    
    async def _process_job(self, job_id: str, payload: Dict[str, Any], db: Session):
        """Process a task generation job or simulate_ai_respondents job"""
        try:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if not job:
                return

            # Dispatch by job_type for non-task-generation jobs
            if payload.get('job_type') == 'simulate_ai_respondents':
                await self._run_simulate_ai_respondents_async(job_id, payload, db)
                return

            job.status = JobStatus.PROCESSING
            job.message = "Generating tasks..."
            db.commit()
            
            # Process based on study type (task generation)
            study_type = payload.get('study_type')
            result = None
            
            if study_type in ('grid', 'text'):
                result = await self._generate_grid_tasks_async(job, payload, db)
            elif study_type == 'layer':
                result = await self._generate_layer_tasks_async(job, payload, db)
            elif study_type == 'hybrid':
                result = await self._generate_hybrid_tasks_async(job, payload, db)
            else:
                raise ValueError(f"Unsupported study type: {study_type}")
            
            # Use a fresh DB session for save phase (original session may be stale after long-running generation)
            db_fresh = SessionLocal()
            try:
                await self._save_results(job, payload, result, db_fresh)
                await self._launch_study(job, payload, db_fresh)
                job_fresh = db_fresh.query(Job).filter(Job.job_id == job_id).first()
                if job_fresh:
                    job_fresh.status = JobStatus.COMPLETED
                    job_fresh.completed_at = datetime.utcnow()
                    job_fresh.progress = 100.0
                    job_fresh.message = "Task generation completed successfully"
                    job_fresh.result = None
                    db_fresh.commit()
                logger.info(f"Job {job_id} completed successfully")
            finally:
                db_fresh.close()
            
        except RuntimeError as e:
            # Use fresh session in case original db connection is stale
            db_err = SessionLocal()
            try:
                db.rollback()
            except Exception:
                pass
            try:
                job_err = db_err.query(Job).filter(Job.job_id == job_id).first()
                if job_err:
                    job_err.status = JobStatus.FAILED
                    job_err.completed_at = datetime.utcnow()
                    job_err.error = str(e)
                    job_err.message = f"Study configuration error: {str(e)}"
                    db_err.commit()
                logger.error(f"Job {job_id} failed due to study configuration: {e}")
            finally:
                db_err.close()

        except Exception as e:
            # Use fresh session in case original db connection is stale (e.g. after long run)
            db_err = SessionLocal()
            try:
                db.rollback()
            except Exception:
                pass
            try:
                job_err = db_err.query(Job).filter(Job.job_id == job_id).first()
                if job_err:
                    job_err.status = JobStatus.FAILED
                    job_err.completed_at = datetime.utcnow()
                    job_err.error = str(e)
                    job_err.message = f"Task generation failed: {str(e)}"
                    db_err.commit()
                logger.error(f"Job {job_id} failed: {e}")
                import traceback
                logger.error(f"Job {job_id} traceback: {traceback.format_exc()}")
            finally:
                db_err.close()
            
        finally:
            if job_id in self.running_tasks:
                del self.running_tasks[job_id]

    async def _run_simulate_ai_respondents_async(self, job_id: str, payload: Dict[str, Any], db: Session):
        """Run AI respondent simulation in a thread and update job status."""
        from uuid import UUID
        from app.db.session import SessionLocal
        from app.services.synthetic_simulation_service import run_simulation

        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            return
        job.status = JobStatus.PROCESSING
        job.message = "Simulating AI respondents..."
        db.commit()

        def run_sync():
            db_sim = SessionLocal()
            try:
                study_id = UUID(job.study_id)
                import os
                progress_stdout = os.environ.get("SYNTHETIC_PROGRESS_STDOUT", "").strip().lower() in ("1", "true", "yes")

                def progress(done: int, total: int, msg: str):
                    try:
                        pct = (100.0 * done / total) if total else 0
                        line = f"[Simulate AI respondents] {msg} — progress: {pct:.0f}% ({done}/{total})"
                        logger.info(line)
                        if progress_stdout:
                            print(line, flush=True)
                        j = db_sim.query(Job).filter(Job.job_id == job_id).first()
                        if j:
                            j.progress = pct
                            j.message = msg
                            db_sim.commit()
                    except Exception as e:
                        try:
                            logger.warning("Progress callback error (non-fatal): %s", e)
                        except Exception:
                            pass
                        try:
                            db_sim.rollback()
                        except Exception:
                            pass
                return run_simulation(
                    db_sim,
                    study_id=study_id,
                    max_respondents=payload.get("max_respondents"),
                    progress_callback=progress,
                    max_panelist_workers=payload.get("max_panelist_workers"),
                )
            finally:
                db_sim.close()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_sync)
        except Exception as e:
            result = {"success": False, "message": str(e), "error": str(e)}

        job = db.query(Job).filter(Job.job_id == job_id).first()
        if job:
            job.completed_at = datetime.utcnow()
            job.progress = 100.0
            job.result = self._make_json_serializable(result)
            if result.get("success"):
                job.status = JobStatus.COMPLETED
                job.message = result.get("message", "Simulation completed.")
                job.error = None
            else:
                job.status = JobStatus.FAILED
                job.message = result.get("message", "Simulation failed.")
                job.error = result.get("error")
            db.commit()
            logger.info(
                f"Job {job_id} simulate_ai_respondents finished: success={result.get('success')}, "
                f"message={result.get('message', '')}"
            )
    
    # --- Helper methods need to accept payload explicitly now since job is DB model ---
    
    async def _generate_grid_tasks_async(self, job: Job, payload: Dict, db: Session):
        return await self._generate_tasks_for_phase(job, payload, db, phase_type=None)

    async def _generate_tasks_for_phase(self, job: Job, payload: Dict, db: Session, phase_type: Optional[str] = None, progress_range: tuple = (20.0, 90.0)):
        from app.services.task_generation_core import generate_grid_tasks_v2
        
        categories_data = []
        payload_categories = payload.get('categories') or []
        payload_elements = payload.get('elements') or []
        
        filtered_categories = payload_categories
        if phase_type:
            filtered_categories = [c for c in payload_categories if c.get('phase_type') == phase_type]
            
        if not filtered_categories:
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
        
        p_start, p_end = progress_range
        
        # We can't easily update DB on every callback if it's too frequent, 
        # but let's try. Or use a throttle.
        # Capturing db/job in closure might be tricky with sessions.
        # Ideally we don't commit every percent. 
        # For now, let's just do it.
        
        def on_progress(done: int, N: int):
            try:
                frac = max(0.0, min(1.0, done / max(1, N)))
                current_progress = p_start + ((p_end - p_start) * frac)
                
                db_progress = SessionLocal()
                try:
                    job_progress = db_progress.query(Job).filter(Job.job_id == job.job_id).first()
                    if job_progress and abs(job_progress.progress - current_progress) > 1.0:
                        job_progress.progress = current_progress
                        job_progress.message = f"Building respondents{f' ({phase_type})' if phase_type else ''} {done}/{N}..."
                        db_progress.commit()
                except Exception:
                    db_progress.rollback()
                finally:
                    db_progress.close()
            except Exception:
                pass
        
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: generate_grid_tasks_v2(
                        categories_data=categories_data,
                        number_of_respondents=payload.get('audience_segmentation', {}).get('number_of_respondents', 0),
                        exposure_tolerance_cv=payload.get('exposure_tolerance_cv', 1.0),
                        seed=payload.get('seed'),
                        progress_callback=on_progress
                    )
                ),
                timeout=settings.TASK_GENERATION_TIMEOUT
            )
        except RuntimeError as e:
            if "Preflight failed" in str(e) or "consider more elements" in str(e):
                 raise RuntimeError(f"Study configuration error in {phase_type} phase: {str(e)}.")
            raise
        
        return result

    async def _generate_hybrid_tasks_async(self, job: Job, payload: Dict, db: Session):
        phase_order = payload.get('phase_order', [])
        if not phase_order:
            raise ValueError("Hybrid study requires phase_order")
            
        job.progress = 10.0
        job.message = "Initializing hybrid task generation..."
        db.commit()
        
        combined_tasks = {}
        combined_metadata = {"phases": []}
        
        is_mix = "mix" in phase_order
        payload_categories = payload.get('categories') or []
        
        target_phases = []
        if is_mix:
            target_phases = list(set(c.get('phase_type') for i, c in enumerate(payload_categories) if c.get('phase_type')))
        else:
            target_phases = phase_order

        num_phases = len(target_phases)
        tasks_per_respondent_mix = {}

        for i, phase_type in enumerate(target_phases):
            p_start = 10.0 + (i * (80.0 / num_phases))
            p_end = 10.0 + ((i + 1) * (80.0 / num_phases))
            
            job.message = f"Starting generation for phase: {phase_type}..."
            db.commit()
            
            phase_result = await self._generate_tasks_for_phase(job, payload, db, phase_type=phase_type, progress_range=(p_start, p_end))
            
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
            db.commit()
            for resp_id, t_list in tasks_per_respondent_mix.items():
                random.shuffle(t_list)
                for idx, t in enumerate(t_list):
                    t['task_index'] = idx
                combined_tasks[resp_id] = t_list

        job.progress = 90.0
        job.message = "Combining all phases..."
        db.commit()
        
        return {
            "tasks": combined_tasks,
            "metadata": combined_metadata
        }
    
    async def _generate_layer_tasks_async(self, job: Job, payload: Dict, db: Session):
        job.progress = 10.0
        job.message = "Planning layer task generation..."
        db.commit()
        
        layers = []
        for layer in payload.get('study_layers', []):
            layer_obj = type('Layer', (), {
                'name': layer.get('name', ''),
                'images': [type('Image', (), {
                    'name': img.get('name', ''),
                    'url': img.get('url', '')
                }) for img in layer.get('images', [])],
                'z_index': layer.get('z_index', 0),
                'order': layer.get('order', 0)
            })()
            layers.append(layer_obj)
        
        job.progress = 20.0
        job.message = "Starting layer task generation..."
        db.commit()
        
        p_start = 20.0
        p_end = 90.0
        
        # Prepare progress callback with DB updates
        def on_progress(done: int, N: int):
            try:
                # Calculate progress
                frac = max(0.0, min(1.0, done / max(1, N)))
                current_progress = p_start + ((p_end - p_start) * frac)
                
                # Update DB with fresh session
                # Throttling could be added here if needed, but for now we update
                # safely using a new session for this thread
                db_progress = SessionLocal()
                try:
                    job_progress = db_progress.query(Job).filter(Job.job_id == job.job_id).first()
                    if job_progress:
                        # Only update if progress changed significantly or it's been a while?
                        # For simplicity, we update every time but we could check against current value.
                        if abs(job_progress.progress - current_progress) > 1.0: # Update every 1%
                            job_progress.progress = current_progress
                            job_progress.message = f"Building respondents {done}/{N}..."
                            db_progress.commit()
                except Exception:
                    db_progress.rollback()
                finally:
                    db_progress.close()
            except Exception:
                pass
        
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
                        number_of_respondents=payload.get('audience_segmentation', {}).get('number_of_respondents', 0),
                        exposure_tolerance_pct=payload.get('exposure_tolerance_pct', 2.0),
                        seed=payload.get('seed'),
                        progress_callback=on_progress
                    )
                ),
                timeout=settings.TASK_GENERATION_TIMEOUT
            )
        except RuntimeError as e:
            if "Preflight failed" in str(e) or "consider more elements" in str(e):
                raise RuntimeError(f"Study configuration error: {str(e)}.")
            raise
        
        job.progress = 90.0
        job.message = "Saving generated tasks..."
        db.commit()
        
        # Attach background image URL
        if isinstance(result, dict):
            try:
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
    
    async def _save_results(self, job: Job, payload: Dict, result: Dict[str, Any], db: Session):
        """Save generated tasks to the study"""
        from app.models.study_model import Study
        
        study = db.query(Study).filter(
            Study.id == job.study_id,
            Study.creator_id == job.user_id
        ).first()
        
        if not study:
           # Try to find by member access if creator check fails (unified access)
           # We use the same unified logic as implemented earlier
           from app.models.study_model import StudyMember
           stmt_member = db.query(Study).join(StudyMember).filter(
               Study.id == job.study_id,
               StudyMember.user_id == job.user_id
           )
           study = stmt_member.first()
        
        if not study:
            raise ValueError(f"Study {job.study_id} not found or access denied")
        
        study.tasks = result.get('tasks', {})
        
        last_step_val = payload.get('last_step')
        if last_step_val is not None and isinstance(last_step_val, int):
            current_step = getattr(study, 'last_step', 1) or 1
            if last_step_val > current_step:
                study.last_step = last_step_val
        
        db.commit()
        db.refresh(study)
        
        # Update our temporary result object to include last_step so the final update picks it up
        if result:
            result['last_step'] = study.last_step
        
        logger.info(f"Saved tasks for study {job.study_id}")
    
    async def _launch_study(self, job: Job, payload: Dict, db: Session):
        """Launch study"""
        try:
            from app.services.study import update_study
            from app.schemas.study_schema import StudyUpdate
            
            study_updates = {}
            # Map payload fields to update
            for field in ['title', 'background', 'language', 'main_question', 'orientation_text', 
                          'background_image_url', 'rating_scale', 'audience_segmentation', 'phase_order']:
                if payload.get(field):
                    study_updates[field] = payload.get(field)
            
            if study_updates:
                update_payload = StudyUpdate(**study_updates)
                update_study(
                    db=db,
                    study_id=uuid.UUID(job.study_id),
                    owner_id=uuid.UUID(job.user_id),
                    payload=update_payload
                )
        except Exception as e:
            logger.error(f"Failed to launch study {job.study_id}: {e}")
    
    def get_job_status(self, job_id: str) -> Optional[Job]:
        """Get the status of a job from DB"""
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            return job
        finally:
            db.close()
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if not job:
                return False
                
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return False
            
            # Cancel asyncio task if on this instance
            if job_id in self.running_tasks:
                self.running_tasks[job_id].cancel()
                del self.running_tasks[job_id]
            
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            job.message = "Job cancelled by user"
            db.commit()
            return True
        finally:
            db.close()
    
    def get_user_jobs(self, user_id: str) -> List[Job]:
        """Get all jobs for a user"""
        db = SessionLocal()
        try:
            jobs = db.query(Job).filter(Job.user_id == user_id).order_by(Job.created_at.desc()).all()
            return jobs
        finally:
            db.close()

background_task_service = BackgroundTaskService()
