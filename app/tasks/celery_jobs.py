"""
Celery jobs for background task processing.
- celery_job.generate_tasks: Task generation (grid, layer, hybrid)
- celery_job.simulate_synthetic_respondents: AI/random respondent simulation
"""
import logging
import random
import time
from typing import Dict, Any, Optional
from datetime import datetime

from app.celery_app import celery_app

# Ensure all SQLAlchemy models are loaded before DB operations
import app.models.user_model  # noqa: F401
import app.models.project_model  # noqa: F401
import app.models.study_model  # noqa: F401
import app.models.response_model  # noqa: F401
import app.models.job_model  # noqa: F401

logger = logging.getLogger(__name__)


def _update_job_progress(job_id: str, progress: float, message: str) -> None:
    """Update job progress in DB and notify WebSocket subscribers via Redis."""
    from app.db.session import SessionLocal
    from app.models.job_model import Job
    from app.websocket.job_notifier import job_progress_notifier

    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if job and abs(job.progress - progress) > 1.0:
            job.progress = progress
            job.message = message
            db.commit()

            # Notify WebSocket subscribers (Redis pub/sub for cross-worker)
            job_progress_notifier.notify(
                job_id,
                {"type": "progress", "progress": progress, "message": message},
            )
    except Exception as e:
        logger.warning(f"Progress update failed (non-fatal): {e}")
        db.rollback()
    finally:
        try:
            db.close()
        except Exception:
            pass


def _generate_grid_tasks(
    job_id: str,
    payload: Dict,
    phase_type: Optional[str] = None,
    progress_range: tuple = (20.0, 90.0),
) -> Dict[str, Any]:
    """Generate grid-style tasks for a phase."""
    from app.services.task_generation_core import generate_grid_tasks_v2

    categories_data = []
    payload_categories = payload.get("categories") or []
    payload_elements = payload.get("elements") or []

    filtered_categories = payload_categories
    if phase_type:
        filtered_categories = [c for c in payload_categories if c.get("phase_type") == phase_type]

    if not filtered_categories:
        return {"tasks": {}, "metadata": {}}

    for cat in filtered_categories:
        cat_elements = [e for e in payload_elements if e.get("category_id") == cat.get("category_id")]
        categories_data.append(
            {
                "category_name": cat.get("name"),
                "elements": [
                    {
                        "element_id": str(el.get("element_id")),
                        "name": el.get("name"),
                        "content": el.get("content"),
                        "alt_text": el.get("alt_text") or el.get("name"),
                        "element_type": el.get("element_type"),
                    }
                    for el in cat_elements
                ],
            }
        )

    p_start, p_end = progress_range

    def on_progress(done: int, N: int) -> None:
        try:
            frac = max(0.0, min(1.0, done / max(1, N)))
            current_progress = p_start + ((p_end - p_start) * frac)
            phase_msg = f" ({phase_type})" if phase_type else ""
            _update_job_progress(job_id, current_progress, f"Building respondents{phase_msg} {done}/{N}...")
        except Exception:
            pass

    try:
        result = generate_grid_tasks_v2(
            categories_data=categories_data,
            number_of_respondents=payload.get("audience_segmentation", {}).get("number_of_respondents", 0),
            exposure_tolerance_cv=payload.get("exposure_tolerance_cv", 1.0),
            seed=payload.get("seed"),
            progress_callback=on_progress,
        )
        return result
    except RuntimeError as e:
        if "Preflight failed" in str(e) or "consider more elements" in str(e):
            raise RuntimeError(
                f"Study configuration error{' in ' + phase_type + ' phase' if phase_type else ''}: {str(e)}."
            )
        raise


def _generate_layer_tasks(job_id: str, payload: Dict) -> Dict[str, Any]:
    """Generate layer-style tasks."""
    from app.db.session import SessionLocal
    from app.models.study_model import Study
    from app.services.task_generation_core import generate_layer_tasks_v2

    _update_job_progress(job_id, 10.0, "Planning layer task generation...")

    layers = []
    for layer in payload.get("study_layers", []):
        layer_obj = {
            "name": layer.get("name", ""),
            "z_index": layer.get("z_index", 0),
            "order": layer.get("order", 0),
            "images": [{"name": img.get("name", ""), "url": img.get("url", "")} for img in layer.get("images", [])],
        }
        layers.append(layer_obj)

    _update_job_progress(job_id, 20.0, "Starting layer task generation...")

    p_start, p_end = 20.0, 90.0

    def on_progress(done: int, N: int) -> None:
        try:
            frac = max(0.0, min(1.0, done / max(1, N)))
            current_progress = p_start + ((p_end - p_start) * frac)
            _update_job_progress(job_id, current_progress, f"Building respondents {done}/{N}...")
        except Exception:
            pass

    try:
        result = generate_layer_tasks_v2(
            layers_data=layers,
            number_of_respondents=payload.get("audience_segmentation", {}).get("number_of_respondents", 0),
            exposure_tolerance_pct=payload.get("exposure_tolerance_pct", 2.0),
            seed=payload.get("seed"),
            progress_callback=on_progress,
        )
    except RuntimeError as e:
        if "Preflight failed" in str(e) or "consider more elements" in str(e):
            raise RuntimeError(f"Study configuration error: {str(e)}.")
        raise

    _update_job_progress(job_id, 90.0, "Saving generated tasks...")

    # Attach background image URL
    if isinstance(result, dict):
        try:
            study_id = payload.get("study_id")
            if study_id:
                db = SessionLocal()
                try:
                    from uuid import UUID
                    study = db.query(Study).filter(Study.id == UUID(str(study_id))).first()
                    if study and hasattr(study, "background_image_url"):
                        bg_url = getattr(study, "background_image_url", None)
                        if bg_url is not None:
                            meta = result.get("metadata", {})
                            if isinstance(meta, dict):
                                meta["background_image_url"] = bg_url
                                result["metadata"] = meta
                finally:
                    try:
                        db.close()
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Could not attach background image URL: {e}")

    return result


def _generate_hybrid_tasks(job_id: str, payload: Dict) -> Dict[str, Any]:
    """Generate hybrid (multi-phase) tasks."""
    phase_order = payload.get("phase_order", [])
    if not phase_order:
        raise ValueError("Hybrid study requires phase_order")

    _update_job_progress(job_id, 10.0, "Initializing hybrid task generation...")

    combined_tasks = {}
    combined_metadata = {"phases": []}

    is_mix = "mix" in phase_order
    payload_categories = payload.get("categories") or []

    target_phases = []
    if is_mix:
        target_phases = list(set(c.get("phase_type") for c in payload_categories if c.get("phase_type")))
    else:
        target_phases = phase_order

    num_phases = len(target_phases)
    tasks_per_respondent_mix = {}

    for i, phase_type in enumerate(target_phases):
        p_start = 10.0 + (i * (80.0 / num_phases))
        p_end = 10.0 + ((i + 1) * (80.0 / num_phases))

        _update_job_progress(job_id, p_start, f"Starting generation for phase: {phase_type}...")

        phase_result = _generate_grid_tasks(
            job_id, payload, phase_type=phase_type, progress_range=(p_start, p_end)
        )

        phase_tasks = phase_result.get("tasks", {})

        if is_mix:
            for resp_id, t_list in phase_tasks.items():
                if resp_id not in tasks_per_respondent_mix:
                    tasks_per_respondent_mix[resp_id] = []
                for t in t_list:
                    t["phase_type"] = phase_type
                    tasks_per_respondent_mix[resp_id].append(t)
        else:
            for resp_id, tasks in phase_tasks.items():
                if resp_id not in combined_tasks:
                    combined_tasks[resp_id] = []

                offset = len(combined_tasks[resp_id])
                for t in tasks:
                    t["task_index"] = int(t.get("task_index", 0)) + offset
                    t["phase_type"] = phase_type

                combined_tasks[resp_id].extend(tasks)

        combined_metadata["phases"].append(
            {"phase_type": phase_type, "metadata": phase_result.get("metadata", {})}
        )

    if is_mix:
        _update_job_progress(job_id, 85.0, "Randomizing mixed tasks...")
        for resp_id, t_list in tasks_per_respondent_mix.items():
            random.shuffle(t_list)
            for idx, t in enumerate(t_list):
                t["task_index"] = idx
            combined_tasks[resp_id] = t_list

    _update_job_progress(job_id, 90.0, "Combining all phases...")

    return {"tasks": combined_tasks, "metadata": combined_metadata}


def _save_results(job_id: str, study_id: str, user_id: str, payload: Dict, result: Dict[str, Any]) -> None:
    """Save generated tasks to the study."""
    from uuid import UUID
    from app.db.session import SessionLocal
    from app.models.study_model import Study, StudyMember

    db = SessionLocal()
    try:
        study = (
            db.query(Study)
            .filter(Study.id == UUID(study_id), Study.creator_id == UUID(user_id))
            .first()
        )

        if not study:
            stmt_member = db.query(Study).join(StudyMember).filter(
                Study.id == UUID(study_id), StudyMember.user_id == UUID(user_id)
            )
            study = stmt_member.first()

        if not study:
            raise ValueError(f"Study {study_id} not found or access denied")

        study.tasks = result.get("tasks", {})

        last_step_val = payload.get("last_step")
        if last_step_val is not None and isinstance(last_step_val, int):
            current_step = getattr(study, "last_step", 1) or 1
            if last_step_val > current_step:
                study.last_step = last_step_val

        db.commit()
        logger.info(f"Saved tasks for study {study_id}")
    finally:
        try:
            db.close()
        except Exception:
            pass


def _launch_study(job_id: str, study_id: str, user_id: str, payload: Dict) -> None:
    """Launch study: update title, background, etc. from payload."""
    try:
        from uuid import UUID
        from app.db.session import SessionLocal
        from app.services.study import update_study
        from app.schemas.study_schema import StudyUpdate

        study_updates = {}
        for field in [
            "title",
            "background",
            "language",
            "main_question",
            "orientation_text",
            "background_image_url",
            "rating_scale",
            "audience_segmentation",
            "phase_order",
        ]:
            if payload.get(field) is not None:
                study_updates[field] = payload.get(field)

        if study_updates:
            db = SessionLocal()
            try:
                update_payload = StudyUpdate(**study_updates)
                update_study(
                    db=db,
                    study_id=UUID(study_id),
                    owner_id=UUID(user_id),
                    payload=update_payload,
                )
            finally:
                try:
                    db.close()
                except Exception:
                    pass
    except Exception as e:
        logger.error(f"Failed to launch study {study_id}: {e}")


@celery_app.task(bind=True, name="celery_job.generate_tasks")
def generate_tasks_celery(
    self, job_id: str, study_id: str, user_id: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Celery job for generating study tasks in background.
    Runs in a separate worker process, freeing web workers.
    """
    if payload.get("seed") is None:
        payload = {**payload, "seed": int(time.time() * 1000) % (2**31)}

    # Ensure study_id is in payload for layer/background URL lookup
    payload = {**payload, "study_id": study_id}

    from app.db.session import SessionLocal
    from app.models.job_model import Job, JobStatus
    from app.websocket.job_notifier import job_progress_notifier

    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return {"success": False, "error": "Job not found"}

        job.status = JobStatus.PROCESSING
        job.message = "Generating tasks..."
        job.started_at = datetime.utcnow()
        db.commit()
        try:
            db.close()
        except Exception:
            pass

        study_type = payload.get("study_type")
        result = None

        if study_type in ("grid", "text"):
            result = _generate_grid_tasks(job_id, payload)
        elif study_type == "layer":
            result = _generate_layer_tasks(job_id, payload)
        elif study_type == "hybrid":
            result = _generate_hybrid_tasks(job_id, payload)
        else:
            raise ValueError(f"Unsupported study type: {study_type}")

        _save_results(job_id, study_id, user_id, payload, result)
        _launch_study(job_id, study_id, user_id, payload)

        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.progress = 100.0
                job.message = "Task generation completed successfully"
                job.result = None
                db.commit()

            job_progress_notifier.notify(
                job_id,
                {
                    "type": "completed",
                    "progress": 100.0,
                    "message": "Task generation completed successfully",
                },
            )
            logger.info(f"Job {job_id} completed successfully")
        finally:
            try:
                db.close()
            except Exception:
                pass

        return {"success": True, "job_id": job_id}

    except RuntimeError as e:
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.utcnow()
                job.error = str(e)
                job.message = f"Study configuration error: {str(e)}"
                db.commit()

            job_progress_notifier.notify(
                job_id,
                {"type": "failed", "error": str(e), "message": f"Study configuration error: {str(e)}"},
            )
            logger.error(f"Job {job_id} failed due to study configuration: {e}")
        finally:
            try:
                db.close()
            except Exception:
                pass
        return {"success": False, "error": str(e)}

    except Exception as e:
        import traceback

        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.utcnow()
                job.error = str(e)
                job.message = f"Task generation failed: {str(e)}"
                db.commit()

            job_progress_notifier.notify(
                job_id,
                {"type": "failed", "error": str(e), "message": f"Task generation failed: {str(e)}"},
            )
            logger.error(f"Job {job_id} failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            try:
                db.close()
            except Exception:
                pass
        return {"success": False, "error": str(e)}


@celery_app.task(bind=True, name="celery_job.simulate_synthetic_respondents")
def simulate_synthetic_respondents_celery(
    self, job_id: str, study_id: str, user_id: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Celery job for simulating synthetic (AI or randomized) respondents.
    Runs in a separate worker process, freeing web workers.
    """
    from uuid import UUID
    from app.db.session import SessionLocal
    from app.models.job_model import Job, JobStatus
    from app.services.synthetic_simulation_service import run_simulation
    from app.websocket.job_notifier import job_progress_notifier

    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return {"success": False, "error": "Job not found"}

        job.status = JobStatus.PROCESSING
        job.message = "Simulating AI respondents..."
        job.started_at = datetime.utcnow()
        db.commit()
    finally:
        try:
            db.close()
        except Exception:
            pass

    def progress_callback(done: int, total: int, msg: str) -> None:
        try:
            pct = (100.0 * done / total) if total else 0
            _update_job_progress(job_id, pct, msg)
            logger.info(f"[Simulate AI respondents] {msg} — progress: {pct:.0f}% ({done}/{total})")

            job_progress_notifier.notify(
                job_id,
                {
                    "type": "progress",
                    "progress": pct,
                    "message": msg,
                    "respondents_completed": done,
                    "respondents_requested": total,
                },
            )
        except Exception as e:
            logger.warning(f"Progress callback error (non-fatal): {e}")

    try:
        db_sim = SessionLocal()
        try:
            result = run_simulation(
                db=db_sim,
                study_id=UUID(study_id),
                max_respondents=payload.get("max_respondents"),
                progress_callback=progress_callback,
                max_panelist_workers=payload.get("max_panelist_workers", 10),
                is_special_creator=payload.get("is_special_creator", False),
                randomize=payload.get("randomize", False),
            )
        finally:
            try:
                db_sim.close()
            except Exception:
                pass

        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                job.completed_at = datetime.utcnow()
                job.progress = 100.0
                if result.get("success"):
                    job.status = JobStatus.COMPLETED
                    job.message = result.get("message", "Simulation completed.")
                    job.error = None
                else:
                    job.status = JobStatus.FAILED
                    job.message = result.get("message", "Simulation failed.")
                    job.error = result.get("error")
                db.commit()

                max_respondents = payload.get("max_respondents", 0)
                if result.get("success"):
                    job_progress_notifier.notify(
                        job_id,
                        {
                            "type": "completed",
                            "progress": 100.0,
                            "message": job.message,
                            "respondents_completed": max_respondents,
                            "respondents_requested": max_respondents,
                        },
                    )
                else:
                    job_progress_notifier.notify(
                        job_id,
                        {
                            "type": "failed",
                            "error": job.error,
                            "message": job.message,
                            "respondents_completed": 0,
                            "respondents_requested": max_respondents,
                        },
                    )

            logger.info(
                f"Job {job_id} simulate_synthetic_respondents finished: success={result.get('success')}, "
                f"message={result.get('message', '')}"
            )
        finally:
            try:
                db.close()
            except Exception:
                pass

        return {
            "success": result.get("success", False),
            "job_id": job_id,
            "respondents_simulated": result.get("respondents_simulated", 0),
            "message": result.get("message", ""),
        }

    except Exception as e:
        import traceback

        logger.error(f"Job {job_id} failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.utcnow()
                job.error = str(e)
                job.message = f"Simulation failed: {str(e)}"
                db.commit()

            job_progress_notifier.notify(
                job_id,
                {
                    "type": "failed",
                    "error": str(e),
                    "message": f"Simulation failed: {str(e)}",
                    "respondents_completed": 0,
                    "respondents_requested": payload.get("max_respondents", 0),
                },
            )
        finally:
            try:
                db.close()
            except Exception:
                pass
        return {"success": False, "error": str(e)}
