"""WebSocket endpoints for real-time streaming (analytics and task generation)."""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.security import verify_token
from app.db.session import get_db, SessionLocal
from app.models.user_model import User
from app.models.study_model import Study, StudyMember
from app.models.job_model import Job, JobStatus
from app.services.response import StudyResponseService
from app.websocket.manager import analytics_manager
from app.websocket.job_notifier import job_progress_notifier

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_user_from_token(token: str, db: Session) -> User | None:
    """Validate JWT token and return the user if valid."""
    if not token:
        return None
    
    payload = verify_token(token, "access")
    if not payload:
        return None
    
    user_id = payload.get("sub")
    if not user_id:
        return None
    
    try:
        user = db.query(User).filter(User.id == UUID(user_id)).first()
        if user and user.is_active:
            return user
    except (ValueError, Exception) as e:
        logger.debug(f"Token validation error: {e}")
    
    return None


def check_study_access(db: Session, study_id: UUID, user_id: UUID) -> bool:
    """Check if user has access to the study (owner or member)."""
    ownership_check = select(Study.creator_id).where(Study.id == study_id)
    result = db.execute(ownership_check).first()
    
    if not result:
        return False
    
    if result.creator_id == user_id:
        return True
    
    member = db.scalar(
        select(StudyMember).where(
            StudyMember.study_id == study_id,
            StudyMember.user_id == user_id
        )
    )
    return member is not None


@router.websocket("/analytics/{study_id}")
async def websocket_analytics(
    websocket: WebSocket,
    study_id: UUID,
    token: str = Query(..., description="JWT access token"),
    interval_seconds: int = Query(5, ge=5, le=60, description="Update interval in seconds"),
):
    """
    WebSocket endpoint for real-time analytics streaming.
    
    Connect with: ws://host/api/v1/ws/analytics/{study_id}?token={jwt_token}
    
    Sends analytics data every `interval_seconds` and a ping every 30 seconds
    to keep the connection alive (Azure has 240s timeout).
    Uses a short-lived DB session per loop iteration to avoid stale connections.
    """
    # Auth phase: one session only for token + access check, then close
    db_auth = next(get_db())
    try:
        user = await get_user_from_token(token, db_auth)
        if not user:
            await websocket.close(code=4001, reason="Invalid or expired token")
            return
        if not check_study_access(db_auth, study_id, user.id):
            await websocket.close(code=4003, reason="Access denied to study")
            return
    finally:
        db_auth.close()

    try:
        await analytics_manager.connect(str(study_id), websocket)
        last_payload: str | None = None
        ping_counter = 0

        try:
            while True:
                try:
                    # Per-iteration session: create, use, close (no long-lived session)
                    db = SessionLocal()
                    try:
                        service = StudyResponseService(db)
                        analytics = service.get_study_analytics(study_id)
                        payload = analytics.model_dump()
                        payload_str = str(payload)

                        if payload_str != last_payload:
                            await analytics_manager.send_to_connection(
                                websocket,
                                {"type": "data", "payload": payload}
                            )
                            last_payload = payload_str
                    finally:
                        db.close()

                    ping_counter += interval_seconds
                    if ping_counter >= 30:
                        await analytics_manager.send_to_connection(
                            websocket,
                            {"type": "ping"}
                        )
                        ping_counter = 0

                    await asyncio.sleep(interval_seconds)

                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error in WebSocket analytics loop: {e}")
                    await asyncio.sleep(interval_seconds)

        finally:
            analytics_manager.disconnect(str(study_id), websocket)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for study {study_id}")
    except Exception as e:
        logger.error(f"WebSocket error for study {study_id}: {e}")
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except Exception:
            pass


@router.websocket("/task-generation/{job_id}")
async def websocket_task_generation(
    websocket: WebSocket,
    job_id: str,
    token: str = Query(..., description="JWT access token"),
):
    """
    WebSocket endpoint for real-time task generation progress.
    
    Connect with: ws://host/api/v1/ws/task-generation/{job_id}?token={jwt_token}
    
    Receives progress updates as they happen (event-driven via asyncio.Queue).
    Messages: {"type": "progress|completed|failed", "progress": float, "message": str, "error"?: str}
    
    Connection closes automatically when job completes or fails.
    Timeout after 15 minutes to prevent orphaned connections.
    """
    # Auth phase: verify token and job ownership
    db_auth = next(get_db())
    try:
        user = await get_user_from_token(token, db_auth)
        if not user:
            await websocket.close(code=4001, reason="Invalid or expired token")
            return
        
        # Check if job exists and user owns it
        job = db_auth.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            await websocket.close(code=4004, reason="Job not found")
            return
        
        # Allow any study creator or member to see job progress (not only job starter)
        if not check_study_access(db_auth, UUID(job.study_id), user.id):
            await websocket.close(code=4003, reason="Access denied to study")
            return
        
        # Check if job is already completed or failed
        if job.status == JobStatus.COMPLETED:
            await websocket.accept()
            await websocket.send_json({
                "type": "completed",
                "progress": 100.0,
                "message": job.message or "Task generation completed"
            })
            await websocket.close()
            return
        
        if job.status == JobStatus.FAILED:
            await websocket.accept()
            await websocket.send_json({
                "type": "failed",
                "error": job.error or "Unknown error",
                "message": job.message or "Task generation failed"
            })
            await websocket.close()
            return
        
        if job.status == JobStatus.CANCELLED:
            await websocket.accept()
            await websocket.send_json({
                "type": "failed",
                "error": "Job was cancelled",
                "message": "Task generation was cancelled"
            })
            await websocket.close()
            return
        
        # Get current progress to send initial state
        initial_progress = job.progress or 0.0
        initial_message = job.message or "Task generation in progress..."
        
    finally:
        db_auth.close()

    # Subscribe to job progress notifications
    queue = await job_progress_notifier.subscribe(job_id)

    # Re-check job status after subscribe: job may have completed before we subscribed (race fix)
    db_recheck = next(get_db())
    try:
        job_recheck = db_recheck.query(Job).filter(Job.job_id == job_id).first()
        if job_recheck:
            if job_recheck.status == JobStatus.COMPLETED:
                await websocket.accept()
                await websocket.send_json({
                    "type": "completed",
                    "progress": 100.0,
                    "message": job_recheck.message or "Task generation completed"
                })
                await websocket.close()
                await job_progress_notifier.unsubscribe(job_id, queue)
                return
            if job_recheck.status == JobStatus.FAILED:
                await websocket.accept()
                await websocket.send_json({
                    "type": "failed",
                    "error": job_recheck.error or "Unknown error",
                    "message": job_recheck.message or "Task generation failed"
                })
                await websocket.close()
                await job_progress_notifier.unsubscribe(job_id, queue)
                return
            if job_recheck.status == JobStatus.CANCELLED:
                await websocket.accept()
                await websocket.send_json({
                    "type": "failed",
                    "error": "Job was cancelled",
                    "message": "Task generation was cancelled"
                })
                await websocket.close()
                await job_progress_notifier.unsubscribe(job_id, queue)
                return
    finally:
        db_recheck.close()

    try:
        await websocket.accept()
        logger.info(f"WebSocket connected for task generation job {job_id}")
        
        # Send initial progress state
        await websocket.send_json({
            "type": "progress",
            "progress": initial_progress,
            "message": initial_message
        })
        
        # Wait for updates with a 15-minute timeout (max job duration)
        timeout_seconds = 15 * 60  # 15 minutes
        ping_interval = 30  # Send ping every 30 seconds
        last_ping = asyncio.get_event_loop().time()
        
        while True:
            try:
                # Wait for next update with timeout
                try:
                    update = await asyncio.wait_for(queue.get(), timeout=ping_interval)
                except asyncio.TimeoutError:
                    # No update received, send ping to keep connection alive
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_ping >= ping_interval:
                        await websocket.send_json({"type": "ping"})
                        last_ping = current_time
                    
                    # Check overall timeout
                    elapsed = current_time - (asyncio.get_event_loop().time() - timeout_seconds)
                    if elapsed <= 0:
                        logger.warning(f"WebSocket timeout for job {job_id}")
                        await websocket.send_json({
                            "type": "failed",
                            "error": "Connection timeout",
                            "message": "Task generation is taking too long"
                        })
                        break
                    continue
                
                # Send update to client
                await websocket.send_json(update)
                
                # Close connection if job completed or failed
                if update.get("type") in ("completed", "failed"):
                    logger.info(f"Job {job_id} finished with status: {update.get('type')}")
                    break
                    
            except WebSocketDisconnect:
                logger.info(f"Client disconnected from job {job_id}")
                break
            except Exception as e:
                logger.error(f"Error in task generation WebSocket loop: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except Exception:
            pass
    finally:
        # Unsubscribe from notifications
        await job_progress_notifier.unsubscribe(job_id, queue)
