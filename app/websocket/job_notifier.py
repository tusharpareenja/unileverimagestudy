"""Job progress notifier for event-driven WebSocket updates.

Uses Redis pub/sub for cross-process communication (multiple Gunicorn workers).
Falls back to in-memory queues if Redis is not configured.
"""

import asyncio
import logging
from typing import Dict, Set, Any, AsyncGenerator

from app.core.redis import publish_job_update, subscribe_to_job, is_redis_configured

logger = logging.getLogger(__name__)


class JobProgressNotifier:
    """
    Event-driven job progress notification system.
    
    Uses Redis pub/sub when configured (for multi-worker deployments).
    Falls back to in-memory asyncio.Queue when Redis is not available.
    
    Background jobs call notify() to push progress updates.
    WebSocket handlers use subscribe_redis() or subscribe() to receive updates.
    """

    def __init__(self):
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        """
        Subscribe to progress updates for a job (in-memory fallback).
        Returns an asyncio.Queue that receives progress notifications.
        
        Note: For multi-worker deployments, use subscribe_redis() instead.
        """
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            if job_id not in self._subscribers:
                self._subscribers[job_id] = set()
            self._subscribers[job_id].add(queue)
        logger.debug(f"Subscribed to job {job_id} (in-memory), total subscribers: {len(self._subscribers[job_id])}")
        return queue

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        """Remove a subscriber's queue from a job (in-memory fallback)."""
        async with self._lock:
            if job_id in self._subscribers:
                self._subscribers[job_id].discard(queue)
                if not self._subscribers[job_id]:
                    del self._subscribers[job_id]
        logger.debug(f"Unsubscribed from job {job_id} (in-memory)")

    def notify(self, job_id: str, data: Dict[str, Any]) -> None:
        """
        Push a progress update to all subscribers of a job.
        
        This is called from background job progress callbacks.
        
        When Redis is configured:
        - Publishes to Redis channel for cross-worker delivery
        - Also notifies in-memory subscribers (same-worker fallback)
        
        When Redis is not configured:
        - Only notifies in-memory subscribers
        
        data should contain:
        - type: "progress" | "completed" | "failed"
        - progress: float (0-100)
        - message: str
        - error: str (for failed type)
        """
        # Always try Redis first (for cross-worker delivery)
        if is_redis_configured():
            publish_job_update(job_id, data)
        
        # Also notify in-memory subscribers (for same-worker or fallback)
        if job_id not in self._subscribers:
            return
        
        dead_queues: Set[asyncio.Queue] = set()
        for queue in self._subscribers[job_id]:
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                logger.warning(f"Queue full for job {job_id}, dropping message")
            except Exception as e:
                logger.debug(f"Failed to notify subscriber for job {job_id}: {e}")
                dead_queues.add(queue)
        
        for queue in dead_queues:
            self._subscribers[job_id].discard(queue)
        
        if job_id in self._subscribers and not self._subscribers[job_id]:
            del self._subscribers[job_id]

    async def subscribe_redis(self, job_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Subscribe to job progress updates via Redis pub/sub.
        
        Use this for multi-worker deployments. Returns an async generator
        that yields progress updates as they arrive from any worker.
        
        Falls back to in-memory subscription if Redis is not configured.
        
        Args:
            job_id: The job identifier to subscribe to
            
        Yields:
            Dict with type, progress, message, error fields
        """
        if is_redis_configured():
            logger.debug(f"Using Redis subscription for job {job_id}")
            async for update in subscribe_to_job(job_id):
                yield update
        else:
            logger.debug(f"Redis not configured, using in-memory subscription for job {job_id}")
            queue = await self.subscribe(job_id)
            try:
                while True:
                    update = await queue.get()
                    yield update
            finally:
                await self.unsubscribe(job_id, queue)

    def has_subscribers(self, job_id: str) -> bool:
        """Check if a job has any in-memory subscribers."""
        return job_id in self._subscribers and len(self._subscribers[job_id]) > 0

    def get_subscriber_count(self, job_id: str) -> int:
        """Get the number of in-memory subscribers for a job."""
        return len(self._subscribers.get(job_id, set()))

    async def cleanup_job(self, job_id: str) -> None:
        """Remove all in-memory subscribers for a completed/failed job."""
        async with self._lock:
            if job_id in self._subscribers:
                del self._subscribers[job_id]
        logger.debug(f"Cleaned up in-memory subscribers for job {job_id}")


# Global singleton instance
job_progress_notifier = JobProgressNotifier()
