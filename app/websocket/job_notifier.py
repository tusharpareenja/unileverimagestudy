"""Job progress notifier for event-driven WebSocket updates."""

import asyncio
import logging
from typing import Dict, Set, Any

logger = logging.getLogger(__name__)


class JobProgressNotifier:
    """
    Event-driven job progress notification system using asyncio.Queue.
    
    Background jobs call notify() to push progress updates.
    WebSocket handlers subscribe() to receive updates via a queue
    """

    def __init__(self):
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        """
        Subscribe to progress updates for a job.
        Returns an asyncio.Queue that receives progress notifications.
        """
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            if job_id not in self._subscribers:
                self._subscribers[job_id] = set()
            self._subscribers[job_id].add(queue)
        logger.debug(f"Subscribed to job {job_id}, total subscribers: {len(self._subscribers[job_id])}")
        return queue

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        """Remove a subscriber's queue from a job."""
        async with self._lock:
            if job_id in self._subscribers:
                self._subscribers[job_id].discard(queue)
                if not self._subscribers[job_id]:
                    del self._subscribers[job_id]
        logger.debug(f"Unsubscribed from job {job_id}")

    def notify(self, job_id: str, data: Dict[str, Any]) -> None:
        """
        Push a progress update to all subscribers of a job.
        
        This is called from background job progress callbacks.
        Uses put_nowait since background jobs may not be in async context.
        
        data should contain:
        - type: "progress" | "completed" | "failed"
        - progress: float (0-100)
        - message: str
        - error: str (for failed type)
        """
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

    def has_subscribers(self, job_id: str) -> bool:
        """Check if a job has any active subscribers."""
        return job_id in self._subscribers and len(self._subscribers[job_id]) > 0

    def get_subscriber_count(self, job_id: str) -> int:
        """Get the number of subscribers for a job."""
        return len(self._subscribers.get(job_id, set()))

    async def cleanup_job(self, job_id: str) -> None:
        """Remove all subscribers for a completed/failed job."""
        async with self._lock:
            if job_id in self._subscribers:
                del self._subscribers[job_id]
        logger.debug(f"Cleaned up subscribers for job {job_id}")


# Global singleton instance
job_progress_notifier = JobProgressNotifier()
