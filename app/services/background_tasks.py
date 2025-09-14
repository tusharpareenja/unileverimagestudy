# app/services/background_tasks.py
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import logging

from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.response import StudyResponseService

logger = logging.getLogger(__name__)

class BackgroundTaskService:
    """Service for running background tasks."""
    
    def __init__(self):
        self.running = False
        self.task: Optional[asyncio.Task] = None
    
    async def start_abandonment_checker(self, interval_minutes: int = 30):
        """Start the background task to check for abandoned sessions."""
        if self.running:
            logger.warning("Abandonment checker is already running")
            return
        
        self.running = True
        logger.info(f"Starting abandonment checker with {interval_minutes} minute intervals")
        
        try:
            while self.running:
                await self._check_abandoned_sessions()
                await asyncio.sleep(interval_minutes * 60)  # Convert minutes to seconds
        except asyncio.CancelledError:
            logger.info("Abandonment checker was cancelled")
        except Exception as e:
            logger.error(f"Error in abandonment checker: {e}")
        finally:
            self.running = False
    
    async def _check_abandoned_sessions(self):
        """Check for abandoned sessions and mark them."""
        try:
            # Get a new database session
            db = next(get_db())
            try:
                service = StudyResponseService(db)
                count = service.check_and_mark_abandoned_sessions()
                
                if count > 0:
                    logger.info(f"Marked {count} sessions as abandoned")
                else:
                    logger.debug("No sessions marked as abandoned")
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error checking abandoned sessions: {e}")
    
    def stop_abandonment_checker(self):
        """Stop the background abandonment checker."""
        if self.task and not self.task.done():
            self.task.cancel()
            logger.info("Stopping abandonment checker")
    
    async def run_once(self):
        """Run the abandonment check once (useful for testing)."""
        await self._check_abandoned_sessions()

# Global instance
background_task_service = BackgroundTaskService()
