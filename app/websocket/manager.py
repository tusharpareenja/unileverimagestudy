"""WebSocket connection manager for analytics streaming."""

from typing import Dict, Set
from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)


class AnalyticsConnectionManager:
    """
    Manages WebSocket connections for real-time analytics streaming.
    Tracks connections per study_id for efficient broadcasting
    """

    def __init__(self):
        self._connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, study_id: str, websocket: WebSocket) -> None:
        """Accept and register a WebSocket connection for a study."""
        await websocket.accept()
        if study_id not in self._connections:
            self._connections[study_id] = set()
        self._connections[study_id].add(websocket)
        logger.info(f"WebSocket connected for study {study_id}. Total connections: {len(self._connections[study_id])}")

    def disconnect(self, study_id: str, websocket: WebSocket) -> None:
        """Remove a WebSocket connection from tracking."""
        if study_id in self._connections:
            self._connections[study_id].discard(websocket)
            if not self._connections[study_id]:
                del self._connections[study_id]
            logger.info(f"WebSocket disconnected for study {study_id}")

    async def send_to_connection(self, websocket: WebSocket, data: dict) -> bool:
        """
        Send data to a specific WebSocket connection.
        Returns True if successful, False if connection is dead.
        """
        try:
            await websocket.send_json(data)
            return True
        except Exception as e:
            logger.debug(f"Failed to send to WebSocket: {e}")
            return False

    async def broadcast_to_study(self, study_id: str, data: dict) -> None:
        """
        Broadcast data to all connections subscribed to a study.
        Automatically cleans up dead connections.
        """
        if study_id not in self._connections:
            return

        dead_connections: Set[WebSocket] = set()
        for websocket in self._connections[study_id]:
            success = await self.send_to_connection(websocket, data)
            if not success:
                dead_connections.add(websocket)

        for websocket in dead_connections:
            self._connections[study_id].discard(websocket)

        if study_id in self._connections and not self._connections[study_id]:
            del self._connections[study_id]

    def get_connection_count(self, study_id: str) -> int:
        """Get the number of active connections for a study."""
        return len(self._connections.get(study_id, set()))

    def get_total_connections(self) -> int:
        """Get total number of active WebSocket connections across all studies."""
        return sum(len(conns) for conns in self._connections.values())


# Global singleton instance
analytics_manager = AnalyticsConnectionManager()
