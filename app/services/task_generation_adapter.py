# app/services/task_generation_adapter.py
"""Re-exports Golden Matrix–backed generators (golden_adapter)."""
from __future__ import annotations

from app.services.golden_adapter import generate_grid_tasks, generate_layer_tasks

__all__ = ["generate_grid_tasks", "generate_layer_tasks"]
