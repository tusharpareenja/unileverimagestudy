# app/services/task_generation_adapter.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from app.models.study_model import StudyElement, StudyLayer
from app.services.task_generation_core import (
    generate_grid_tasks as _grid,
    generate_layer_tasks as _layer,
)

def generate_grid_tasks(
    num_elements: int,
    tasks_per_consumer: int,
    number_of_respondents: int,
    exposure_tolerance_cv: float,
    seed: Optional[int],
    elements: List[StudyElement],
) -> Dict[str, Any]:
    return _grid(
        num_elements=num_elements,
        tasks_per_consumer=tasks_per_consumer,
        number_of_respondents=number_of_respondents,
        exposure_tolerance_cv=exposure_tolerance_cv,
        seed=seed,
        elements=elements,
    )

def generate_layer_tasks(
    layers: List[StudyLayer],
    number_of_respondents: int,
    exposure_tolerance_pct: float,
    seed: Optional[int],
) -> Dict[str, Any]:
    category_info: Dict[str, List[str]] = {}
    elements_dict: Dict[str, List[Any]] = {}
    for layer in layers:
        category_info[layer.name] = [img.name for img in layer.images]
        elements_dict[layer.name] = list(layer.images)
    return _layer(
        category_info=category_info,
        number_of_respondents=number_of_respondents,
        exposure_tolerance_pct=exposure_tolerance_pct,
        seed=seed,
        elements=elements_dict,
    )