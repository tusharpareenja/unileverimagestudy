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
    tasks_per_consumer: Optional[int],
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
    # Convert StudyLayer objects to the format expected by core function
    layers_data = []
    for layer in layers:
        layer_dict = {
            'name': str(getattr(layer, 'name')),
            'z_index': int(getattr(layer, 'z_index', 0)),
            'order': int(getattr(layer, 'order', 0)),
            'images': [
                {
                    'name': str(getattr(img, 'name')),
                    'url': str(getattr(img, 'url', '')),
                    'alt_text': str(getattr(img, 'alt_text', ''))
                }
                for img in layer.images
            ]
        }
        layers_data.append(layer_dict)
    
    return _layer(
        layers_data=layers_data,
        number_of_respondents=number_of_respondents,
        exposure_tolerance_pct=exposure_tolerance_pct,
        seed=seed,
    )