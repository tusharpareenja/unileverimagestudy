# app/services/task_generation_adapter.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select
from app.models.study_model import StudyElement, StudyLayer, StudyCategory
from app.services.task_generation_core import (
    generate_grid_tasks_v2 as _grid,
    generate_layer_tasks_v2 as _layer,
)

def generate_grid_tasks(
    num_elements: int,
    tasks_per_consumer: Optional[int],  # unused by v2; T is auto-planned
    number_of_respondents: int,
    exposure_tolerance_cv: float,
    seed: Optional[int],
    elements: List[StudyElement],
    db: Session,
    study_id: str,
) -> Dict[str, Any]:
    # Load categories with elements for proper multi-category support
    categories = db.scalars(
        select(StudyCategory)
        .options(selectinload(StudyCategory.elements))
        .where(StudyCategory.study_id == study_id)
        .order_by(StudyCategory.order)
    ).all()
    
    if categories:
        # Use actual categories
        categories_data = [
            {
                "category_name": cat.name,
                "elements": [
                    {
                        "element_id": str(el.element_id),  # Convert UUID to string
                        "name": el.name,
                        "content": el.content,
                        "alt_text": el.alt_text or el.name,
                        "element_type": el.element_type,
                    }
                    for el in cat.elements
                ]
            }
            for cat in categories
        ]
    else:
        # Fallback: wrap flat elements into one category for backward compatibility
        categories_data = [
            {
                "category_name": "All",
                "elements": [
                    {
                        "element_id": getattr(el, "id", getattr(el, "element_id", None)),
                        "name": str(getattr(el, "name", f"E{i+1}")),
                        "content": str(getattr(el, "content", "")),
                        "alt_text": str(getattr(el, "alt_text", getattr(el, "name", f"E{i+1}"))),
                        "element_type": str(getattr(el, "element_type", "image")),
                    }
                    for i, el in enumerate(elements[: num_elements])
                ],
            }
        ]

    return _grid(
        categories_data=categories_data,
        number_of_respondents=number_of_respondents,
        exposure_tolerance_cv=exposure_tolerance_cv,
        seed=seed,
    )

def generate_layer_tasks(
    layers: List[StudyLayer],
    number_of_respondents: int,
    exposure_tolerance_pct: float,
    seed: Optional[int],
) -> Dict[str, Any]:
    # Convert StudyLayer objects to the format expected by v2
    layers_data: List[Dict[str, Any]] = []
    for layer in layers:
        layer_dict = {
            "name": str(getattr(layer, "name")),
            "z_index": int(getattr(layer, "z_index", 0)),
            "order": int(getattr(layer, "order", 0)),
            "images": [
                {
                    "name": str(getattr(img, "name")),
                    "url": str(getattr(img, "url", "")),
                    "alt_text": str(getattr(img, "alt_text", "")),
                }
                for img in getattr(layer, "images", [])
            ],
        }
        layers_data.append(layer_dict)

    return _layer(
        layers_data=layers_data,
        number_of_respondents=number_of_respondents,
        exposure_tolerance_pct=exposure_tolerance_pct,
        seed=seed,
    )