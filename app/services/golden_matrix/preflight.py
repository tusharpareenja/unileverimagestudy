"""
Pre-flight validation for RDE design configurations.

Checks feasibility *before* burning through generation attempts:
  - Row variety (distinct possible rows given silo structure)
  - Structural VIF (geometry-driven collinearity floor)
  - T/P headroom and exposure density
"""
import numpy as np
import logging
from itertools import combinations
from math import prod
from typing import Dict, List, Any

from .config import DesignConfig
from .row_generator import generate_random_row
from .diagnostics import compute_max_vif

logger = logging.getLogger("RDE_Generator")

_VIF_SAMPLE_SIZE = 500


def _count_row_variety(config: DesignConfig) -> int:
    """
    Count distinct possible rows given silo structure and activation range.

    For each k in [min_actives, max_actives]:
        C(C, k) category combos x prod(E_c) for each combo.
    """
    total = 0
    for k in range(config.min_actives_per_row, config.max_actives_per_row + 1):
        for active_cats in combinations(range(config.n_categories), k):
            total += prod(config.elements_per_category[c] for c in active_cats)
    return total


def _estimate_structural_vif(config: DesignConfig) -> float:
    """
    Estimate the VIF floor by generating a random sample matrix.

    This is the best-case VIF achievable — inherent collinearity
    from the silo constraint cannot be designed away.
    """
    rng = np.random.default_rng(0)
    rows: List[np.ndarray] = []
    seen: set = set()
    budget = _VIF_SAMPLE_SIZE * 10

    for _ in range(budget):
        if len(rows) >= _VIF_SAMPLE_SIZE:
            break
        row = generate_random_row(config, rng)
        h = hash(row.tobytes())
        if h not in seen:
            seen.add(h)
            rows.append(row)

    if len(rows) < config.total_elements:
        return np.inf

    return compute_max_vif(np.vstack(rows))


def preflight_check(config: DesignConfig) -> Dict[str, Any]:
    """
    Validate a DesignConfig before generation.

    Returns dict with:
        feasible, row_variety, structural_vif, tp_ratio, tp_class,
        target_exposure_local, warnings, recommendations.
    """
    warnings: List[str] = []
    recommendations: List[str] = []
    feasible = True

    # --- Row variety ---
    row_variety = _count_row_variety(config)
    if row_variety < 2 * config.tasks_per_respondent:
        warnings.append(
            f"Low row variety: {row_variety} distinct rows < {2 * config.tasks_per_respondent} (2*T)."
        )
        feasible = False

    # --- T / P headroom ---
    tp_ratio = config.tasks_per_respondent / config.total_elements
    if tp_ratio == 1.0:
        tp_class = "saturated"
        warnings.append("Saturated design (T=P): zero degrees of freedom.")
    elif tp_ratio < 1.5:
        tp_class = "tight"
        warnings.append(f"Tight design (T/P={tp_ratio:.2f}): may struggle with VIF.")
    else:
        tp_class = "comfortable"

    # --- Exposure density ---
    if config.target_exposure_local < 2.0:
        warnings.append(
            f"Sparse exposure: ~{config.target_exposure_local:.1f} per element per respondent (want >= 2)."
        )

    # --- Structural VIF ---
    structural_vif = _estimate_structural_vif(config)
    if not np.isfinite(structural_vif):
        warnings.append("Could not estimate structural VIF (singular sample matrix).")
        feasible = False
    elif structural_vif > config.max_vif:
        warnings.append(
            f"Structural VIF floor ({structural_vif:.1f}) exceeds max_vif ({config.max_vif})."
        )
        feasible = False
        if config.min_actives_per_row == config.max_actives_per_row:
            recommendations.append(
                f"Widen activation range: try min_actives=1, max_actives={config.n_categories}."
            )

    logger.info(
        "Preflight: variety=%d, VIF_floor=%.1f, T/P=%.2f (%s), "
        "exposure_local=%.1f, feasible=%s",
        row_variety, structural_vif, tp_ratio, tp_class,
        config.target_exposure_local, feasible,
    )
    for w in warnings:
        logger.warning("Preflight: %s", w)

    return {
        "feasible": feasible,
        "row_variety": row_variety,
        "structural_vif": structural_vif,
        "tp_ratio": tp_ratio,
        "tp_class": tp_class,
        "target_exposure_local": config.target_exposure_local,
        "warnings": warnings,
        "recommendations": recommendations,
    }
