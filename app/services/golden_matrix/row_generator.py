from __future__ import annotations
import numpy as np
from .config import DesignConfig


def row_violates_incompatible_pairs(row: np.ndarray, incompatible_pairs: list) -> bool:
    """Return True if row has both cols active for any incompatible pair."""
    if not incompatible_pairs:
        return False
    for (a, b) in incompatible_pairs:
        if row[a] == 1 and row[b] == 1:
            return True
    return False


def generate_random_row(config: DesignConfig, rng: np.random.Generator) -> np.ndarray:
    """Stateless factory: Generates a single binary row vector."""
    row = np.zeros(config.total_elements, dtype=int)
    
    n_actives = rng.integers(config.min_actives_per_row, config.max_actives_per_row, endpoint=True)
    active_categories = rng.choice(config.n_categories, size=n_actives, replace=False)
    
    for cat_idx in active_categories:
        col_indices = config.category_map[cat_idx]
        chosen_col = rng.choice(col_indices)
        row[chosen_col] = 1
        
    return row
