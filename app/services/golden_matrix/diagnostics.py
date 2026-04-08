"""
Post-generation diagnostics for the Golden Matrix.

Checks: individual-level rank (solvability), duplicate columns,
dead/fixed columns, and global VIF (multicollinearity).
"""
import numpy as np
import logging
from .config import DesignConfig
from .exceptions import DiagnosticsFailedError

logger = logging.getLogger("RDE_Generator")


def compute_max_vif(X: np.ndarray) -> float:
    """
    Max VIF (variance inflation factor) for design matrix X (N×P).

    Returns np.inf when the matrix is under-determined (rows < cols)
    or the correlation matrix is singular/contains NaN.
    """
    N, P = X.shape
    if N < P:
        return np.inf
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(X, rowvar=False)
        if not np.all(np.isfinite(corr)):
            return np.inf
        inv_corr = np.linalg.inv(corr)
        return float(np.max(np.diag(inv_corr)))
    except np.linalg.LinAlgError:
        return np.inf


def run_all_diagnostics(X_all: np.ndarray, config: DesignConfig) -> dict:
    """
    Run all QC checks on the generated matrix.

    Checks:
      - Individual solvability: each respondent block has full rank (= P).
      - No duplicate columns globally.
      - No dead (all-0) or fixed (all-1) columns.
      - Global VIF within threshold.

    Returns diagnostics dict on success; raises DiagnosticsFailedError on failure.
    """
    N, P = X_all.shape
    T = config.tasks_per_respondent

    # --- Individual rank check ---
    rank_failures = 0
    for r in range(config.n_respondents):
        X_r = X_all[r * T : (r + 1) * T, :]
        if np.linalg.matrix_rank(X_r) < P:
            rank_failures += 1
            if config.verbose:
                logger.warning("Respondent %d rank deficient: %d/%d",
                               r + 1, np.linalg.matrix_rank(X_r), P)

    if rank_failures > 0:
        raise DiagnosticsFailedError(
            f"Individual Solvability Failed: {rank_failures}/{config.n_respondents} "
            f"respondents have Rank < {P}. Increase T or loosen constraints."
        )

    # --- Duplicate columns (hash-based, O(P) memory) ---
    col_hashes = set()
    for j in range(P):
        h = hash(X_all[:, j].tobytes())
        if h in col_hashes:
            raise DiagnosticsFailedError("Duplicate columns in global stack.")
        col_hashes.add(h)

    # --- Dead / fixed columns ---
    col_sums = X_all.sum(axis=0)
    if np.any(col_sums == 0) or np.any(col_sums == N):
        raise DiagnosticsFailedError("Dead or fixed column detected.")

    # --- Global VIF ---
    max_vif = compute_max_vif(X_all)
    if max_vif > config.max_vif:
        raise DiagnosticsFailedError(
            f"Global Max VIF {max_vif:.2f} > {config.max_vif}"
        )

    return {
        "status": "PASS",
        "rank_check": "INDIVIDUAL_PASSED",
        "global_max_vif": max_vif,
        "n_rows": N,
        "n_cols": P,
    }
