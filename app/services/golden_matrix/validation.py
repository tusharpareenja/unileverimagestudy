"""
Design validation via simulation.

Tests whether the design can distinguish signal from noise by comparing
regression results under two conditions:
  - Null: all coefficients are zero (pure noise)
  - Signal: planted coefficients with respondent-level heterogeneity

A good design produces clearly separable results between the two conditions.
"""
import numpy as np
import logging
from typing import Dict, Any

from .config import DesignConfig

logger = logging.getLogger("RDE_Generator")


def validate_design(X: np.ndarray, config: DesignConfig,
                    n_simulations: int = 50, noise_scale: float = 30.0,
                    seed: int = 42) -> Dict[str, Any]:
    """
    Validate a golden matrix design by comparing signal vs null conditions.

    Signal condition: each respondent gets their own coefficients drawn from
    a shared mean + respondent-level variation (simulating mind-sets).
    Null condition: all coefficients are zero (pure noise).

    A good design should show:
      - High signal recovery (r > 0.4 individual, r > 0.7 group)
      - Low false-positive rate under null (few significant coefficients)
      - Clear separation between signal and null R-squared

    Args:
        X:              N x P binary design matrix (from report["X"])
        config:         DesignConfig used to generate X
        n_simulations:  simulation rounds per condition
        noise_scale:    std dev of response noise (0-100 scale)
        seed:           random seed

    Returns dict with signal metrics, null metrics, and verdict.
    """
    rng = np.random.default_rng(seed)
    T = config.tasks_per_respondent
    P = config.total_elements
    R = config.n_respondents

    signal_ind_corrs = []
    signal_grp_corrs = []
    signal_grp_r2s = []
    null_false_pos_rates = []
    null_grp_r2s = []
    rank_ok = 0
    rank_total = 0

    for sim in range(n_simulations):
        # --- Signal condition ---
        # Shared mean coefficients + per-respondent variation (mind-sets)
        mean_betas = rng.uniform(-20, 20, size=P)
        respondent_betas = []
        for r in range(R):
            personal = mean_betas + rng.normal(0, 10, size=P)
            respondent_betas.append(personal)

        # Individual-level signal recovery
        sim_ind_corrs = []
        Y_all_signal = np.zeros(X.shape[0])
        for r in range(R):
            block = X[r * T : (r + 1) * T]
            rank = np.linalg.matrix_rank(block)
            rank_total += 1
            if rank < P:
                continue
            rank_ok += 1

            betas_r = respondent_betas[r]
            Y = np.clip(block @ betas_r + rng.normal(0, noise_scale, T), 0, 100)
            Y_all_signal[r * T : (r + 1) * T] = Y

            XtX = block.T @ block
            beta_hat = np.linalg.solve(XtX, block.T @ Y)
            corr = np.corrcoef(betas_r, beta_hat)[0, 1]
            if np.isfinite(corr):
                sim_ind_corrs.append(corr)

        if sim_ind_corrs:
            signal_ind_corrs.append(np.mean(sim_ind_corrs))

        # Group-level signal recovery (with intercept)
        X_int = np.column_stack([np.ones(X.shape[0]), X])
        try:
            beta_g = np.linalg.solve(X_int.T @ X_int, X_int.T @ Y_all_signal)
            corr_g = np.corrcoef(mean_betas, beta_g[1:])[0, 1]
            if np.isfinite(corr_g):
                signal_grp_corrs.append(corr_g)
            ss_res = np.sum((Y_all_signal - X_int @ beta_g) ** 2)
            ss_tot = np.sum((Y_all_signal - np.mean(Y_all_signal)) ** 2)
            signal_grp_r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
        except np.linalg.LinAlgError:
            pass

        # --- Null condition (same design, zero signal) ---
        Y_null = np.clip(rng.normal(50, noise_scale, size=X.shape[0]), 0, 100)

        # Count false positives per respondent at p<0.10
        sim_fp = []
        for r in range(R):
            block = X[r * T : (r + 1) * T]
            if np.linalg.matrix_rank(block) < P:
                continue
            Y_r = Y_null[r * T : (r + 1) * T]
            XtX = block.T @ block
            beta_hat = np.linalg.solve(XtX, block.T @ Y_r)
            resid = Y_r - block @ beta_hat
            s2 = np.sum(resid ** 2) / max(T - P, 1)
            se = np.sqrt(np.abs(s2 * np.diag(np.linalg.inv(XtX))))
            se[se == 0] = 1e-10
            t_stats = np.abs(beta_hat / se)
            fp_rate = np.mean(t_stats > 1.645)  # p<0.10 two-tailed approx
            sim_fp.append(fp_rate)

        if sim_fp:
            null_false_pos_rates.append(np.mean(sim_fp))

        # Null group R-squared
        try:
            beta_g_null = np.linalg.solve(X_int.T @ X_int, X_int.T @ Y_null)
            ss_res_n = np.sum((Y_null - X_int @ beta_g_null) ** 2)
            ss_tot_n = np.sum((Y_null - np.mean(Y_null)) ** 2)
            null_grp_r2s.append(1 - ss_res_n / ss_tot_n if ss_tot_n > 0 else 0)
        except np.linalg.LinAlgError:
            pass

    # --- Compile ---
    ind_recovery = float(np.mean(signal_ind_corrs)) if signal_ind_corrs else 0.0
    grp_recovery = float(np.mean(signal_grp_corrs)) if signal_grp_corrs else 0.0
    grp_r2_signal = float(np.mean(signal_grp_r2s)) if signal_grp_r2s else 0.0
    grp_r2_null = float(np.mean(null_grp_r2s)) if null_grp_r2s else 0.0
    false_pos = float(np.mean(null_false_pos_rates)) if null_false_pos_rates else 0.0
    rank_frac = rank_ok / rank_total if rank_total > 0 else 0.0

    # Verdict — FP threshold is DOF-aware (tight DOF inflates t-distribution tails)
    dof = T - P
    fp_threshold = 0.20 if dof >= 20 else 0.30

    if rank_frac < 1.0:
        verdict = f"FAIL: {100*(1-rank_frac):.0f}% of respondent blocks are rank-deficient"
    elif grp_recovery < 0.7:
        verdict = f"WEAK: group recovery r={grp_recovery:.2f} (want >= 0.70)"
    elif ind_recovery < 0.3:
        verdict = f"WEAK: individual recovery r={ind_recovery:.2f} (want >= 0.30)"
    elif false_pos > fp_threshold:
        verdict = f"WEAK: false positive rate {false_pos:.0%} (want <= {fp_threshold:.0%} at {dof} DOF)"
    else:
        verdict = f"PASS: signal r={ind_recovery:.2f}/{grp_recovery:.2f}, null FP={false_pos:.0%}, DOF={dof}"

    result = {
        "individual_recovery": round(ind_recovery, 3),
        "individual_rank_ok": round(rank_frac, 3),
        "group_recovery": round(grp_recovery, 3),
        "signal_r_squared": round(grp_r2_signal, 3),
        "null_r_squared": round(grp_r2_null, 3),
        "false_positive_rate": round(false_pos, 3),
        "n_simulations": n_simulations,
        "verdict": verdict,
    }

    logger.info("Validation: %s", verdict)
    return result
