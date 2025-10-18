from __future__ import annotations

import math
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import multiprocessing as mp
import numpy as np
import pandas as pd

# ============================ Globals & Constants (IDENTICAL) ============================

_rng = np.random.default_rng()

# Worker helper
def _build_one_worker(args):
    """
    Worker function that runs in a separate process.
    Returns: (resp_id, rows, X_df, report)
    """
    (
        resp_id, T, category_info, E, A_min_used, base_seed, log_every_rows,
        study_mode, max_active_per_row
    ) = args

    global LOG_EVERY_ROWS
    LOG_EVERY_ROWS = log_every_rows

    rows, X, report = build_respondent_with_uniqueness(
        T, category_info, E, A_min_used, base_seed, resp_id,
        study_mode, max_active_per_row
    )
    return (resp_id, rows, X, report)

# Core constants (match codebase)
PER_ELEM_EXPOSURES = 3     # minimum exposures per element
MIN_ACTIVE_PER_ROW = 2     # min actives per vignette
SAFETY_ROWS        = 3     # OLS safety above P
GLOBAL_RESTARTS    = 200   # rebuild attempts per respondent
SWAP_TRIES         = 8000  # swap tries per row
HARD_CAP_SWAP_TRIES = 30000  # extra attempts for hard cap
BASE_SEED          = 12345
LOG_EVERY_ROWS     = 25

# Study modes
DEFAULT_STUDY_MODE = "layout"   # "layout" or "grid"
GRID_MAX_ACTIVE    = 4          # grid constraint

# Absence policy
ABSENCE_RATIO      = 2.0

# Row-mix shaping
ROW_MIX_MODE       = "wide"  # "dense" (old behavior) or "wide" (more 3s & 5s)
ROW_MIX_WIDEN      = 0.40

# “inflate T” ratio
T_RATIO            = 1.10

# Per-respondent safety: enforce no duplicate columns
PER_RESP_UNIQUE_COLS_ENFORCE = True

# Planner preference
CAPACITY_SLACK     = 1

# QC (“Professor Gate”) knobs
QC_RANDOM_Y_K      = 5       # number of random dependent variables per respondent
QC_KAPPA_MAX       = 1e6     # condition number upper bound (tight)
QC_RANK_TOL        = 1e-12   # singular value cutoff for rank
QC_LS_TOL          = 1e-6    # tolerance on normal equations residual
VIF_TOL_MIN        = 1e-6    # per-column tolerance = 1 / max(VIF) threshold


def set_seed(seed: Optional[int] = None):
    """Set global RNGs for reproducibility."""
    global _rng
    if seed is not None:
        _rng = np.random.default_rng(seed)
        random.seed(seed)


# ============================ Helpers (IDENTICAL) ============================

def is_abs(val: str) -> bool:
    return isinstance(val, str) and str(val).startswith("__ABS__")


def sig_pair(cat: str, val: str) -> Tuple[str, str]:
    return (cat, "__ABS__") if is_abs(val) else (cat, str(val))


def params_main_effects(category_info: Dict[str, List[str]]) -> int:
    C = len(category_info)
    M = sum(len(v) for v in category_info.values())
    return M - C + 1  # intercept + (n_c - 1) per category


def visible_capacity(category_info: Dict[str, List[str]],
                     min_active: int,
                     max_active: Optional[int] = None) -> int:
    """Absence-collapsed count of distinct row patterns with ≥ min_active actives,
       optionally capped at ≤ max_active actives per row."""
    cats = list(category_info.keys())
    m = [len(category_info[c]) for c in cats]
    C = len(m)
    coeff = [0]*(C+1); coeff[0] = 1
    for mi in m:
        nxt=[0]*(C+1)
        for k in range(C+1):
            if coeff[k]==0: continue
            nxt[k] += coeff[k]           # ABS choice
            if k+1<=C: nxt[k+1] += coeff[k]*mi  # choose an element
        coeff = nxt
    hi = C if max_active is None else min(max_active, C)
    lo = max(min_active, 0)
    if lo > hi:
        return 0
    return sum(coeff[k] for k in range(lo, hi+1))


def one_hot_df_from_rows(rows: List[Dict[str,str]], category_info: Dict[str,List[str]]) -> pd.DataFrame:
    cats = list(category_info.keys())
    elems = [e for es in category_info.values() for e in es]
    recs = []
    for row in rows:
        d = {e: 0 for e in elems}
        for c in cats:
            ch = row[c]
            if ch in elems: d[ch] = 1
        recs.append(d)
    return pd.DataFrame(recs, columns=elems)


def per_respondent_duplicate_pairs(X: pd.DataFrame) -> List[Tuple[str,str]]:
    """Return list of (colA, colB) that are exact duplicates within this respondent."""
    col_map = {}
    dups = []
    for col in X.columns:
        key = tuple(int(v) for v in X[col].tolist())
        if key in col_map:
            dups.append((col_map[key], col))
        else:
            col_map[key] = col
    return dups


def max_cross_category_correlation(X: pd.DataFrame, category_info: Dict[str,List[str]]) -> Tuple[float, Tuple[str,str]]:
    """Compute max absolute Pearson r across elements from different categories."""
    cat_of = {}
    for c, es in category_info.items():
        for e in es:
            cat_of[e] = c
    max_abs_r = 0.0
    max_pair = ("","")
    cols = list(X.columns)
    for i in range(len(cols)):
        e1 = cols[i]
        for j in range(i+1, len(cols)):
            e2 = cols[j]
            if cat_of[e1] == cat_of[e2]:
                continue
            r = np.corrcoef(X[e1].to_numpy(), X[e2].to_numpy())[0,1]
            if abs(r) > max_abs_r:
                max_abs_r = abs(r)
                max_pair = (e1, e2)
    return float(max_abs_r), max_pair


def professor_gate(X: pd.DataFrame,
                   k: int = QC_RANDOM_Y_K,
                   kappa_max: float = QC_KAPPA_MAX,
                   rank_tol: float = QC_RANK_TOL,
                   ls_tol: float = QC_LS_TOL,
                   rng_seed: Optional[int] = None) -> Tuple[bool, dict]:
    """
    Gate (no intercept), optimized:
      - zero-variance columns = 0
      - full rank by SVD (rank == p)
      - condition number ≤ kappa_max
      - min tolerance (1 / max VIF) ≥ VIF_TOL_MIN
      - K random-DV fits: rank==p and normal-eq residual small, all K/ K
    """
    Xn = X.to_numpy(dtype=float)
    n, p = Xn.shape

    # ---------- one SVD (rank & kappa) ----------
    U, s, Vt = np.linalg.svd(Xn, full_matrices=False)
    rank = int(np.sum(s > QC_RANK_TOL))
    full_rank = (rank == p)
    s_min = float(np.min(s)) if s.size else float('nan')
    s_max = float(np.max(s)) if s.size else float('nan')
    kappa = float(s_max / s_min) if s_min > 0 else float('inf')
    if (not full_rank) or (kappa > kappa_max):
        return False, {
            "zero_var": 0, "rank": int(rank), "p": int(p),
            "s_min": float(s_min), "s_max": float(s_max), "kappa": float(kappa),
            "dup_pairs": 0, "max_vif": float("inf"), "min_tolerance": 0.0,
            "tolerance_ok": False, "corr_inv_exact": False,
            "fit_rank_passes": 0, "ls_passes": 0, "ls_total": int(k), "passed": False,
        }

    # ---------- tolerance / VIF via correlation inverse ----------
    XT = Xn.T
    G  = XT @ Xn
    norms = np.sqrt(np.diag(G))
    eps = 1e-15
    norms = np.where(norms < eps, eps, norms)
    D_inv = np.diag(1.0 / norms)
    Rcorr = D_inv @ G @ D_inv
    try:
        Rinv = np.linalg.inv(Rcorr)
        inv_ok = True
    except np.linalg.LinAlgError:
        Rinv = np.linalg.pinv(Rcorr, rcond=QC_RANK_TOL)
        inv_ok = False
    vif = np.maximum(np.diag(Rinv), 1.0)
    max_vif = float(np.max(vif))
    min_tolerance = float(1.0 / max_vif)
    tolerance_ok = (min_tolerance >= VIF_TOL_MIN)
    if not tolerance_ok:
        return False, {
            "zero_var": 0, "rank": int(rank), "p": int(p),
            "s_min": float(s_min), "s_max": float(s_max), "kappa": float(kappa),
            "dup_pairs": 0, "max_vif": float(max_vif), "min_tolerance": float(min_tolerance),
            "tolerance_ok": False, "corr_inv_exact": bool(inv_ok),
            "fit_rank_passes": 0, "ls_passes": 0, "ls_total": int(k), "passed": False,
        }

    # ---------- batched stress tests (QR once, K solves) ----------
    Q, R = np.linalg.qr(Xn, mode="reduced")  # X = Q R
    rng = np.random.default_rng(rng_seed)
    Y = rng.integers(1, 10, size=(n, k)).astype(float)  # n x K
    QTy = Q.T @ Y                                        # p x K
    B = np.linalg.solve(R, QTy)

    # rank check per RHS
    rdiag_ok = np.all(np.abs(np.diag(R)) > QC_RANK_TOL)
    fit_rank_passes = int(k if rdiag_ok else 0)

    # normal-equations residuals for all K
    XT = Xn.T
    G_B   = (XT @ Xn) @ B
    XTy   = XT @ Y
    RES   = G_B - XTy
    lhs   = np.linalg.norm(RES, axis=0)
    rhs   = np.linalg.norm(XTy, axis=0) + 1e-12
    ls_passes = int(np.sum(lhs <= QC_LS_TOL * rhs))

    passed = (fit_rank_passes == k) and (ls_passes == k)
    stats = {
        "zero_var": 0,
        "rank": int(rank), "p": int(p),
        "s_min": float(s_min), "s_max": float(s_max), "kappa": float(kappa),
        "dup_pairs": 0,
        "max_vif": float(max_vif), "min_tolerance": float(min_tolerance),
        "tolerance_ok": bool(tolerance_ok), "corr_inv_exact": bool(inv_ok),
        "fit_rank_passes": int(fit_rank_passes),
        "ls_passes": int(ls_passes), "ls_total": int(k),
        "passed": bool(passed),
    }
    return passed, stats


# ============================ Planner (IDENTICAL) ============================

def plan_T_E_auto(category_info: Dict[str, List[str]],
                  study_mode: str = "layout",
                  max_active_per_row: Optional[int] = None) -> Tuple[int, int, Dict[str, int], float, int]:
    """Plan T and E automatically with advanced algorithm."""
    cats = list(category_info.keys())
    q = {c: len(category_info[c]) for c in cats}
    C = len(cats)
    M = sum(q.values())
    P = params_main_effects(category_info)
    cap = visible_capacity(category_info, MIN_ACTIVE_PER_ROW, max_active_per_row)

    # For small studies, use simpler approach
    if cap < 10:
        T = min(12, cap)
        E = max(1, T // M)  # Simple exposure calculation
        A_min_used = max(1, int(math.ceil(ABSENCE_RATIO * E)))
        A_map = {c: max(A_min_used, T - q[c]*E) for c in cats}
        avg_k = (M * E) / T if T > 0 else C
        return T, E, A_map, avg_k, A_min_used

    # Start from identifiability floor and scale by T_RATIO
    T = max(P + SAFETY_ROWS, 2)
    if T_RATIO and T_RATIO > 1.0:
        T = int(math.ceil(T * float(T_RATIO)))

    # Helper: maximum feasible E at a given T
    def E_upper_at_T(T_try: int) -> int:
        # For each category c: T - q[c]*E >= ceil(ABSENCE_RATIO * E)
        bound_ratio = min(int(math.floor(T_try / (q[c] + ABSENCE_RATIO))) for c in cats)
        # Per-row active cap: total 1s = M*E <= T * rowcap
        rowcap = (max_active_per_row if (study_mode == "grid" and max_active_per_row is not None) else C)
        bound_rowcap = int(math.floor(T_try * rowcap / M))
        return min(bound_ratio, bound_rowcap)

    slack = CAPACITY_SLACK
    while True:
        if T > max(cap - slack, 0):
            if T > cap:
                # Fallback to simple approach for small studies
                T = min(12, cap)
                E = max(1, T // M)
                A_min_used = max(1, int(math.ceil(ABSENCE_RATIO * E)))
                A_map = {c: max(A_min_used, T - q[c]*E) for c in cats}
                avg_k = (M * E) / T if T > 0 else C
                return T, E, A_map, avg_k, A_min_used
            slack = 0
        E_up = E_upper_at_T(T)
        if E_up >= PER_ELEM_EXPOSURES:
            E = E_up
            break
        T += 1

    A_min_used = int(math.ceil(ABSENCE_RATIO * E))
    A_map = {c: T - q[c]*E for c in cats}  # by construction >= A_min_used
    avg_k = (M * E) / T
    return T, E, A_map, avg_k, A_min_used


def plan_row_mix(T: int, total_ones: int, min_k: int, max_k: int, mode: str = "wide", widen: float = 0.40) -> List[int]:
    """Plan row mix for variation in active categories per row."""
    if total_ones < T*min_k or total_ones > T*max_k:
        if T <= 50:
            base = total_ones // T
            remainder = total_ones % T
            targets = [base] * T
            for i in range(remainder):
                targets[i] += 1
            return targets
        else:
            raise ValueError("total_ones not representable; adjust T/E.")

    if mode == "dense":
        k0 = min(max_k, max(min_k+1, 4))
        targets = [k0] * T
        deficit = total_ones - sum(targets)
        i = 0
        while deficit > 0:
            if targets[i] < max_k:
                step = min(max_k - targets[i], deficit)
                targets[i] += step
                deficit -= step
            i = (i + 1) % T
        i = 0
        while deficit < 0:
            if targets[i] > min_k:
                step = min(targets[i] - min_k, -deficit)
                targets[i] -= step
                deficit += step
            i = (i + 1) % T
        return targets

    # wide mode
    base = total_ones // T
    deficit = total_ones - base*T
    base = max(min_k, min(max_k, base))
    targets = [base] * T
    down_cap = T if (base - 1) >= min_k else 0
    pair_count = min(max(0, int(round(widen * T))), down_cap, T)

    # push some rows to base-1
    idx = 0
    down_done = 0
    while down_done < pair_count and idx < T:
        if targets[idx] - 1 >= min_k:
            targets[idx] -= 1
            down_done += 1
        idx += 1

    # balance with +1s: one per downshift plus the global deficit
    up_needed = pair_count + deficit
    idx = 0
    up_done = 0
    while up_done < up_needed and idx < T:
        if targets[idx] + 1 <= max_k:
            targets[idx] += 1
            up_done += 1
        idx += 1
    if up_done < up_needed:
        for i in range(T):
            if up_done >= up_needed: break
            if targets[i] + 1 <= max_k:
                targets[i] += 1
                up_done += 1

    return targets


# ============================ Builders (IDENTICAL) ============================

def build_once_advanced(T: int, category_info: Dict[str, List[str]], E: int, A_min: int,
                        rng: np.random.Generator, study_mode: str = "layout",
                        max_active_per_row: Optional[int] = None) -> Optional[List[Dict[str, str]]]:
    """
    Build rows with exact exposures/absences using advanced algorithm.
    Returns rows or None.
    """
    cats = list(category_info.keys())
    M = sum(len(category_info[c]) for c in cats)
    total_ones = M * E

    # Build token pools with EXACT exposures and required ABS (derived from T & E)
    pools: Dict[str, List[str]] = {}
    for c in cats:
        elems = category_info[c]
        tokens = []
        for e in elems:
            tokens.extend([e] * E)
        abs_count = T - E * len(elems)
        if abs_count < A_min:
            return None
        tokens.extend([f"__ABS__{c}"] * int(abs_count))
        if len(tokens) != T:
            return None
        rng.shuffle(tokens)
        pools[c] = tokens

    # Target actives per row
    allowed_max = (max_active_per_row if (study_mode == "grid" and max_active_per_row is not None) else len(cats))
    targets = plan_row_mix(
        T, total_ones, MIN_ACTIVE_PER_ROW, allowed_max,
        mode=ROW_MIX_MODE, widen=ROW_MIX_WIDEN
    )

    # Assemble rows by aligned index
    rows = [{c: pools[c][t] for c in cats} for t in range(T)]

    def active_count(rr: int) -> int:
        return sum(1 for c in cats if not is_abs(rows[rr][c]))

    def vis_sig(rr: int) -> Tuple[Tuple[str, str], ...]:
        return tuple(sig_pair(c, rows[rr][c]) for c in cats)

    seen: Counter = Counter()
    sig_of: Dict[int, Tuple[Tuple[str,str], ...]] = {}

    # Phase 1: build with uniqueness and bounds
    for r in range(T):
        s_r = vis_sig(r)
        within_cap = (active_count(r) <= allowed_max)
        ok = (MIN_ACTIVE_PER_ROW <= active_count(r) and within_cap and seen[s_r] == 0)
        if ok:
            seen[s_r] += 1
            sig_of[r] = s_r
        else:
            best = None
            best_gain = -10**9
            for _ in range(SWAP_TRIES):
                rc = int(rng.integers(0, r+1))
                c = rng.choice(cats)

                rows[r][c], rows[rc][c] = rows[rc][c], rows[r][c]
                new_r_sig = vis_sig(r)
                new_rc_sig = vis_sig(rc) if rc < r else None

                valid = True
                # enforce bounds at both rows
                if active_count(r) < MIN_ACTIVE_PER_ROW or active_count(r) > allowed_max: valid = False
                if rc < r and (active_count(rc) < MIN_ACTIVE_PER_ROW or active_count(rc) > allowed_max): valid = False

                if valid:
                    if new_r_sig != s_r and seen.get(new_r_sig, 0) > 0: valid = False
                    if valid and rc < r:
                        old_rc_sig = sig_of[rc]
                        if new_rc_sig != old_rc_sig and seen.get(new_rc_sig, 0) > 0: valid = False

                if valid:
                    gain = -abs(active_count(r) - targets[r])
                    if rc < r:
                        gain += -abs(active_count(rc) - targets[rc])
                    if gain > best_gain:
                        best_gain = gain
                        best = (rc, c, new_r_sig, new_rc_sig)

                rows[r][c], rows[rc][c] = rows[rc][c], rows[r][c]  # rollback

            if best is None:
                return None
            rc, c, new_r_sig, new_rc_sig = best
            rows[r][c], rows[rc][c] = rows[rc][c], rows[r][c]
            if s_r in seen:
                seen[s_r] -= 1
                if seen[s_r] <= 0: del seen[s_r]
            seen[new_r_sig] = seen.get(new_r_sig, 0) + 1
            sig_of[r] = new_r_sig
            if rc < r:
                old_rc_sig = sig_of[rc]
                if old_rc_sig in seen:
                    seen[old_rc_sig] -= 1
                    if seen[old_rc_sig] <= 0: del seen[old_rc_sig]
                seen[new_rc_sig] = seen.get(new_rc_sig, 0) + 1
                sig_of[rc] = new_rc_sig

    # Phase 2: Hard-cap enforcement (grid only)
    if study_mode == "grid":
        over_idx = [i for i in range(T) if active_count(i) > allowed_max]
        under_ok_idx = [i for i in range(T) if active_count(i) < allowed_max]

        tries = 0
        while over_idx and tries < HARD_CAP_SWAP_TRIES:
            tries += 1
            r = over_idx[tries % len(over_idx)]
            act_cats = [c for c in cats if not is_abs(rows[r][c])]
            if not act_cats:
                over_idx = [i for i in range(T) if active_count(i) > allowed_max]
                continue
            c = act_cats[tries % len(act_cats)]

            # prefer rows with absence in c and currently under the cap
            candidates = [i for i in under_ok_idx if is_abs(rows[i][c])]
            if not candidates:
                candidates = [i for i in range(T) if i != r and is_abs(rows[i][c])]
            found = False
            for s in candidates:
                if s == r: continue
                rows[r][c], rows[s][c] = rows[s][c], rows[r][c]
                if (MIN_ACTIVE_PER_ROW <= active_count(r) <= allowed_max) and (MIN_ACTIVE_PER_ROW <= active_count(s) <= allowed_max):
                    found = True
                    break
                rows[r][c], rows[s][c] = rows[s][c], rows[r][c]

            over_idx = [i for i in range(T) if active_count(i) > allowed_max]
            under_ok_idx = [i for i in range(T) if active_count(i) < allowed_max]

        if over_idx:
            return None  # fail this attempt; caller will restart

    # Final checks
    sigs = [vis_sig(i) for i in range(T)]
    if len(set(sigs)) != T:
        return None
    if any(active_count(i) < MIN_ACTIVE_PER_ROW for i in range(T)):
        return None
    if study_mode == "grid" and any(active_count(i) > allowed_max for i in range(T)):
        return None
    return rows


def build_with_restarts_advanced(T: int, category_info: Dict[str, List[str]], E: int, A_min: int,
                                 rng: np.random.Generator, study_mode: str = "layout",
                                 max_active_per_row: Optional[int] = None) -> Optional[List[Dict[str, str]]]:
    """Fixed-T builder with heavy retries. Returns rows or None."""
    for _ in range(1, GLOBAL_RESTARTS+1):
        rows = build_once_advanced(T, category_info, E, A_min, rng, study_mode, max_active_per_row)
        if rows is not None:
            return rows
    return None


def build_respondent_with_uniqueness(T: int, category_info: Dict[str, List[str]], E: int, A_min: int,
                                     base_seed: int, resp_index: int, study_mode: str,
                                     max_active_per_row: Optional[int]):
    """
    Build a single respondent, enforcing:
      • no duplicate columns
      • Professor Gate (rank, κ, K random-y) — LOOPS UNTIL PASS (infinite rebuild).
    This function never raises on QC/build failure; it keeps trying new RNG streams until success.
    """
    tries = 0
    last_status_print = 0
    while True:
        tries += 1
        rng_seed = base_seed + resp_index*7919 + tries*104729  # big prime stride
        rng = np.random.default_rng(rng_seed)

        rows = None
        try:
            rows = build_with_restarts_advanced(T, category_info, E, A_min, rng, study_mode, max_active_per_row)
        except Exception:
            rows = None

        if rows is None:
            if tries - last_status_print >= 20:
                last_status_print = tries
            continue

        # Per-respondent uniqueness + QC gate
        X = one_hot_df_from_rows(rows, category_info)
        dup_pairs = per_respondent_duplicate_pairs(X)
        gate_ok, gate_stats = professor_gate(
            X,
            k=QC_RANDOM_Y_K,
            kappa_max=QC_KAPPA_MAX,
            rank_tol=QC_RANK_TOL,
            ls_tol=QC_LS_TOL,
            rng_seed=rng_seed + 17
        )

        enforce_dups_ok = (not PER_RESP_UNIQUE_COLS_ENFORCE) or (len(dup_pairs) == 0)
        all_ok = gate_ok and enforce_dups_ok

        if all_ok:
            max_r, pair = max_cross_category_correlation(X, category_info)
            report = {
                "zero_var": gate_stats.get("zero_var", 0),
                "dup_pairs": len(dups if (dups:=dup_pairs) else []),
                "rank": gate_stats.get("rank", 0),
                "p": gate_stats.get("p", 0),
                "s_min": gate_stats.get("s_min", float("nan")),
                "s_max": gate_stats.get("s_max", float("nan")),
                "kappa": gate_stats.get("kappa", float("nan")),
                "max_vif": gate_stats.get("max_vif", float("nan")),
                "min_tolerance": gate_stats.get("min_tolerance", float("nan")),
                "tolerance_ok": gate_stats.get("tolerance_ok", True),
                "fit_rank_passes": gate_stats.get("fit_rank_passes", 0),
                "ls_passes": gate_stats.get("ls_passes", 0),
                "ls_total": gate_stats.get("ls_total", QC_RANDOM_Y_K),
                "max_abs_r": float(max_r),
                "max_r_pair": pair,
                "gate_passed": True,
            }
            return rows, X, report

        # loop continues until pass


def preflight_lock_T(T: int, category_info: Dict[str, List[str]], E: int, A_min_used: int,
                     study_mode: str, max_active_per_row: Optional[int]) -> Tuple[int, Dict[str,int]]:
    """Try T, then T+1..T+3, locking a feasible T if possible."""
    rng = np.random.default_rng(BASE_SEED + 999_999)
    cap = visible_capacity(category_info,
                           MIN_ACTIVE_PER_ROW,
                           (max_active_per_row if study_mode == "grid" else None))

    def try_build(T_try: int) -> bool:
        for _ in range(GLOBAL_RESTARTS):
            if build_once_advanced(T_try, category_info, E, A_min_used, rng, study_mode, max_active_per_row) is not None:
                return True
        return False

    if try_build(T):
        A_map = {c: max(A_min_used, T - len(category_info[c]) * E) for c in category_info}
        return T, A_map

    for extra in range(1, 4):
        T2 = T + extra
        if T2 > cap:
            break
        if try_build(T2):
            A_map = {c: max(A_min_used, T2 - len(category_info[c]) * E) for c in category_info}
            return T2, A_map

    raise RuntimeError("Preflight failed even after a small global T bump; consider more elements/categories or adjust ABSENCE_RATIO/T_RATIO.")


# ============================ Public APIs (IDENTICAL OUTPUT) ============================

from typing import Optional, Dict, Any, List, Callable

def generate_grid_tasks_v2(categories_data: List[Dict], number_of_respondents: int,
                           exposure_tolerance_cv: float = 1.0, seed: Optional[int] = None,
                           progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    """
    Generate tasks for grid studies using EXACT logic from the codebase.
    Returns structure identical to utils.task_generation.generate_grid_tasks_v2 (final exact version).
    """
    print(f"🚀 Starting grid task generation for {number_of_respondents} respondents...")
    start_time = time.time()
    print(f"⏰ Start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")

    # Set seed if provided
    if seed is not None:
        global BASE_SEED
        BASE_SEED = seed

    # Convert categories_data to category_info format (EXACT same naming)
    category_info: Dict[str, List[str]] = {}
    for category in categories_data:
        category_name = category['category_name']
        category_info[category_name] = [f"{category_name}_{j+1}" for j in range(len(category['elements']))]

    C = len(category_info)
    # Over-generate respondents by 5x and later slice to requested count
    N = number_of_respondents * 5

    # Study mode & per-row active cap
    mode = "grid"
    max_active_per_row = min(GRID_MAX_ACTIVE, C)

    if max_active_per_row < MIN_ACTIVE_PER_ROW:
        raise ValueError(
            f"MAX_ACTIVE_PER_ROW ({max_active_per_row}) < MIN_ACTIVE_PER_ROW ({MIN_ACTIVE_PER_ROW}). "
            "Increase categories or relax settings."
        )

    # Plan automatically
    planning_start = time.time()
    print(f"📊 Starting planning phase at {time.strftime('%H:%M:%S', time.localtime(planning_start))}")
    T, E, A_map, avg_k, A_min_used = plan_T_E_auto(category_info, mode, max_active_per_row)
    planning_duration = time.time() - planning_start
    print(f"⏱️ Planning completed in {planning_duration:.2f} seconds")

    # Logging cadence
    global LOG_EVERY_ROWS
    LOG_EVERY_ROWS = max(T, 25)

    # Preflight lock T
    preflight_start = time.time()
    print(f"🔒 Starting preflight lock at {time.strftime('%H:%M:%S', time.localtime(preflight_start))}")
    try:
        T, A_map = preflight_lock_T(T, category_info, E, A_min_used, mode, max_active_per_row)
        preflight_duration = time.time() - preflight_start
        print(f"✅ Preflight lock successful in {preflight_duration:.2f} seconds")
    except RuntimeError as e:
        preflight_duration = time.time() - preflight_start
        print(f"⚠️ Preflight could not lock T ({e}) after {preflight_duration:.2f} seconds. Proceeding with T={T} in rebuild-until-pass mode.")
        A_map = {c: max(A_min_used, T - len(category_info[c]) * E) for c in category_info}

    # Recompute avg_k in case T was bumped
    M = sum(len(category_info[c]) for c in category_info)
    avg_k = (M * E) / T

    print(f"\n[Plan] FINAL T={T}, E={E}, avg actives ≈ {avg_k:.2f} (ABSENCE_RATIO={ABSENCE_RATIO}, A_min used = {A_min_used}, T_RATIO={T_RATIO})")
    print(f"[Plan] Row-mix mode: {ROW_MIX_MODE}, widen={ROW_MIX_WIDEN}")
    print("[Plan] Absences per category:", ", ".join([f"{c}:{A_map[c]}" for c in category_info]))

    # Build each respondent in parallel
    all_rows_per_resp: List[Tuple[int, List[Dict[str, str]]]] = []
    per_resp_reports: Dict[int, dict] = {}

    max_workers = min(os.cpu_count() or 1, N)
    print(f"👥 Using {max_workers} workers for parallel processing")

    parallel_start = time.time()
    print(f"🔄 Starting parallel processing at {time.strftime('%H:%M:%S', time.localtime(parallel_start))}")

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for resp_id in range(N):
            future = executor.submit(_build_one_worker, (
                resp_id, T, category_info, E, A_min_used, BASE_SEED, LOG_EVERY_ROWS,
                mode, max_active_per_row
            ))
            futures.append((resp_id, future))

        future_to_resp_id = {future: resp_id for resp_id, future in futures}

        completed_count = 0
        for future in as_completed([f[1] for f in futures]):
            try:
                resp_id, rows, X_df, report = future.result()
                all_rows_per_resp.append((resp_id, rows))
                per_resp_reports[resp_id] = report
                completed_count += 1
                current_time = time.strftime('%H:%M:%S', time.localtime())
                print(f"✅ Built respondent {resp_id+1}/{N} at {current_time} ({completed_count}/{N} completed)")
                if progress_callback is not None:
                    try:
                        progress_callback(completed_count, N)
                    except Exception:
                        pass
            except Exception as e:
                completed_count += 1
                current_time = time.strftime('%H:%M:%S', time.localtime())
                print(f"❌ Failed to build respondent at {current_time}: {e}")
                per_resp_reports[future_to_resp_id[future]] = {'status': 'failed', 'error': str(e)}
                if progress_callback is not None:
                    try:
                        progress_callback(completed_count, N)
                    except Exception:
                        pass

    parallel_duration = time.time() - parallel_start
    print(f"⏱️ Parallel processing completed in {parallel_duration:.2f} seconds")

    # Sort results by respondent ID
    all_rows_per_resp.sort(key=lambda t: t[0])

    # Convert to design matrix
    if not all_rows_per_resp:
        raise RuntimeError("Failed to generate any valid designs")

    design_df = one_hot_df_from_rows(all_rows_per_resp[0][1], category_info)
    for resp_id, rows in all_rows_per_resp[1:]:
        resp_df = one_hot_df_from_rows(rows, category_info)
        design_df = pd.concat([design_df, resp_df], ignore_index=True)

    # Add Consumer_ID
    consumer_ids = []
    for resp_id, rows in all_rows_per_resp:
        consumer_ids.extend([resp_id] * len(rows))
    design_df['Consumer_ID'] = consumer_ids

    # Derive tasks_per_consumer / capacity
    tasks_per_consumer = max(1, len(design_df) // N)
    capacity = len(design_df)

    # Convert to tasks with content mapping back to categories_data
    tasks_structure: Dict[str, List[Dict[str, Any]]] = {}
    all_elements = [e for es in category_info.values() for e in es]

    for respondent_id in range(number_of_respondents):
        respondent_tasks: List[Dict[str, Any]] = []
        start_idx = respondent_id * tasks_per_consumer
        end_idx = start_idx + tasks_per_consumer
        respondent_data = design_df.iloc[start_idx:end_idx]

        for task_index, (_, task_row) in enumerate(respondent_data.iterrows()):
            elements_shown: Dict[str, int] = {}
            elements_shown_content: Dict[str, Optional[Dict[str, Any]]] = {}

            for element_name in all_elements:
                element_active = int(task_row[element_name])
                elements_shown[element_name] = element_active

                if element_active:
                    # Parse "CategoryName_ElementIndex"
                    if '_' in element_name:
                        category_name, elem_index_str = element_name.rsplit('_', 1)
                        try:
                            elem_index = int(elem_index_str) - 1  # 0-based
                            for category in categories_data:
                                if category['category_name'] == category_name and elem_index < len(category['elements']):
                                    element = category['elements'][elem_index]
                                    elements_shown_content[element_name] = {
                                        'element_id': element['element_id'],
                                        'name': element['name'],
                                        'content': element['content'],
                                        'alt_text': element.get('alt_text', element['name']),
                                        'element_type': element['element_type'],
                                        'category_name': category_name
                                    }
                                    break
                            else:
                                elements_shown_content[element_name] = None
                        except ValueError:
                            elements_shown_content[element_name] = None
                    else:
                        elements_shown_content[element_name] = None
                else:
                    elements_shown_content[element_name] = None

            # Clean up any _ref entries
            elements_shown = {k: v for k, v in elements_shown.items() if not k.endswith('_ref')}
            elements_shown_content = {k: v for k, v in elements_shown_content.items() if not k.endswith('_ref')}

            task_obj = {
                "task_id": f"{respondent_id}_{task_index}",
                "elements_shown": elements_shown,
                "elements_shown_content": elements_shown_content,
                "task_index": task_index
            }
            respondent_tasks.append(task_obj)

        tasks_structure[str(respondent_id)] = respondent_tasks

    end_time = time.time()
    total_duration = end_time - start_time
    end_time_str = time.strftime('%H:%M:%S', time.localtime(end_time))
    print(f"✅ Grid task generation completed at {end_time_str}")
    print(f"⏱️ Total duration: {total_duration:.2f} seconds")
    print(f"📊 Performance: {number_of_respondents} respondents in {total_duration:.2f}s = {number_of_respondents/total_duration:.2f} respondents/second")

    return {
        'tasks': tasks_structure,
        'metadata': {
            'study_type': 'grid_v2',
            'categories_data': categories_data,
            'category_info': category_info,
            'tasks_per_consumer': tasks_per_consumer,
            'number_of_respondents': number_of_respondents,
            'exposure_tolerance_cv': exposure_tolerance_cv,
            'capacity': capacity,
            'algorithm': 'final_builder_parallel_exact'
        }
    }


def generate_layer_tasks_v2(layers_data: List[Dict], number_of_respondents: int,
                            exposure_tolerance_pct: float = 2.0, seed: Optional[int] = None,
                            progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    """
    Generate tasks for the new layer structure using advanced algorithms (IDENTICAL OUTPUT).
    """
    print(f"🚀 Starting layer task generation for {number_of_respondents} respondents...")
    start_time = time.time()
    print(f"⏰ Start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")

    if seed is not None:
        set_seed(seed)

    # Convert layers data to category_info format
    category_info = {}
    for layer in layers_data:
        layer_name = layer['name']
        elements = [f"{layer_name}_{i+1}" for i in range(len(layer['images']))]
        category_info[layer_name] = elements

    C = len(category_info)
    # Over-generate respondents by 5x and later slice to requested count
    N = number_of_respondents * 5

    mode = "layout"
    max_active_per_row = C

    # Seed base for workers
    if seed is not None:
        global BASE_SEED
        BASE_SEED = seed

    # Plan automatically
    T, E, A_map, avg_k, A_min_used = plan_T_E_auto(category_info, mode, max_active_per_row)

    # Logging cadence
    global LOG_EVERY_ROWS
    LOG_EVERY_ROWS = max(T, 25)

    # Preflight lock T
    try:
        T, A_map = preflight_lock_T(T, category_info, E, A_min_used, mode, max_active_per_row)
    except RuntimeError as e:
        print(f"⚠️ Preflight could not lock T ({e}). Proceeding with T={T} in rebuild-until-pass mode.")
        A_map = {c: max(A_min_used, T - len(category_info[c]) * E) for c in category_info}

    # Recompute avg_k in case T was bumped
    M = sum(len(category_info[c]) for c in category_info)
    avg_k = (M * E) / T

    # Build respondents in parallel
    all_rows_per_resp: List[Tuple[int, List[Dict[str, str]]]] = []
    per_resp_reports: Dict[int, dict] = {}

    max_workers = min(os.cpu_count() or 1, N)
    print(f"🚀 Building {N} respondents concurrently with {max_workers} workers...")

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    tasks = []
    for r in range(1, N+1):
        tasks.append((r, T, category_info, E, A_min_used, BASE_SEED, LOG_EVERY_ROWS,
                      mode, max_active_per_row))

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_build_one_worker, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futures):
            resp_id = futures[fut]
            rid, rows, X, report = fut.result()
            per_resp_reports[rid] = report
            all_rows_per_resp.append((rid, rows))
            done += 1
            if done % 5 == 0 or done == N:
                print(f"  • Completed {done}/{N} respondents")
            if progress_callback is not None:
                try:
                    progress_callback(done, N)
                except Exception:
                    pass

    # Sort results
    all_rows_per_resp.sort(key=lambda t: t[0])

    # Convert to design matrix
    design_df = one_hot_df_from_rows(all_rows_per_resp[0][1], category_info)
    for resp_id, rows in all_rows_per_resp[1:]:
        resp_df = one_hot_df_from_rows(rows, category_info)
        design_df = pd.concat([design_df, resp_df], ignore_index=True)

    # Add Consumer_ID
    consumer_ids = []
    for resp_id, rows in all_rows_per_resp:
        consumer_ids.extend([resp_id] * len(rows))
    design_df['Consumer_ID'] = consumer_ids

    # Derive tasks_per_consumer / capacity
    tasks_per_consumer = max(1, len(design_df) // N)
    capacity = len(design_df)

    # Convert to tasks with content mapping back to layers_data
    tasks_structure: Dict[str, List[Dict[str, Any]]] = {}
    all_elements = [e for es in category_info.values() for e in es]

    for respondent_id in range(number_of_respondents):
        respondent_tasks: List[Dict[str, Any]] = []
        start_idx = respondent_id * tasks_per_consumer
        end_idx = start_idx + tasks_per_consumer
        respondent_data = design_df.iloc[start_idx:end_idx]

        for task_index, (_, task_row) in enumerate(respondent_data.iterrows()):
            elements_shown: Dict[str, int] = {}
            elements_shown_content: Dict[str, Optional[Dict[str, Any]]] = {}

            for element_name in all_elements:
                element_active = int(task_row[element_name])
                elements_shown[element_name] = element_active

                if element_active:
                    # Parse "LayerName_Index"
                    if '_' in element_name:
                        layer_name, img_index_str = element_name.rsplit('_', 1)
                        try:
                            img_index = int(img_index_str) - 1  # 0-based
                            for layer in layers_data:
                                if layer['name'] == layer_name and img_index < len(layer['images']):
                                    image = layer['images'][img_index]
                                    elements_shown_content[element_name] = {
                                        'url': image['url'],
                                        'name': image['name'],
                                        'alt_text': image.get('alt', image.get('alt_text', '')),
                                        'layer_name': layer_name,
                                        'z_index': layer['z_index']
                                    }
                                    break
                            else:
                                elements_shown_content[element_name] = None
                        except ValueError:
                            elements_shown_content[element_name] = None
                    else:
                        elements_shown_content[element_name] = None
                else:
                    elements_shown_content[element_name] = None

            # Clean up any _ref entries
            elements_shown = {k: v for k, v in elements_shown.items() if not k.endswith('_ref')}
            elements_shown_content = {k: v for k, v in elements_shown_content.items() if not k.endswith('_ref')}

            task_obj = {
                "task_id": f"{respondent_id}_{task_index}",
                "elements_shown": elements_shown,
                "elements_shown_content": elements_shown_content,
                "task_index": task_index
            }
            respondent_tasks.append(task_obj)

        tasks_structure[str(respondent_id)] = respondent_tasks

    end_time = time.time()
    total_duration = end_time - start_time
    end_time_str = time.strftime('%H:%M:%S', time.localtime(end_time))
    print(f"✅ Layer task generation completed at {end_time_str}")
    print(f"⏱️ Total duration: {total_duration:.2f} seconds")
    print(f"📊 Performance: {number_of_respondents} respondents in {total_duration:.2f}s = {number_of_respondents/total_duration:.2f} respondents/second")

    return {
        'tasks': tasks_structure,
        'metadata': {
            'study_type': 'layer_v2',
            'layers_data': layers_data,
            'category_info': category_info,
            'tasks_per_consumer': tasks_per_consumer,
            'number_of_respondents': number_of_respondents,
            'exposure_tolerance_pct': exposure_tolerance_pct,
            'capacity': capacity,
            # Pass through optional background image URL from study, if any. The caller may include it.
            'background_image_url': None
        }
    }