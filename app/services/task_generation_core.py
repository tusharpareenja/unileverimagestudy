# app/services/task_generation_core.py
from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple, cast
import numpy as np
import math
import random
import pandas as pd

_rng = np.random.default_rng()

def set_seed(seed: Optional[int] = None) -> None:
    global _rng
    if seed is not None:
        _rng = np.random.default_rng(seed)
        random.seed(seed)
    else:
        _rng = np.random.default_rng()
        random.seed()

# ---------------- GRID ----------------

def _target_k_from_e(E: int) -> int:
    if E <= 8:
        return 2
    elif E <= 16:
        return 3
    return 4

def _choose_k_t_policy(num_consumers: int, num_elements: int, maxT: int = 24, exposure_tol_cv: float = 0.01):
    E = int(num_elements)
    if E < 4:
        raise ValueError("E must be at least 4 (K in [2,4]).")
    if 4 <= E <= 8:
        K = 2
    elif 9 <= E <= 16:
        K = 3
    else:
        K = 4
    K = min(max(2, K), min(4, E))
    cap = math.comb(E, K)
    T = min(maxT, cap)
    notes = []
    if cap < maxT:
        notes.append(f"Capacity C({E},{K})={cap} < {maxT}; T clipped to {T}.")
    else:
        notes.append(f"T set to {T} (soft exposure target {100*exposure_tol_cv:.2f}% CV).")
    return K, K, T, cap, notes

def _compute_exposure_stats(df: pd.DataFrame, elem_names: List[str]) -> Tuple[List[float], float, float, float]:
    counts = df[elem_names].sum().astype(float).to_numpy()
    mean = float(np.mean(counts)) if counts.size else 0.0
    std = float(np.std(counts, ddof=0)) if counts.size else 0.0
    cv = (std / mean) if mean > 0 else 0.0
    return counts, mean, std, cv

def _soft_repair_grid_counts(design_df: pd.DataFrame, elem_names: List[str], exposure_tol_cv=0.01, max_passes=8) -> pd.DataFrame:
    totals = design_df[elem_names].sum().astype(int).to_dict()
    X = design_df[elem_names].to_numpy()
    row_to_cid = design_df["Consumer ID"].astype(str).to_numpy()

    def sig_from_row(row: np.ndarray) -> Tuple[str, ...]:
        return tuple([elem_names[i] for i in np.flatnonzero(row == 1)])

    seen_by_cid = {}
    row_sig = []
    for i in range(X.shape[0]):
        sig = tuple(sorted(sig_from_row(X[i])))
        row_sig.append(sig)
        s = seen_by_cid.setdefault(row_to_cid[i], set()); s.add(sig)

    rows_by_elem = {e: set(np.flatnonzero(design_df[e].to_numpy() == 1)) for e in elem_names}

    _, mean, std, cv = _compute_exposure_stats(design_df, elem_names)
    if cv <= exposure_tol_cv:
        return design_df

    moved_any = True
    passes = 0
    while moved_any and passes < max_passes:
        moved_any = False; passes += 1
        diffs = {e: (totals[e] - mean) for e in elem_names}
        donors = [(e, diffs[e]) for e in elem_names if diffs[e] > 0]
        recvs  = [(e, -diffs[e]) for e in elem_names if diffs[e] < 0]
        donors.sort(key=lambda x: x[1], reverse=True)
        recvs.sort(key=lambda x: x[1], reverse=True)
        if not donors or not recvs:
            break

        recv_list = [e for e, _ in recvs]
        for don, _over in donors:
            _, mean, std, cv = _compute_exposure_stats(design_df, elem_names)
            if cv <= exposure_tol_cv:
                return design_df
            candidate_rows = list(rows_by_elem[don]); _rng.shuffle(candidate_rows)
            for r in candidate_rows:
                # Pick receiver not in row
                receivers = [e for e in recv_list if totals[e] < mean and design_df.at[r, e] == 0]
                if not receivers:
                    continue
                for rec in receivers:
                    cid = row_to_cid[r]
                    new_sig = tuple(sorted([e for e in elem_names if (design_df.at[r, e] == 1 and e != don)] + [rec]))
                    if new_sig in seen_by_cid[cid]:
                        continue
                    # swap
                    design_df.at[r, don] = 0; design_df.at[r, rec] = 1
                    rows_by_elem[don].remove(r); rows_by_elem[rec].add(r)
                    totals[don] -= 1; totals[rec] += 1
                    old_sig = row_sig[r]
                    seen_by_cid[cid].remove(old_sig); seen_by_cid[cid].add(new_sig)
                    row_sig[r] = new_sig
                    moved_any = True
                    break
                if moved_any:
                    break
    return design_df

def _generate_grid_mode(num_consumers: int, tasks_per_consumer: int, num_elements: int, K: int, exposure_tol_cv=0.01, elem_names: Optional[List[str]] = None):
    E = int(num_elements)
    total_tasks = num_consumers * tasks_per_consumer
    # Allow caller to provide explicit element names (e.g., image names). Fallback to E1..EN.
    if not elem_names or len(elem_names) != E:
        elem_names = [f"E{i+1}" for i in range(E)]
    used_elem = {e: 0 for e in elem_names}
    col_index = {e: i for i, e in enumerate(elem_names)}
    design_data = np.zeros((total_tasks, E), dtype=int)

    def ranked_elements(r_mean: float):
        return sorted(
            [((r_mean - used_elem[e]), _rng.random(), e) for e in elem_names],
            key=lambda x: (x[0], x[1]), reverse=True
        )

    row = 0
    for cid in range(1, num_consumers + 1):
        seen = set()
        for _ in range(tasks_per_consumer):
            r_mean = (sum(used_elem.values()) + K) / E
            ranked = ranked_elements(r_mean)
            pool_size = min(E, max(6, 2 * K + 4))
            pool = [e for _, _, e in ranked[:pool_size]]
            attempts = 0
            while attempts < 200:
                chosen = _rng.choice(pool, size=K, replace=False)
                sig = tuple(sorted(chosen))
                if sig not in seen:
                    for e in chosen:
                        used_elem[e] += 1
                        design_data[row, col_index[e]] = 1
                    seen.add(sig); row += 1
                    break
                attempts += 1
            if attempts >= 200:
                raise RuntimeError("Grid: failed to build unique vignette; adjust parameters.")
    df = pd.DataFrame(design_data, columns=elem_names)  # type: ignore[arg-type]
    df.insert(0, "Consumer ID", [f"C{i+1}" for i in range(num_consumers) for _ in range(tasks_per_consumer)])  # type: ignore[arg-type]
    df = _soft_repair_grid_counts(df, elem_names, exposure_tol_cv=exposure_tol_cv)
    _, mean, std, cv = _compute_exposure_stats(df, elem_names)
    r_stats = {"mean": mean, "std": std, "cv_pct": 100.0 * cv}
    return df, r_stats, elem_names

def generate_grid_tasks(
    num_elements: int,
    tasks_per_consumer: Optional[int],
    number_of_respondents: int,
    exposure_tolerance_cv: float = 1.0,
    seed: Optional[int] = None,
    elements: Optional[List[Any]] = None
) -> Dict[str, Any]:
    if seed is not None:
        set_seed(seed)
    exposure_tol_cv = exposure_tolerance_cv / 100.0
    minK, maxK, T, cap, notes = _choose_k_t_policy(number_of_respondents, num_elements, maxT=24, exposure_tol_cv=exposure_tol_cv)
    if tasks_per_consumer is None:
        tasks_per_consumer = T
        notes.append(f"tasks_per_consumer auto-picked to {T}.")
    if tasks_per_consumer > cap:
        raise ValueError(f"T ({tasks_per_consumer}) exceeds capacity ({cap}) for E={num_elements}, K={minK}")
    # If StudyElement list is provided, use their display names as the element keys
    provided_names: Optional[List[str]] = None
    if elements:
        try:
            provided_names = [str(getattr(el, 'name')) for el in elements][:num_elements]
        except Exception:
            provided_names = None

    df, r_stats, elem_names = _generate_grid_mode(
        number_of_respondents,
        tasks_per_consumer,
        num_elements,
        K=minK,
        exposure_tol_cv=exposure_tol_cv,
        elem_names=provided_names,
    )

    tasks_structure: Dict[str, List[Dict[str, Any]]] = {}
    for respondent_id in range(number_of_respondents):
        respondent_tasks = []
        start_idx = respondent_id * tasks_per_consumer
        end_idx = start_idx + tasks_per_consumer
        respondent_data = df.iloc[start_idx:end_idx]
        for task_index, (_, task_row) in enumerate(respondent_data.iterrows()):
            elements_shown = {}
            for i, element_name in enumerate(elem_names):
                active = int(task_row[element_name])
                elements_shown[element_name] = active
                key_content = f"{element_name}_content"
                if active and elements and i < len(elements):
                    elements_shown[key_content] = getattr(elements[i], 'content', '')
                else:
                    elements_shown[key_content] = ""
            elements_shown = {k: v for k, v in elements_shown.items() if not k.endswith('_ref')}
            respondent_tasks.append({
                "task_id": f"{respondent_id}_{task_index}",
                "elements_shown": elements_shown,
                "task_index": task_index
            })
        tasks_structure[str(respondent_id)] = respondent_tasks

    return {
        "tasks": tasks_structure,
        "metadata": {
            "study_type": "grid",
            "num_elements": num_elements,
            "tasks_per_consumer": tasks_per_consumer,
            "number_of_respondents": number_of_respondents,
            "K": minK,
            "exposure_tolerance_cv": exposure_tolerance_cv,
            "exposure_stats": r_stats,
            "notes": notes
        }
    }

# ---------------- LAYER ----------------

def _auto_pick_t_for_layer(category_info: Dict[str, List[str]], baseline=24) -> Tuple[int, int]:
    sizes = [len(v) for v in category_info.values()]
    cap = 1
    for s in sizes:
        cap *= s
    return min(baseline, cap), cap

def _repair_layer_counts(design_df: pd.DataFrame, category_info: Dict[str, List[str]], tol_pct=0.02) -> pd.DataFrame:
    total_tasks = len(design_df)
    cats = list(category_info.keys())

    chosen = {c: cast(pd.Series, design_df[category_info[c]].idxmax(axis=1)).copy() for c in cats}
    row_to_cid = design_df["Consumer ID"].astype(str).to_numpy()
    row_sig = [None] * total_tasks
    seen_by_cid = {}
    for i in range(total_tasks):
        pairs = [(c, chosen[c].iat[i]) for c in cats]
        sig = tuple(sorted(pairs))
        row_sig[i] = sig
        s = seen_by_cid.setdefault(row_to_cid[i], set()); s.add(sig)

    all_elems = [e for es in category_info.values() for e in es]
    totals = design_df[all_elems].sum().astype(int).to_dict()

    lower, upper, target = {}, {}, {}
    for c in cats:
        n = len(category_info[c]); t = total_tasks / n
        tol_cnt = max(1, int(round(tol_pct * t)))
        for e in category_info[c]:
            target[e] = t
            lower[e] = int(np.ceil(t - tol_cnt))
            upper[e] = int(np.floor(t + tol_cnt))

    rows_by_elem = {e: set(np.flatnonzero(design_df[e].to_numpy() == 1)) for e in all_elems}

    moved_any = True; passes = 0
    while moved_any and passes < 4:
        moved_any = False; passes += 1
        for c in cats:
            elems = category_info[c]
            lo = {e: lower[e] for e in elems}
            donors_hi = [e for e in elems if totals[e] > upper[e]]
            donors_lo = [e for e in elems if (e not in donors_hi) and (totals[e] > lo[e])]
            donors = donors_hi + donors_lo
            receivers = [e for e in elems if totals[e] < lo[e]]
            if not donors or not receivers:
                continue
            receivers.sort(key=lambda e: (lo[e] - totals[e]), reverse=True)
            donors.sort(key=lambda e: (totals[e] - lo[e]), reverse=True)
            for rec in receivers:
                need = lo[rec] - totals[rec]
                if need <= 0:
                    continue
                d_idx = 0
                while need > 0 and d_idx < len(donors):
                    don = donors[d_idx]
                    give_cap = totals[don] - (upper[don] if don in donors_hi else lo[don])
                    if give_cap <= 0:
                        d_idx += 1
                        continue
                    candidates = list(rows_by_elem[don]); _rng.shuffle(candidates)
                    gave_here = 0
                    for r in candidates:
                        if give_cap <= 0 or need <= 0:
                            break
                        cid = design_df.at[r, "Consumer ID"]
                        pairs = [(cc, (rec if cc == c else chosen[cc].iat[r])) for cc in cats]
                        new_sig = tuple(sorted(pairs))
                        if new_sig in seen_by_cid[cid]:
                            continue
                        design_df.at[r, don] = 0; design_df.at[r, rec] = 1
                        rows_by_elem[don].remove(r); rows_by_elem[rec].add(r)
                        totals[don] -= 1; totals[rec] += 1
                        old_sig = row_sig[r]
                        seen_by_cid[cid].remove(old_sig); seen_by_cid[cid].add(new_sig)
                        row_sig[r] = new_sig; chosen[c].iat[r] = rec
                        give_cap -= 1; need -= 1; gave_here += 1; moved_any = True
                    if gave_here == 0:
                        d_idx += 1
    return design_df

def _generate_layer_mode(num_consumers: int, tasks_per_consumer: int, category_info: Dict[str, List[str]], tol_pct=0.02):
    all_factors = [e for es in category_info.values() for e in es]
    factor_index = {f: i for i, f in enumerate(all_factors)}
    design_data = np.zeros((num_consumers * tasks_per_consumer, len(all_factors)), dtype=int)
    used_elem = {e: 0 for e in all_factors}
    cats = list(category_info.keys())

    def top_candidates(cat: str, width: int, allow_overflow: bool):
        ranked = sorted(
            [((num_consumers * tasks_per_consumer / len(category_info[cat]) - used_elem[e]), _rng.random(), e) for e in category_info[cat]],
            key=lambda x: (x[0], x[1]), reverse=True,
        )
        if allow_overflow:
            base = [e for _, _, e in ranked]
        else:
            base = [e for _, _, e in ranked if used_elem[e] < math.ceil(num_consumers * tasks_per_consumer / len(category_info[cat]))]
            if not base:
                base = [e for _, _, e in ranked]
        return base[: min(4, len(base))]

    row = 0
    for cid in range(1, num_consumers + 1):
        seen = set()
        for _ in range(tasks_per_consumer):
            success = False
            for allow_overflow in (False, True):
                for _attempt in range(600):
                    pairs = []
                    for cat in cats:
                        cands = top_candidates(cat, 4, allow_overflow)
                        e = _rng.choice(cands)
                        pairs.append((cat, e))
                    sig = tuple(sorted(pairs))
                    if sig not in seen:
                        for _, e in pairs:
                            used_elem[e] += 1
                            design_data[row, factor_index[e]] = 1
                        seen.add(sig); row += 1; success = True
                        break
                if success:
                    break
            if not success:
                raise RuntimeError("Layer: could not build a unique vignette; adjust parameters/tolerance/retries.")

    df = pd.DataFrame(design_data, columns=all_factors)  # type: ignore[arg-type]
    df.insert(0, "Consumer ID", [f"C{i+1}" for i in range(num_consumers) for _ in range(tasks_per_consumer)])  # type: ignore[arg-type]
    df = _repair_layer_counts(df, category_info, tol_pct=tol_pct)
    Ks = [len(cats)] * (num_consumers * tasks_per_consumer)
    return df, Ks, all_factors

def generate_layer_tasks(layers_data: List[Dict], number_of_respondents: int,
                           exposure_tolerance_pct: float = 2.0, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate tasks for the new layer structure (vignette-based approach).
    
    Args:
        layers_data: List of layer objects with images
        number_of_respondents: Number of respondents (N)
        exposure_tolerance_pct: Exposure tolerance as percentage
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing task matrix and metadata
    """
    # Set seed if provided
    if seed is not None:
        set_seed(seed)
    
    # Convert layers data to category_info format for the existing algorithm
    category_info = {}
    for layer in layers_data:
        layer_name = layer['name']
        # Create element names for this layer (e.g., "Background_1", "Background_2")
        elements = [f"{layer_name}_{i+1}" for i in range(len(layer['images']))]
        category_info[layer_name] = elements
    
    # Auto-calculate tasks per consumer
    tasks_per_consumer, capacity = _auto_pick_t_for_layer(category_info, baseline=24)
    
    # Generate design matrix using existing algorithm
    design_df, Ks, _ = _generate_layer_mode(
        num_consumers=number_of_respondents,
        tasks_per_consumer=tasks_per_consumer,
        category_info=category_info,
        tol_pct=exposure_tolerance_pct / 100.0
    )
    
    # Convert to task structure with image content
    tasks_structure = {}
    all_elements = [e for es in category_info.values() for e in es]
    
    for respondent_id in range(number_of_respondents):
        respondent_tasks = []
        start_idx = respondent_id * tasks_per_consumer
        end_idx = start_idx + tasks_per_consumer
        
        # Get tasks for this specific respondent
        respondent_data = design_df.iloc[start_idx:end_idx]
        
        for task_index, (_, task_row) in enumerate(respondent_data.iterrows()):
            # Create elements_shown dictionary
            elements_shown = {}
            elements_shown_content = {}
            
            for element_name in all_elements:
                # Element is only shown if it's active in this task
                element_active = int(task_row[element_name])
                elements_shown[element_name] = element_active
                
                # Find the corresponding image for this element
                if element_active:
                    # Parse element name to find layer and image index
                    # Format: "LayerName_ImageIndex" (e.g., "Background_1")
                    if '_' in element_name:
                        layer_name, img_index_str = element_name.rsplit('_', 1)
                        try:
                            img_index = int(img_index_str) - 1  # Convert to 0-based index
                            
                            # Find the layer and image
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
    
    return {
        'tasks': tasks_structure,
        'metadata': {
            'study_type': 'layer_v2',
            'layers_data': layers_data,
            'category_info': category_info,
            'tasks_per_consumer': tasks_per_consumer,
            'number_of_respondents': number_of_respondents,
            'exposure_tolerance_pct': exposure_tolerance_pct,
            'capacity': capacity
        }
    }
