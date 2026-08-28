from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Literal
import time

import numpy as np
from tqdm import tqdm

try:
    from sklearn.ensemble import GradientBoostingRegressor
except Exception:
    GradientBoostingRegressor = None  # type: ignore

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None  # type: ignore

from surrogatePermOptim import SurrogateWrapper

Perm = List[int]

def _repair_perm(x: Sequence[int], n: int) -> Perm:
    out = [int(v) for v in x]
    seen = set()
    for i, v in enumerate(out):
        if 0 <= v < n and v not in seen:
            seen.add(v)
        else:
            out[i] = -1
    missing = [g for g in range(n) if g not in seen]
    mi = 0
    for i, v in enumerate(out):
        if v == -1:
            out[i] = missing[mi]
            mi += 1
    return out

def _pmx_crossover(p1: Sequence[int], p2: Sequence[int], a: int, b: int) -> Perm:
    n = len(p1)
    if n <= 1:
        return [int(v) for v in p1]

    p1 = [int(v) for v in p1]
    p2 = [int(v) for v in p2]
    child = [-1] * n
    child[a:b] = p1[a:b]
    seg = set(child[a:b])

    # map from p2 gene -> p1 gene on the crossover segment
    mapping: Dict[int, int] = {}
    for i in range(a, b):
        mapping[p2[i]] = p1[i]

    for i in range(n):
        if a <= i < b:
            continue
        g = p2[i]
        seen = set()
        while g in seg and g in mapping and g not in seen:
            seen.add(g)
            g = mapping[g]
        child[i] = g

    return _repair_perm(child, n)

def _ox_crossover(p1: Sequence[int], p2: Sequence[int], a: int, b: int) -> Perm:
    n = len(p1)
    if n <= 1:
        return [int(v) for v in p1]

    p1 = [int(v) for v in p1]
    p2 = [int(v) for v in p2]
    child = [-1] * n
    child[a:b] = p1[a:b]
    seg = set(child[a:b])

    j = b % n
    for i in range(n):
        idx = (b + i) % n
        g = p2[idx]
        if g not in seg:
            child[j] = g
            j = (j + 1) % n

    return _repair_perm(child, n)

# def _pmx_crossover(
#     p1: Sequence[int],
#     p2: Sequence[int],
#     rng: np.random.Generator,
#     a: int,
#     b: int
# ) -> list[int]:
#     """Partially Mapped Crossover (PMX). Returns one valid permutation child."""

#     n = len(p1)
#     if n <= 1:
#         return list(p1)

#     child = [-1] * n

#     child[a:b] = p1[a:b]

#     mapping = {}
#     for i in range(a, b):
#         g1 = p1[i]
#         g2 = p2[i]
#         mapping[g1] = g2
#         mapping[g2] = g1

#     for i in range(n):
#         if a <= i < b:
#             continue

#         g = p2[i]

#         # resolve conflicts
#         while g in child[a:b]:
#             g = mapping[g]

#         child[i] = g

#     return child


def _operator_ga_perm(pop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    OperatorGAperm(..., 'CrossType','PMX') equivalent:
    from |pop| parents, return |pop| offspring.
    """
    n_pop = int(pop.shape[0])
    if n_pop == 0:
        return np.zeros((0, 0), dtype=np.int32)
    if n_pop == 1:
        return pop.copy().astype(np.int32)

    off: List[Perm] = []
    for i in range(0, n_pop, 2):
        p1 = pop[i]
        p2 = pop[i + 1] if i + 1 < n_pop else pop[0]
        n = len(p1)
        a = int(rng.integers(0, n - 1))
        b = int(rng.integers(a + 1, n))
        off.append(_pmx_crossover(p1, p2,a,b))
        # off.append(_ox_crossover(p1, p2,a,b))
        if len(off) < n_pop:
            off.append(_pmx_crossover(p2, p1,a,b))
            # off.append(_ox_crossover(p2, p1,a,b))
    return np.asarray(off[:n_pop], dtype=np.int32)

def _swap_neighbor(p: Sequence[int], rng: np.random.Generator) -> Perm:
    n = len(p)
    if n <= 1:
        return list(p)
    i, j = rng.choice(n, size=2, replace=False)
    q = list(p)
    q[i], q[j] = q[j], q[i]
    return q


def _multi_swap_neighbor(p: Sequence[int], rng: np.random.Generator, depth: int) -> Perm:
    q = list(p)
    for _ in range(max(1, int(depth))):
        q = _swap_neighbor(q, rng)
    return q

def _insert_neighbor(p: Sequence[int], rng: np.random.Generator) -> Perm:
    n = len(p)
    if n <= 1:
        return list(p)
    i, j = rng.choice(n, size=2, replace=False)
    q = list(p)
    gene = q.pop(i)
    q.insert(j, gene)
    return q

def _multi_insert_neighbor(p: Sequence[int], rng: np.random.Generator, depth: int) -> Perm:
    q = list(p)
    for _ in range(max(1, int(depth))):
        q = _insert_neighbor(q, rng)
    return q


def _make_incumbent_pool(
    best_perm: Sequence[int],
    rng: np.random.Generator,
    pool_size: int,
    max_depth: int,
) -> np.ndarray:
    pool: List[Perm] = []
    pool_size = max(1, int(pool_size))
    max_depth = max(1, int(max_depth))
    for _ in range(pool_size):
        depth = int(rng.integers(1, max_depth + 1))
        # pool.append(_multi_swap_neighbor(best_perm, rng, depth))
        pool.append(_multi_insert_neighbor(best_perm, rng, depth))
    return np.asarray(pool, dtype=np.int32)


def _incumbent_insert_internal_search(
    best_perm: Sequence[int],
    mdl: Any,
    rng: np.random.Generator,
    pool_size: int,
    max_depth: int,
    sigma_steps: int,
    local_acq_kind: str,
    local_kappa: float,
    arc_set: set[Tuple[int, ...]],
    chosen_set: set[Tuple[int, ...]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    sigma_steps = max(1, int(sigma_steps))
    inc_dec = _make_incumbent_pool(
        best_perm=best_perm,
        rng=rng,
        pool_size=pool_size,
        max_depth=max_depth,
    )
    inc_dec, _ = _unique_rows_stable(inc_dec)
    if inc_dec.shape[0] == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return inc_dec, empty, empty, empty, None, {
            "sigma": int(sigma_steps),
            "initial_pool_size": 0,
            "final_pool_size": 0,
            "updates_total": 0,
            "updates_by_step": np.zeros((sigma_steps,), dtype=np.int64),
            "known_child_skips_by_step": np.zeros((sigma_steps,), dtype=np.int64),
            "duplicate_child_skips_by_step": np.zeros((sigma_steps,), dtype=np.int64),
            "birth_step": np.zeros((0,), dtype=np.int64),
            "update_count": np.zeros((0,), dtype=np.int64),
            "initial_score": empty.copy(),
            "final_score": empty.copy(),
            "comparison_step": np.zeros((0,), dtype=np.int64),
            "comparison_parent_idx": np.zeros((0,), dtype=np.int64),
            "comparison_parent_perm": np.zeros((0, len(best_perm)), dtype=np.int32),
            "comparison_child_perm": np.zeros((0, len(best_perm)), dtype=np.int32),
            "comparison_parent_score": empty.copy(),
            "comparison_child_score": empty.copy(),
            "comparison_surrogate_update": np.zeros((0,), dtype=bool),
        }

    mu_cur, sigma_cur, _ = _predict_with_optional_embedding(mdl, inc_dec.tolist())
    score_cur = _compute_acquisition(mu_cur, sigma_cur, local_acq_kind, local_kappa)
    initial_score = score_cur.copy()
    birth_step = np.ones((inc_dec.shape[0],), dtype=np.int64)
    update_count = np.zeros((inc_dec.shape[0],), dtype=np.int64)
    updates_by_step = np.zeros((sigma_steps,), dtype=np.int64)
    known_child_skips_by_step = np.zeros((sigma_steps,), dtype=np.int64)
    duplicate_child_skips_by_step = np.zeros((sigma_steps,), dtype=np.int64)
    comparison_step: List[int] = []
    comparison_parent_idx: List[int] = []
    comparison_parent_perm: List[np.ndarray] = []
    comparison_child_perm: List[np.ndarray] = []
    comparison_parent_score: List[float] = []
    comparison_child_score: List[float] = []
    comparison_surrogate_update: List[bool] = []

    unavailable = set(arc_set) | set(chosen_set)
    for step in range(2, sigma_steps + 1):
        child_list: List[Perm] = []
        child_parent_idx: List[int] = []
        seen_child: set[Tuple[int, ...]] = set()
        for parent_idx, parent in enumerate(inc_dec):
            child = _insert_neighbor(parent.tolist(), rng)
            key = tuple(int(v) for v in child)
            if key in unavailable:
                known_child_skips_by_step[step - 1] += 1
                continue
            if key in seen_child:
                duplicate_child_skips_by_step[step - 1] += 1
                continue
            seen_child.add(key)
            child_list.append(child)
            child_parent_idx.append(int(parent_idx))

        if not child_list:
            continue

        child_dec = np.asarray(child_list, dtype=np.int32)
        mu_child, sigma_child, _ = _predict_with_optional_embedding(mdl, child_dec.tolist())
        score_child = _compute_acquisition(mu_child, sigma_child, local_acq_kind, local_kappa)
        for child_pos, parent_idx in enumerate(child_parent_idx):
            parent_score_before = float(score_cur[parent_idx])
            child_score_now = float(score_child[child_pos])
            surrogate_update = child_score_now < parent_score_before
            comparison_step.append(int(step))
            comparison_parent_idx.append(int(parent_idx))
            comparison_parent_perm.append(inc_dec[parent_idx].astype(np.int32).copy())
            comparison_child_perm.append(child_dec[child_pos].astype(np.int32).copy())
            comparison_parent_score.append(parent_score_before)
            comparison_child_score.append(child_score_now)
            comparison_surrogate_update.append(bool(surrogate_update))
            if surrogate_update:
                inc_dec[parent_idx] = child_dec[child_pos]
                mu_cur[parent_idx] = mu_child[child_pos]
                sigma_cur[parent_idx] = sigma_child[child_pos]
                score_cur[parent_idx] = score_child[child_pos]
                birth_step[parent_idx] = step
                update_count[parent_idx] += 1
                updates_by_step[step - 1] += 1

    inc_dec, keep = _unique_rows_stable(inc_dec)
    mu_cur = mu_cur[keep]
    sigma_cur = sigma_cur[keep]
    score_cur = score_cur[keep]
    initial_score = initial_score[keep]
    birth_step = birth_step[keep]
    update_count = update_count[keep]
    _, _, emb_cur = _predict_with_optional_embedding(mdl, inc_dec.tolist())
    if comparison_parent_perm:
        comparison_parent_perm_arr = np.vstack(comparison_parent_perm).astype(np.int32)
        comparison_child_perm_arr = np.vstack(comparison_child_perm).astype(np.int32)
    else:
        comparison_parent_perm_arr = np.zeros((0, len(best_perm)), dtype=np.int32)
        comparison_child_perm_arr = np.zeros((0, len(best_perm)), dtype=np.int32)
    diag: Dict[str, Any] = {
        "sigma": int(sigma_steps),
        "initial_pool_size": int(initial_score.shape[0]),
        "final_pool_size": int(inc_dec.shape[0]),
        "updates_total": int(np.sum(updates_by_step)),
        "updates_by_step": updates_by_step.copy(),
        "known_child_skips_by_step": known_child_skips_by_step.copy(),
        "duplicate_child_skips_by_step": duplicate_child_skips_by_step.copy(),
        "birth_step": birth_step.copy(),
        "update_count": update_count.copy(),
        "initial_score": initial_score.copy(),
        "final_score": score_cur.copy(),
        "comparison_step": np.asarray(comparison_step, dtype=np.int64),
        "comparison_parent_idx": np.asarray(comparison_parent_idx, dtype=np.int64),
        "comparison_parent_perm": comparison_parent_perm_arr,
        "comparison_child_perm": comparison_child_perm_arr,
        "comparison_parent_score": np.asarray(comparison_parent_score, dtype=np.float64),
        "comparison_child_score": np.asarray(comparison_child_score, dtype=np.float64),
        "comparison_surrogate_update": np.asarray(comparison_surrogate_update, dtype=bool),
    }
    return inc_dec, mu_cur, sigma_cur, score_cur, emb_cur, diag

def _incumbent_insert_drifting_search(
    best_perm: Sequence[int],
    best_val: float,
    mdl: Any,
    rng: np.random.Generator,
    pool_size: int,
    max_depth: int,
    sigma_steps: int,
    local_acq_kind: str,
    local_kappa: float,
    arc_set: set[Tuple[int, ...]],
    chosen_set: set[Tuple[int, ...]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    sigma_steps = max(1, int(sigma_steps))
    pool_size = max(1, int(pool_size))
    max_depth = max(1, int(max_depth))

    current_center = [int(v) for v in best_perm]
    current_center_value = float(best_val)
    unavailable = set(arc_set) | set(chosen_set)
    seen_generated: set[Tuple[int, ...]] = set()

    cand_blocks: List[np.ndarray] = []
    mu_blocks: List[np.ndarray] = []
    sigma_blocks: List[np.ndarray] = []
    score_blocks: List[np.ndarray] = []
    birth_blocks: List[np.ndarray] = []
    center_update_blocks: List[np.ndarray] = []

    updates_by_step = np.zeros((sigma_steps,), dtype=np.int64)
    known_child_skips_by_step = np.zeros((sigma_steps,), dtype=np.int64)
    duplicate_child_skips_by_step = np.zeros((sigma_steps,), dtype=np.int64)
    center_score_history = np.full((sigma_steps + 1,), np.nan, dtype=np.float64)
    center_score_history[0] = current_center_value
    center_update_count = 0

    for step in range(1, sigma_steps + 1):
        raw_child_dec = _make_incumbent_pool(
            best_perm=current_center,
            rng=rng,
            pool_size=pool_size,
            max_depth=max_depth,
        )
        raw_child_dec, _ = _unique_rows_stable(raw_child_dec)
        child_list: List[Perm] = []
        seen_step: set[Tuple[int, ...]] = set()
        for child in raw_child_dec:
            child_perm = [int(v) for v in child]
            key = tuple(child_perm)
            if key in unavailable:
                known_child_skips_by_step[step - 1] += 1
                continue
            if key in seen_step or key in seen_generated:
                duplicate_child_skips_by_step[step - 1] += 1
                continue
            seen_step.add(key)
            seen_generated.add(key)
            child_list.append(child_perm)

        if not child_list:
            center_score_history[step] = current_center_value
            continue

        child_dec = np.asarray(child_list, dtype=np.int32)
        mu_child, sigma_child, _ = _predict_with_optional_embedding(mdl, child_dec.tolist())
        score_child = _compute_acquisition(mu_child, sigma_child, local_acq_kind, local_kappa)

        cand_blocks.append(child_dec.copy())
        mu_blocks.append(mu_child.copy())
        sigma_blocks.append(sigma_child.copy())
        score_blocks.append(score_child.copy())
        birth_blocks.append(np.full((child_dec.shape[0],), step, dtype=np.int64))
        center_update_blocks.append(np.full((child_dec.shape[0],), center_update_count, dtype=np.int64))

        best_child_idx = int(np.argmin(mu_child))
        best_child_mu = float(mu_child[best_child_idx])
        if best_child_mu < current_center_value:
            current_center = child_dec[best_child_idx].tolist()
            current_center_value = best_child_mu
            center_update_count += 1
            updates_by_step[step - 1] += 1
        center_score_history[step] = current_center_value

    if not cand_blocks:
        empty = np.zeros((0,), dtype=np.float64)
        return np.zeros((0, len(best_perm)), dtype=np.int32), empty, empty, empty, None, {
            "sigma": int(sigma_steps),
            "search_mode": "mu_plus_lambda_drifting_center",
            "initial_pool_size": 0,
            "final_pool_size": 0,
            "updates_total": int(np.sum(updates_by_step)),
            "updates_by_step": updates_by_step.copy(),
            "known_child_skips_by_step": known_child_skips_by_step.copy(),
            "duplicate_child_skips_by_step": duplicate_child_skips_by_step.copy(),
            "birth_step": np.zeros((0,), dtype=np.int64),
            "update_count": np.zeros((0,), dtype=np.int64),
            "initial_score": empty.copy(),
            "final_score": empty.copy(),
            "center_score_history": center_score_history.copy(),
        }

    inc_dec = np.vstack(cand_blocks).astype(np.int32)
    mu_cur = np.concatenate(mu_blocks).astype(np.float64)
    sigma_cur = np.concatenate(sigma_blocks).astype(np.float64)
    score_cur = np.concatenate(score_blocks).astype(np.float64)
    birth_step = np.concatenate(birth_blocks).astype(np.int64)
    update_count = np.concatenate(center_update_blocks).astype(np.int64)

    inc_dec, keep = _unique_rows_stable(inc_dec)
    mu_cur = mu_cur[keep]
    sigma_cur = sigma_cur[keep]
    score_cur = score_cur[keep]
    birth_step = birth_step[keep]
    update_count = update_count[keep]
    _, _, emb_cur = _predict_with_optional_embedding(mdl, inc_dec.tolist())

    diag: Dict[str, Any] = {
        "sigma": int(sigma_steps),
        "search_mode": "mu_plus_lambda_drifting_center",
        "initial_pool_size": int(pool_size),
        "final_pool_size": int(inc_dec.shape[0]),
        "updates_total": int(np.sum(updates_by_step)),
        "updates_by_step": updates_by_step.copy(),
        "known_child_skips_by_step": known_child_skips_by_step.copy(),
        "duplicate_child_skips_by_step": duplicate_child_skips_by_step.copy(),
        "birth_step": birth_step.copy(),
        "update_count": update_count.copy(),
        "initial_score": score_cur.copy(),
        "final_score": score_cur.copy(),
        "center_score_history": center_score_history.copy(),
    }
    return inc_dec, mu_cur, sigma_cur, score_cur, emb_cur, diag


def _append_incumbent_comparison_truth(
    inc_diag: Dict[str, Any],
    true_eval_fn: Callable[[Sequence[int]], float],
    truth_cache: Dict[Tuple[int, ...], float],
) -> None:
    """Attach true-value diagnostics for incumbent parent-child comparisons.

    These evaluations are for logging/analysis only. The caller must not count
    them as function evaluations.
    """
    parent_perm = np.asarray(inc_diag.get("comparison_parent_perm", []), dtype=np.int32)
    child_perm = np.asarray(inc_diag.get("comparison_child_perm", []), dtype=np.int32)
    if parent_perm.ndim == 1:
        parent_perm = parent_perm.reshape(1, -1) if parent_perm.size else np.zeros((0, 0), dtype=np.int32)
    if child_perm.ndim == 1:
        child_perm = child_perm.reshape(1, -1) if child_perm.size else np.zeros((0, 0), dtype=np.int32)

    n_comp = int(min(parent_perm.shape[0], child_perm.shape[0]))
    if n_comp == 0:
        empty = np.zeros((0,), dtype=np.float64)
        inc_diag["comparison_parent_true"] = empty.copy()
        inc_diag["comparison_child_true"] = empty.copy()
        inc_diag["comparison_true_update"] = np.zeros((0,), dtype=bool)
        inc_diag["comparison_decision_correct"] = np.zeros((0,), dtype=bool)
        inc_diag["comparison_count"] = 0
        inc_diag["comparison_correct_count"] = 0
        inc_diag["comparison_accuracy"] = np.nan
        inc_diag["comparison_tp"] = 0
        inc_diag["comparison_fp"] = 0
        inc_diag["comparison_tn"] = 0
        inc_diag["comparison_fn"] = 0
        return

    parent_perm = parent_perm[:n_comp]
    child_perm = child_perm[:n_comp]

    def eval_cached(row: np.ndarray) -> float:
        key = tuple(int(v) for v in row)
        if key not in truth_cache:
            truth_cache[key] = float(true_eval_fn([int(v) for v in row]))
        return truth_cache[key]

    parent_true = np.asarray([eval_cached(row) for row in parent_perm], dtype=np.float64)
    child_true = np.asarray([eval_cached(row) for row in child_perm], dtype=np.float64)
    surrogate_update = np.asarray(
        inc_diag.get("comparison_surrogate_update", np.zeros((n_comp,), dtype=bool)),
        dtype=bool,
    ).reshape(-1)[:n_comp]
    true_update = child_true < parent_true
    correct = surrogate_update == true_update

    inc_diag["comparison_parent_true"] = parent_true
    inc_diag["comparison_child_true"] = child_true
    inc_diag["comparison_true_update"] = true_update
    inc_diag["comparison_decision_correct"] = correct
    inc_diag["comparison_count"] = int(n_comp)
    inc_diag["comparison_correct_count"] = int(np.sum(correct))
    inc_diag["comparison_accuracy"] = float(np.mean(correct))
    inc_diag["comparison_tp"] = int(np.sum(surrogate_update & true_update))
    inc_diag["comparison_fp"] = int(np.sum(surrogate_update & ~true_update))
    inc_diag["comparison_tn"] = int(np.sum(~surrogate_update & ~true_update))
    inc_diag["comparison_fn"] = int(np.sum(~surrogate_update & true_update))


def _as_perm_list(perms: Sequence[Sequence[int]] | np.ndarray) -> List[Perm]:
    arr = np.asarray(perms, dtype=np.int32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return [[int(v) for v in row] for row in arr]


def _unpack_prediction_result(res: Any, n: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    emb: Optional[np.ndarray] = None
    if isinstance(res, dict):
        mu = res.get("mu", res.get("mean", res.get("pred")))
        sigma = res.get("sigma", res.get("std", None))
        emb = res.get("emb", res.get("embedding", res.get("z", None)))
    elif isinstance(res, tuple) or isinstance(res, list):
        if len(res) >= 3:
            mu, sigma, emb = res[0], res[1], res[2]
        elif len(res) == 2:
            mu, sigma = res[0], res[1]
        elif len(res) == 1:
            mu, sigma = res[0], None
        else:
            raise ValueError("Empty prediction result.")
    else:
        mu, sigma = res, None

    mu_arr = np.asarray(mu, dtype=np.float64).reshape(-1)
    if sigma is None:
        sigma_arr = np.zeros_like(mu_arr, dtype=np.float64)
    else:
        sigma_arr = np.asarray(sigma, dtype=np.float64).reshape(-1)

    if mu_arr.shape[0] != n:
        raise ValueError(f"Prediction length mismatch: expected {n}, got {mu_arr.shape[0]}.")
    if sigma_arr.shape[0] != n:
        sigma_arr = np.zeros_like(mu_arr, dtype=np.float64)

    if emb is not None:
        emb_arr = np.asarray(emb, dtype=np.float64)
        if emb_arr.ndim == 1:
            emb_arr = emb_arr.reshape(n, -1)
        if emb_arr.shape[0] != n:
            emb = None
        else:
            emb = emb_arr
    return mu_arr, sigma_arr, emb


def _predict_with_optional_embedding(
    mdl: Any,
    perms: Sequence[Sequence[int]] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Return (mu, sigma, embedding) if the surrogate exposes embeddings.

    Supported optional APIs on SurrogateWrapper:
      - predict_with_embedding(perms) -> (mu, sigma, emb) or dict
      - predict_return_embedding(perms) -> (mu, sigma, emb) or dict
      - predict(perms, return_embedding=True) -> (mu, sigma, emb) or dict
      - embed(perms), get_embedding(perms), or get_embeddings(perms) after predict(perms)

    If none is available, embedding is None and the caller falls back to score-only selection.
    """
    perm_list = _as_perm_list(perms)
    n = len(perm_list)

    for name in ("predict_with_embedding", "predict_return_embedding"):
        if hasattr(mdl, name):
            try:
                return _unpack_prediction_result(getattr(mdl, name)(perm_list), n)
            except TypeError:
                pass

    try:
        return _unpack_prediction_result(mdl.predict(perm_list, return_embedding=True), n)
    except TypeError:
        pass

    mu, sigma, _ = _unpack_prediction_result(mdl.predict(perm_list), n)

    emb: Optional[np.ndarray] = None
    for name in ("embed", "get_embedding", "get_embeddings"):
        if hasattr(mdl, name):
            try:
                emb_tmp = np.asarray(getattr(mdl, name)(perm_list), dtype=np.float64)
                if emb_tmp.ndim == 1:
                    emb_tmp = emb_tmp.reshape(n, -1)
                if emb_tmp.shape[0] == n:
                    emb = emb_tmp
                    break
            except Exception:
                pass
    return mu, sigma, emb


def _compute_acquisition(
    mu: np.ndarray,
    sigma: np.ndarray,
    acq: str,
    kappa: float,
) -> np.ndarray:
    if acq == "lcb":
        return np.asarray(mu, dtype=np.float64) - float(kappa) * np.asarray(sigma, dtype=np.float64)
    return np.asarray(mu, dtype=np.float64)


def _embedding_distance_to_selected(
    emb: np.ndarray,
    idx: int,
    selected: Sequence[int],
    metric: str,
) -> float:
    if len(selected) == 0:
        return float("inf")
    if metric == "cosine":
        e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        sims = e[idx] @ e[np.asarray(selected, dtype=np.int64)].T
        dists = 1.0 - sims
    else:
        dists = np.linalg.norm(emb[idx] - emb[np.asarray(selected, dtype=np.int64)], axis=1)
    return float(np.min(dists))

def _embedding_distance_to_reference(
    emb: np.ndarray,
    idx: int,
    reference_emb: Optional[np.ndarray],
    metric: str,
) -> float:
    if reference_emb is None:
        return float("inf")
    ref = np.asarray(reference_emb, dtype=np.float64)
    if ref.size == 0:
        return float("inf")
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
    if metric == "cosine":
        e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        r = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12)
        dists = 1.0 - (e[idx] @ r.T)
    else:
        dists = np.linalg.norm(emb[idx] - ref, axis=1)
    return float(np.min(dists))

def _embedding_knn_mean_dist_to_reference(
    emb: np.ndarray,
    idx: int,
    ref_emb: Optional[np.ndarray],
    metric: str,
    knn_k: int = 1,
) -> float:
    if ref_emb is None:
        return float("inf")
    arc = np.asarray(ref_emb, dtype=np.float64)
    if arc.size == 0:
        return float("inf")
    if arc.ndim == 1:
        arc = arc.reshape(1, -1)
    if metric == "cosine":
        e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        a = arc / (np.linalg.norm(arc, axis=1, keepdims=True) + 1e-12)
        dists = 1.0 - (e[idx] @ a.T)
    else:
        dists = np.linalg.norm(emb[idx] - arc, axis=1)
    if dists.size == 0:
        return float("inf")
    k_eff = min(max(1, int(knn_k)), dists.size)
    nearest = np.partition(dists, kth=k_eff - 1)[:k_eff]
    return float(np.mean(nearest))


def _combine_archive_and_selected_embeddings(
    emb: np.ndarray,
    selected: Sequence[int],
    arc_emb: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    blocks: List[np.ndarray] = []
    if arc_emb is not None:
        arc = np.asarray(arc_emb, dtype=np.float64)
        if arc.size > 0:
            if arc.ndim == 1:
                arc = arc.reshape(1, -1)
            blocks.append(arc)
    if selected:
        emb_arr = np.asarray(emb, dtype=np.float64)
        blocks.append(emb_arr[np.asarray(selected, dtype=np.int64)])
    if not blocks:
        return None
    return np.vstack(blocks)

def _estimate_latent_min_dist(
    emb: np.ndarray,
    order: np.ndarray,
    pool_size: int,
    q: float,
    metric: str,
) -> float:
    pool = order[: max(2, min(int(pool_size), len(order)))]
    if pool.size < 2:
        return 0.0
    z = np.asarray(emb[pool], dtype=np.float64)
    if metric == "cosine":
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        dist = 1.0 - (z @ z.T)
    else:
        diff = z[:, None, :] - z[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
    tri = dist[np.triu_indices(dist.shape[0], k=1)]
    if tri.size == 0:
        return 0.0
    return float(np.quantile(tri, float(q)))

def _estimate_latent_min_dist_pop(
    emb: np.ndarray,
    q: float,
    metric: str,
) -> float:
    if emb.shape[0] < 2:
        return 0.0
    z = np.asarray(emb, dtype=np.float64)
    if metric == "cosine":
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        dist = 1.0 - (z @ z.T)
    else:
        diff = z[:, None, :] - z[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
    tri = dist[np.triu_indices(dist.shape[0], k=1)]
    if tri.size == 0:
        return 0.0
    return float(np.quantile(tri, float(q)))

def _estimate_latent_min_dist_knn_density(
    emb: np.ndarray,
    q: float,
    metric: str,
    knn_k: int = 1,
) -> float:
    if emb.shape[0] < 2:
        return 0.0
    z = np.asarray(emb, dtype=np.float64)
    if metric == "cosine":
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        dist = 1.0 - (z @ z.T)
    else:
        diff = z[:, None, :] - z[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    k_eff = min(max(1, int(knn_k)), dist.shape[1] - 1)
    if k_eff <= 0:
        return 0.0
    knn_dist = np.partition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
    return float(np.quantile(np.mean(knn_dist, axis=1), float(q)))

def _topk_excluding(
    score: np.ndarray,
    k: int,
    exclude: Optional[set[int]] = None,
) -> np.ndarray:
    exclude = exclude or set()
    out: List[int] = []
    for idx in np.argsort(score):
        ii = int(idx)
        if ii in exclude:
            continue
        out.append(ii)
        if len(out) >= k:
            break
    return np.asarray(out, dtype=np.int64)

def _pairwise_embedding_distances(
    x: np.ndarray,
    y: np.ndarray,
    metric: str,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(1, -1)
    if y_arr.shape[0] == 0:
        return np.zeros((x_arr.shape[0], 0), dtype=np.float64)
    if x_arr.shape[1] != y_arr.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"x={x_arr.shape}, y={y_arr.shape}"
        )
    if metric == "cosine":
        x_norm = x_arr / (
            np.linalg.norm(x_arr, axis=1, keepdims=True) + 1e-12
        )
        y_norm = y_arr / (
            np.linalg.norm(y_arr, axis=1, keepdims=True) + 1e-12
        )
        dist = 1.0 - (x_norm @ y_norm.T)
        return np.clip(dist, 0.0, 2.0)
    if metric == "euclidean":
        diff = x_arr[:, None, :] - y_arr[None, :, :]
        return np.linalg.norm(diff, axis=2)
    raise ValueError(f"Unknown embedding metric: {metric}")

def _embedding_knn_mean_dist_to_archive(
    emb: np.ndarray,
    candidate_idx: np.ndarray,
    arc_emb: Optional[np.ndarray],
    metric: str,
    knn_k: int,
) -> np.ndarray:
    """
    蜷・呵｣懊↓縺､縺・※縲∬ｩ穂ｾ｡貂医∩繧｢繝ｼ繧ｫ繧､繝門・縺ｮk霑大ｍ縺ｸ縺ｮ蟷ｳ蝮・ｷ晞屬繧定ｿ斐☆縲・
    arc_emb 縺悟ｭ伜惠縺励↑縺・ｴ蜷医・ inf 繧定ｿ斐☆縲・    縺薙ｌ縺ｫ繧医ｊ縲∵怙蛻昴・蛟呵｣懊・score鬆・・蛟狗岼莉･髯阪・
    selected髢楢ｷ晞屬縺ｫ繧医▲縺ｦ驕ｸ謚槭〒縺阪ｋ縲・    """
    idx = np.asarray(candidate_idx, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if arc_emb is None:
        return np.full(idx.shape[0], np.inf, dtype=np.float64)
    arc = np.asarray(arc_emb, dtype=np.float64)
    if arc.size == 0:
        return np.full(idx.shape[0], np.inf, dtype=np.float64)
    if arc.ndim == 1:
        arc = arc.reshape(1, -1)
    candidate_emb = np.asarray(emb, dtype=np.float64)[idx]
    dist = _pairwise_embedding_distances(
        candidate_emb,
        arc,
        metric,
    )
    if dist.shape[1] == 0:
        return np.full(idx.shape[0], np.inf, dtype=np.float64)
    k_eff = min(
        max(1, int(knn_k)),
        dist.shape[1],
    )
    # 蜈ｨ霍晞屬繧痴ort縺吶ｋ繧医ｊpartition縺ｮ譁ｹ縺悟柑邇・噪
    nearest = np.partition(
        dist,
        kth=k_eff - 1,
        axis=1,
    )[:, :k_eff]
    return np.mean(nearest, axis=1)

def _knn_promising_farthest_select(
    score: np.ndarray,
    emb: np.ndarray,
    arc_emb: Optional[np.ndarray],
    k: int,
    promising_q: float,
    archive_knn_k: int,
    metric: str,
    exclude: Optional[set[int]] = None,
) -> np.ndarray:
    k = max(0, int(k))
    if k == 0:
        return np.zeros((0,), dtype=np.int64)
    exclude = exclude or set()
    score_arr = np.asarray(score, dtype=np.float64).reshape(-1)
    emb_arr = np.asarray(emb, dtype=np.float64)
    if emb_arr.ndim == 1:
        emb_arr = emb_arr.reshape(score_arr.shape[0], -1)
    if emb_arr.shape[0] != score_arr.shape[0]:
        raise ValueError(
            f"score and embedding length mismatch: "
            f"score={score_arr.shape[0]}, emb={emb_arr.shape[0]}"
        )
    promising_q = float(promising_q)
    # Build acquisition order after removing excluded candidates.
    valid_order = np.asarray(
        [
            int(i)
            for i in np.argsort(score_arr)
            if int(i) not in exclude
        ],
        dtype=np.int64,
    )
    if valid_order.size == 0:
        return np.zeros((0,), dtype=np.int64)
    select_num = min(k, valid_order.size)
    # Keep at least enough candidates to fill the requested selections.
    promising_pool_size = max(
        select_num,
        int(np.ceil(promising_q * valid_order.size)),
    )
    promising_pool_size = min(
        promising_pool_size,
        valid_order.size,
    )
    promising_idx = valid_order[:promising_pool_size]
    # 蜷аromising candidate縺ｮ繧｢繝ｼ繧ｫ繧､繝悶↓蟇ｾ縺吶ｋnovelty
    archive_novelty = _embedding_knn_mean_dist_to_archive(
        emb=emb_arr,
        candidate_idx=promising_idx,
        arc_emb=arc_emb,
        metric=metric,
        knn_k=archive_knn_k,
    )
    selected: List[int] = []
    # Track remaining positions inside promising_idx.
    remaining_mask = np.ones(
        promising_pool_size,
        dtype=bool,
    )
    for _ in range(select_num):
        remaining_pos = np.flatnonzero(remaining_mask)
        if remaining_pos.size == 0:
            break
        remaining_idx = promising_idx[remaining_pos]
        if len(selected) == 0:
            selected_novelty = np.full(
                remaining_pos.shape[0],
                np.inf,
                dtype=np.float64,
            )
        else:
            dist_to_selected = _pairwise_embedding_distances(
                emb_arr[remaining_idx],
                emb_arr[np.asarray(selected, dtype=np.int64)],
                metric,
            )
            selected_novelty = np.min(
                dist_to_selected,
                axis=1,
            )
        # Prefer candidates far from both archive and already selected candidates.
        novelty = np.minimum(
            archive_novelty[remaining_pos],
            selected_novelty,
        )
        novelty = np.nan_to_num(
            novelty,
            nan=-np.inf,
            posinf=np.inf,
            neginf=-np.inf,
        )
        # Break ties by the acquisition order already encoded in remaining_idx.
        best_relative_pos = int(np.argmax(novelty))
        best_pool_pos = int(remaining_pos[best_relative_pos])
        best_idx = int(promising_idx[best_pool_pos])
        selected.append(best_idx)
        remaining_mask[best_pool_pos] = False
    return np.asarray(selected, dtype=np.int64)

def _global_selection_diagnostics(
    score: np.ndarray,
    emb: Optional[np.ndarray],
    selected_idx: Sequence[int],
    reference_emb: Optional[np.ndarray],
    pop_emb: Optional[np.ndarray],
    arc_emb: Optional[np.ndarray],
    cfg: Any,
) -> Dict[str, np.ndarray]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    order = np.argsort(score)
    selected = np.asarray([int(i) for i in selected_idx], dtype=np.int64)
    rank = np.full(score.shape[0], -1, dtype=np.int64)
    rank[order] = np.arange(order.shape[0], dtype=np.int64)
    top_idx = order[: selected.shape[0]] if selected.shape[0] else np.zeros((0,), dtype=np.int64)

    min_dist = np.nan
    dref = np.full(selected.shape[0], np.nan, dtype=np.float64)
    dprev = np.full(selected.shape[0], np.nan, dtype=np.float64)
    darchive = np.full(selected.shape[0], np.nan, dtype=np.float64)
    dgate = np.full(selected.shape[0], np.nan, dtype=np.float64)
    if emb is not None and selected.shape[0] > 0:
        emb_arr = np.asarray(emb, dtype=np.float64)
        pool_size = max(selected.shape[0], selected.shape[0] * max(1, int(getattr(cfg, "latent_pool_mult", 20))))
        metric = str(getattr(cfg, "latent_metric", "cosine"))
        q = float(getattr(cfg, "latent_min_dist_q", 0.10))
        archive_knn_k = int(getattr(cfg, "latent_archive_knn_k", getattr(cfg, "archive_knn_k", 1)))
        try:
            if cfg.min_dist_type == "pop" and pop_emb is not None:
                min_dist = _estimate_latent_min_dist_pop(pop_emb, q, metric)
            elif cfg.min_dist_type == "min_both" and pop_emb is not None:
                min_dist = min(
                    _estimate_latent_min_dist(emb_arr, order, pool_size, q, metric),
                    _estimate_latent_min_dist_pop(pop_emb, q, metric),
                )
            elif cfg.min_dist_type == "knn_density" and arc_emb is not None:
                min_dist = _estimate_latent_min_dist_knn_density(arc_emb, q, metric, archive_knn_k)
            else:
                min_dist = _estimate_latent_min_dist(emb_arr, order, pool_size, q, metric)
        except Exception:
            min_dist = np.nan

        prev: List[int] = []
        for j, idx in enumerate(selected.tolist()):
            dref[j] = _embedding_distance_to_reference(emb_arr, int(idx), reference_emb, metric)
            dprev[j] = _embedding_distance_to_selected(emb_arr, int(idx), prev, metric) if prev else np.inf
            if cfg.min_dist_type == "knn_density" and arc_emb is not None:
                ref_emb = _combine_archive_and_selected_embeddings(emb_arr, prev, arc_emb)
                darchive[j] = _embedding_knn_mean_dist_to_reference(emb_arr, int(idx), arc_emb, metric, archive_knn_k)
                dgate[j] = _embedding_knn_mean_dist_to_reference(emb_arr, int(idx), ref_emb, metric, archive_knn_k)
            else:
                dgate[j] = min(dref[j], dprev[j])
            prev.append(int(idx))

    return {
        "score_order": order.astype(np.int64),
        "score_top_idx": top_idx.astype(np.int64),
        "selected_idx": selected.astype(np.int64),
        "selected_score_rank": (rank[selected] if selected.shape[0] else np.zeros((0,), dtype=np.int64)),
        "selected_rank_delta": ((rank[selected] - np.arange(selected.shape[0], dtype=np.int64)) if selected.shape[0] else np.zeros((0,), dtype=np.int64)),
        "selected_dist_to_incumbent": dref,
        "selected_dist_to_prev_selected": dprev,
        "selected_dist_to_archive": darchive,
        "selected_gate_dist": dgate,
        "min_dist": np.asarray([min_dist], dtype=np.float64),
    }


def _latent_nms_select(
    score: np.ndarray,
    pop_emb: Optional[np.ndarray],
    arc_emb: Optional[np.ndarray],
    emb: Optional[np.ndarray],
    k: int,
    min_dist_q: float,
    pool_mult: int,
    metric: str,
    exclude: Optional[set[int]] = None,
    reference_emb: Optional[np.ndarray] = None,
    min_dist_type: Literal['candidate','pop', 'min_both','knn_density'] = 'candidate',
    promising_q: float = 0.10,
    archive_knn_k: int = 1,
) -> np.ndarray:
    k = max(0, int(k))
    if k == 0:
        return np.zeros((0,), dtype=np.int64)
    exclude = exclude or set()
    if emb is None:
        return _topk_excluding(score, k, exclude)

    if min_dist_type == 'knn_promisingFar':
        return _knn_promising_farthest_select(
            score=score,
            emb=emb,
            arc_emb=arc_emb,
            k=k,
            promising_q=promising_q,
            archive_knn_k=archive_knn_k,
            metric=metric,
            exclude=exclude,
        )

    order = np.argsort(score)
    pool_size = max(k, int(k) * max(1, int(pool_mult)))
    if min_dist_type == 'pop':
        min_dist = _estimate_latent_min_dist_pop(pop_emb, min_dist_q, metric)
    elif min_dist_type == 'min_both':
        min_dist_cand = _estimate_latent_min_dist(emb, order, pool_size, min_dist_q, metric)
        min_dist_pop = _estimate_latent_min_dist_pop(pop_emb, min_dist_q, metric)
        min_dist = min(min_dist_cand, min_dist_pop)
    elif min_dist_type == 'candidate':
        min_dist = _estimate_latent_min_dist(emb, order, pool_size, min_dist_q, metric)
    elif min_dist_type == 'knn_density':
        min_dist = _estimate_latent_min_dist_knn_density(arc_emb, min_dist_q, metric, archive_knn_k)
    else:
        raise ValueError(f"Unknown min_dist_type: {min_dist_type}")

    selected: List[int] = []
    for idx in order:
        ii = int(idx)
        if ii in exclude:
            continue

        if min_dist_type == 'pop' or min_dist_type == 'min_both' or min_dist_type == 'candidate':
            dref = _embedding_distance_to_reference(emb, ii, reference_emb, metric)
            if len(selected) == 0:
                if dref >= min_dist:
                    selected.append(ii)
            else:
                dmin = min(dref, _embedding_distance_to_selected(emb, ii, selected, metric))
                if dmin >= min_dist:
                    selected.append(ii)
            if len(selected) >= k:
                break
        elif min_dist_type == "knn_density":
            ref_emb = _combine_archive_and_selected_embeddings(emb, selected, arc_emb)
            dmin = _embedding_knn_mean_dist_to_reference(emb, ii, ref_emb, metric, archive_knn_k)
            if dmin >= min_dist:
                selected.append(ii)
            if len(selected) >= k:
                break

    if len(selected) < k:
        selected_set = set(selected) | exclude
        for idx in order:
            ii = int(idx)
            if ii in selected_set:
                continue
            selected.append(ii)
            selected_set.add(ii)
            if len(selected) >= k:
                break
    return np.asarray(selected[:k], dtype=np.int64)


def _latent_cluster_select(
    score: np.ndarray,
    emb: Optional[np.ndarray],
    k: int,
    pool_mult: int,
    metric: str,
    seed: int,
    exclude: Optional[set[int]] = None,
) -> np.ndarray:
    k = max(0, int(k))
    if k == 0:
        return np.zeros((0,), dtype=np.int64)
    exclude = exclude or set()
    if emb is None or KMeans is None:
        return _latent_nms_select(score, emb, k, 0.10, pool_mult, metric, exclude)

    order = np.asarray([int(i) for i in np.argsort(score) if int(i) not in exclude], dtype=np.int64)
    if order.size <= k:
        return order[:k]

    pool_size = max(k, k * max(1, int(pool_mult)))
    pool = order[: min(pool_size, order.size)]
    z = np.asarray(emb[pool], dtype=np.float64)
    if metric == "cosine":
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)

    n_clusters = min(k, z.shape[0])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=int(seed))
    labels = km.fit_predict(z)

    selected: List[int] = []
    for c in range(n_clusters):
        member_pos = np.where(labels == c)[0]
        if member_pos.size == 0:
            continue
        member_idx = pool[member_pos]
        best = int(member_idx[np.argmin(score[member_idx])])
        selected.append(best)

    selected = sorted(set(selected), key=lambda i: float(score[i]))
    if len(selected) < k:
        selected_set = set(selected) | exclude
        for idx in order:
            ii = int(idx)
            if ii in selected_set:
                continue
            selected.append(ii)
            selected_set.add(ii)
            if len(selected) >= k:
                break
    return np.asarray(selected[:k], dtype=np.int64)


def _select_global_indices(
    score: np.ndarray,
    emb: Optional[np.ndarray],
    pop_emb: Optional[np.ndarray],
    arc_emb: Optional[np.ndarray],
    k: int,
    cfg: Any,
    seed: int,
    exclude: Optional[set[int]] = None,
    reference_emb: Optional[np.ndarray] = None,
) -> np.ndarray:
    mode = str(getattr(cfg, "global_selection", "topk"))
    if mode == "latent_nms":
        return _latent_nms_select(
            score=score,
            pop_emb=pop_emb,
            arc_emb=arc_emb,
            emb=emb,
            k=k,
            min_dist_q=float(getattr(cfg, "latent_min_dist_q", 0.10)),
            pool_mult=int(getattr(cfg, "latent_pool_mult", 20)),
            metric=str(getattr(cfg, "latent_metric", "cosine")),
            exclude=exclude,
            reference_emb=reference_emb,
            min_dist_type=cfg.min_dist_type,
            promising_q=float(getattr(cfg, "latent_promising_q", 0.10)),
            archive_knn_k=int(getattr(cfg, "latent_archive_knn_k", getattr(cfg, "archive_knn_k", 1))),
        )
    if mode == "latent_cluster":
        return _latent_cluster_select(
            score=score,
            emb=emb,
            k=k,
            pool_mult=int(getattr(cfg, "latent_pool_mult", 20)),
            metric=str(getattr(cfg, "latent_metric", "cosine")),
            seed=seed,
            exclude=exclude,
        )
    return _topk_excluding(score, k, exclude)


def _select_uncertainty_indices(
    mu: np.ndarray,
    sigma: np.ndarray,
    emb: Optional[np.ndarray],
    k: int,
    cfg: Any,
    already_selected: Optional[set[int]] = None,
) -> np.ndarray:
    k = max(0, int(k))
    if k == 0:
        return np.zeros((0,), dtype=np.int64)
    already_selected = already_selected or set()
    if mu.size == 0:
        return np.zeros((0,), dtype=np.int64)

    q = float(getattr(cfg, "uncertainty_mu_quantile", 0.50))
    q = min(1.0, max(0.0, q))
    reasonable = mu <= np.quantile(mu, q)
    pool = [int(i) for i in np.where(reasonable)[0] if int(i) not in already_selected]
    if len(pool) == 0:
        pool = [int(i) for i in range(mu.shape[0]) if int(i) not in already_selected]

    # Larger sigma is better, so use negative sigma with the same minimization selectors.
    score = -np.asarray(sigma, dtype=np.float64)
    masked_score = np.full_like(score, np.inf, dtype=np.float64)
    masked_score[np.asarray(pool, dtype=np.int64)] = score[np.asarray(pool, dtype=np.int64)]

    return _latent_nms_select(
        score=masked_score,
        emb=emb,
        k=k,
        min_dist_q=float(getattr(cfg, "latent_min_dist_q", 0.10)),
        pool_mult=int(getattr(cfg, "latent_pool_mult", 20)),
        metric=str(getattr(cfg, "latent_metric", "cosine")),
        exclude=already_selected,
    )


def _filter_unseen_unique(
    dec: np.ndarray,
    score: np.ndarray,
    sigma: np.ndarray,
    arc_set: set[Tuple[int, ...]],
    chosen_set: Optional[set[Tuple[int, ...]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen_set = chosen_set or set()
    keep: List[int] = []
    seen_local: set[Tuple[int, ...]] = set()
    for i in range(dec.shape[0]):
        key = tuple(int(v) for v in dec[i])
        if key in arc_set or key in chosen_set or key in seen_local:
            continue
        seen_local.add(key)
        keep.append(i)
    if not keep:
        return dec[:0].copy(), score[:0].copy(), sigma[:0].copy()
    idx = np.asarray(keep, dtype=np.int64)
    return dec[idx].copy(), score[idx].copy(), sigma[idx].copy()


def _first_index_by_perm(dec: np.ndarray) -> Dict[Tuple[int, ...], int]:
    out: Dict[Tuple[int, ...], int] = {}
    for i in range(dec.shape[0]):
        key = tuple(int(v) for v in dec[i])
        if key not in out:
            out[key] = int(i)
    return out


def _take_optional_embedding(
    emb: Optional[np.ndarray],
    idx: np.ndarray,
) -> Optional[np.ndarray]:
    if emb is None:
        return None
    if idx.size == 0:
        return np.zeros((0, int(emb.shape[1])), dtype=np.float32)
    return np.asarray(emb[idx], dtype=np.float32).copy()


def _concat_optional_embeddings(
    blocks: Sequence[Optional[np.ndarray]],
    row_counts: Sequence[int],
) -> Optional[np.ndarray]:
    valid = [b for b in blocks if b is not None]
    if len(valid) == 0:
        return None
    dim = int(valid[0].shape[1])
    total = int(sum(row_counts))
    out = np.full((total, dim), np.nan, dtype=np.float32)
    pos = 0
    for block, rows in zip(blocks, row_counts):
        rows = int(rows)
        if block is not None and block.ndim == 2 and block.shape[0] == rows and block.shape[1] == dim:
            out[pos:pos + rows] = np.asarray(block, dtype=np.float32)
        pos += rows
    return out


def _stable_union_indices(primary: np.ndarray, extra: Sequence[int]) -> np.ndarray:
    seen: set[int] = set()
    out: List[int] = []
    for i in primary.tolist() + [int(v) for v in extra]:
        ii = int(i)
        if ii in seen:
            continue
        seen.add(ii)
        out.append(ii)
    return np.asarray(out, dtype=np.int64)


def _unique_rows_stable(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    seen = set()
    keep_idx: List[int] = []
    for i in range(x.shape[0]):
        key = tuple(int(v) for v in x[i])
        if key in seen:
            continue
        seen.add(key)
        keep_idx.append(i)
    keep = np.asarray(keep_idx, dtype=np.int64)
    return x[keep], keep


def popupdate_elanddiv(
    arc_perm: np.ndarray,
    arc_obj: np.ndarray,
    elite_rate: float,
    pop_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    PopUpdate_ElandDiv.m equivalent:
    elite = idx(1:elr*Popsize), other = datasample(idx(elr*Popsize+1:end), round((1-elr)*Popsize))
    """
    pop_size = max(1, int(pop_size))
    idx = np.argsort(arc_obj, axis=0)

    elite_num = int(np.floor(float(elite_rate) * pop_size))
    elite_num = max(1, min(elite_num, pop_size))
    other_num = int(np.round((1.0 - float(elite_rate)) * pop_size))
    other_num = max(0, other_num)

    elite_idx = idx[:elite_num]
    rest_idx = idx[elite_num:]

    if other_num > 0:
        if rest_idx.size == 0:
            other = rng.choice(idx, size=other_num, replace=True)
        else:
            other = rng.choice(rest_idx, size=other_num, replace=True)
        sel = np.concatenate([elite_idx, other])
    else:
        sel = elite_idx

    if sel.size > pop_size:
        sel = sel[:pop_size]
    elif sel.size < pop_size:
        pad = rng.choice(idx, size=(pop_size - sel.size), replace=True)
        sel = np.concatenate([sel, pad])
    return arc_perm[sel.astype(np.int64)]


@dataclass
class Latent2WaysConfig:
    # MATLAB defaults: K=5, lambda=10, delta=0.2, t=100, r=10
    K: int = 1
    lam: int = 10
    delta: float = 0.2
    n_estimators: int = 100
    r: int = 0 #! i'd rather it be 0 for clearer methodology, but 10 is the default in GBDTMA.m
    fit_per: int = 10

    # PlatEMO default population size is often 100.
    pop_size: int = 100

    # sklearn options (close to LSBoost behavior)
    learning_rate: float = 0.1 #matlab縺縺ｨ1縺後ョ繝輔か
    max_depth: int = 3
    subsample: float = 1.0
    min_samples_leaf: int = 1
    surrogate_kind: Literal['gbdt','rf','gcn','gat','sage','gin','jkgcn','permformer','rand','oracle'] = 'gbdt'
    gnn_epochs: int = 200
    batch_size: int = 256
    acq: Literal['mean','lcb'] = 'mean'
    kappa: float = 2.0
    min_dist_type: Literal['candidate','pop', 'min_both','knn_density','knn_promisingFar'] = 'candidate'
    latent_promising_q: float = 0.10
    archive_knn_k: int = 5

    # Candidate-budget allocation. K is the total number of true evaluations per outer loop.
    # global_k is computed as K - incumbent_k - uncertainty_k.
    incumbent_k: int = 1
    uncertainty_k: int = 0

    # Global candidate selection.
    # topk: original behavior.
    # latent_nms: score-first greedy selection while avoiding near-duplicate graph embeddings.
    # latent_cluster: cluster high-score embeddings and select the best in each cluster.
    global_selection: Literal['topk','latent_nms','latent_cluster'] = 'latent_nms'
    latent_metric: Literal['cosine','euclidean'] = 'cosine'
    latent_pool_mult: int = 20
    latent_min_dist_q: float = 0.20

    # Incumbent-centered exploitation.
    local_kappa: float = 0.0
    incumbent_pool_size: int = 100
    incumbent_max_depth: int = 1
    incumbent_sigma: int = 1

    mu_plus_lambda: bool = True

    # Uncertainty-only slot. Candidates with too poor predicted mean are filtered out first.
    uncertainty_mu_quantile: float = 0.50

    problem: Literal['TSP','ATSP','QAP','LOP','PFSP'] = 'TSP'


def latent_2ways(
    dim: int,
    true_eval_fn: Callable[[Perm], float],
    budget: int,
    cfg: Optional[Latent2WaysConfig] = None,
    seed: int = 0,
    internal_gen: Optional[int] = 1,
) -> Tuple[Perm, float, List[Perm], List[float], List[float], Dict[str, Any]]:
    """
    Python implementation aligned to GBDTMA.m flow (minimization).
    """


    if cfg is None:
        cfg = Latent2WaysConfig()

    if GradientBoostingRegressor is None and cfg.surrogate_kind == "gbdt":
        raise ImportError("scikit-learn is required for GBDTMA (GradientBoostingRegressor).")

    rng = np.random.default_rng(seed)
    n = int(dim)
    pop_size = max(1, int(cfg.pop_size))
    budget = max(1, int(budget))

    pop_perm = np.vstack([rng.permutation(n) for _ in range(pop_size)]).astype(np.int32)

    arc_perm: List[Perm] = []
    arc_obj: List[float] = []
    arc_set = set()

    eval_perms: List[Perm] = []
    eval_vals: List[float] = []
    history_best: List[float] = []

    best_val = float("inf")
    best_perm: Perm = pop_perm[0].tolist()

    if cfg.fit_per % cfg.K != 0:
        raise ValueError(f"Select A correct Fit_rate {cfg.fit_per}, and K {cfg.K}")

    nfev = 0
    elapsed_time_log: List[float] = []
    step_time_log: List[float] = []
    eval_count_log: List[int] = []
    start_time = time.perf_counter()
    last_eval_time = start_time
    for i in range(pop_size):
        if nfev >= budget:
            break
        p = pop_perm[i].tolist()
        v = float(true_eval_fn(p))
        nfev += 1
        now_time = time.perf_counter()
        eval_count_log.append(int(nfev))
        elapsed_time_log.append(float(now_time - start_time))
        step_time_log.append(float(now_time - last_eval_time))
        last_eval_time = now_time

        arc_perm.append(p)
        arc_obj.append(v)
        arc_set.add(tuple(p))
        eval_perms.append(p)
        eval_vals.append(v)

        if v < best_val:
            best_val = v
            best_perm = p
        history_best.append(best_val)

    cand_log: List[np.ndarray] = []
    pred_log: List[np.ndarray] = []
    sigma_log: List[np.ndarray] = []
    label_log: List[np.ndarray] = []
    source_log: List[np.ndarray] = []

    # Medium-weight analysis logs. These do not affect candidate generation or selection.
    # They store (i) a compact top global pool for latent-space analysis and
    # (ii) the actually evaluated candidates with mu/sigma/pred/embedding/source.
    mu_log: List[np.ndarray] = []
    emb_log: List[Optional[np.ndarray]] = []
    selected_mask_log: List[np.ndarray] = []
    selected_order_log: List[np.ndarray] = []
    selected_eval_source_log: List[np.ndarray] = []

    medium_top_pool_cand_log: List[np.ndarray] = []
    medium_top_pool_mu_log: List[np.ndarray] = []
    medium_top_pool_sigma_log: List[np.ndarray] = []
    medium_top_pool_pred_log: List[np.ndarray] = []
    medium_top_pool_emb_log: List[Optional[np.ndarray]] = []
    medium_top_pool_root_selected_log: List[np.ndarray] = []
    medium_top_pool_root_source_log: List[np.ndarray] = []

    medium_selected_cand_log: List[np.ndarray] = []
    medium_selected_mu_log: List[np.ndarray] = []
    medium_selected_sigma_log: List[np.ndarray] = []
    medium_selected_pred_log: List[np.ndarray] = []
    medium_selected_emb_log: List[Optional[np.ndarray]] = []
    medium_selected_label_log: List[np.ndarray] = []
    medium_selected_source_log: List[np.ndarray] = []

    detail_global_cand_log: List[np.ndarray] = []
    detail_global_mu_log: List[np.ndarray] = []
    detail_global_sigma_log: List[np.ndarray] = []
    detail_global_pred_log: List[np.ndarray] = []
    detail_global_label_log: List[np.ndarray] = []

    detail_local_cand_log: List[np.ndarray] = []
    detail_local_mu_log: List[np.ndarray] = []
    detail_local_sigma_log: List[np.ndarray] = []
    detail_local_pred_log: List[np.ndarray] = []
    detail_local_label_log: List[np.ndarray] = []
    detail_local_source_log: List[np.ndarray] = []
    incumbent_internal_log: List[Dict[str, Any]] = []
    incumbent_truth_cache: Dict[Tuple[int, ...], float] = {
        tuple(int(v) for v in p): float(v)
        for p, v in zip(arc_perm, arc_obj)
    }

    global_score_order_log: List[np.ndarray] = []
    global_score_top_idx_log: List[np.ndarray] = []
    global_selected_idx_log: List[np.ndarray] = []
    global_selected_score_rank_log: List[np.ndarray] = []
    global_selected_rank_delta_log: List[np.ndarray] = []
    global_selected_dist_to_incumbent_log: List[np.ndarray] = []
    global_selected_dist_to_prev_log: List[np.ndarray] = []
    global_selected_dist_to_archive_log: List[np.ndarray] = []
    global_selected_gate_dist_log: List[np.ndarray] = []
    global_min_dist_log: List[np.ndarray] = []

    archive_perm_log: List[np.ndarray] = []
    archive_fx_log: List[np.ndarray] = []
    archive_mu_log: List[np.ndarray] = []
    archive_sigma_log: List[np.ndarray] = []
    archive_pred_log: List[np.ndarray] = []
    archive_emb_log: List[Optional[np.ndarray]] = []
    cand_true_log: List[np.ndarray] = []

    # if cfg.surrogate_kind == "gbdt":
    #     mdl = GradientBoostingRegressor(
    #         n_estimators=int(cfg.n_estimators),
    #         learning_rate=float(cfg.learning_rate),
    #         max_depth=int(cfg.max_depth),
    #         subsample=float(cfg.subsample),
    #         min_samples_leaf=int(cfg.min_samples_leaf),
    #         random_state=int(seed),
    #     )
    # else:
    #     p = str(cfg.problem).upper()
    #     if p == "TSP":
    #         make_cycle, undirected = True, True
    #         every_connect, top_k = False, dim
    #         each_connect = False
    #     elif p == "ATSP":
    #         make_cycle, undirected = True, False
    #         every_connect, top_k = False, dim
    #         each_connect = False
    #     elif p == "QAP":
    #         make_cycle, undirected = False, False
    #         every_connect, top_k = False, dim
    #         each_connect = True
    #     elif p in ("LOP", "PFSP"):
    #         make_cycle, undirected = False, False
    #         every_connect, top_k = True, dim
    #         each_connect = False
    #     else:
    #         raise ValueError(f"Unsupported problem type: {cfg.problem}")
    #     mdl = SurrogateWrapper(
    #         kind=cfg.surrogate_kind,
    #         dim=dim,
    #         epochs=cfg.gnn_epochs,
    #         batch_size=cfg.batch_size,
    #         make_cycle=make_cycle,
    #         undirected=undirected,
    #         every_connect=every_connect,
    #         each_connect=each_connect,
    #         top_k=top_k,
    #         seed=seed,
    #     )

    p = str(cfg.problem).upper()
    if p == "TSP":
        make_cycle, undirected = False, False
        every_connect, top_k = False, dim
        each_connect = False
    elif p == "ATSP":
        make_cycle, undirected = False, False
        every_connect, top_k = False, dim
        each_connect = False
    elif p == "QAP":
        make_cycle, undirected = False, False
        every_connect, top_k = False, dim
        each_connect = False
    elif p in ("LOP", "PFSP"):
        make_cycle, undirected = False, False
        every_connect, top_k = False, dim
        each_connect = False
    else:
        raise ValueError(f"Unsupported problem type: {cfg.problem}")

    mdl = SurrogateWrapper(
        kind=cfg.surrogate_kind,
        dim=dim,
        epochs=cfg.gnn_epochs,
        batch_size=cfg.batch_size,
        make_cycle=make_cycle,
        undirected=undirected,
        every_connect=every_connect,
        each_connect=each_connect,
        top_k=top_k,
        seed=seed,
        true_eval_fn=true_eval_fn,
    )

    with tqdm(total=budget, desc="True evaluations") as pbar:
        pbar.update(nfev)
        while nfev < budget:
            x_train = np.asarray(arc_perm, dtype=np.int32)
            y_train = np.asarray(arc_obj, dtype=np.float64)
            if cfg.surrogate_kind == "gbdt":
                # mdl.fit(x_train, y_train)
                mdl.fit(x_train.tolist(), y_train.tolist())
            else:
                if (nfev - 100) % cfg.fit_per == 0:
                    mdl.fit(x_train.tolist(), y_train.tolist())

            pop_perm_np = np.asarray(pop_perm, dtype=np.int32)

            total_k = max(1, int(cfg.K))
            incumbent_k = max(0, min(int(cfg.incumbent_k), total_k))
            uncertainty_k = max(0, min(int(cfg.uncertainty_k), total_k - incumbent_k))
            global_k = max(0, total_k - incumbent_k - uncertainty_k)
            if global_k + incumbent_k + uncertainty_k <= 0:
                global_k = 1

            acq_kind = "mean" if cfg.surrogate_kind == "gbdt" else cfg.acq

            # GA global search. The last internal generation is used as the global candidate pool,
            # matching the original code structure. Candidate selection is now diversity-aware.
            for _ in range(internal_gen or 1):
                off_list: List[Perm] = []
                for _ in range(int(cfg.lam)):
                    order = rng.permutation(pop_perm_np.shape[0])
                    shuffled = pop_perm_np[order]
                    off = _operator_ga_perm(shuffled, rng)
                    off_list.extend(off.tolist())
                off_dec = np.asarray(off_list, dtype=np.int32)

                # unique(OffDec,"rows",'stable')
                off_dec, _ = _unique_rows_stable(off_dec)
                unseen_mask = np.asarray([
                    tuple(int(v) for v in row) not in arc_set
                    for row in off_dec
                ])

                off_dec = off_dec[unseen_mask]

                mu_all, sigma_all, emb_all = _predict_with_optional_embedding(mdl, off_dec.tolist())
                off_pred = _compute_acquisition(mu_all, sigma_all, acq_kind, float(cfg.kappa))

                selected_idx: List[int] = []
                selected_src: List[str] = []
                selected_idx_set: set[int] = set()
                global_idx_list: List[int] = []
                global_diag: Dict[str, np.ndarray] = {}

                if global_k > 0:
                    _, _, incumbent_ref_emb = _predict_with_optional_embedding(mdl, [best_perm])
                    if cfg.min_dist_type == 'pop' or cfg.min_dist_type == 'min_both':
                        _, _, pop_emb = _predict_with_optional_embedding(mdl, pop_perm_np.tolist())
                    elif cfg.min_dist_type == 'knn_density' or cfg.min_dist_type == 'knn_promisingFar':
                        _, _, arc_emb = _predict_with_optional_embedding(mdl, np.asarray(arc_perm, dtype=np.int32).tolist())

                    global_idx = _select_global_indices(
                        score=off_pred,
                        pop_emb=pop_emb if cfg.min_dist_type in ['pop', 'min_both'] else None,
                        arc_emb=arc_emb if cfg.min_dist_type in ['knn_density', 'knn_promisingFar'] else None,
                        emb=emb_all,
                        k=min(global_k, off_dec.shape[0]),
                        cfg=cfg,
                        seed=seed,
                        exclude=None,
                        reference_emb=incumbent_ref_emb,
                    )
                    global_idx_list = [int(i) for i in global_idx]
                    global_diag = _global_selection_diagnostics(
                        score=off_pred,
                        emb=emb_all,
                        selected_idx=global_idx_list,
                        reference_emb=incumbent_ref_emb,
                        pop_emb=pop_emb if cfg.min_dist_type in ["pop", "min_both"] else None,
                        arc_emb=arc_emb if cfg.min_dist_type in ["knn_density", "knn_promisingFar"] else None,
                        cfg=cfg,
                    )
                    selected_idx.extend(global_idx_list)
                    selected_src.extend(["global"] * len(global_idx_list))
                    selected_idx_set.update(global_idx_list)

                if uncertainty_k > 0:
                    uncertainty_idx = _select_uncertainty_indices(
                        mu=mu_all,
                        sigma=sigma_all,
                        emb=emb_all,
                        k=min(uncertainty_k, max(0, off_dec.shape[0] - len(selected_idx_set))),
                        cfg=cfg,
                        already_selected=selected_idx_set,
                    )
                    uncertainty_idx_list = [int(i) for i in uncertainty_idx]
                    selected_idx.extend(uncertainty_idx_list)
                    selected_src.extend(["uncertainty"] * len(uncertainty_idx_list))
                    selected_idx_set.update(uncertainty_idx_list)

                # If diversity/uncertainty filters return too few candidates, fill by acquisition score.
                if len(selected_idx) < global_k + uncertainty_k:
                    fill_idx = _topk_excluding(
                        score=off_pred,
                        k=(global_k + uncertainty_k) - len(selected_idx),
                        exclude=selected_idx_set,
                    )
                    fill_idx_list = [int(i) for i in fill_idx]
                    selected_idx.extend(fill_idx_list)
                    selected_src.extend(["fallback_root"] * len(fill_idx_list))
                    selected_idx_set.update(fill_idx_list)

                root_dec = off_dec[np.asarray(selected_idx, dtype=np.int64)] if selected_idx else np.zeros((0, n), dtype=np.int32)
                root_source = selected_src

                # Medium log: compact global candidate pool.
                # Use score-top pool, but always include root-selected candidates even if
                # latent diversity selected them outside the top-score slice.
                medium_pool_size = min(
                    off_dec.shape[0],
                    max(total_k, total_k * max(1, int(getattr(cfg, "latent_pool_mult", 20)))),
                )
                medium_top_idx = _stable_union_indices(
                    np.argsort(off_pred)[:medium_pool_size],
                    selected_idx,
                )
                medium_root_source_by_idx = {int(idx): str(src) for idx, src in zip(selected_idx, selected_src)}
                if not global_diag:
                    global_diag = _global_selection_diagnostics(
                        score=off_pred,
                        emb=emb_all,
                        selected_idx=[],
                        reference_emb=None,
                        pop_emb=None,
                        arc_emb=None,
                        cfg=cfg,
                    )

            new_dec_list: List[Perm] = []
            new_dec_source: List[str] = []
            chosen_set: set[Tuple[int, ...]] = set()

            local_cand_blocks: List[np.ndarray] = []
            local_mu_blocks: List[np.ndarray] = []
            local_pred_blocks: List[np.ndarray] = []
            local_sigma_blocks: List[np.ndarray] = []
            local_emb_blocks: List[Optional[np.ndarray]] = []
            local_source_blocks: List[np.ndarray] = []

            def _append_best_from_pool(
                dec_pool: np.ndarray,
                score_pool: np.ndarray,
                sigma_pool: np.ndarray,
                source: str,
                pick_num: int = 1,
                mu_pool: Optional[np.ndarray] = None,
                emb_pool: Optional[np.ndarray] = None,
            ) -> None:
                nonlocal new_dec_list, new_dec_source, chosen_set
                if dec_pool.shape[0] == 0 or pick_num <= 0:
                    return
                dec_f, score_f, sigma_f = _filter_unseen_unique(
                    dec=dec_pool,
                    score=score_pool,
                    sigma=sigma_pool,
                    arc_set=arc_set,
                    chosen_set=chosen_set,
                )
                if dec_f.shape[0] == 0:
                    return

                # Logging-only alignment from the filtered pool back to original prediction arrays.
                key_to_pool_idx = _first_index_by_perm(dec_pool)
                pool_idx_f = np.asarray(
                    [key_to_pool_idx[tuple(int(v) for v in row)] for row in dec_f],
                    dtype=np.int64,
                )
                if mu_pool is None:
                    mu_f = np.full((dec_f.shape[0],), np.nan, dtype=np.float64)
                else:
                    mu_f = np.asarray(mu_pool, dtype=np.float64).reshape(-1)[pool_idx_f].copy()
                emb_f = _take_optional_embedding(emb_pool, pool_idx_f)

                local_cand_blocks.append(dec_f.copy())
                local_mu_blocks.append(mu_f.copy())
                local_pred_blocks.append(score_f.copy())
                local_sigma_blocks.append(sigma_f.copy())
                local_emb_blocks.append(emb_f)
                local_source_blocks.append(np.full((dec_f.shape[0],), source, dtype=object))

                for j in np.argsort(score_f)[:pick_num]:
                    chosen = dec_f[int(j)].tolist()
                    key = tuple(int(v) for v in chosen)
                    if key in arc_set or key in chosen_set:
                        continue
                    new_dec_list.append(chosen)
                    new_dec_source.append(source)
                    chosen_set.add(key)

            # Local refinement around global/uncertainty roots.
            for i in range(root_dec.shape[0]):
                now = root_dec[i].tolist()
                source = root_source[i] if i < len(root_source) else "global"
                nei_list: List[Perm] = [now]
                for _ in range(int(cfg.r)):
                    # nei_list.append(_swap_neighbor(now, rng))
                    nei_list.append(_insert_neighbor(now, rng))
                nei_dec = np.asarray(nei_list, dtype=np.int32)
                mu, sigma, emb = _predict_with_optional_embedding(mdl, nei_dec.tolist())

                if source == "uncertainty":
                    # In the uncertainty slot, prefer high-sigma candidates, but avoid candidates
                    # whose predicted mean is too poor compared with this local pool.
                    q = min(1.0, max(0.0, float(cfg.uncertainty_mu_quantile)))
                    reasonable = mu <= np.quantile(mu, q)
                    nei_score = -sigma.copy()
                    if np.any(reasonable):
                        nei_score[~reasonable] = np.inf
                else:
                    nei_score = _compute_acquisition(mu, sigma, acq_kind, float(cfg.kappa))

                _append_best_from_pool(
                    dec_pool=nei_dec,
                    score_pool=nei_score,
                    sigma_pool=sigma,
                    source=source,
                    pick_num=1,
                    mu_pool=mu,
                    emb_pool=emb,
                )

            # Incumbent-centered exploitation: directly search around the current true best.
            if incumbent_k > 0:
                local_acq_kind = "mean" if cfg.surrogate_kind == "gbdt" else cfg.acq
                incumbent_sigma = max(1, int(getattr(cfg, "incumbent_sigma", 1)))
                if incumbent_sigma == 1:
                    inc_dec = _make_incumbent_pool(
                        best_perm=best_perm,
                        rng=rng,
                        pool_size=int(cfg.incumbent_pool_size),
                        max_depth=int(cfg.incumbent_max_depth),
                    )
                    inc_dec, _ = _unique_rows_stable(inc_dec)
                    mu_inc, sigma_inc, emb_inc = _predict_with_optional_embedding(mdl, inc_dec.tolist())
                    inc_score = _compute_acquisition(mu_inc, sigma_inc, local_acq_kind, float(cfg.local_kappa))
                    inc_diag: Dict[str, Any] = {
                        "generation": int(len(cand_log)),
                        "sigma": int(incumbent_sigma),
                        "initial_pool_size": int(inc_dec.shape[0]),
                        "final_pool_size": int(inc_dec.shape[0]),
                        "updates_total": 0,
                        "updates_by_step": np.zeros((incumbent_sigma,), dtype=np.int64),
                        "known_child_skips_by_step": np.zeros((incumbent_sigma,), dtype=np.int64),
                        "duplicate_child_skips_by_step": np.zeros((incumbent_sigma,), dtype=np.int64),
                        "birth_step": np.ones((inc_dec.shape[0],), dtype=np.int64),
                        "update_count": np.zeros((inc_dec.shape[0],), dtype=np.int64),
                        "initial_score": inc_score.copy(),
                        "final_score": inc_score.copy(),
                        "comparison_step": np.zeros((0,), dtype=np.int64),
                        "comparison_parent_idx": np.zeros((0,), dtype=np.int64),
                        "comparison_parent_perm": np.zeros((0, n), dtype=np.int32),
                        "comparison_child_perm": np.zeros((0, n), dtype=np.int32),
                        "comparison_parent_score": np.zeros((0,), dtype=np.float64),
                        "comparison_child_score": np.zeros((0,), dtype=np.float64),
                        "comparison_surrogate_update": np.zeros((0,), dtype=bool),
                    }
                else:
                    if cfg.mu_plus_lambda:
                        inc_dec, mu_inc, sigma_inc, inc_score, emb_inc, inc_diag = _incumbent_insert_drifting_search(
                            best_perm=best_perm,
                            best_val=best_val,
                            mdl=mdl,
                            rng=rng,
                            pool_size=int(cfg.incumbent_pool_size),
                            max_depth=int(cfg.incumbent_max_depth),
                            sigma_steps=incumbent_sigma,
                            local_acq_kind=local_acq_kind,
                            local_kappa=float(cfg.local_kappa),
                            arc_set=arc_set,
                            chosen_set=chosen_set,
                        )
                        inc_diag["generation"] = int(len(cand_log))
                    else:
                        inc_dec, mu_inc, sigma_inc, inc_score, emb_inc, inc_diag = _incumbent_insert_internal_search(
                            best_perm=best_perm,
                            mdl=mdl,
                            rng=rng,
                            pool_size=int(cfg.incumbent_pool_size),
                            max_depth=int(cfg.incumbent_max_depth),
                            sigma_steps=incumbent_sigma,
                            local_acq_kind=local_acq_kind,
                            local_kappa=float(cfg.local_kappa),
                            arc_set=arc_set,
                            chosen_set=chosen_set,
                        )
                        inc_diag["generation"] = int(len(cand_log))

                _append_incumbent_comparison_truth(
                    inc_diag=inc_diag,
                    true_eval_fn=true_eval_fn,
                    truth_cache=incumbent_truth_cache,
                )

                before_incumbent_len = len(new_dec_list)
                _append_best_from_pool(
                    dec_pool=inc_dec,
                    score_pool=inc_score,
                    sigma_pool=sigma_inc,
                    source="incumbent",
                    pick_num=incumbent_k,
                    mu_pool=mu_inc,
                    emb_pool=emb_inc,
                )
                key_to_inc_idx = _first_index_by_perm(inc_dec)
                selected_inc_idx: List[int] = []
                for chosen_perm, chosen_source in zip(
                    new_dec_list[before_incumbent_len:],
                    new_dec_source[before_incumbent_len:],
                ):
                    if chosen_source != "incumbent":
                        continue
                    key = tuple(int(v) for v in chosen_perm)
                    if key in key_to_inc_idx:
                        selected_inc_idx.append(int(key_to_inc_idx[key]))
                selected_inc_idx_arr = np.asarray(selected_inc_idx, dtype=np.int64)
                inc_diag["selected_pool_idx"] = selected_inc_idx_arr.copy()
                if selected_inc_idx_arr.size > 0:
                    inc_diag["selected_birth_step"] = np.asarray(inc_diag["birth_step"], dtype=np.int64)[selected_inc_idx_arr].copy()
                    inc_diag["selected_update_count"] = np.asarray(inc_diag["update_count"], dtype=np.int64)[selected_inc_idx_arr].copy()
                    inc_diag["selected_score"] = np.asarray(inc_score, dtype=np.float64)[selected_inc_idx_arr].copy()
                    rank = np.empty((inc_score.shape[0],), dtype=np.int64)
                    rank[np.argsort(inc_score)] = np.arange(inc_score.shape[0], dtype=np.int64)
                    inc_diag["selected_score_rank"] = rank[selected_inc_idx_arr].copy()
                else:
                    inc_diag["selected_birth_step"] = np.zeros((0,), dtype=np.int64)
                    inc_diag["selected_update_count"] = np.zeros((0,), dtype=np.int64)
                    inc_diag["selected_score"] = np.zeros((0,), dtype=np.float64)
                    inc_diag["selected_score_rank"] = np.zeros((0,), dtype=np.int64)
                incumbent_internal_log.append(inc_diag)

            # Final fallback: if all local filters were exhausted, evaluate best unseen global candidates.
            if len(new_dec_list) < total_k:
                dec_f, score_f, sigma_f = _filter_unseen_unique(
                    dec=off_dec,
                    score=off_pred,
                    sigma=sigma_all,
                    arc_set=arc_set,
                    chosen_set=chosen_set,
                )
                if dec_f.shape[0] > 0:
                    key_to_off_idx = _first_index_by_perm(off_dec)
                    off_idx_f = np.asarray(
                        [key_to_off_idx[tuple(int(v) for v in row)] for row in dec_f],
                        dtype=np.int64,
                    )
                    local_cand_blocks.append(dec_f.copy())
                    local_mu_blocks.append(np.asarray(mu_all, dtype=np.float64).reshape(-1)[off_idx_f].copy())
                    local_pred_blocks.append(score_f.copy())
                    local_sigma_blocks.append(sigma_f.copy())
                    local_emb_blocks.append(_take_optional_embedding(emb_all, off_idx_f))
                    local_source_blocks.append(np.full((dec_f.shape[0],), "fallback", dtype=object))
                    for j in np.argsort(score_f)[: total_k - len(new_dec_list)]:
                        chosen = dec_f[int(j)].tolist()
                        key = tuple(int(v) for v in chosen)
                        if key in arc_set or key in chosen_set:
                            continue
                        new_dec_list.append(chosen)
                        new_dec_source.append("fallback")
                        chosen_set.add(key)

            # Logs: raw global pool + all filtered local/slot pools.
            if len(local_cand_blocks) > 0:
                cand_local = np.vstack(local_cand_blocks).astype(np.int32)
                mu_local = np.concatenate(local_mu_blocks).astype(np.float64)
                pred_local = np.concatenate(local_pred_blocks).astype(np.float64)
                sigma_local = np.concatenate(local_sigma_blocks).astype(np.float64)
                source_local = np.concatenate(local_source_blocks).astype(object)
                emb_local = _concat_optional_embeddings(
                    local_emb_blocks,
                    [block.shape[0] for block in local_cand_blocks],
                )
            else:
                cand_local = np.zeros((0, n), dtype=np.int32)
                mu_local = np.zeros((0,), dtype=np.float64)
                pred_local = np.zeros((0,), dtype=np.float64)
                sigma_local = np.zeros((0,), dtype=np.float64)
                source_local = np.zeros((0,), dtype=object)
                emb_local = None

            cand_all = np.vstack([off_dec.astype(np.int32), cand_local])
            mu_all_log_arr = np.concatenate([mu_all.astype(np.float64), mu_local])
            pred_all = np.concatenate([off_pred.astype(np.float64), pred_local])
            sigma_all_log = np.concatenate([sigma_all.astype(np.float64), sigma_local])
            emb_all_log_arr = _concat_optional_embeddings(
                [None if emb_all is None else np.asarray(emb_all, dtype=np.float32), emb_local],
                [off_dec.shape[0], cand_local.shape[0]],
            )
            source_all = np.concatenate([
                np.full((off_dec.shape[0],), "global_pool", dtype=object),
                source_local,
            ])
            cand_index = {tuple(int(v) for v in cand_all[i]): i for i in range(cand_all.shape[0])}
            global_index = {tuple(int(v) for v in off_dec[i]): i for i in range(off_dec.shape[0])}
            local_index = {tuple(int(v) for v in cand_local[i]): i for i in range(cand_local.shape[0])}
            label_all = np.full((cand_all.shape[0],), np.nan, dtype=np.float64)
            label_global = np.full((off_dec.shape[0],), np.nan, dtype=np.float64)
            label_local = np.full((cand_local.shape[0],), np.nan, dtype=np.float64)
            selected_mask_all = np.zeros((cand_all.shape[0],), dtype=bool)
            selected_order_all = np.full((cand_all.shape[0],), -1, dtype=np.int64)
            selected_eval_source_all = np.full((cand_all.shape[0],), "", dtype=object)

            # true evaluation and archive update
            for eval_order, p in enumerate(new_dec_list):
                if nfev >= budget:
                    break
                key = tuple(int(v) for v in p)
                if key in arc_set:
                    continue
                v = float(true_eval_fn(p))
                nfev += 1
                now_time = time.perf_counter()
                eval_count_log.append(int(nfev))
                elapsed_time_log.append(float(now_time - start_time))
                step_time_log.append(float(now_time - last_eval_time))
                last_eval_time = now_time
                pbar.update(1)

                arc_perm.append(p)
                arc_obj.append(v)
                arc_set.add(key)
                incumbent_truth_cache[key] = v
                eval_perms.append(p)
                eval_vals.append(v)

                if v < best_val:
                    best_val = v
                    best_perm = p
                history_best.append(best_val)

                if key in cand_index:
                    ci = cand_index[key]
                    label_all[ci] = v
                    selected_mask_all[ci] = True
                    selected_order_all[ci] = int(eval_order)
                    if eval_order < len(new_dec_source):
                        selected_eval_source_all[ci] = str(new_dec_source[eval_order])
                if key in global_index:
                    label_global[global_index[key]] = v
                if key in local_index:
                    label_local[local_index[key]] = v

            known_true = dict(incumbent_truth_cache)
            for p, v in zip(arc_perm, arc_obj):
                known_true[tuple(int(vv) for vv in p)] = float(v)
            cand_true = np.empty((cand_all.shape[0],), dtype=np.float64)
            for ci, p in enumerate(cand_all):
                key = tuple(int(v) for v in p)
                if key not in known_true:
                    known_true[key] = float(true_eval_fn([int(v) for v in p]))
                    incumbent_truth_cache[key] = known_true[key]
                cand_true[ci] = known_true[key]

            arc_perm_snapshot = np.asarray(arc_perm, dtype=np.int32)
            arc_fx_snapshot = np.asarray(arc_obj, dtype=np.float64)
            arc_mu_snapshot, arc_sigma_snapshot, arc_emb_snapshot = _predict_with_optional_embedding(
                mdl,
                arc_perm_snapshot.tolist(),
            )
            arc_pred_snapshot = _compute_acquisition(
                arc_mu_snapshot,
                arc_sigma_snapshot,
                acq_kind,
                float(cfg.kappa),
            )

            cand_log.append(cand_all.copy())
            mu_log.append(mu_all_log_arr.copy())
            pred_log.append(pred_all.copy())
            sigma_log.append(sigma_all_log.copy())
            emb_log.append(None if emb_all_log_arr is None else emb_all_log_arr.copy())
            label_log.append(label_all.copy())
            source_log.append(source_all.copy())
            selected_mask_log.append(selected_mask_all.copy())
            selected_order_log.append(selected_order_all.copy())
            selected_eval_source_log.append(selected_eval_source_all.copy())
            cand_true_log.append(cand_true.copy())
            archive_perm_log.append(arc_perm_snapshot.copy())
            archive_fx_log.append(arc_fx_snapshot.copy())
            archive_mu_log.append(arc_mu_snapshot.astype(np.float64).copy())
            archive_sigma_log.append(arc_sigma_snapshot.astype(np.float64).copy())
            archive_pred_log.append(arc_pred_snapshot.astype(np.float64).copy())
            archive_emb_log.append(None if arc_emb_snapshot is None else arc_emb_snapshot.astype(np.float32).copy())

            detail_global_cand_log.append(off_dec.astype(np.int32).copy())
            detail_global_mu_log.append(mu_all.astype(np.float64).copy())
            detail_global_sigma_log.append(sigma_all.astype(np.float64).copy())
            detail_global_pred_log.append(off_pred.astype(np.float64).copy())
            detail_global_label_log.append(label_global.copy())

            detail_local_cand_log.append(cand_local.astype(np.int32).copy())
            detail_local_mu_log.append(mu_local.astype(np.float64).copy())
            detail_local_sigma_log.append(sigma_local.astype(np.float64).copy())
            detail_local_pred_log.append(pred_local.astype(np.float64).copy())
            detail_local_label_log.append(label_local.copy())
            detail_local_source_log.append(source_local.copy())

            global_score_order_log.append(global_diag["score_order"].copy())
            global_score_top_idx_log.append(global_diag["score_top_idx"].copy())
            global_selected_idx_log.append(global_diag["selected_idx"].copy())
            global_selected_score_rank_log.append(global_diag["selected_score_rank"].copy())
            global_selected_rank_delta_log.append(global_diag["selected_rank_delta"].copy())
            global_selected_dist_to_incumbent_log.append(global_diag["selected_dist_to_incumbent"].copy())
            global_selected_dist_to_prev_log.append(global_diag["selected_dist_to_prev_selected"].copy())
            global_selected_dist_to_archive_log.append(global_diag["selected_dist_to_archive"].copy())
            global_selected_gate_dist_log.append(global_diag["selected_gate_dist"].copy())
            global_min_dist_log.append(global_diag["min_dist"].copy())

            medium_top_pool_cand_log.append(off_dec[medium_top_idx].astype(np.int32).copy())
            medium_top_pool_mu_log.append(mu_all[medium_top_idx].astype(np.float64).copy())
            medium_top_pool_sigma_log.append(sigma_all[medium_top_idx].astype(np.float64).copy())
            medium_top_pool_pred_log.append(off_pred[medium_top_idx].astype(np.float64).copy())
            medium_top_pool_emb_log.append(_take_optional_embedding(emb_all, medium_top_idx))
            medium_top_pool_root_selected_log.append(
                np.asarray([int(i) in selected_idx_set for i in medium_top_idx], dtype=bool)
            )
            medium_top_pool_root_source_log.append(
                np.asarray([medium_root_source_by_idx.get(int(i), "") for i in medium_top_idx], dtype=object)
            )

            sel_idx = np.where(selected_mask_all)[0]
            medium_selected_cand_log.append(cand_all[sel_idx].astype(np.int32).copy())
            medium_selected_mu_log.append(mu_all_log_arr[sel_idx].astype(np.float64).copy())
            medium_selected_sigma_log.append(sigma_all_log[sel_idx].astype(np.float64).copy())
            medium_selected_pred_log.append(pred_all[sel_idx].astype(np.float64).copy())
            medium_selected_emb_log.append(
                None if emb_all_log_arr is None else emb_all_log_arr[sel_idx].astype(np.float32).copy()
            )
            medium_selected_label_log.append(label_all[sel_idx].astype(np.float64).copy())
            medium_selected_source_log.append(selected_eval_source_all[sel_idx].copy())

            # Population = PopUpdate_ElandDiv(Arc,delta,Problem.N)
            arc_perm_np = np.asarray(arc_perm, dtype=np.int32)
            arc_obj_np = np.asarray(arc_obj, dtype=np.float64)
            pop_perm = popupdate_elanddiv(
                arc_perm=arc_perm_np,
                arc_obj=arc_obj_np,
                elite_rate=float(cfg.delta),
                pop_size=pop_size,
                rng=rng,
            )
            pbar.set_postfix(best=float(best_val))

    logs: Dict[str, Any] = {
        "eval_count": np.asarray(eval_count_log, dtype=np.int64),
        "elapsed_time": np.asarray(elapsed_time_log, dtype=np.float64),
        "step_time": np.asarray(step_time_log, dtype=np.float64),
        "cand": cand_log,
        "mu": mu_log,
        "pred": pred_log,
        "sigma": sigma_log,
        "emb": emb_log,
        "label": label_log,
        "source": source_log,
        "selected_mask": selected_mask_log,
        "selected_order": selected_order_log,
        "selected_eval_source": selected_eval_source_log,
        "cand_true": cand_true_log,
        "archive_perm_by_gen": archive_perm_log,
        "archive_fx_by_gen": archive_fx_log,
        "archive_mu_by_gen": archive_mu_log,
        "archive_sigma_by_gen": archive_sigma_log,
        "archive_pred_by_gen": archive_pred_log,
        "archive_emb_by_gen": archive_emb_log,
        "incumbent_internal": incumbent_internal_log,
        "detail": {
            "cand_true": cand_true_log,
            "archive_perm": archive_perm_log,
            "archive_fx": archive_fx_log,
            "archive_mu": archive_mu_log,
            "archive_sigma": archive_sigma_log,
            "archive_pred": archive_pred_log,
            "archive_emb": archive_emb_log,
            "top_pool_cand": medium_top_pool_cand_log,
            "top_pool_mu": medium_top_pool_mu_log,
            "top_pool_sigma": medium_top_pool_sigma_log,
            "top_pool_pred": medium_top_pool_pred_log,
            "top_pool_emb": medium_top_pool_emb_log,
            "top_pool_root_selected": medium_top_pool_root_selected_log,
            "top_pool_root_source": medium_top_pool_root_source_log,
            "selected_cand": medium_selected_cand_log,
            "selected_mu": medium_selected_mu_log,
            "selected_sigma": medium_selected_sigma_log,
            "selected_pred": medium_selected_pred_log,
            "selected_emb": medium_selected_emb_log,
            "selected_label": medium_selected_label_log,
            "selected_source": medium_selected_source_log,
            "global_cand": detail_global_cand_log,
            "global_mu": detail_global_mu_log,
            "global_sigma": detail_global_sigma_log,
            "global_pred": detail_global_pred_log,
            "global_label": detail_global_label_log,
            "local_cand": detail_local_cand_log,
            "local_mu": detail_local_mu_log,
            "local_sigma": detail_local_sigma_log,
            "local_pred": detail_local_pred_log,
            "local_label": detail_local_label_log,
            "local_source": detail_local_source_log,
            "incumbent_internal": incumbent_internal_log,
            "global_score_order": global_score_order_log,
            "global_score_top_idx": global_score_top_idx_log,
            "global_selected_idx": global_selected_idx_log,
            "global_selected_score_rank": global_selected_score_rank_log,
            "global_selected_rank_delta": global_selected_rank_delta_log,
            "global_selected_dist_to_incumbent": global_selected_dist_to_incumbent_log,
            "global_selected_dist_to_prev": global_selected_dist_to_prev_log,
            "global_selected_dist_to_archive": global_selected_dist_to_archive_log,
            "global_selected_gate_dist": global_selected_gate_dist_log,
            "global_min_dist": global_min_dist_log,
        },
    }
    return best_perm, float(best_val), eval_perms, eval_vals, history_best, logs
