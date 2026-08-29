from __future__ import annotations

import random
from collections import deque
from typing import Callable, List, Optional, Tuple

import numpy as np

def s_beta(p: float, beta: float) -> float:
    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)
    return 1.0 - 1.0 / (1.0 + ((1.0 - p) / p) ** beta)


def _apply_insertion(perm: List[int], i: int, j: int) -> List[int]:
    out = perm.copy()
    item = out.pop(i)
    out.insert(j, item)
    return out


def _gen_candidates_insertion(n: int, distance: int) -> List[Tuple[int, int]]:
    candidates: List[Tuple[int, int]] = []
    for i in range(n):
        if i - distance >= 0:
            candidates.append((i, i - distance))
        if i + distance < n:
            candidates.append((i, i + distance))
    return candidates


def fat_rls(
    dim: int,
    true_eval_fn: Callable[[List[int]], float],
    budget: int,
    move_type: str = "insertion",
    seed: Optional[int] = None,
    beta: float = 1.2,
    tabu_len: Optional[int] = None,
    tabu_mode: str = "item",
    init_count: int = 100,
) -> Tuple[List[int], float, List[List[int]], List[float], List[float]]:
    if move_type != "insertion":
        raise ValueError("FAT-RLS only supports insertion moves.")
    if tabu_mode != "item":
        raise ValueError("FAT-RLS only supports item tabu.")
    if budget <= 0:
        raise ValueError("budget must be positive.")

    n = int(dim)
    tabu_len = n if tabu_len is None else int(tabu_len)
    init_evals = max(1, min(int(init_count), int(budget)))
    rng = np.random.default_rng(seed)
    if seed is not None:
        random.seed(seed)

    eval_perms: List[List[int]] = []
    eval_vals: List[float] = []
    history_best: List[float] = []
    best_val = float("inf")
    best_perm = list(range(n))

    for _ in range(init_evals):
        perm = rng.permutation(n).astype(np.int32).tolist()
        value = float(true_eval_fn(perm))
        eval_perms.append(perm)
        eval_vals.append(value)
        if value < best_val:
            best_val = value
            best_perm = perm.copy()
        history_best.append(best_val)

    current = best_perm.copy()
    current_value = best_val
    nfev = init_evals
    tabu_queue: deque[int] = deque(maxlen=tabu_len)
    d_ini = max(1, n // 2)

    while nfev < budget:
        schedule_budget = max(1, budget - init_evals)
        progress = max(0, nfev - init_evals) / float(schedule_budget)
        distance = int(round(1.0 + s_beta(progress, beta) * (d_ini - 1)))
        distance = max(1, min(distance, n - 1))
        candidates = [
            move for move in _gen_candidates_insertion(n, distance)
            if current[move[0]] not in tabu_queue
        ]
        if not candidates:
            candidates = _gen_candidates_insertion(n, distance)
        i, j = random.choice(candidates)
        candidate = _apply_insertion(current, i, j)
        value = float(true_eval_fn(candidate))
        nfev += 1
        eval_perms.append(candidate.copy())
        eval_vals.append(value)
        if value < current_value:
            current = candidate
            current_value = value
        if current_value < best_val:
            best_val = current_value
            best_perm = current.copy()
        history_best.append(best_val)
        tabu_queue.append(candidate[j])

    return best_perm, float(best_val), eval_perms, eval_vals, history_best
