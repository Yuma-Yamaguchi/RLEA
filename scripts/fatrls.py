# fat_rls.py (improved)
from __future__ import annotations

from typing import List, Tuple, Callable, Optional, Literal, Any
from collections import deque
import random

import numpy as np

MoveType = Literal["insertion", "swap", "invert"]


def s_beta(p: float, beta: float) -> float:
    """
    skewed S-shaped function s_beta(p)
        s_beta(p) = 1 - 1 / (1 + ((1-p)/p)^beta)
    p in (0, 1], beta >= 1
    """
    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)
    ratio = (1.0 - p) / p
    return 1.0 - 1.0 / (1.0 + ratio**beta)


def _normalize_pair(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _apply_insertion(perm: List[int], i: int, j: int) -> List[int]:
    """
    Move element at position i to position j (insertion).
    """
    sigma = perm.copy()
    item = sigma.pop(i)
    sigma.insert(j, item)
    return sigma


def _apply_swap(perm: List[int], i: int, j: int) -> List[int]:
    """
    Swap positions i and j.
    """
    sigma = perm.copy()
    sigma[i], sigma[j] = sigma[j], sigma[i]
    return sigma


def _apply_invert(perm: List[int], i: int, j: int) -> List[int]:
    """
    Reverse the segment between i and j inclusive (2-opt style inversion).
    """
    a, b = _normalize_pair(i, j)
    if a == b:
        return perm.copy()
    sigma = perm.copy()
    sigma[a : b + 1] = reversed(sigma[a : b + 1])
    return sigma


def _gen_candidates_insertion(n: int, d: int) -> List[Tuple[int, int]]:
    cands: List[Tuple[int, int]] = []
    for i in range(n):
        if i - d >= 0:
            cands.append((i, i - d))
        if i + d < n:
            cands.append((i, i + d))
    return cands


def _gen_candidates_swap(n: int, d: int) -> List[Tuple[int, int]]:
    # swap 繧・|i-j|=d 縺ｨ縺・≧縲瑚ｷ晞屬蛻ｶ蠕｡縲阪ｒ蜷梧ｧ倥↓菴ｿ縺・
    cands: List[Tuple[int, int]] = []
    for i in range(n):
        j1 = i - d
        j2 = i + d
        if j1 >= 0:
            cands.append(_normalize_pair(i, j1))
        if j2 < n:
            cands.append(_normalize_pair(i, j2))
    # 驥崎､・勁蜴ｻ
    cands = list(dict.fromkeys(cands))
    return cands


def _gen_candidates_invert(n: int, d: int) -> List[Tuple[int, int]]:
    # invert(2-opt) 縺ｯ蛹ｺ髢馴聞繧偵慧+1縲阪￥繧峨＞縺ｫ蝗ｺ螳壹＠縺ｦ蜍輔°縺呻ｼ育岼螳会ｼ・
    # 萓・ (i, i+d) 縺ｮ蜿崎ｻ｢
    cands: List[Tuple[int, int]] = []
    for i in range(n):
        j = i + d
        if j < n:
            cands.append((i, j))
    return cands


def fat_rls(
    dim: int,
    true_eval_fn: Callable[[List[int]], float],  # minimization
    budget: int,
    move_type: MoveType = "insertion",
    seed: Optional[int] = None,
    beta: float = 1.2,
    tabu_len: Optional[int] = None,
    tabu_mode: Literal["item", "move"] = "move",
    init_count: int = 1,
) -> Tuple[List[int], float, List[List[int]], List[float], List[float]]:
    """
    FAT-RLS (single trajectory) with configurable neighborhood move.

    Parameters
    ----------
    dim : int
        permutation length n
    true_eval_fn : callable
        f(perm) -> float (minimization)
    budget : int
        number of true evaluations (>=1)
    move_type : {"insertion","swap","invert"}
        neighborhood operator
    seed : int or None
        random seed
    beta : float
        shape parameter in intensity schedule
    tabu_len : int or None
        tabu queue length. default: n
    tabu_mode : {"item","move"}
        - "item": keep tabu by element ID (similar to your original; weakly operator-dependent)
        - "move": keep tabu by move signature (recommended when using swap/invert)
    init_count : int
        Number of random initial solutions to true-evaluate before FAT-RLS moves.

    Returns
    -------
    best_perm, best_val, eval_perms, eval_vals, history_best
    """
    if budget <= 0:
        raise ValueError("budget must be positive.")
    n = dim
    if tabu_len is None:
        tabu_len = n

    if seed is not None:
        random.seed(seed)

    # intensity schedule parameters
    d_ini = n // 2

    # --- initial solution(s) ---
    init_evals = max(1, min(int(init_count), int(budget)))
    eval_perms: List[List[int]] = []
    eval_vals: List[float] = []
    history_best: List[float] = []

    best_val = float("inf")
    best_perm: List[int] = list(range(n))

    if init_evals == 1:
        pi = list(range(n))
        random.shuffle(pi)
        f_pi = true_eval_fn(pi)
        best_perm = pi.copy()
        best_val = f_pi
        eval_perms.append(pi.copy())
        eval_vals.append(f_pi)
        history_best.append(best_val)
    else:
        rng = np.random.default_rng(seed)
        for _ in range(init_evals):
            p0 = rng.permutation(n).astype(np.int32).tolist()
            v0 = float(true_eval_fn(p0))
            eval_perms.append(p0)
            eval_vals.append(v0)
            if v0 < best_val:
                best_val = v0
                best_perm = p0.copy()
            history_best.append(best_val)
        pi = best_perm.copy()
        f_pi = best_val

    nfev = init_evals

    tabu_q: deque[Any] = deque(maxlen=tabu_len)

    # operator config
    if move_type == "insertion":
        gen_cands = _gen_candidates_insertion
        apply_move = _apply_insertion
    elif move_type == "swap":
        gen_cands = _gen_candidates_swap
        apply_move = _apply_swap
    elif move_type == "invert":
        gen_cands = _gen_candidates_invert
        apply_move = _apply_invert
    else:
        raise ValueError(f"Unknown move_type: {move_type}")

    while nfev < budget:
        # 1) compute intensity d. For init_count > 1, initial evaluations
        # count against budget but the schedule starts from zero after them.
        if init_evals <= 1:
            p = nfev / float(budget)
        else:
            schedule_budget = max(1, int(budget) - init_evals)
            schedule_nfev = max(0, nfev - init_evals)
            p = schedule_nfev / float(schedule_budget)
        d = int(round(1.0 + s_beta(p, beta) * (d_ini - 1)))
        d = max(1, min(d, n - 1))

        # 2) generate candidates
        candidates = gen_cands(n, d)

        # 3) tabu filtering
        filtered: List[Tuple[int, int]] = []

        if tabu_mode == "item":
            # old style: forbid moves involving tabu "item ids"
            # for insertion: we can mimic "moved item" check by sigma[i]
            # for swap/invert: we forbid if either endpoint items are tabu (simple heuristic)
            for (i, j) in candidates:
                if move_type == "insertion":
                    moved_item = pi[i]
                    if moved_item in tabu_q:
                        continue
                else:
                    a, b = _normalize_pair(i, j)
                    if pi[a] in tabu_q or pi[b] in tabu_q:
                        continue
                filtered.append((i, j))

        elif tabu_mode == "move":
            # recommended: forbid repeating the same "move signature"
            # - insertion: (item_id, target_pos)
            # - swap: (pos_a, pos_b) (position-based; cheap and stable)
            # - invert: (l, r) segment
            for (i, j) in candidates:
                if move_type == "insertion":
                    sig = ("ins", pi[i], j)
                elif move_type == "swap":
                    a, b = _normalize_pair(i, j)
                    sig = ("swp", a, b)
                else:  # invert
                    a, b = _normalize_pair(i, j)
                    sig = ("inv", a, b)

                if sig in tabu_q:
                    continue
                filtered.append((i, j))
        else:
            raise ValueError(f"Unknown tabu_mode: {tabu_mode}")

        if not filtered:
            # tabu ignored fallback
            filtered = candidates

        i, j = random.choice(filtered)

        # 4) apply move & evaluate
        sigma = apply_move(pi, i, j)
        f_sigma = true_eval_fn(sigma)

        nfev += 1
        eval_perms.append(sigma.copy())
        eval_vals.append(f_sigma)

        # 5) accept if improved (RLS style)
        if f_sigma < f_pi:
            pi = sigma
            f_pi = f_sigma

        history_best.append(f_pi)

        # 6) update tabu queue
        if tabu_mode == "item":
            if move_type == "insertion":
                # moved item now at position j in sigma
                shifted_item = sigma[j]
                tabu_q.append(shifted_item)
            else:
                # push both endpoint items as tabu (heuristic)
                a, b = _normalize_pair(i, j)
                tabu_q.append(pi[a])
                tabu_q.append(pi[b])
        else:
            if move_type == "insertion":
                sig = ("ins", pi[i], j)  # note: pi might have changed if accepted; keep consistent with current pi
            elif move_type == "swap":
                a, b = _normalize_pair(i, j)
                sig = ("swp", a, b)
            else:
                a, b = _normalize_pair(i, j)
                sig = ("inv", a, b)
            tabu_q.append(sig)

    best_perm = pi
    best_val = f_pi
    return best_perm, best_val, eval_perms, eval_vals, history_best

def fat_rls_init(
    dim: int,
    true_eval_fn: Callable[[List[int]], float],  # minimization
    budget: int,
    move_type: MoveType = "insertion",
    seed: Optional[int] = None,
    init: int = 100,
    beta: float = 1.2,
    tabu_len: Optional[int] = None,
    tabu_mode: Literal["item", "move"] = "move",
    init_count: int = 1,
) -> Tuple[List[int], float, List[List[int]], List[float], List[float]]:
    """
    FAT-RLS (single trajectory) with configurable neighborhood move.

    Parameters
    ----------
    dim : int
        permutation length n
    true_eval_fn : callable
        f(perm) -> float (minimization)
    budget : int
        number of true evaluations (>=1)
    move_type : {"insertion","swap","invert"}
        neighborhood operator
    seed : int or None
        random seed
    beta : float
        shape parameter in intensity schedule
    tabu_len : int or None
        tabu queue length. default: n
    tabu_mode : {"item","move"}
        - "item": keep tabu by element ID (similar to your original; weakly operator-dependent)
        - "move": keep tabu by move signature (recommended when using swap/invert)
    init_count : int
        Number of random initial solutions to true-evaluate before FAT-RLS moves.

    Returns
    -------
    best_perm, best_val, eval_perms, eval_vals, history_best
    """
    if budget <= 0:
        raise ValueError("budget must be positive.")
    n = dim
    if tabu_len is None:
        tabu_len = n

    if seed is not None:
        random.seed(seed)

    # intensity schedule parameters
    d_ini = n // 2

    # --- initial solution(s) ---
    init_evals = max(1, min(int(init_count), int(budget)))
    eval_perms: List[List[int]] = []
    eval_vals: List[float] = []
    history_best: List[float] = []

    best_val = float("inf")
    best_perm: List[int] = list(range(n))

    if init_evals == 1:
        pi = list(range(n))
        random.shuffle(pi)
        f_pi = true_eval_fn(pi)
        best_perm = pi.copy()
        best_val = f_pi
        eval_perms.append(pi.copy())
        eval_vals.append(f_pi)
        history_best.append(best_val)
    else:
        rng = np.random.default_rng(seed)
        for _ in range(init_evals):
            p0 = rng.permutation(n).astype(np.int32).tolist()
            v0 = float(true_eval_fn(p0))
            eval_perms.append(p0)
            eval_vals.append(v0)
            if v0 < best_val:
                best_val = v0
                best_perm = p0.copy()
            history_best.append(best_val)
        pi = best_perm.copy()
        f_pi = best_val

    nfev = init_evals

    tabu_q: deque[Any] = deque(maxlen=tabu_len)

    # operator config
    if move_type == "insertion":
        gen_cands = _gen_candidates_insertion
        apply_move = _apply_insertion
    elif move_type == "swap":
        gen_cands = _gen_candidates_swap
        apply_move = _apply_swap
    elif move_type == "invert":
        gen_cands = _gen_candidates_invert
        apply_move = _apply_invert
    else:
        raise ValueError(f"Unknown move_type: {move_type}")

    while nfev < budget and nfev < init:
        # 1) compute intensity d. For init_count > 1, initial evaluations
        # count against budget but the schedule starts from zero after them.
        if init_evals <= 1:
            p = nfev / float(budget)
        else:
            schedule_budget = max(1, int(budget) - init_evals)
            schedule_nfev = max(0, nfev - init_evals)
            p = schedule_nfev / float(schedule_budget)
        d = int(round(1.0 + s_beta(p, beta) * (d_ini - 1)))
        d = max(1, min(d, n - 1))

        # 2) generate candidates
        candidates = gen_cands(n, d)

        # 3) tabu filtering
        filtered: List[Tuple[int, int]] = []

        if tabu_mode == "item":
            # old style: forbid moves involving tabu "item ids"
            # for insertion: we can mimic "moved item" check by sigma[i]
            # for swap/invert: we forbid if either endpoint items are tabu (simple heuristic)
            for (i, j) in candidates:
                if move_type == "insertion":
                    moved_item = pi[i]
                    if moved_item in tabu_q:
                        continue
                else:
                    a, b = _normalize_pair(i, j)
                    if pi[a] in tabu_q or pi[b] in tabu_q:
                        continue
                filtered.append((i, j))

        elif tabu_mode == "move":
            # recommended: forbid repeating the same "move signature"
            # - insertion: (item_id, target_pos)
            # - swap: (pos_a, pos_b) (position-based; cheap and stable)
            # - invert: (l, r) segment
            for (i, j) in candidates:
                if move_type == "insertion":
                    sig = ("ins", pi[i], j)
                elif move_type == "swap":
                    a, b = _normalize_pair(i, j)
                    sig = ("swp", a, b)
                else:  # invert
                    a, b = _normalize_pair(i, j)
                    sig = ("inv", a, b)

                if sig in tabu_q:
                    continue
                filtered.append((i, j))
        else:
            raise ValueError(f"Unknown tabu_mode: {tabu_mode}")

        if not filtered:
            # tabu ignored fallback
            filtered = candidates

        i, j = random.choice(filtered)

        # 4) apply move & evaluate
        sigma = apply_move(pi, i, j)
        f_sigma = true_eval_fn(sigma)

        nfev += 1
        eval_perms.append(sigma.copy())
        eval_vals.append(f_sigma)

        # 5) accept if improved (RLS style)
        if f_sigma < f_pi:
            pi = sigma
            f_pi = f_sigma

        history_best.append(f_pi)

        # 6) update tabu queue
        if tabu_mode == "item":
            if move_type == "insertion":
                # moved item now at position j in sigma
                shifted_item = sigma[j]
                tabu_q.append(shifted_item)
            else:
                # push both endpoint items as tabu (heuristic)
                a, b = _normalize_pair(i, j)
                tabu_q.append(pi[a])
                tabu_q.append(pi[b])
        else:
            if move_type == "insertion":
                sig = ("ins", pi[i], j)  # note: pi might have changed if accepted; keep consistent with current pi
            elif move_type == "swap":
                a, b = _normalize_pair(i, j)
                sig = ("swp", a, b)
            else:
                a, b = _normalize_pair(i, j)
                sig = ("inv", a, b)
            tabu_q.append(sig)

    best_perm = pi
    best_val = f_pi
    return eval_perms, eval_vals

