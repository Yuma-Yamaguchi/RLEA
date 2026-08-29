from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm

Perm = List[int]

def _repair_perm(x: Sequence[int], n: int) -> Perm:
    out = [int(v) for v in x]
    seen = set()
    for i, value in enumerate(out):
        if 0 <= value < n and value not in seen:
            seen.add(value)
        else:
            out[i] = -1
    missing = [value for value in range(n) if value not in seen]
    for i, value in enumerate(out):
        if value == -1:
            out[i] = missing.pop(0)
    return out


def _pmx_crossover(p1: Sequence[int], p2: Sequence[int], rng: np.random.Generator) -> Perm:
    n = len(p1)
    if n <= 1:
        return [int(value) for value in p1]
    a = int(rng.integers(0, n - 1))
    b = int(rng.integers(a + 1, n))
    child = [-1] * n
    child[a:b] = [int(value) for value in p1[a:b]]
    segment = set(child[a:b])
    mapping = {int(p2[i]): int(p1[i]) for i in range(a, b)}
    for i in range(n):
        if a <= i < b:
            continue
        value = int(p2[i])
        seen = set()
        while value in segment and value in mapping and value not in seen:
            seen.add(value)
            value = mapping[value]
        child[i] = value
    return _repair_perm(child, n)


def _operator_ga_perm(pop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if pop.shape[0] <= 1:
        return pop.copy().astype(np.int32)
    offspring: List[Perm] = []
    for i in range(0, pop.shape[0], 2):
        p1 = pop[i]
        p2 = pop[i + 1] if i + 1 < pop.shape[0] else pop[0]
        offspring.append(_pmx_crossover(p1, p2, rng))
        if len(offspring) < pop.shape[0]:
            offspring.append(_pmx_crossover(p2, p1, rng))
    return np.asarray(offspring[: pop.shape[0]], dtype=np.int32)


def _swap_neighbor(p: Sequence[int], rng: np.random.Generator) -> Perm:
    q = [int(value) for value in p]
    if len(q) <= 1:
        return q
    i, j = rng.choice(len(q), size=2, replace=False)
    q[int(i)], q[int(j)] = q[int(j)], q[int(i)]
    return q


def _unique_rows_stable(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    seen = set()
    keep: List[int] = []
    for i, row in enumerate(x):
        key = tuple(int(value) for value in row)
        if key not in seen:
            seen.add(key)
            keep.append(i)
    idx = np.asarray(keep, dtype=np.int64)
    return x[idx], idx


def popupdate_elanddiv(
    arc_perm: np.ndarray,
    arc_obj: np.ndarray,
    elite_rate: float,
    pop_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pop_size = max(1, int(pop_size))
    ranked = np.argsort(arc_obj.reshape(-1))
    elite_num = max(1, min(int(np.floor(elite_rate * pop_size)), pop_size, ranked.size))
    selected = ranked[:elite_num]
    remaining = ranked[elite_num:]
    other_num = max(0, pop_size - selected.size)
    if other_num:
        pool = remaining if remaining.size else ranked
        other = rng.choice(pool, size=other_num, replace=pool.size < other_num)
        selected = np.concatenate([selected, other])
    return arc_perm[selected[:pop_size].astype(np.int64)]


@dataclass
class GBDTMAConfig:
    K: int = 5
    lam: int = 10
    delta: float = 0.2
    n_estimators: int = 100
    r: int = 10
    pop_size: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    subsample: float = 1.0
    min_samples_leaf: int = 1


def _make_model(cfg: GBDTMAConfig, seed: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=int(cfg.n_estimators),
        learning_rate=float(cfg.learning_rate),
        max_depth=int(cfg.max_depth),
        subsample=float(cfg.subsample),
        min_samples_leaf=int(cfg.min_samples_leaf),
        random_state=int(seed),
    )


def gbdtma(
    dim: int,
    true_eval_fn: Callable[[Perm], float],
    budget: int,
    cfg: Optional[GBDTMAConfig] = None,
    seed: int = 0,
    internal_gen: Optional[int] = 1,
) -> Tuple[Perm, float, List[Perm], List[float], List[float], Dict[str, Any]]:
    cfg = cfg or GBDTMAConfig()
    rng = np.random.default_rng(seed)
    n = int(dim)
    budget = max(1, int(budget))
    pop_size = max(1, int(cfg.pop_size))
    pop_perm = np.vstack([rng.permutation(n) for _ in range(pop_size)]).astype(np.int32)
    arc_perm: List[Perm] = []
    arc_obj: List[float] = []
    arc_set: set[Tuple[int, ...]] = set()
    eval_perms: List[Perm] = []
    eval_vals: List[float] = []
    history_best: List[float] = []
    best_perm = pop_perm[0].tolist()
    best_val = float("inf")
    nfev = 0

    for row in pop_perm:
        if nfev >= budget:
            break
        perm = row.tolist()
        value = float(true_eval_fn(perm))
        nfev += 1
        arc_perm.append(perm)
        arc_obj.append(value)
        arc_set.add(tuple(perm))
        eval_perms.append(perm)
        eval_vals.append(value)
        if value < best_val:
            best_val = value
            best_perm = perm
        history_best.append(best_val)

    cand_log: List[np.ndarray] = []
    pred_log: List[np.ndarray] = []
    sigma_log: List[np.ndarray] = []
    label_log: List[np.ndarray] = []

    with tqdm(total=budget, desc="True evaluations") as pbar:
        pbar.update(nfev)
        while nfev < budget:
            model = _make_model(cfg, seed)
            model.fit(np.asarray(arc_perm, dtype=np.int32), np.asarray(arc_obj, dtype=np.float64))
            pop_perm_np = np.asarray(pop_perm, dtype=np.int32)
            off_dec = pop_perm_np.copy()
            for _ in range(internal_gen or 1):
                off_list: List[Perm] = []
                for _ in range(int(cfg.lam)):
                    shuffled = pop_perm_np[rng.permutation(pop_perm_np.shape[0])]
                    off_list.extend(_operator_ga_perm(shuffled, rng).tolist())
                off_dec, _ = _unique_rows_stable(np.asarray(off_list, dtype=np.int32))
            mu = model.predict(off_dec).astype(np.float64)
            k = min(max(1, int(cfg.K)), off_dec.shape[0])
            good_dec = off_dec[np.argsort(mu)[:k]]
            new_dec_list: List[Perm] = []
            local_blocks: List[np.ndarray] = []
            local_pred_blocks: List[np.ndarray] = []

            for root in good_dec:
                neighbors = [root.tolist()]
                for _ in range(int(cfg.r)):
                    neighbors.append(_swap_neighbor(root, rng))
                nei_dec = np.asarray(neighbors, dtype=np.int32)
                nei_mu = model.predict(nei_dec).astype(np.float64)
                unseen = np.asarray([tuple(row.tolist()) not in arc_set for row in nei_dec], dtype=bool)
                if not np.any(unseen):
                    unseen[-1] = True
                filtered_dec = nei_dec[unseen]
                filtered_mu = nei_mu[unseen]
                new_dec_list.append(filtered_dec[int(np.argmin(filtered_mu))].tolist())
                local_blocks.append(filtered_dec)
                local_pred_blocks.append(filtered_mu)

            cand_local = np.vstack(local_blocks) if local_blocks else np.zeros((0, n), dtype=np.int32)
            pred_local = np.concatenate(local_pred_blocks) if local_pred_blocks else np.zeros((0,), dtype=np.float64)
            cand_all = np.vstack([off_dec, cand_local])
            pred_all = np.concatenate([mu, pred_local])
            label = np.full(cand_all.shape[0], np.nan, dtype=np.float64)
            cand_index = {tuple(row.tolist()): i for i, row in enumerate(cand_all)}

            for perm in new_dec_list:
                if nfev >= budget:
                    break
                value = float(true_eval_fn(perm))
                nfev += 1
                arc_perm.append(perm)
                arc_obj.append(value)
                arc_set.add(tuple(perm))
                eval_perms.append(perm)
                eval_vals.append(value)
                if value < best_val:
                    best_val = value
                    best_perm = perm
                history_best.append(best_val)
                if tuple(perm) in cand_index:
                    label[cand_index[tuple(perm)]] = value
                pbar.update(1)

            cand_log.append(cand_all.copy())
            pred_log.append(pred_all.copy())
            sigma_log.append(np.zeros_like(pred_all))
            label_log.append(label.copy())
            pop_perm = popupdate_elanddiv(
                np.asarray(arc_perm, dtype=np.int32),
                np.asarray(arc_obj, dtype=np.float64),
                float(cfg.delta),
                pop_size,
                rng,
            )
            pbar.set_postfix(best=float(best_val))

    logs: Dict[str, Any] = {
        "cand": cand_log,
        "pred": pred_log,
        "sigma": sigma_log,
        "label": label_log,
    }
    return best_perm, float(best_val), eval_perms, eval_vals, history_best, logs
