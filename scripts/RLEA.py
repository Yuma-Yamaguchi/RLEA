from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm

from gbdtma import popupdate_elanddiv

Perm = List[int]


def _insert_neighbor(perm: Sequence[int], rng: np.random.Generator) -> Perm:
    out = [int(value) for value in perm]
    if len(out) <= 1:
        return out
    i, j = rng.choice(len(out), size=2, replace=False)
    value = out.pop(int(i))
    out.insert(int(j), value)
    return out


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


def _make_incumbent_pool(
    best_perm: Sequence[int],
    rng: np.random.Generator,
    pool_size: int,
    max_depth: int,
) -> np.ndarray:
    pool: List[Perm] = []
    for _ in range(max(1, int(pool_size))):
        child = [int(value) for value in best_perm]
        for _ in range(max(1, int(max_depth))):
            child = _insert_neighbor(child, rng)
        pool.append(child)
    return _unique_rows_stable(np.asarray(pool, dtype=np.int32))[0]


def _score_pool(model: GradientBoostingRegressor, pool: np.ndarray) -> np.ndarray:
    if pool.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    return model.predict(pool).astype(np.float64)


def _incumbent_search(
    best_perm: Sequence[int],
    model: GradientBoostingRegressor,
    rng: np.random.Generator,
    pool_size: int,
    max_depth: int,
    sigma_steps: int,
    arc_set: set[Tuple[int, ...]],
) -> Tuple[np.ndarray, np.ndarray]:
    pool = _make_incumbent_pool(best_perm, rng, pool_size, max_depth)
    scores = _score_pool(model, pool)
    for _ in range(max(1, int(sigma_steps)) - 1):
        if pool.shape[0] == 0:
            break
        child_list = [_insert_neighbor(row, rng) for row in pool]
        children = np.asarray(child_list, dtype=np.int32)
        child_scores = _score_pool(model, children)
        next_pool = pool.copy()
        next_scores = scores.copy()
        for i, child in enumerate(child_list):
            key = tuple(child)
            if key in arc_set:
                continue
            if child_scores[i] < next_scores[i]:
                next_pool[i] = children[i]
                next_scores[i] = child_scores[i]
        pool, keep = _unique_rows_stable(next_pool)
        scores = next_scores[keep]
    return pool, scores


def _make_model(cfg: "RLEAConfig", seed: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=int(cfg.n_estimators),
        learning_rate=float(cfg.learning_rate),
        max_depth=int(cfg.max_depth),
        subsample=float(cfg.subsample),
        min_samples_leaf=int(cfg.min_samples_leaf),
        random_state=int(seed),
    )


@dataclass
class RLEAConfig:
    K: int = 1
    lam: int = 10
    delta: float = 0.2
    fit_per: int = 5
    pop_size: int = 100
    incumbent_pool_size: int = 50
    incumbent_max_depth: int = 1
    incumbent_sigma: int = 10
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    subsample: float = 1.0
    min_samples_leaf: int = 1


def rlea(
    dim: int,
    true_eval_fn: Callable[[Perm], float],
    budget: int,
    cfg: Optional[RLEAConfig] = None,
    seed: int = 0,
) -> Tuple[Perm, float, List[Perm], List[float], List[float], Dict[str, Any]]:
    cfg = cfg or RLEAConfig()
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

    model = _make_model(cfg, seed)
    fitted = False
    cand_log: List[np.ndarray] = []
    pred_log: List[np.ndarray] = []
    sigma_log: List[np.ndarray] = []
    label_log: List[np.ndarray] = []
    source_log: List[np.ndarray] = []

    with tqdm(total=budget, desc="True evaluations") as pbar:
        pbar.update(nfev)
        while nfev < budget:
            if not fitted or (nfev - pop_size) % max(1, int(cfg.fit_per)) == 0:
                model = _make_model(cfg, seed)
                model.fit(np.asarray(arc_perm, dtype=np.int32), np.asarray(arc_obj, dtype=np.float64))
                fitted = True

            pool, scores = _incumbent_search(
                best_perm=best_perm,
                model=model,
                rng=rng,
                pool_size=cfg.incumbent_pool_size,
                max_depth=cfg.incumbent_max_depth,
                sigma_steps=cfg.incumbent_sigma,
                arc_set=arc_set,
            )
            unseen = [
                i for i, row in enumerate(pool)
                if tuple(int(value) for value in row) not in arc_set
            ]
            if unseen:
                unseen_idx = np.asarray(unseen, dtype=np.int64)
                chosen_idx = unseen_idx[np.argsort(scores[unseen_idx])[: max(1, int(cfg.K))]]
                chosen = pool[chosen_idx].tolist()
            else:
                chosen = []
                while len(chosen) < max(1, int(cfg.K)):
                    perm = rng.permutation(n).astype(np.int32).tolist()
                    if tuple(perm) not in arc_set:
                        chosen.append(perm)

            label = np.full(pool.shape[0], np.nan, dtype=np.float64)
            pool_index = {tuple(row.tolist()): i for i, row in enumerate(pool)}
            for perm in chosen:
                if nfev >= budget:
                    break
                key = tuple(int(value) for value in perm)
                if key in arc_set:
                    continue
                value = float(true_eval_fn(perm))
                nfev += 1
                arc_perm.append(perm)
                arc_obj.append(value)
                arc_set.add(key)
                eval_perms.append(perm)
                eval_vals.append(value)
                if value < best_val:
                    best_val = value
                    best_perm = perm
                history_best.append(best_val)
                if key in pool_index:
                    label[pool_index[key]] = value
                pbar.update(1)

            cand_log.append(pool.copy())
            pred_log.append(scores.copy())
            sigma_log.append(np.zeros_like(scores))
            label_log.append(label.copy())
            source_log.append(np.full(pool.shape[0], "incumbent", dtype=object))
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
        "source": source_log,
    }
    return best_perm, float(best_val), eval_perms, eval_vals, history_best, logs
