from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Literal

import numpy as np
from tqdm import tqdm

try:
    from sklearn.ensemble import GradientBoostingRegressor
except Exception:
    GradientBoostingRegressor = None  # type: ignore

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
        if len(off) < n_pop:
            off.append(_pmx_crossover(p2, p1,a,b))
    return np.asarray(off[:n_pop], dtype=np.int32)

def _swap_neighbor(p: Sequence[int], rng: np.random.Generator) -> Perm:
    n = len(p)
    if n <= 1:
        return list(p)
    i, j = rng.choice(n, size=2, replace=False)
    q = list(p)
    q[i], q[j] = q[j], q[i]
    return q


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
class GBDTMAConfig:
    # MATLAB defaults: K=5, lambda=10, delta=0.2, t=100, r=10
    K: int = 5
    lam: int = 10
    delta: float = 0.2
    n_estimators: int = 100
    r: int = 10

    # PlatEMO default population size is often 100.
    pop_size: int = 100

    # sklearn options (close to LSBoost behavior)
    learning_rate: float = 0.1 #matlabだと1がデフォ
    max_depth: int = 3
    subsample: float = 1.0
    min_samples_leaf: int = 1
    surrogate_kind: Literal['gbdt','rf','gcn','gat','sage','gin','jkgcn','permformer','rand'] = 'gbdt'
    gnn_epochs: int = 200
    batch_size: int = 256
    acq: Literal['mean','lcb'] = 'lcb'
    kappa: float = 2.0
    problem: Literal['TSP','ATSP','QAP','LOP','PFSP'] = 'TSP'


def gbdtma(
    dim: int,
    true_eval_fn: Callable[[Perm], float],
    budget: int,
    cfg: Optional[GBDTMAConfig] = None,
    seed: int = 0,
    internal_gen: Optional[int] = 1,
) -> Tuple[Perm, float, List[Perm], List[float], List[float], Dict[str, Any]]:
    """
    Python implementation aligned to GBDTMA.m flow (minimization).
    """
    if cfg is None:
        cfg = GBDTMAConfig()

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

    nfev = 0
    for i in range(pop_size):
        if nfev >= budget:
            break
        p = pop_perm[i].tolist()
        v = float(true_eval_fn(p))
        nfev += 1

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
        make_cycle, undirected = True, True
        every_connect, top_k = False, dim
        each_connect = False
    elif p == "ATSP":
        make_cycle, undirected = True, False
        every_connect, top_k = False, dim
        each_connect = False
    elif p == "QAP":
        make_cycle, undirected = False, False
        every_connect, top_k = False, dim
        each_connect = True
    elif p in ("LOP", "PFSP"):
        make_cycle, undirected = False, False
        every_connect, top_k = True, dim
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
                mdl.fit(x_train.tolist(), y_train.tolist())

            pop_perm_np = np.asarray(pop_perm, dtype=np.int32)

            # GA global search
            for i in range(internal_gen or 1):
                off_list: List[Perm] = []
                for _ in range(int(cfg.lam)):
                    order = rng.permutation(pop_perm_np.shape[0])
                    shuffled = pop_perm_np[order]
                    off = _operator_ga_perm(shuffled, rng)
                    off_list.extend(off.tolist())
                off_dec = np.asarray(off_list, dtype=np.int32)

                # unique(OffDec,"rows",'stable')
                off_dec, ndid = _unique_rows_stable(off_dec)
                # if cfg.surrogate_kind == "gbdt":
                #     mu_all = mdl.predict(np.asarray(off_list, dtype=np.int32)).astype(np.float64)
                #     sigma_all = np.zeros_like(mu_all)
                # else:
                mu_all, sigma_all = mdl.predict(off_list)
                mu_all = np.asarray(mu_all, dtype=np.float64)
                sigma_all = np.asarray(sigma_all, dtype=np.float64)

                # If using GBDT surrogate, keep original GBDTMA behavior (mean only)
                acq_kind = "mean" if cfg.surrogate_kind == "gbdt" else cfg.acq
                if acq_kind == "lcb":
                    off_pred_all = mu_all - float(cfg.kappa) * sigma_all
                else:
                    off_pred_all = mu_all
                off_pred = off_pred_all[ndid]

                # GoodDec = OffDec(goodid(1:K),:)
                k = min(max(1, int(cfg.K)), off_dec.shape[0])
                good_idx = np.argsort(off_pred)[:k]
                good_dec = off_dec[good_idx]

            new_dec_list: List[Perm] = []
            local_cand_blocks: List[np.ndarray] = []
            local_pred_blocks: List[np.ndarray] = []
            local_sigma_blocks: List[np.ndarray] = []

            # local search
            for i in range(good_dec.shape[0]):
                now = good_dec[i].tolist()
                nei_list: List[Perm] = [now]
                for _ in range(int(cfg.r)):
                    nei_list.append(_swap_neighbor(now, rng))

                nei_dec = np.asarray(nei_list, dtype=np.int32)
                # if cfg.surrogate_kind == "gbdt":
                #     mu = mdl.predict(nei_dec).astype(np.float64)
                #     sigma = np.zeros_like(mu)
                # else:
                mu, sigma = mdl.predict(nei_dec.tolist())
                mu = np.asarray(mu, dtype=np.float64)
                sigma = np.asarray(sigma, dtype=np.float64)
                acq_kind = "mean" if cfg.surrogate_kind == "gbdt" else cfg.acq
                if acq_kind == "lcb":
                    nei_pred = mu - float(cfg.kappa) * sigma
                else:
                    nei_pred = mu

                mask = np.ones(nei_dec.shape[0], dtype=bool)
                for j in range(nei_dec.shape[0]):
                    if tuple(int(v) for v in nei_dec[j]) in arc_set:
                        mask[j] = False

                if np.any(mask):
                    nei_dec_f = nei_dec[mask]
                    nei_pred_f = nei_pred[mask]
                    sigma_f = sigma[mask]
                else:
                    # MATLAB fallback:
                    # [ndarcid,~] = size(NeiDec); NeiDec = NeiDec(ndarcid,:)
                    # -> keep only the last row
                    nei_dec_f = nei_dec[-1:, :]
                    nei_pred_f = nei_pred[-1:]
                    sigma_f = sigma[-1:]

                best_j = int(np.argmin(nei_pred_f))
                chosen = nei_dec_f[best_j].tolist()
                new_dec_list.append(chosen)

                local_cand_blocks.append(nei_dec_f.copy())
                local_pred_blocks.append(nei_pred_f.copy())
                local_sigma_blocks.append(sigma_f.copy())

            if len(local_cand_blocks) > 0:
                cand_local = np.vstack(local_cand_blocks).astype(np.int32)
                pred_local = np.concatenate(local_pred_blocks).astype(np.float64)
            else:
                cand_local = np.zeros((0, n), dtype=np.int32)
                pred_local = np.zeros((0,), dtype=np.float64)

            cand_all = np.vstack([off_dec.astype(np.int32), cand_local])
            pred_all = np.concatenate([off_pred.astype(np.float64), pred_local])
            # sigma aligns with pred_all (global candidates + local candidates)
            off_sigma = sigma_all[ndid].astype(np.float64)
            if len(local_sigma_blocks) > 0:
                sigma_local = np.concatenate(local_sigma_blocks).astype(np.float64)
            else:
                sigma_local = np.zeros((0,), dtype=np.float64)
            sigma_all_log = np.concatenate([off_sigma, sigma_local])
            cand_index = {tuple(int(v) for v in cand_all[i]): i for i in range(cand_all.shape[0])}
            label_all = np.full((cand_all.shape[0],), np.nan, dtype=np.float64)

            # true evaluation and archive update
            for p in new_dec_list:
                if nfev >= budget:
                    break
                if nfev == 300:
                    a = 1
                v = float(true_eval_fn(p))
                nfev += 1
                pbar.update(1)

                arc_perm.append(p)
                arc_obj.append(v)
                arc_set.add(tuple(p))
                eval_perms.append(p)
                eval_vals.append(v)

                if v < best_val:
                    best_val = v
                    best_perm = p
                history_best.append(best_val)

                key = tuple(p)
                if key in cand_index:
                    label_all[cand_index[key]] = v

            cand_log.append(cand_all.copy())
            pred_log.append(pred_all.copy())
            sigma_log.append(sigma_all_log.copy())
            label_log.append(label_all.copy())

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

    logs: Dict[str, Any] = {"cand": cand_log, "pred": pred_log, "sigma": sigma_log, "label": label_log}
    return best_perm, float(best_val), eval_perms, eval_vals, history_best, logs
