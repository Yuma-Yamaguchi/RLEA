from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor

class SupportsAlgorithm(Protocol):
    metric: SimpleNamespace

    def not_terminated(self, arc: "Population") -> bool:
        ...

class SupportsProblem(Protocol):
    N: int
    D: int

    def evaluation(self, decs: np.ndarray, log: Optional[list[float]] = None) -> "Population":
        ...

    def cal_obj(self, decs: np.ndarray) -> np.ndarray:
        ...

@dataclass
class Population:
    decs: np.ndarray
    objs: np.ndarray

    def __post_init__(self) -> None:
        self.decs = np.asarray(self.decs)
        self.objs = np.asarray(self.objs).reshape(-1, 1)

    def __add__(self, other: "Population") -> "Population":
        return Population(
            decs=np.vstack([self.decs, other.decs]),
            objs=np.vstack([self.objs, other.objs]),
        )


def garkencoder_gen(n: int, d: int, rng: np.random.Generator) -> Tuple[None, np.ndarray]:
    decs = np.empty((n, d), dtype=int)
    base = np.arange(1, d + 1, dtype=int)
    for i in range(n):
        decs[i] = rng.permutation(base)
    return None, decs


def two_exch(pop_dec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    off_dec = np.array(pop_dec, copy=True)
    n, d = off_dec.shape
    for i in range(n):
        a, b = rng.choice(d, size=2, replace=False)
        off_dec[i, [a, b]] = off_dec[i, [b, a]]
    return off_dec


def topkchecker(true_obj: np.ndarray, better_id: np.ndarray, k: int) -> float:
    if k <= 0:
        return float("nan")
    true_rank = np.argsort(true_obj.reshape(-1))
    true_topk = true_rank[:k]
    pred_topk = np.asarray(better_id).reshape(-1)[:k]
    hit = np.intersect1d(true_topk, pred_topk).size
    return hit / k


def pop_update_elanddiv(arc: Population, elr: float, n_keep: int) -> Population:
    decs = arc.decs
    objs = arc.objs.reshape(-1)
    n_all = decs.shape[0]

    elite_n = int(np.ceil(elr * n_keep))
    elite_n = min(max(elite_n, 1), n_keep, n_all)

    ranked = np.argsort(objs)
    selected = list(ranked[:elite_n])

    if len(selected) < min(n_keep, n_all):
        remain = [idx for idx in ranked if idx not in selected]
        while len(selected) < min(n_keep, n_all) and remain:
            sel_decs = decs[selected]
            rem_decs = decs[remain]
            dist = cdist(rem_decs, sel_decs, metric="hamming") * decs.shape[1]
            min_dist = dist.min(axis=1)
            pick_pos = int(np.argmax(min_dist))
            selected.append(remain[pick_pos])
            remain.pop(pick_pos)

    selected = np.asarray(selected[:n_keep], dtype=int)
    return Population(decs=decs[selected], objs=arc.objs[selected])


class RFLoS:

    def __init__(
        self,
        k: int = 10,
        wmax: int = 10,
        elr: float = 0.2,
        random_state: Optional[int] = None,
        garkencoder_fn: Optional[Callable[[int, int, np.random.Generator], Tuple[None, np.ndarray]]] = None,
        two_exch_fn: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
        pop_update_fn: Optional[Callable[[Population, float, int], Population]] = None,
        topkchecker_fn: Optional[Callable[[np.ndarray, np.ndarray, int], float]] = None,
        model_factory: Optional[Callable[[], object]] = None,
    ) -> None:
        self.k = k
        self.wmax = wmax
        self.elr = elr
        self.rng = np.random.default_rng(random_state)
        self.garkencoder_fn = garkencoder_fn or garkencoder_gen
        self.two_exch_fn = two_exch_fn or two_exch
        self.pop_update_fn = pop_update_elanddiv or pop_update_fn
        self.topkchecker_fn = topkchecker_fn or topkchecker
        self.model_factory = model_factory or (
            lambda: RandomForestRegressor(random_state=random_state)
        )

    def main(self, algorithm: SupportsAlgorithm, problem: SupportsProblem) -> None:
        if not hasattr(algorithm, "metric"):
            algorithm.metric = SimpleNamespace(runtime=0.0)
        if not hasattr(algorithm.metric, "runtime"):
            algorithm.metric.runtime = 0.0

        _, inidec = self.garkencoder_fn(problem.N, problem.D, self.rng)
        population = problem.evaluation(inidec)
        arc = population

        tic_at = time.perf_counter()

        while algorithm.not_terminated(arc):
            train_decs = arc.decs
            train_objs = arc.objs.reshape(-1)
            pop_dec = population.decs.copy()

            model = self.model_factory()
            model.fit(train_decs, train_objs)
            dec0 = pop_dec.copy()

            w = 1
            recall_list = np.full(self.wmax, np.nan, dtype=float)

            while w <= self.wmax:
                off_dec = self.two_exch_fn(pop_dec, self.rng)

                pre_obj = model.predict(np.vstack([pop_dec, off_dec]))
                pre_pop_obj = pre_obj[: problem.N]
                pre_off_obj = pre_obj[problem.N :]

                selection = pre_off_obj < pre_pop_obj

                true_pop_obj = problem.cal_obj(pop_dec).reshape(-1)
                true_off_obj = problem.cal_obj(off_dec).reshape(-1)
                true_select = true_off_obj < true_pop_obj
                rid = true_select == 1

                denom = np.sum(rid)
                if denom == 0:
                    recall_list[w - 1] = np.nan
                else:
                    recall_list[w - 1] = np.sum(selection[rid]) / denom

                pop_dec[selection, :] = off_dec[selection, :]
                w += 1

            recall = float(np.mean(recall_list))

            pre_pop_obj[selection] = pre_off_obj[selection]
            dec = pop_dec.copy()

            unique_id = ~np.any(np.all(pop_dec[:, None, :] == arc.decs[None, :, :], axis=2), axis=1)
            pop_dec = pop_dec[unique_id, :]
            pre_pop_obj = pre_pop_obj[unique_id]

            better_id = np.argsort(pre_pop_obj)
            if int(np.sum(unique_id)) < self.k:
                new_dec = pop_dec[better_id, :]
            else:
                new_dec = pop_dec[better_id[: self.k], :]

            algorithm.metric.runtime = algorithm.metric.runtime + (time.perf_counter() - tic_at)
            delta = float(np.mean(np.diag(cdist(dec, dec0, metric="hamming")) * problem.D))
            true_obj = problem.cal_obj(pop_dec).reshape(-1)
            kt = float(kendalltau(true_obj, pre_pop_obj).statistic)
            topk_id = better_id[np.arange(self.k)]
            prob = float(self.topkchecker_fn(true_obj, topk_id, self.k))
            tic_at = time.perf_counter()

            for i in range(new_dec.shape[0]):
                solution = new_dec[i : i + 1, :]
                new = problem.evaluation(
                    solution,
                    [algorithm.metric.runtime, prob, kt, recall, delta],
                )
                arc = arc + new
                algorithm.not_terminated(arc)

            population = self.pop_update_fn(arc, self.elr, problem.N)

@dataclass
class RFLoSConfig:
    k: int = 10
    wmax: int = 10
    elr: float = 0.2
    pop_size: int = 100


class _RFLoSProblemAdapter:
    def __init__(self, dim: int, pop_size: int, true_eval_fn: Callable[[List[int]], float], budget: int):
        self.D = int(dim)
        self.N = int(pop_size)
        self._true_eval_fn = true_eval_fn
        self._budget = int(budget)

        self.nfev = 0
        self.eval_perms: List[List[int]] = []
        self.eval_vals: List[float] = []
        self.history_best: List[float] = []
        self.best_val = float('inf')
        self.best_perm: List[int] = list(range(self.D))
        self.metric_log: List[List[float]] = []

    def _to_zero_based(self, p1: np.ndarray) -> List[int]:
        return (np.asarray(p1, dtype=np.int64) - 1).astype(np.int64).tolist()

    def _eval_one(self, p1: np.ndarray, log: Optional[list[float]] = None) -> float:
        p0 = self._to_zero_based(p1)
        val = float(self._true_eval_fn(p0))

        self.nfev += 1
        self.eval_perms.append(p0)
        self.eval_vals.append(val)
        if val < self.best_val:
            self.best_val = val
            self.best_perm = p0
        self.history_best.append(self.best_val)

        if log is not None:
            self.metric_log.append([float(x) for x in log])
        else:
            self.metric_log.append([np.nan, np.nan, np.nan, np.nan, np.nan])
        return val

    def evaluation(self, decs: np.ndarray, log: Optional[list[float]] = None) -> Population:
        decs = np.asarray(decs, dtype=np.int64)
        if decs.ndim == 1:
            decs = decs.reshape(1, -1)

        rem = self._budget - self.nfev
        if rem <= 0 or decs.shape[0] == 0:
            return Population(decs=np.zeros((0, self.D), dtype=np.int64), objs=np.zeros((0, 1), dtype=np.float64))

        if decs.shape[0] > rem:
            decs = decs[:rem]

        vals = np.asarray([self._eval_one(decs[i], log=log) for i in range(decs.shape[0])], dtype=np.float64)
        return Population(decs=decs, objs=vals.reshape(-1, 1))

    def cal_obj(self, decs: np.ndarray) -> np.ndarray:
        decs = np.asarray(decs, dtype=np.int64)
        if decs.ndim == 1:
            decs = decs.reshape(1, -1)
        vals = [float(self._true_eval_fn(self._to_zero_based(decs[i]))) for i in range(decs.shape[0])]
        return np.asarray(vals, dtype=np.float64).reshape(-1, 1)


class _RFLoSAlgorithmAdapter:
    def __init__(self, problem: _RFLoSProblemAdapter):
        self.problem = problem
        self.metric = SimpleNamespace(runtime=0.0)

    def not_terminated(self, arc: Population) -> bool:
        return self.problem.nfev < self.problem._budget


def rflos(
    dim: int,
    true_eval_fn: Callable[[List[int]], float],
    budget: int,
    cfg: Optional[RFLoSConfig] = None,
    seed: int = 0,
) -> Tuple[List[int], float, List[List[int]], List[float], List[float], Dict[str, Any]]:

    if cfg is None:
        cfg = RFLoSConfig()

    budget = max(1, int(budget))
    problem = _RFLoSProblemAdapter(
        dim=int(dim),
        pop_size=max(1, int(cfg.pop_size)),
        true_eval_fn=true_eval_fn,
        budget=budget,
    )
    algo = _RFLoSAlgorithmAdapter(problem)

    solver = RFLoS(
        k=int(cfg.k),
        wmax=int(cfg.wmax),
        elr=float(cfg.elr),
        random_state=int(seed),
    )
    solver.main(algo, problem)

    logs: Dict[str, Any] = {
        'cand': [],
        'pred': [],
        'sigma': [],
        'label': [],
        'metric': problem.metric_log,
    }

    return (
        problem.best_perm,
        float(problem.best_val),
        problem.eval_perms,
        problem.eval_vals,
        problem.history_best,
        logs,
    )
