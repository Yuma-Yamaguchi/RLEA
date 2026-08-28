
# cego.py
"""
Python implementation of Combinatorial Efficient Global Optimization (CEGO) for permutation problems.

Design goals for this project:
- Same return signature as other baselines (fat_rls / gbdtma):
    (best_perm, best_val, eval_perms, eval_vals, history_best, logs)
- Keep `main` simple: CEGO-specific details live here (distance, surrogate, EI, optimizer).

Faithfulness (practical level):
- Mirrors the high-level control flow of R/CEGO's `optimCEGO`:
    1) initial design (evalInit)
    2) build kriging model on evaluated archive
    3) optimize infill criterion (EI) using an EA (optimEA-style) in permutation space
    4) if candidate is duplicate (or optimization fails), fall back to exploration by Max-Min distance
    5) evaluate exactly ONE new solution per outer iteration (until budget)

Notes:
- We implement *ordinary kriging with constant mean* and a distance-based correlation:
      R_ij = exp(-theta * d(x_i, x_j)) + nugget*I
  Theta is fitted by grid-search MLE (like a simple MLE routine).
- Distance can be swapped among: interchange(Cayley), Hamming, Kendall tau.
- The EA here is intended to be close to the R `optimEA` spirit (tournament selection, crossover, mutation),
  but it is specialized to optimize EI over permutations and uses a surrogate-evaluation budget
  that does NOT consume the expensive evaluation budget.

If you need bitwise-identical behavior to the R package, you'll have to port more of the CEGO R stack
(indefinite-kernel repairs, distance-parameter MLE, etc.). This file targets a solid, swappable baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Any, Optional, Sequence

import math
import numpy as np

try:
    from scipy.stats import norm
except Exception:
    norm = None  # type: ignore

Perm = List[int]


# =========================
# Distances (normalized)
# =========================

def distance_hamming_scaled(x: Sequence[int], y: Sequence[int]) -> float:
    """Hamming distance on positions, scaled to [0,1]."""
    n = len(x)
    if n == 0:
        return 0.0
    diff = 0
    for i in range(n):
        if int(x[i]) != int(y[i]):
            diff += 1
    return float(diff) / float(n)


def _invert_perm(p: Sequence[int]) -> List[int]:
    """Return inverse permutation inv where inv[p[i]] = i."""
    n = len(p)
    inv = [0] * n
    for i, v in enumerate(p):
        inv[int(v)] = i
    return inv


def distance_interchange_scaled(x: Sequence[int], y: Sequence[int]) -> float:
    """
    Interchange (Cayley) distance between permutations x,y, scaled to [0,1].

    Cayley distance: d = n - (#cycles of permutation inv(x) o y).
    Max Cayley distance for S_n is n-1, so scaled by (n-1).
    """
    n = len(x)
    if n <= 1:
        return 0.0

    invx = _invert_perm(x)
    # p = inv(x) o y as mapping on {0..n-1}: i -> invx[y[i]]
    p = [invx[int(v)] for v in y]

    visited = [False] * n
    cycles = 0
    for i in range(n):
        if visited[i]:
            continue
        cycles += 1
        j = i
        while not visited[j]:
            visited[j] = True
            j = p[j]

    d = n - cycles
    return float(d) / float(n - 1)


class _Fenwick:
    __slots__ = ("n", "bit")
    def __init__(self, n: int):
        self.n = n
        self.bit = [0] * (n + 1)
    def add(self, i: int, delta: int) -> None:
        i += 1
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i
    def sum(self, i: int) -> int:
        # sum [0, i)
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s


def distance_kendall_scaled(x: Sequence[int], y: Sequence[int]) -> float:
    """
    Kendall tau distance (number of discordant pairs), scaled to [0,1].

    Implementation:
      Let inv_x map value -> position in x.
      Transform y into positions in x: a[i] = inv_x[y[i]].
      Kendall distance = inversion count of a.
    """
    n = len(x)
    if n <= 1:
        return 0.0

    invx = _invert_perm(x)
    a = [invx[int(v)] for v in y]

    fw = _Fenwick(n)
    inv_count = 0
    seen = 0
    for v in a:
        # how many seen so far are > v ?
        leq = fw.sum(v + 1)
        inv_count += (seen - leq)
        fw.add(v, 1)
        seen += 1

    max_inv = n * (n - 1) // 2
    return float(inv_count) / float(max_inv)


def get_distance_fn(name: str) -> Callable[[Sequence[int], Sequence[int]], float]:
    name_l = name.strip().lower()
    if name_l in {"interchange", "cayley"}:
        return distance_interchange_scaled
    if name_l in {"hamming"}:
        return distance_hamming_scaled
    if name_l in {"kendall", "kendalltau", "kendall-tau"}:
        return distance_kendall_scaled
    raise ValueError(f"Unknown distance '{name}'. Use interchange|hamming|kendall.")


# =========================
# Kriging (ordinary kriging)
# =========================

def _pairwise_distance_matrix(X: np.ndarray, dist_fn: Callable[[Sequence[int], Sequence[int]], float]) -> np.ndarray:
    """Compute pairwise distances for X (m,n) permutations."""
    m = int(X.shape[0])
    D = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        xi = X[i].tolist()
        for j in range(i + 1, m):
            dj = float(dist_fn(xi, X[j].tolist()))
            D[i, j] = dj
            D[j, i] = dj
    return D


def _distance_vector_to_set(
    x: np.ndarray,
    X_train: np.ndarray,
    dist_fn: Callable[[Sequence[int], Sequence[int]], float],
) -> np.ndarray:
    """Distances from a single perm x (n,) to each row of X_train (m,n)."""
    m = int(X_train.shape[0])
    xi = x.tolist()
    d = np.zeros((m,), dtype=np.float64)
    for i in range(m):
        d[i] = float(dist_fn(xi, X_train[i].tolist()))
    return d


def _corr_from_dist(dist: np.ndarray, theta: float, nugget: float) -> np.ndarray:
    """R = exp(-theta * dist) + nugget*I."""
    R = np.exp(-theta * dist, dtype=np.float64)
    if nugget > 0:
        R = R + np.eye(R.shape[0], dtype=np.float64) * nugget
    return R


def _neg_loglik_theta(dist_mat: np.ndarray, y: np.ndarray, theta: float, nugget: float) -> float:
    """
    Negative concentrated log-likelihood for ordinary kriging with constant mean.
    Uses Cholesky; returns +inf if not PSD.
    """
    n = int(y.shape[0])
    R = _corr_from_dist(dist_mat, theta=float(theta), nugget=float(nugget))

    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        return float("inf")

    # Solve R^{-1} y via cholesky
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

    ones = np.ones((n,), dtype=np.float64)
    beta = np.linalg.solve(L.T, np.linalg.solve(L, ones))

    denom = float(ones @ beta)
    if denom <= 0.0 or (not np.isfinite(denom)):
        return float("inf")

    mu = float((ones @ alpha) / denom)
    resid = y - mu * ones
    gamma = np.linalg.solve(L.T, np.linalg.solve(L, resid))

    sigma2 = float((resid @ gamma) / n)
    if sigma2 <= 0.0 or (not np.isfinite(sigma2)):
        return float("inf")

    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return 0.5 * (n * math.log(sigma2) + logdet)


def _kriging_fit_theta(
    dist_mat: np.ndarray,
    y: np.ndarray,
    log_theta_min: float,
    log_theta_max: float,
    n_grid: int,
    nugget: float,
    prev_theta: Optional[float] = None,
) -> float:
    """
    Fit theta (positive) by a small logspace grid-search over the concentrated NLL.

    Performance note:
      The full CEGO stack can spend a lot of time in MLE. For practical runs we:
        - search locally around prev_theta once available
        - keep grid sizes modest (configurable)
    """
    n_grid = max(7, int(n_grid))

    if prev_theta is not None and np.isfinite(prev_theta) and prev_theta > 0:
        c = float(math.log10(prev_theta))
        lo = max(float(log_theta_min), c - 2.0)
        hi = min(float(log_theta_max), c + 2.0)
        grid = np.logspace(lo, hi, n_grid, base=10.0)
    else:
        grid = np.logspace(float(log_theta_min), float(log_theta_max), n_grid, base=10.0)

    best_theta = float(grid[0])
    best_nll = float("inf")
    for theta in grid:
        nll = _neg_loglik_theta(dist_mat, y, theta=float(theta), nugget=float(nugget))
        if nll < best_nll:
            best_nll = nll
            best_theta = float(theta)
    return best_theta


def _kriging_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dist_mat: np.ndarray,
    theta: float,
    nugget: float,
    dist_fn: Callable[[Sequence[int], Sequence[int]], float],
    X_cand: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ordinary kriging prediction at candidate points.
    Returns (mu_hat, s_hat).
    """
    n = int(y_train.shape[0])
    R = _corr_from_dist(dist_mat, theta=float(theta), nugget=float(nugget))

    # Cholesky
    L = np.linalg.cholesky(R)

    ones = np.ones((n,), dtype=np.float64)
    Rinvy = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    Rinv1 = np.linalg.solve(L.T, np.linalg.solve(L, ones))

    denom = float(ones @ Rinv1)
    mu = float((ones @ Rinvy) / denom)

    resid = y_train - mu * ones
    Rinv_resid = np.linalg.solve(L.T, np.linalg.solve(L, resid))
    sigma2 = float((resid @ Rinv_resid) / n)
    sigma2 = max(sigma2, 1e-16)

    m = int(X_cand.shape[0])
    mu_hat = np.zeros((m,), dtype=np.float64)
    s_hat = np.zeros((m,), dtype=np.float64)

    for i in range(m):
        d = _distance_vector_to_set(X_cand[i], X_train, dist_fn)
        r = np.exp(-float(theta) * d, dtype=np.float64)

        # compute R^{-1} r
        Rinv_r = np.linalg.solve(L.T, np.linalg.solve(L, r))

        mu_hat[i] = mu + float(r @ Rinv_resid)

        # ordinary kriging variance
        u = 1.0 - float(ones @ Rinv_r)
        s2 = sigma2 * (1.0 - float(r @ Rinv_r) + (u * u) / denom)
        s_hat[i] = math.sqrt(max(s2, 0.0))

    
def _kriging_prepare(
    dist_mat: np.ndarray,
    y_train: np.ndarray,
    theta: float,
    nugget: float,
) -> Dict[str, Any]:
    """
    Precompute matrices/vectors for fast repeated ordinary-kriging predictions
    with fixed (dist_mat, y_train, theta).
    """
    n = int(y_train.shape[0])
    R = _corr_from_dist(dist_mat, theta=float(theta), nugget=float(nugget))
    L = np.linalg.cholesky(R)

    ones = np.ones((n,), dtype=np.float64)
    Rinvy = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    Rinv1 = np.linalg.solve(L.T, np.linalg.solve(L, ones))

    denom = float(ones @ Rinv1)
    mu = float((ones @ Rinvy) / denom)

    resid = y_train - mu * ones
    Rinv_resid = np.linalg.solve(L.T, np.linalg.solve(L, resid))
    sigma2 = float((resid @ Rinv_resid) / n)
    sigma2 = max(sigma2, 1e-16)

    return {
        "L": L,
        "ones": ones,
        "denom": denom,
        "mu": mu,
        "Rinv_resid": Rinv_resid,
        "sigma2": sigma2,
    }


def _kriging_predict_prepared(
    prep: Dict[str, Any],
    X_train: np.ndarray,
    dist_fn: Callable[[Sequence[int], Sequence[int]], float],
    theta: float,
    X_cand: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict using prepared Cholesky etc. Avoids recomputing factorization.
    """
    L = prep["L"]
    ones = prep["ones"]
    denom = float(prep["denom"])
    mu = float(prep["mu"])
    Rinv_resid = prep["Rinv_resid"]
    sigma2 = float(prep["sigma2"])

    m = int(X_cand.shape[0])
    mu_hat = np.zeros((m,), dtype=np.float64)
    s_hat = np.zeros((m,), dtype=np.float64)

    for i in range(m):
        d = _distance_vector_to_set(X_cand[i], X_train, dist_fn)
        r = np.exp(-float(theta) * d, dtype=np.float64)

        Rinv_r = np.linalg.solve(L.T, np.linalg.solve(L, r))
        mu_hat[i] = mu + float(r @ Rinv_resid)

        u = 1.0 - float(ones @ Rinv_r)
        s2 = sigma2 * (1.0 - float(r @ Rinv_r) + (u * u) / denom)
        s_hat[i] = math.sqrt(max(s2, 0.0))

    return mu_hat, s_hat



# =========================
# EI
# =========================

def _ei(y_min: float, mu: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Expected Improvement for minimization (vectorized)."""
    if norm is None:
        raise ImportError("scipy is required for EI computation (scipy.stats.norm).")
    s = np.maximum(s, 1e-12)
    z = (y_min - mu) / s
    return (y_min - mu) * norm.cdf(z) + s * norm.pdf(z)


# =========================
# Permutation operators (EA)
# =========================

def _swap_mutation(p: Sequence[int], rng: np.random.Generator, n_swaps: int = 1) -> Perm:
    q = list(p)
    n = len(q)
    if n <= 1:
        return q
    n_swaps = max(1, int(n_swaps))
    for _ in range(n_swaps):
        i, j = rng.choice(n, size=2, replace=False)
        q[i], q[j] = q[j], q[i]
    return q


def _cycle_crossover(p1: Sequence[int], p2: Sequence[int], rng: np.random.Generator) -> Perm:
    """
    Cycle crossover (CX) for permutations.
    Produces one child:
      - pick a random start index
      - copy that cycle from p1, rest from p2
    """
    n = len(p1)
    if n <= 1:
        return list(p1)

    start = int(rng.integers(0, n))
    child = [-1] * n

    # position lookup in p1 for values (needed to follow cycles using values in p2)
    pos_in_p1 = [0] * n
    for i, v in enumerate(p1):
        pos_in_p1[int(v)] = i

    idx = start
    while child[idx] == -1:
        child[idx] = int(p1[idx])
        next_val = int(p2[idx])
        idx = int(pos_in_p1[next_val])

    # fill remaining from p2
    for i in range(n):
        if child[i] == -1:
            child[i] = int(p2[i])

    return child  # type: ignore


def _tournament_select(
    fitness: np.ndarray,
    rng: np.random.Generator,
    tournament_size: int,
    tournament_prob: float,
    n_select: int,
) -> np.ndarray:
    """
    Tournament selection with probability (like CEGO/optimEA style):
      - sample k contestants
      - sort by fitness (min)
      - pick best with prob p, else 2nd best with prob p*(1-p), etc.
    """
    popsize = int(fitness.shape[0])
    k = max(2, int(tournament_size))
    p = float(tournament_prob)
    n_select = int(n_select)

    out = np.zeros((n_select,), dtype=np.int64)
    for t in range(n_select):
        cand = rng.choice(popsize, size=k, replace=False)
        order = cand[np.argsort(fitness[cand])]
        # probabilistic pick among ordered
        r = rng.random()
        prob = 0.0
        chosen = int(order[-1])
        for i in range(k):
            prob += p * ((1.0 - p) ** i)
            if r <= prob:
                chosen = int(order[i])
                break
        out[t] = chosen
    return out


def _ea_optimize_infill(
    n: int,
    rng: np.random.Generator,
    creation_fn: Callable[[], Perm],
    fitness_fn: Callable[[Perm], float],
    popsize: int,
    budget: int,
    recombination_rate: float,
    mutation_rate: float,
    tournament_size: int,
    tournament_prob: float,
    n_swaps_mut: int,
) -> Tuple[Perm, float, List[Perm], List[float]]:
    """
    Optimize a (cheap) fitness function over permutations with an EA.

    Returns:
      best_x, best_f, visited_x, visited_f
    visited_* are all individuals that got fitness evaluated (for logging).
    """
    popsize = max(2, int(popsize))
    budget = max(popsize, int(budget))

    # init
    population: List[Perm] = [creation_fn() for _ in range(popsize)]
    fit_cache: Dict[Tuple[int, ...], float] = {}

    def eval_fit(p: Perm) -> float:
        k = tuple(p)
        if k in fit_cache:
            return fit_cache[k]
        v = float(fitness_fn(p))
        fit_cache[k] = v
        return v

    fitness = np.array([eval_fit(ind) for ind in population], dtype=np.float64)
    eval_count = popsize

    visited_x: List[Perm] = [ind.copy() for ind in population]
    visited_f: List[float] = [float(v) for v in fitness.tolist()]

    best_idx = int(np.argmin(fitness))
    best_x = population[best_idx].copy()
    best_f = float(fitness[best_idx])

    while eval_count < budget:
        # parent selection
        n_parents = max(int(math.floor(popsize * float(recombination_rate)) * 2), 2)
        parents_idx = _tournament_select(
            fitness=fitness,
            rng=rng,
            tournament_size=tournament_size,
            tournament_prob=tournament_prob,
            n_select=n_parents,
        )
        parents_idx = parents_idx[rng.permutation(parents_idx.shape[0])]

        # recombine pairs -> offspring
        offspring: List[Perm] = []
        for i in range(0, parents_idx.shape[0], 2):
            p1 = population[int(parents_idx[i])]
            if i + 1 < parents_idx.shape[0]:
                p2 = population[int(parents_idx[i + 1])]
            else:
                p2 = population[int(parents_idx[0])]
            child = _cycle_crossover(p1, p2, rng)
            offspring.append(child)

        # mutate offspring
        mutated: List[Perm] = []
        for ind in offspring:
            if rng.random() < float(mutation_rate):
                mutated.append(_swap_mutation(ind, rng, n_swaps=int(n_swaps_mut)))
            else:
                mutated.append(ind)

        # evaluate offspring (respect budget)
        off_fit = []
        for ind in mutated:
            if eval_count >= budget:
                break
            v = eval_fit(ind)
            off_fit.append(v)
            visited_x.append(ind.copy())
            visited_f.append(float(v))
            eval_count += 1

        if len(off_fit) == 0:
            break

        off_fit_np = np.asarray(off_fit, dtype=np.float64)

        # survivor selection: elitist (mu+lambda) to keep popsize best
        # combine
        comb = population + mutated[: len(off_fit_np)]
        comb_fit = np.concatenate([fitness, off_fit_np], axis=0)

        order = np.argsort(comb_fit)[:popsize]
        population = [comb[i].copy() for i in order]
        fitness = comb_fit[order]

        if float(fitness[0]) < best_f:
            best_f = float(fitness[0])
            best_x = population[0].copy()

    return best_x, best_f, visited_x, visited_f


# =========================
# Max-Min exploration design
# =========================

def _design_max_min_dist_next(
    existing: List[Perm],
    creation_fn: Callable[[], Perm],
    dist_fn: Callable[[Sequence[int], Sequence[int]], float],
    retries: int,
    rng: np.random.Generator,
) -> Perm:
    """
    Create ONE new design point by maximizing the minimum distance to `existing`.
    Mimics the idea used in optimCEGO's exploration branch with `designMaxMinDist`.
    """
    if len(existing) == 0:
        return creation_fn()

    best = None
    best_score = -1.0
    retries = max(1, int(retries))

    for _ in range(retries):
        cand = creation_fn()
        # compute min distance to existing
        md = float("inf")
        for x in existing:
            d = float(dist_fn(cand, x))
            if d < md:
                md = d
                if md <= best_score:
                    break
        if md > best_score:
            best_score = md
            best = cand

    if best is None:
        best = creation_fn()
    return best


# =========================
# CEGO main
# =========================

@dataclass
class CEGOConfig:
    # R/optimCEGO default evalInit=2, budget=100, creationRetries=100
    eval_init: int = 20          # practical default (often needs >2); set to 2 to match R default
    creation_retries: int = 100

    # distance: "interchange" | "hamming" | "kendall"
    distance: str = "interchange"

    # theta grid (logspace)
    log_theta_min: float = -5.0
    log_theta_max: float = 5.0
    n_theta_grid: int = 21

    # numerical stability (nugget)
    nugget: float = 1e-10

    # optimizer for EI (EA settings)
    ea_popsize: int = 10
    ea_budget: int = 200            # surrogate (EI) evaluations per CEGO iteration
    ea_recombination_rate: float = 0.9
    ea_mutation_rate: float = 0.9
    ea_tournament_size: int = 2
    ea_tournament_prob: float = 0.9
    ea_mut_n_swaps: int = 1


def cego(
    dim: int,
    true_eval_fn: Callable[[Perm], float],
    budget: int,
    cfg: Optional[CEGOConfig] = None,
    seed: int = 0,
) -> Tuple[Perm, float, List[Perm], List[float], List[float], Dict[str, Any]]:
    """
    Run CEGO (minimization).

    Returns
    -------
    best_perm, best_val, eval_perms, eval_vals, history_best, logs

    logs:
      - "cand": list[np.ndarray]  candidates *scored* by surrogate (EI optimization search trace), per outer iter
      - "pred": list[np.ndarray]  predicted means for "cand" (same length)
      - "label": list[np.ndarray] true objective for "cand" where available, else nan (same length)
      - "ei": list[np.ndarray]    EI values for "cand" (same length)
      - "theta": list[float]      fitted theta per outer iter
      - "picked": list[np.ndarray]  picked x_next per iter (shape (dim,))
    """
    if cfg is None:
        cfg = CEGOConfig()

    if norm is None:
        raise ImportError("scipy is required for CEGO (scipy.stats.norm).")

    rng = np.random.default_rng(int(seed))
    n = int(dim)
    budget = max(1, int(budget))

    dist_fn = get_distance_fn(cfg.distance)

    def creation_fn() -> Perm:
        return rng.permutation(n).tolist()

    # evaluated dataset
    X_list: List[Perm] = []
    y_list: List[float] = []
    seen: set[Tuple[int, ...]] = set()

    eval_perms: List[Perm] = []
    eval_vals: List[float] = []
    history_best: List[float] = []

    best_val = float("inf")
    best_perm: Perm = list(range(n))

    # --- initial design ---
    n_init = min(int(cfg.eval_init), budget)
    while len(X_list) < n_init:
        p = creation_fn()
        key = tuple(p)
        if key in seen:
            continue
        v = float(true_eval_fn(p))

        X_list.append(p)
        y_list.append(v)
        seen.add(key)

        eval_perms.append(p)
        eval_vals.append(v)

        if v < best_val:
            best_val = v
            best_perm = p
        history_best.append(best_val)

    nfev = len(eval_vals)
    if nfev >= budget:
        return best_perm, float(best_val), eval_perms, eval_vals, history_best, {
            "cand": [],
            "pred": [],
            "label": [],
            "ei": [],
            "theta": [],
            "picked": [],
        }

    # logs per outer iteration
    cand_log: List[np.ndarray] = []
    pred_log: List[np.ndarray] = []
    label_log: List[np.ndarray] = []
    ei_log: List[np.ndarray] = []
    theta_log: List[float] = []
    picked_log: List[np.ndarray] = []

    while nfev < budget:
        # ---- fit kriging (theta by grid MLE) ----
        X_train = np.asarray(X_list, dtype=np.int32)
        y_train = np.asarray(y_list, dtype=np.float64)

        dist_mat = _pairwise_distance_matrix(X_train, dist_fn)
        theta = _kriging_fit_theta(
            dist_mat=dist_mat,
            y=y_train,
            log_theta_min=float(cfg.log_theta_min),
            log_theta_max=float(cfg.log_theta_max),
            n_grid=int(cfg.n_theta_grid),
            nugget=float(cfg.nugget),
            prev_theta=(theta_log[-1] if len(theta_log) > 0 else None),
        )
        theta_log.append(float(theta))

        prep = _kriging_prepare(dist_mat=dist_mat, y_train=y_train, theta=float(theta), nugget=float(cfg.nugget))

        y_min = float(np.min(y_train))

        # ---- define infill (EI) fitness: minimize -EI ----
        def fitness_fn(p: Perm) -> float:
            X_cand = np.asarray([p], dtype=np.int32)
            mu_hat, s_hat = _kriging_predict_prepared(prep=prep, X_train=X_train, dist_fn=dist_fn, theta=float(theta), X_cand=X_cand)
            ei_val = float(_ei(y_min, mu_hat, s_hat)[0])
            return -ei_val  # minimize

        # ---- optimize EI with EA ----
        xbest_ea, fbest_ea, visited_x, visited_f = _ea_optimize_infill(
            n=n,
            rng=rng,
            creation_fn=creation_fn,
            fitness_fn=fitness_fn,
            popsize=int(cfg.ea_popsize),
            budget=int(cfg.ea_budget),
            recombination_rate=float(cfg.ea_recombination_rate),
            mutation_rate=float(cfg.ea_mutation_rate),
            tournament_size=int(cfg.ea_tournament_size),
            tournament_prob=float(cfg.ea_tournament_prob),
            n_swaps_mut=int(cfg.ea_mut_n_swaps),
        )

        # visited_f are -EI (min); convert
        visited_ei = -np.asarray(visited_f, dtype=np.float64)

        # For logging predicted mean too, compute it in one batch
        X_vis = np.asarray(visited_x, dtype=np.int32)
        mu_vis, s_vis = _kriging_predict_prepared(prep=prep, X_train=X_train, dist_fn=dist_fn, theta=float(theta), X_cand=X_vis)

        # label: known y for candidates already evaluated, else nan
        label_vis = np.full((X_vis.shape[0],), np.nan, dtype=np.float64)
        for i, p in enumerate(visited_x):
            key = tuple(p)
            if key in seen:
                # find its true value (archive lookup)
                # (small archive, linear scan is fine)
                for xp, yp in zip(X_list, y_list):
                    if tuple(xp) == key:
                        label_vis[i] = float(yp)
                        break

        cand_log.append(X_vis.copy())
        pred_log.append(mu_vis.copy())
        ei_log.append(visited_ei.copy())
        label_log.append(label_vis.copy())

        # pick candidate suggested by EA
        x_next = xbest_ea

        # ---- duplicate handling & exploration ----
        if tuple(x_next) in seen:
            x_next = _design_max_min_dist_next(
                existing=X_list,
                creation_fn=creation_fn,
                dist_fn=dist_fn,
                retries=int(cfg.creation_retries),
                rng=rng,
            )
            # still might duplicate in tiny spaces; last resort force random unseen
            if tuple(x_next) in seen:
                for _ in range(int(cfg.creation_retries) * 5):
                    p = creation_fn()
                    if tuple(p) not in seen:
                        x_next = p
                        break

        picked_log.append(np.asarray(x_next, dtype=np.int32))

        # ---- expensive evaluation (exactly one per outer loop) ----
        y_next = float(true_eval_fn(x_next))
        nfev += 1

        X_list.append(x_next)
        y_list.append(y_next)
        seen.add(tuple(x_next))

        eval_perms.append(x_next)
        eval_vals.append(y_next)

        if y_next < best_val:
            best_val = y_next
            best_perm = x_next
        history_best.append(best_val)

    logs: Dict[str, Any] = {
        "cand": cand_log,
        "pred": pred_log,
        "label": label_log,
        "ei": ei_log,
        "theta": theta_log,
        "picked": picked_log,
        "distance": cfg.distance,
    }
    return best_perm, float(best_val), eval_perms, eval_vals, history_best, logs
