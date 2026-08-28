from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Dict, Any
import math

import numpy as np
import mallows_kendall as mk
import pandas as pd


def binary_search_rho(
    w,
    ratio_samples_learn,
    weight_mass_learn,
    # 0 <= w_i <= 1, w is sorted increasingly,
    rho_ini=1,
    rho_end=0,
    tol=0.001,
):
    w = np.asarray(w)
    assert np.all(w >= 0.0)
    assert np.all(w <= 1.0)

    # Find rho so that top ratio of weights captures weight_mass of cumulative mass.
    pos = int(len(w) * ratio_samples_learn)
    rho_med = (rho_ini + rho_end) / 2
    if abs(rho_ini - rho_end) < 1e-20:
        return rho_med

    try:
        acum = np.cumsum(rho_med**w)
        a = acum[pos]
        b = acum[-1]
        if b < tol:
            return 1.0
        if abs(a / b - weight_mass_learn) < tol:
            return rho_med

        if a / b > weight_mass_learn:
            mid, last = rho_ini, rho_med
        else:
            mid, last = rho_med, rho_end
        return binary_search_rho(w, ratio_samples_learn, weight_mass_learn, mid, last)
    except Exception:
        print(w)
        pos = math.floor(len(w) * ratio_samples_learn) - 1
        pos = max(pos, 0)
        print(pos, len(w), ratio_samples_learn)
        rho_med = rho_ini + (rho_end - rho_ini) / 2
        acum = np.cumsum(rho_med**w)
        a = acum[pos]
        b = acum[-1]
        print(
            "binary_search_rho: "
            f"a={a} b={b} a/b={a/b} wml={weight_mass_learn} "
            f"rho_med={rho_med} rho_ini={rho_ini} rho_end={rho_end} w={w}"
        )
        raise


def get_expected_distance(iterat, n, budget):
    # Should this be Kendall max dist?
    N = (n - 1) * n / 2
    f_ini, f_end = N / 4, 1
    iter_decrease = budget - 10
    jump = (f_ini - f_end) / iter_decrease
    a = f_ini - jump * iterat
    return max(a, f_end)


def _umm_core(
    dim: int,
    f_eval: Callable[[np.ndarray], float],
    budget: int,
    m_ini: int,
    ratio_samples_learn: float,
    weight_mass_learn: float,
    distance_to_best_fn: Optional[Callable[[np.ndarray], float]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    if budget <= 0:
        raise ValueError("budget must be positive.")
    if m_ini <= 0:
        raise ValueError("m_ini must be positive.")
    if budget < m_ini:
        raise ValueError("budget must be >= m_ini for UMM.")

    n = dim
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    sample = [rng.permutation(n).astype(np.int32) for _ in range(m_ini)]
    fitnesses = [float(f_eval(perm)) for perm in sample]

    if distance_to_best_fn is None:
        res = [[np.nan, np.nan, np.nan, np.nan] for _ in sample]
    else:
        res = [[np.nan, np.nan, np.nan, float(distance_to_best_fn(perm))] for perm in sample]

    history_best: List[float] = []
    best_so_far = float("inf")
    for fx in fitnesses:
        best_so_far = min(best_so_far, fx)
        history_best.append(best_so_far)

    for m in range(budget - m_ini):
        # ===== learning step =====
        ws = np.asarray(fitnesses, dtype=np.float64).copy()
        ws = ws - ws.min()
        if ws.max() > 0:
            ws = ws / ws.max()
        else:
            ws[:] = 0

        co = ws.copy()
        co.sort()
        rho = binary_search_rho(co, ratio_samples_learn, weight_mass_learn)

        ws = rho**ws
        borda = mk.uborda(np.array(sample), ws)
        phi_estim = mk.u_phi(sample, borda, ws)

        expected_dist = get_expected_distance(m, n, budget)
        phi_sample = mk.find_phi(n, expected_dist, expected_dist + 1)

        # ===== PURE UMM sampling =====
        while True:
            perm = mk.samplingMM(1, n, phi=phi_sample, k=None)[0]
            perm = perm[borda]
            perm = np.asarray(perm, dtype=int)
            if not any(np.array_equal(perm, s) for s in sample):
                break

        # ===== evaluation =====
        sample.append(perm)
        fx = float(f_eval(perm))
        fitnesses.append(fx)
        best_so_far = min(best_so_far, fx)
        history_best.append(best_so_far)

        if distance_to_best_fn is None:
            dist_val = np.nan
        else:
            dist_val = float(distance_to_best_fn(np.asarray(borda, dtype=int)))

        res.append([rho, phi_estim, phi_sample, dist_val])

    best_idx = int(np.argmin(np.asarray(fitnesses, dtype=np.float64)))
    best_perm = np.asarray(sample[best_idx], dtype=int).tolist()
    best_val = float(fitnesses[best_idx])
    eval_perms = [np.asarray(p, dtype=int).tolist() for p in sample]
    eval_vals = [float(v) for v in fitnesses]

    return {
        "sample": sample,
        "fitnesses": fitnesses,
        "res": res,
        "best_perm": best_perm,
        "best_val": best_val,
        "eval_perms": eval_perms,
        "eval_vals": eval_vals,
        "history_best": history_best,
    }


def umm(
    dim: int,
    true_eval_fn: Callable[[List[int]], float],
    budget: int,
    seed: Optional[int] = None,
    m_ini: int = 100,
    ratio_samples_learn: float = 0.1,
    weight_mass_learn: float = 0.9,
) -> Tuple[List[int], float, List[List[int]], List[float], List[float]]:
    """
    UMM runner with FAT-RLS-compatible interface.

    Returns
    -------
    best_perm, best_val, eval_perms, eval_vals, history_best
    """
    if seed is not None:
        np.random.seed(seed)

    def f_eval(perm: np.ndarray) -> float:
        return float(true_eval_fn(np.asarray(perm, dtype=int).tolist()))

    out = _umm_core(
        dim=dim,
        f_eval=f_eval,
        budget=budget,
        m_ini=m_ini,
        ratio_samples_learn=ratio_samples_learn,
        weight_mass_learn=weight_mass_learn,
        distance_to_best_fn=None,
        seed=seed,
    )
    return out["best_perm"], out["best_val"], out["eval_perms"], out["eval_vals"], out["history_best"]


def UMM(
    instance,
    seed,
    budget,
    m_ini,
    ratio_samples_learn,
    weight_mass_learn,
    eval_ranks,
):
    """
    Backward-compatible API used by older scripts.
    """
    np.random.seed(seed)

    if eval_ranks:
        f_eval = lambda p: float(instance.fitness(np.asarray(p, dtype=int)))
    else:
        f_eval = lambda p: float(instance.fitness(np.argsort(np.asarray(p, dtype=int))))

    out = _umm_core(
        dim=int(instance.n),
        f_eval=f_eval,
        budget=int(budget),
        m_ini=int(m_ini),
        ratio_samples_learn=float(ratio_samples_learn),
        weight_mass_learn=float(weight_mass_learn),
        distance_to_best_fn=lambda p: instance.distance_to_best(np.asarray(p, dtype=int)),
        seed=seed,
    )

    df = pd.DataFrame(out["res"], columns=["rho", "phi_estim", "phi_sample", "Distance"])
    df["Fitness"] = out["fitnesses"]
    df["x"] = out["sample"]
    return df

