import numpy as np


def uborda(S, ws):
    S = np.asarray(S, dtype=int)
    ws = np.asarray(ws, dtype=float).reshape(-1)
    if S.ndim != 2:
        raise ValueError("S must be a 2D array (num_samples, n)")
    if S.shape[0] != ws.shape[0]:
        raise ValueError("len(ws) must match number of rows in S")


    B = ws @ S
    idx = np.argsort(B, kind="mergesort")
    sigma0 = np.argsort(idx, kind="mergesort")
    return sigma0.astype(int)


def find_phi(n, dmin, dmax):
    imin = 0.0
    imax = 1.0
    med = 0.5

    for _ in range(500):
        med = (imax + imin) / 2.0

        theta = -np.log(max(med, np.finfo(float).tiny))





        rnge = np.arange(1, n, dtype=float)
        exp_t = np.exp(-theta)
        first = n * exp_t / max(1.0 - exp_t, np.finfo(float).tiny)

        exp_r = np.exp(-rnge * theta)
        denom = np.maximum(1.0 - exp_r, np.finfo(float).tiny)
        second = np.sum(rnge * exp_r / denom)
        d = first - second

        if d < dmin:
            imin = med
        elif d > dmax:
            imax = med
        else:
            return med
    return med


def _v_to_ranking(v, n):
    rem = list(range(n))
    rank = np.empty(n, dtype=int)
    for i, vi in enumerate(v):
        idx = int(vi) - 1
        rank[i] = rem[idx]
        rem.pop(idx)
    return rank


def samplingMM(m, n, phi, k=None):
    if not (0.0 < phi <= 1.0):
        raise ValueError("phi must satisfy 0 < phi <= 1")
    if n <= 0 or m <= 0:
        raise ValueError("m and n must be positive")

    theta = -np.log(phi)
    theta_vec = np.full(n - 1, theta, dtype=float)

    rnge = np.arange(0, n - 1, dtype=float)
    if np.isclose(theta, 0.0):
        psi = n - rnge
    else:
        num = 1.0 - np.exp((-n + rnge) * theta_vec)
        den = 1.0 - np.exp(-theta_vec)
        psi = num / den

    vprobs = np.zeros((n, n), dtype=float)
    for j in range(1, n):
        vprobs[j - 1, 0] = 1.0 / psi[j - 1]

        for r in range(2, n - j + 1):

            vprobs[j - 1, r - 1] = np.exp(-theta_vec[j - 1] * r - 1.0) / psi[j - 1]

        row_sum = vprobs[j - 1].sum()
        if row_sum > 0:
            vprobs[j - 1] /= row_sum

    sample = np.empty((m, n), dtype=int)
    choices = np.arange(1, n + 1, dtype=int)
    for samp in range(m):
        v = np.zeros(n - 1, dtype=int)
        for i in range(n - 1):
            v[i] = np.random.choice(choices, p=vprobs[i])
        v = np.concatenate([v, np.array([1], dtype=int)])
        ranking = _v_to_ranking(v, n)
        sample[samp] = ranking

    if k is not None:
        k = np.asarray(k, dtype=int)
        sample = sample[:, k]

    return sample


def _kendall_distance(rank_a, rank_b):
    rank_a = np.asarray(rank_a, dtype=int)
    rank_b = np.asarray(rank_b, dtype=int)
    n = rank_a.size
    dist = 0
    for i in range(n - 1):
        da = rank_a[i] - rank_a[i + 1 :]
        db = rank_b[i] - rank_b[i + 1 :]
        dist += np.count_nonzero(da * db < 0)
    return int(dist)


def u_phi(S, ranking, ws):
    S = np.asarray(S, dtype=int)
    ranking = np.asarray(ranking, dtype=int).reshape(-1)
    ws = np.asarray(ws, dtype=float).reshape(-1)
    if S.ndim != 2:
        raise ValueError("S must be a 2D array")
    if S.shape[0] != ws.shape[0]:
        raise ValueError("len(ws) must match number of rows in S")
    if S.shape[1] != ranking.shape[0]:
        raise ValueError("ranking length must match S.shape[1]")

    wsum = ws.sum()
    if wsum <= 0:
        return 1.0

    dists = np.array([_kendall_distance(s, ranking) for s in S], dtype=float)
    dist_avg = float(np.dot(dists, ws) / wsum)
    return float(find_phi(S.shape[1], dist_avg, dist_avg + 1.0))
