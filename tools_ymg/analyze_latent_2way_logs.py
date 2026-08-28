#!/usr/bin/env python3
"""
Analysis for latent_2ways logs.

This version is designed for experiments laid out as

    BASE / INSTANCE / ALGORITHM / trial_XX / result.npz

and can compare multiple algorithms on one instance.

Main outputs
------------
NEW FULL-POOL ANALYSIS: uses cand_true for every generated global/incumbent candidate.
1. True best-update contribution by source
   - Counts are computed per trial, never summed across trials in the main plot.
   - A best update means one true evaluation reduced the incumbent objective.
2. Embedding/objective relationship
   - Pairwise embedding distance vs absolute objective difference, within each
     generation only. Embeddings from retrained surrogate models are not mixed
     across generations.
   - 2-D projection of one generation's candidate embeddings, colored by
     predicted objective; true-evaluated candidates are overlaid with true values.
3. Global latent-selection diagnostics
   - overlap with pure score top-k
   - selected score ranks / rank penalty
   - embedding-diversity gain over score top-k
   - gate-distance margin and threshold-fallback rate
4. Surrogate accuracy
   - selected-origin comparison: global-origin vs incumbent
   - exact logged pool comparison: global_pool vs local_pool
   - MAE, RMSE, bias, Pearson, Spearman, Kendall tau-b
5. Additional analyses
   - evaluation allocation and update rate
   - total improvement contribution
   - final best objective by algorithm
   - global selection diagnostics vs same-generation improvement
6. FE-timing and global/local role analysis
   - exact FE positions of global/incumbent best updates
   - early/middle/late and 20%-bin contribution
   - cumulative update/improvement curves
   - stagnation-breaking updates and global-to-incumbent follow-up sequences
   - candidate-pool improvement opportunities and realized improvement

Examples
--------
python analyze_latent_2way_logs_v2.py \
    --base-input E:/results \
    --instance br17 \
    --algorithm lat2way_sage_topk lat2way_sage_latent_nms \
    --outdir E:/results_analysis

python analyze_latent_2way_logs_v2.py \
    --input E:/results/br17/*/trial_*/result.npz \
    --outdir E:/results_analysis \
    --instance br17

Notes
-----
- The optimization is assumed to be minimization.
- Prediction accuracy uses `mu`, not LCB/acquisition `pred`.
- A single-run latent-selection log can prove that NMS changed selection and
  increased embedding diversity, but it cannot prove a counterfactual objective
  gain for unselected score-top candidates. For that claim, compare against a
  separate top-k ablation algorithm.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise ImportError("pandas is required: pip install pandas") from e

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise ImportError("matplotlib is required: pip install matplotlib") from e


EPS = 1e-12
ANALYSIS_SCRIPT_VERSION = "2026-07-30-cand-true-detail-fallback-v3"


# =============================================================================
# Data model and loading
# =============================================================================

@dataclass(frozen=True)
class FileSpec:
    path: Path
    instance: Optional[str] = None
    algorithm: Optional[str] = None


@dataclass
class RunRecord:
    path: str
    instance: str
    algorithm: str
    trial: str
    run: str
    logs: Dict[str, Any]
    eval_vals: Optional[np.ndarray]
    history_best: Optional[np.ndarray]
    best_val: Optional[float]

    @property
    def total_fe(self) -> int:
        if self.eval_vals is not None:
            return int(len(self.eval_vals))
        if self.history_best is not None:
            return int(len(self.history_best))
        return 0


def _unwrap_object(x: Any) -> Any:
    if isinstance(x, np.ndarray) and x.shape == () and x.dtype == object:
        return x.item()
    return x


def _normalize_loaded(x: Any) -> Any:
    x = _unwrap_object(x)
    if isinstance(x, np.ndarray) and x.dtype == object:
        return [_unwrap_object(v) for v in x.tolist()]
    return x


def _as_1d(x: Any, dtype: Any = float) -> np.ndarray:
    if x is None:
        return np.zeros((0,), dtype=dtype)
    try:
        return np.asarray(x, dtype=dtype).reshape(-1)
    except Exception:
        return np.zeros((0,), dtype=dtype)


def _as_2d_or_none(x: Any, dtype: Any = float) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        a = np.asarray(x, dtype=dtype)
    except Exception:
        return None
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        return None
    return a


def _list_get(xs: Any, i: int, default: Any = None) -> Any:
    if xs is None:
        return default
    try:
        return xs[i] if 0 <= i < len(xs) else default
    except Exception:
        return default


def get_detail(logs: Mapping[str, Any]) -> Dict[str, Any]:
    for key in ("detail", "medium"):
        value = logs.get(key)
        if isinstance(value, dict):
            return value
    return {}


def n_generations(logs: Mapping[str, Any]) -> int:
    lengths: List[int] = []
    for key in ("label", "cand", "cand_true", "selected_mask", "emb", "mu", "archive_fx_by_gen"):
        value = logs.get(key)
        try:
            lengths.append(len(value))
        except Exception:
            pass
    detail = get_detail(logs)
    for key in ("selected_label", "top_pool_cand", "global_mu", "local_mu"):
        value = detail.get(key)
        try:
            lengths.append(len(value))
        except Exception:
            pass
    return max(lengths) if lengths else 0


def result_dict_to_logs(obj: Mapping[str, Any]) -> Dict[str, Any]:
    for key in ("logs", "log"):
        value = obj.get(key)
        if isinstance(value, dict):
            return value

    logs: Dict[str, Any] = {}
    for key in (
        "cand", "mu", "pred", "sigma", "emb", "label", "source", "cand_true",
        "selected_mask", "selected_order", "selected_eval_source",
        "archive_perm_by_gen", "archive_fx_by_gen", "archive_mu_by_gen",
        "archive_sigma_by_gen", "archive_pred_by_gen", "archive_emb_by_gen",
    ):
        if key in obj:
            logs[key] = _normalize_loaded(obj[key])

    detail = _normalize_loaded(obj.get("detail"))
    if isinstance(detail, list) and len(detail) == 1 and isinstance(detail[0], dict):
        detail = detail[0]
    if isinstance(detail, dict):
        logs["detail"] = {k: _normalize_loaded(v) for k, v in detail.items()}
    return logs


def _is_trial_name(name: str) -> bool:
    return bool(re.match(r"^(trial|seed|run)[_-]?\d+", name.lower()))


def infer_metadata(path: Path) -> Tuple[str, str, str]:
    """Best-effort inference for .../instance/algorithm/trial/result.npz."""
    parts = list(path.parts)
    trial_idx: Optional[int] = None
    for i in range(len(parts) - 2, -1, -1):
        if _is_trial_name(parts[i]):
            trial_idx = i
            break

    if trial_idx is not None:
        trial = parts[trial_idx]
        algorithm = parts[trial_idx - 1] if trial_idx >= 1 else "default"
        instance = parts[trial_idx - 2] if trial_idx >= 2 else "default"
    else:
        trial = path.parent.name or path.stem
        algorithm = path.parent.parent.name if path.parent.parent.name else "default"
        instance = path.parent.parent.parent.name if path.parent.parent.parent.name else "default"
    return instance, algorithm, trial


def load_run(spec: FileSpec) -> RunRecord:
    path = spec.path
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as npz:
            obj: Any = {k: _normalize_loaded(npz[k]) for k in npz.files}
    elif suffix == ".npy":
        obj = _normalize_loaded(np.load(path, allow_pickle=True))
    elif suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    logs: Optional[Dict[str, Any]] = None
    eval_vals: Optional[np.ndarray] = None
    history_best: Optional[np.ndarray] = None
    best_val: Optional[float] = None

    if isinstance(obj, tuple) and len(obj) >= 6:
        best_val = float(obj[1]) if obj[1] is not None else None
        eval_vals = np.asarray(obj[3], dtype=float).reshape(-1) if obj[3] is not None else None
        history_best = np.asarray(obj[4], dtype=float).reshape(-1) if obj[4] is not None else None
        logs = obj[5]
    elif isinstance(obj, dict):
        logs = result_dict_to_logs(obj)
        for key in ("archive_fx", "eval_vals", "y"):
            if key in obj:
                eval_vals = np.asarray(obj[key], dtype=float).reshape(-1)
                break
        for key in ("history", "history_best"):
            if key in obj:
                history_best = np.asarray(obj[key], dtype=float).reshape(-1)
                break
        for key in ("best_fx", "best_val"):
            if key in obj:
                try:
                    best_val = float(np.asarray(obj[key]).reshape(()))
                except Exception:
                    pass
                break
    else:
        raise ValueError(f"Could not interpret {path}")

    if not isinstance(logs, dict) or not logs:
        raise ValueError(f"No logs found in {path}")

    inferred_instance, inferred_algorithm, inferred_trial = infer_metadata(path)
    instance = spec.instance or inferred_instance
    algorithm = spec.algorithm or inferred_algorithm
    trial = inferred_trial
    run = f"{instance}::{algorithm}::{trial}"

    return RunRecord(
        path=str(path),
        instance=instance,
        algorithm=algorithm,
        trial=trial,
        run=run,
        logs=logs,
        eval_vals=eval_vals,
        history_best=history_best,
        best_val=best_val,
    )


def _expand_path(inp: str) -> List[Path]:
    p = Path(inp)
    if p.is_dir():
        result_files = sorted(p.rglob("result.npz"))
        if result_files:
            return result_files
        out: List[Path] = []
        for pat in ("*.npz", "*.npy", "*.pkl", "*.pickle"):
            out.extend(sorted(p.rglob(pat)))
        return out
    return [Path(x) for x in sorted(glob.glob(inp, recursive=True)) if Path(x).is_file()]


def discover_files(
    inputs: Optional[Sequence[str]],
    base_input: Optional[str],
    instances: Optional[Sequence[str]],
    algorithms: Optional[Sequence[str]],
) -> List[FileSpec]:
    specs: List[FileSpec] = []

    if inputs:
        for inp in inputs:
            for path in _expand_path(inp):
                specs.append(FileSpec(path=path))

    if base_input:
        base = Path(base_input)
        inst_list = list(instances or [])
        if not inst_list:
            raise ValueError("--base-input requires at least one --instance.")
        for inst in inst_list:
            inst_dir = base / inst
            if algorithms:
                for alg_pattern in algorithms:
                    matches = sorted(Path(x) for x in glob.glob(str(inst_dir / alg_pattern)))
                    if not matches and (inst_dir / alg_pattern).exists():
                        matches = [inst_dir / alg_pattern]
                    for alg_dir in matches:
                        if not alg_dir.is_dir():
                            continue
                        for path in sorted(alg_dir.rglob("result.npz")):
                            specs.append(FileSpec(path=path, instance=inst, algorithm=alg_dir.name))
            else:
                for path in sorted(inst_dir.rglob("result.npz")):
                    inferred_instance, inferred_algorithm, _ = infer_metadata(path)
                    specs.append(FileSpec(
                        path=path,
                        instance=inst or inferred_instance,
                        algorithm=inferred_algorithm,
                    ))

    # Stable de-duplication, preserving forced metadata from the first occurrence.
    seen: set[str] = set()
    out: List[FileSpec] = []
    for spec in specs:
        key = str(spec.path.resolve())
        if key in seen or not spec.path.is_file():
            continue
        seen.add(key)
        out.append(spec)
    return out


# =============================================================================
# Basic statistics
# =============================================================================

def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x, dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sx[j] == sx[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=float).reshape(-1)
    ya = np.asarray(y, dtype=float).reshape(-1)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    if len(xa) < 2 or np.std(xa) <= EPS or np.std(ya) <= EPS:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=float).reshape(-1)
    ya = np.asarray(y, dtype=float).reshape(-1)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    if len(xa) < 2:
        return float("nan")
    return pearson_corr(rankdata_average(xa), rankdata_average(ya))


def kendall_tau_b(x: Sequence[float], y: Sequence[float], max_n: int = 3000) -> float:
    """Kendall tau-b without scipy. Deterministically subsamples very large arrays."""
    xa = np.asarray(x, dtype=float).reshape(-1)
    ya = np.asarray(y, dtype=float).reshape(-1)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    n = len(xa)
    if n < 2:
        return float("nan")
    if n > max_n:
        idx = np.linspace(0, n - 1, max_n, dtype=int)
        xa, ya = xa[idx], ya[idx]
        n = len(xa)

    concordant = discordant = ties_x = ties_y = 0
    for i in range(n - 1):
        dx = np.sign(xa[i + 1:] - xa[i])
        dy = np.sign(ya[i + 1:] - ya[i])
        concordant += int(np.sum(dx * dy > 0))
        discordant += int(np.sum(dx * dy < 0))
        ties_x += int(np.sum((dx == 0) & (dy != 0)))
        ties_y += int(np.sum((dy == 0) & (dx != 0)))
    denom = math.sqrt((concordant + discordant + ties_x) *
                      (concordant + discordant + ties_y))
    return float((concordant - discordant) / denom) if denom > 0 else float("nan")


def pairwise_distances(z: Optional[np.ndarray], metric: str) -> np.ndarray:
    if z is None:
        return np.zeros((0,), dtype=float)
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        return np.zeros((0,), dtype=float)
    z = z[np.all(np.isfinite(z), axis=1)]
    if len(z) < 2:
        return np.zeros((0,), dtype=float)
    if metric == "cosine":
        zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + EPS)
        mat = 1.0 - zn @ zn.T
    elif metric == "euclidean":
        mat = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    tri = mat[np.triu_indices(len(z), k=1)]
    return tri[np.isfinite(tri)]


def pairwise_embedding_objective(
    z: np.ndarray,
    y: np.ndarray,
    metric: str,
    max_pairs: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    ok = np.isfinite(y) & np.all(np.isfinite(z), axis=1)
    z, y = z[ok], y[ok]
    n = len(y)
    if n < 2:
        return (np.zeros((0,), dtype=float),) * 3

    total = n * (n - 1) // 2
    if total <= max_pairs:
        ii, jj = np.triu_indices(n, k=1)
    else:
        rng = np.random.default_rng(seed)
        ii = rng.integers(0, n, size=max_pairs)
        jj = rng.integers(0, n - 1, size=max_pairs)
        jj = jj + (jj >= ii)

    if metric == "cosine":
        zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + EPS)
        d = 1.0 - np.sum(zn[ii] * zn[jj], axis=1)
    else:
        d = np.linalg.norm(z[ii] - z[jj], axis=1)
    dy = np.abs(y[ii] - y[jj])
    yrange = float(np.nanmax(y) - np.nanmin(y))
    dy_norm = dy / (yrange + EPS)
    valid = np.isfinite(d) & np.isfinite(dy)
    return d[valid], dy[valid], dy_norm[valid]


def _safe_rmse(err: np.ndarray) -> float:
    err = np.asarray(err, dtype=float)
    err = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan")


def normalize_source(source: Any) -> str:
    s = str(source).strip().lower()
    if s == "global":
        return "global"
    if s == "incumbent":
        return "incumbent"
    if s == "uncertainty":
        return "uncertainty"
    if s in {"fallback", "fallback_root"}:
        return "fallback"
    if not s:
        return "unknown"
    return s


# =============================================================================
# Extract true-evaluated rows and updates
# =============================================================================

def selected_rows(record: RunRecord) -> pd.DataFrame:
    logs = record.logs
    rows: List[Dict[str, Any]] = []
    seq = 0

    for gen in range(n_generations(logs)):
        label = _as_1d(_list_get(logs.get("label"), gen), float)
        if len(label) == 0:
            continue
        mask = _as_1d(_list_get(logs.get("selected_mask"), gen), bool)
        if len(mask) != len(label):
            mask = np.isfinite(label)
        order = _as_1d(_list_get(logs.get("selected_order"), gen), int)
        if len(order) != len(label):
            order = np.full(len(label), -1, dtype=int)
        src_eval = _as_1d(_list_get(logs.get("selected_eval_source"), gen), object)
        src_pool = _as_1d(_list_get(logs.get("source"), gen), object)
        mu = _as_1d(_list_get(logs.get("mu"), gen), float)
        pred = _as_1d(_list_get(logs.get("pred"), gen), float)
        sigma = _as_1d(_list_get(logs.get("sigma"), gen), float)

        idx = np.where(mask & np.isfinite(label))[0]
        if len(idx) == 0:
            idx = np.where(np.isfinite(label))[0]
        if len(idx) and np.any(order[idx] >= 0):
            idx = idx[np.argsort(order[idx], kind="stable")]

        for local_order, ci in enumerate(idx.tolist()):
            source = ""
            if ci < len(src_eval) and str(src_eval[ci]).strip():
                source = str(src_eval[ci])
            elif ci < len(src_pool):
                source = str(src_pool[ci])
            rows.append({
                "run": record.run,
                "instance": record.instance,
                "algorithm": record.algorithm,
                "trial": record.trial,
                "path": record.path,
                "generation": gen,
                "candidate_index": ci,
                "eval_order_in_generation": int(order[ci]) if ci < len(order) else local_order,
                "logged_eval_index": seq,
                "source": source,
                "source_group": normalize_source(source),
                "label": float(label[ci]),
                "mu": float(mu[ci]) if ci < len(mu) else np.nan,
                "acquisition": float(pred[ci]) if ci < len(pred) else np.nan,
                "sigma": float(sigma[ci]) if ci < len(sigma) else np.nan,
            })
            seq += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    total_fe = record.total_fe
    initial_fe = max(0, total_fe - len(df)) if total_fe > 0 else np.nan
    df["total_fe"] = total_fe if total_fe > 0 else np.nan
    df["initial_fe"] = initial_fe
    if np.isfinite(initial_fe):
        df["fe"] = int(initial_fe) + np.arange(len(df), dtype=int) + 1
    else:
        df["fe"] = np.arange(len(df), dtype=int) + 1

    before: List[float] = []
    after: List[float] = []
    improvements: List[float] = []
    updates: List[bool] = []

    use_history = (
        record.history_best is not None
        and total_fe > 0
        and len(record.history_best) >= total_fe
        and np.isfinite(initial_fe)
    )

    if use_history:
        h = np.asarray(record.history_best, dtype=float).reshape(-1)
        init = int(initial_fe)
        for j in range(len(df)):
            pos = init + j
            b = float(h[pos - 1]) if pos > 0 else float("inf")
            a = float(h[pos])
            imp = (b - a) if np.isfinite(b) else 0.0
            is_up = bool(np.isfinite(b) and a < b - EPS)
            before.append(b)
            after.append(a)
            improvements.append(max(0.0, imp))
            updates.append(is_up)
    else:
        if record.eval_vals is not None and np.isfinite(initial_fe) and int(initial_fe) > 0:
            current = float(np.nanmin(record.eval_vals[:int(initial_fe)]))
        else:
            current = float("nan")
        for y in df["label"].to_numpy(float):
            b = current
            is_up = bool(np.isfinite(current) and y < current - EPS)
            if is_up:
                current = float(y)
            before.append(b)
            after.append(current)
            improvements.append(max(0.0, b - current) if np.isfinite(b) else 0.0)
            updates.append(is_up)

    df["best_before"] = before
    df["best_after"] = after
    df["improvement"] = improvements
    df["is_best_update"] = updates
    df["error"] = df["mu"] - df["label"]
    df["abs_error"] = np.abs(df["error"])

    # Alignment sanity check against archive values, if possible.
    if record.eval_vals is not None and np.isfinite(initial_fe):
        tail = np.asarray(record.eval_vals[int(initial_fe):int(initial_fe) + len(df)], dtype=float)
        if len(tail) == len(df):
            df["archive_alignment_error"] = np.abs(tail - df["label"].to_numpy(float))
        else:
            df["archive_alignment_error"] = np.nan
    else:
        df["archive_alignment_error"] = np.nan
    return df


def update_summaries(selected: pd.DataFrame, records: Sequence[RunRecord]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()

    sources = sorted(set(selected["source_group"].astype(str)))
    run_meta = pd.DataFrame([{
        "run": r.run,
        "instance": r.instance,
        "algorithm": r.algorithm,
        "trial": r.trial,
        "total_fe": r.total_fe,
    } for r in records])

    # Include zero-count source/run combinations so means are not biased upward.
    grid = run_meta.assign(_key=1).merge(
        pd.DataFrame({"source_group": sources, "_key": 1}), on="_key"
    ).drop(columns="_key")

    grouped = (
        selected.groupby(["run", "source_group"], as_index=False)
        .agg(
            n_true_eval=("label", "size"),
            n_best_update=("is_best_update", "sum"),
            total_improvement=("improvement", "sum"),
            mean_abs_error=("abs_error", "mean"),
        )
    )
    by_run = grid.merge(grouped, on=["run", "source_group"], how="left")
    for col in ("n_true_eval", "n_best_update", "total_improvement"):
        by_run[col] = by_run[col].fillna(0)
    by_run["update_rate"] = by_run["n_best_update"] / by_run["n_true_eval"].replace(0, np.nan)
    by_run["mean_improvement_per_update"] = (
        by_run["total_improvement"] / by_run["n_best_update"].replace(0, np.nan)
    )

    overall = (
        by_run.groupby(["instance", "algorithm", "source_group"], as_index=False)
        .agg(
            n_trials=("run", "nunique"),
            mean_true_eval=("n_true_eval", "mean"),
            median_true_eval=("n_true_eval", "median"),
            total_true_eval=("n_true_eval", "sum"),
            mean_best_update=("n_best_update", "mean"),
            median_best_update=("n_best_update", "median"),
            min_best_update=("n_best_update", "min"),
            max_best_update=("n_best_update", "max"),
            total_best_update=("n_best_update", "sum"),
            mean_update_rate=("update_rate", "mean"),
            total_improvement=("total_improvement", "sum"),
            mean_total_improvement=("total_improvement", "mean"),
        )
    )
    overall["pooled_update_rate"] = (
        overall["total_best_update"] / overall["total_true_eval"].replace(0, np.nan)
    )
    return by_run, overall


# =============================================================================
# Embedding analyses
# =============================================================================

def _selected_embedding_for_generation(record: RunRecord, gen: int) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    detail = get_detail(record.logs)
    z = _as_2d_or_none(_list_get(detail.get("selected_emb"), gen), float)
    y = _as_1d(_list_get(detail.get("selected_label"), gen), float)
    src = _as_1d(_list_get(detail.get("selected_source"), gen), object)
    if z is not None and len(y) == len(z):
        return z, y, src

    # Fallback to aligned top-level logs.
    emb = _as_2d_or_none(_list_get(record.logs.get("emb"), gen), float)
    label = _as_1d(_list_get(record.logs.get("label"), gen), float)
    mask = _as_1d(_list_get(record.logs.get("selected_mask"), gen), bool)
    source = _as_1d(_list_get(record.logs.get("selected_eval_source"), gen), object)
    if emb is None or len(label) != len(emb):
        return None, np.zeros((0,), dtype=float), np.zeros((0,), dtype=object)
    if len(mask) != len(label):
        mask = np.isfinite(label)
    idx = np.where(mask & np.isfinite(label))[0]
    return emb[idx], label[idx], source[idx] if len(source) == len(label) else np.array([], dtype=object)


def embedding_objective_pairs(
    records: Sequence[RunRecord],
    metric: str,
    max_pairs_per_generation: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pair_frames: List[pd.DataFrame] = []
    summaries: List[Dict[str, Any]] = []

    for rec_i, rec in enumerate(records):
        run_dist: List[np.ndarray] = []
        run_diff: List[np.ndarray] = []
        run_diff_norm: List[np.ndarray] = []
        used_generations = 0
        for gen in range(n_generations(rec.logs)):
            z, y, _ = _selected_embedding_for_generation(rec, gen)
            if z is None or len(y) < 2:
                continue
            d, dy, dyn = pairwise_embedding_objective(
                z, y, metric, max_pairs_per_generation, seed=rec_i * 100000 + gen
            )
            if len(d) == 0:
                continue
            used_generations += 1
            run_dist.append(d)
            run_diff.append(dy)
            run_diff_norm.append(dyn)
            pair_frames.append(pd.DataFrame({
                "run": rec.run,
                "instance": rec.instance,
                "algorithm": rec.algorithm,
                "trial": rec.trial,
                "generation": gen,
                "embedding_distance": d,
                "objective_abs_diff": dy,
                "objective_abs_diff_within_generation_normalized": dyn,
            }))
        if run_dist:
            d_all = np.concatenate(run_dist)
            dy_all = np.concatenate(run_diff)
            dyn_all = np.concatenate(run_diff_norm)
            summaries.append({
                "run": rec.run,
                "instance": rec.instance,
                "algorithm": rec.algorithm,
                "trial": rec.trial,
                "n_generations_used": used_generations,
                "n_pairs": len(d_all),
                "pearson_raw": pearson_corr(d_all, dy_all),
                "spearman_raw": spearman_corr(d_all, dy_all),
                "pearson_generation_normalized": pearson_corr(d_all, dyn_all),
                "spearman_generation_normalized": spearman_corr(d_all, dyn_all),
            })

    pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    return pairs, summary


def _project_2d(z: np.ndarray, method: str, seed: int) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if len(z) == 0:
        return np.zeros((0, 2), dtype=float)
    if len(z) == 1:
        return np.zeros((1, 2), dtype=float)

    # Standardize dimensions before nonlinear projection.
    z_std = (z - np.nanmean(z, axis=0, keepdims=True)) / (np.nanstd(z, axis=0, keepdims=True) + EPS)
    if method == "pca":
        centered = z_std - np.mean(z_std, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        comp = centered @ vt[: min(2, len(vt))].T
        if comp.shape[1] == 1:
            comp = np.column_stack([comp[:, 0], np.zeros(len(comp))])
        return comp[:, :2]
    if method == "umap":
        try:
            import umap  # type: ignore
            n_neighbors = max(2, min(15, len(z_std) - 1))
            return np.asarray(umap.UMAP(
                n_components=2, n_neighbors=n_neighbors, random_state=seed
            ).fit_transform(z_std), dtype=float)
        except Exception as e:
            warnings.warn(
                "UMAP could not run because umap-learn/scikit-learn are unavailable "
                "or version-incompatible. Falling back to PCA. "
                f"Original error: {e}",
                RuntimeWarning,
            )
            centered = z_std - np.mean(z_std, axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            comp = centered @ vt[: min(2, len(vt))].T
            if comp.shape[1] == 1:
                comp = np.column_stack([comp[:, 0], np.zeros(len(comp))])
            return comp[:, :2]
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except Exception as e:
            raise ImportError("t-SNE requested. Install scikit-learn.") from e
        perplexity = max(2.0, min(30.0, (len(z_std) - 1) / 3.0))
        return np.asarray(TSNE(
            n_components=2, perplexity=perplexity, init="pca",
            learning_rate="auto", random_state=seed
        ).fit_transform(z_std), dtype=float)
    raise ValueError(f"Unknown projection method: {method}")


def parse_generation_specs(specs: Sequence[str], ng: int) -> List[int]:
    out: List[int] = []
    for token0 in specs:
        for token in str(token0).split(","):
            token = token.strip().lower()
            if not token:
                continue
            if token == "first":
                idx = 0
            elif token in {"middle", "mid"}:
                idx = max(0, (ng - 1) // 2)
            elif token == "last":
                idx = max(0, ng - 1)
            else:
                idx = int(token)
                if idx < 0:
                    idx = ng + idx
            if 0 <= idx < ng and idx not in out:
                out.append(idx)
    return out



# =============================================================================
# Full candidate-pool and archive embedding analyses
# =============================================================================

def _candidate_generation_arrays(record: RunRecord, gen: int) -> Optional[Dict[str, np.ndarray]]:
    """Return length-aligned arrays for every candidate in one generation.

    ``cand_true`` and archive snapshots may be saved either as top-level NPZ
    fields or inside ``detail``.  Optional/missing arrays must never cause a
    broadcasting error against the candidate pool.
    """
    logs = record.logs
    detail = get_detail(logs)

    cand = _as_2d_or_none(_list_get(logs.get("cand"), gen), int)
    mu = _as_1d(_list_get(logs.get("mu"), gen), float)
    pred = _as_1d(_list_get(logs.get("pred"), gen), float)
    sigma = _as_1d(_list_get(logs.get("sigma"), gen), float)
    emb = _as_2d_or_none(_list_get(logs.get("emb"), gen), float)
    source = _as_1d(_list_get(logs.get("source"), gen), object)

    # Full-pool true values are logged in latent_2way as cand_true.  Some
    # runners save only ``detail`` rather than exporting cand_true as a
    # top-level NPZ field, so support both layouts.
    truth_raw = _list_get(logs.get("cand_true"), gen)
    if truth_raw is None:
        truth_raw = _list_get(detail.get("cand_true"), gen)
    truth = _as_1d(truth_raw, float)

    selected = _as_1d(_list_get(logs.get("selected_mask"), gen), bool)
    eval_source = _as_1d(_list_get(logs.get("selected_eval_source"), gen), object)

    # Determine pool size from actual candidate-defining arrays.  Do not let
    # optional fields such as pred/sigma/source shrink the pool to zero.
    core_lengths: List[int] = []
    if cand is not None and len(cand):
        core_lengths.append(len(cand))
    if len(mu):
        core_lengths.append(len(mu))
    if emb is not None and len(emb):
        core_lengths.append(len(emb))
    if len(truth):
        core_lengths.append(len(truth))
    if not core_lengths:
        return None

    n = min(core_lengths)
    if n <= 0:
        return None

    def pad_1d(arr: np.ndarray, fill: Any, dtype: Any) -> np.ndarray:
        arr = np.asarray(arr, dtype=dtype).reshape(-1)
        if len(arr) >= n:
            return arr[:n].copy()
        out = np.full(n, fill, dtype=dtype)
        if len(arr):
            out[:len(arr)] = arr
        return out

    if cand is None or len(cand) < n:
        width = int(cand.shape[1]) if cand is not None and cand.ndim == 2 else 0
        cand_out = np.zeros((n, width), dtype=int)
        if cand is not None and len(cand):
            cand_out[:len(cand)] = cand
        cand = cand_out
    else:
        cand = np.asarray(cand[:n], dtype=int)

    if emb is None or len(emb) < n:
        width = int(emb.shape[1]) if emb is not None and emb.ndim == 2 else 0
        emb_out = np.full((n, width), np.nan, dtype=float)
        if emb is not None and len(emb):
            emb_out[:len(emb)] = emb
        emb = emb_out
    else:
        emb = np.asarray(emb[:n], dtype=float)

    mu = pad_1d(mu, np.nan, float)
    pred = pad_1d(pred, np.nan, float)
    sigma = pad_1d(sigma, np.nan, float)
    truth = pad_1d(truth, np.nan, float)
    source = pad_1d(source, "unknown", object)
    selected = pad_1d(selected, False, bool)
    eval_source = pad_1d(eval_source, "", object)

    if not np.any(np.isfinite(truth)):
        warnings.warn(
            f"Full candidate true values are unavailable for {record.run}, generation {gen}. "
            "Save logs['cand_true'] as a top-level NPZ field or keep detail['cand_true']. "
            "Full-pool accuracy rows for this generation will contain no valid samples.",
            RuntimeWarning,
        )

    return {
        "cand": cand,
        "mu": mu,
        "pred": pred,
        "sigma": sigma,
        "emb": emb,
        "source": source,
        "truth": truth,
        "selected": selected,
        "eval_source": eval_source,
    }


def _archive_generation_arrays(record: RunRecord, gen: int) -> Optional[Dict[str, np.ndarray]]:
    """Return the archive snapshot embedded/predicted by that generation's surrogate."""
    logs = record.logs
    detail = get_detail(logs)

    def first(keys: Sequence[str]) -> Any:
        for key in keys:
            value = _list_get(logs.get(key), gen)
            if value is not None:
                return value
            value = _list_get(detail.get(key), gen)
            if value is not None:
                return value
        return None

    perm = _as_2d_or_none(first(("archive_perm_by_gen", "archive_perm")), int)
    truth = _as_1d(first(("archive_fx_by_gen", "archive_fx")), float)
    mu = _as_1d(first(("archive_mu_by_gen", "archive_mu")), float)
    pred = _as_1d(first(("archive_pred_by_gen", "archive_pred")), float)
    sigma = _as_1d(first(("archive_sigma_by_gen", "archive_sigma")), float)
    emb = _as_2d_or_none(first(("archive_emb_by_gen", "archive_emb")), float)

    lengths = [len(x) for x in (truth, mu, pred, sigma) if len(x)]
    if perm is not None:
        lengths.append(len(perm))
    if emb is not None:
        lengths.append(len(emb))
    if not lengths:
        return None
    n = min(lengths)
    if n <= 0:
        return None
    if perm is None or len(perm) < n:
        perm = np.zeros((n, 0), dtype=int)
    if emb is None or len(emb) < n:
        emb = np.full((n, 0), np.nan, dtype=float)
    return {
        "perm": np.asarray(perm[:n]),
        "truth": np.asarray(truth[:n], dtype=float),
        "mu": np.asarray(mu[:n], dtype=float),
        "pred": np.asarray(pred[:n], dtype=float),
        "sigma": np.asarray(sigma[:n], dtype=float),
        "emb": np.asarray(emb[:n], dtype=float),
    }


def _pool_mask(source: np.ndarray, scope: str) -> np.ndarray:
    s = np.asarray([str(v).strip().lower() for v in source], dtype=object)
    if scope == "global":
        return s == "global_pool"
    if scope == "incumbent":
        return s == "incumbent"
    if scope == "all_candidates":
        return np.ones(len(s), dtype=bool)
    return s == scope.lower()


def _prediction_metric_dict(y: np.ndarray, p: np.ndarray, top_fraction: float = 0.10) -> Dict[str, float]:
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    n = len(y)
    if n == 0:
        return {
            "n": 0, "pearson": np.nan, "spearman": np.nan, "kendall_tau_b": np.nan,
            "mae": np.nan, "rmse": np.nan, "bias": np.nan, "nrmse_by_std": np.nan,
            "nrmse_by_range": np.nan, "top10_overlap": np.nan,
            "true_best_pred_rank": np.nan, "pred_best_true_rank": np.nan,
            "pred_best_regret": np.nan,
        }
    err = p - y
    ystd = float(np.std(y))
    yrange = float(np.max(y) - np.min(y))
    q = max(1, min(n, int(math.ceil(n * float(top_fraction)))))
    pred_order = np.argsort(p, kind="stable")
    true_order = np.argsort(y, kind="stable")
    overlap = len(set(pred_order[:q].tolist()) & set(true_order[:q].tolist())) / q
    true_best_idx = int(true_order[0])
    pred_best_idx = int(pred_order[0])
    pred_rank_of_true_best = int(np.where(pred_order == true_best_idx)[0][0]) + 1
    true_rank_of_pred_best = int(np.where(true_order == pred_best_idx)[0][0]) + 1
    return {
        "n": int(n),
        "pearson": pearson_corr(p, y),
        "spearman": spearman_corr(p, y),
        "kendall_tau_b": kendall_tau_b(p, y),
        "mae": float(np.mean(np.abs(err))),
        "rmse": _safe_rmse(err),
        "bias": float(np.mean(err)),
        "nrmse_by_std": _safe_rmse(err) / (ystd + EPS),
        "nrmse_by_range": _safe_rmse(err) / (yrange + EPS),
        "top10_overlap": float(overlap),
        "true_best_pred_rank": float(pred_rank_of_true_best),
        "pred_best_true_rank": float(true_rank_of_pred_best),
        "pred_best_regret": float(y[pred_best_idx] - np.min(y)),
    }


def candidate_pool_accuracy_by_generation(records: Sequence[RunRecord]) -> pd.DataFrame:
    """Accuracy on every true-labelled candidate, separated into global and incumbent pools."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        for gen in range(n_generations(rec.logs)):
            data = _candidate_generation_arrays(rec, gen)
            if data is None:
                continue
            for scope in ("global", "incumbent"):
                mask = _pool_mask(data["source"], scope)
                mask &= np.isfinite(data["truth"]) & np.isfinite(data["mu"])
                metrics = _prediction_metric_dict(data["truth"][mask], data["mu"][mask])
                rows.append({
                    "run": rec.run,
                    "instance": rec.instance,
                    "algorithm": rec.algorithm,
                    "trial": rec.trial,
                    "generation": gen,
                    "scope": scope,
                    **metrics,
                })
    return pd.DataFrame(rows)


def summarize_candidate_pool_accuracy(by_gen: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if by_gen.empty:
        return pd.DataFrame(), pd.DataFrame()
    numeric = [
        "n", "pearson", "spearman", "kendall_tau_b", "mae", "rmse", "bias",
        "nrmse_by_std", "nrmse_by_range", "top10_overlap",
        "true_best_pred_rank", "pred_best_true_rank", "pred_best_regret",
    ]
    by_run = (
        by_gen.groupby(["run", "instance", "algorithm", "trial", "scope"], as_index=False)[numeric]
        .mean(numeric_only=True)
    )
    summary = (
        by_run.groupby(["instance", "algorithm", "scope"], as_index=False)
        .agg(
            n_trials=("run", "nunique"),
            mean_candidates_per_generation=("n", "mean"),
            mean_pearson=("pearson", "mean"),
            mean_spearman=("spearman", "mean"),
            mean_kendall_tau_b=("kendall_tau_b", "mean"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_top10_overlap=("top10_overlap", "mean"),
            mean_true_best_pred_rank=("true_best_pred_rank", "mean"),
            mean_pred_best_true_rank=("pred_best_true_rank", "mean"),
            mean_pred_best_regret=("pred_best_regret", "mean"),
        )
    )
    return by_run, summary


def _pairwise_relation(z: np.ndarray, values: np.ndarray, metric: str, max_pairs: int, seed: int) -> Dict[str, float]:
    if z.ndim != 2 or z.shape[1] == 0 or len(z) < 2:
        return {"pearson": np.nan, "spearman": np.nan, "n_pairs": 0}
    d, dv, _ = pairwise_embedding_objective(z, values, metric, max_pairs, seed)
    return {
        "pearson": pearson_corr(d, dv),
        "spearman": spearman_corr(d, dv),
        "n_pairs": int(len(d)),
    }


def embedding_structure_by_generation(
    records: Sequence[RunRecord], metric: str, max_pairs_per_generation: int,
) -> pd.DataFrame:
    """How embedding distance relates to true and predicted objective differences."""
    rows: List[Dict[str, Any]] = []
    for ri, rec in enumerate(records):
        for gen in range(n_generations(rec.logs)):
            cand = _candidate_generation_arrays(rec, gen)
            arc = _archive_generation_arrays(rec, gen)
            datasets: List[Tuple[str, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]] = []
            if arc is not None:
                datasets.append(("archive", arc["emb"], arc["truth"], arc["mu"]))
            if cand is not None:
                for scope in ("global", "incumbent", "all_candidates"):
                    m = _pool_mask(cand["source"], scope)
                    datasets.append((scope, cand["emb"][m], cand["truth"][m], cand["mu"][m]))

            for scope, z0, y0, p0 in datasets:
                if z0 is None or y0 is None or p0 is None:
                    continue
                z = np.asarray(z0, dtype=float)
                y = np.asarray(y0, dtype=float).reshape(-1)
                p = np.asarray(p0, dtype=float).reshape(-1)
                if z.ndim != 2 or len(z) != len(y) or len(y) != len(p):
                    continue
                ok = np.all(np.isfinite(z), axis=1) & np.isfinite(y) & np.isfinite(p)
                z, y, p = z[ok], y[ok], p[ok]
                true_rel = _pairwise_relation(z, y, metric, max_pairs_per_generation, ri * 100000 + gen)
                pred_rel = _pairwise_relation(z, p, metric, max_pairs_per_generation, ri * 100000 + gen + 1)
                pdist = pairwise_distances(z, metric)
                pm = _prediction_metric_dict(y, p)
                rows.append({
                    "run": rec.run,
                    "instance": rec.instance,
                    "algorithm": rec.algorithm,
                    "trial": rec.trial,
                    "generation": gen,
                    "scope": scope,
                    "n": len(y),
                    "mean_pairwise_embedding_distance": float(np.mean(pdist)) if len(pdist) else np.nan,
                    "median_pairwise_embedding_distance": float(np.median(pdist)) if len(pdist) else np.nan,
                    "min_pairwise_embedding_distance": float(np.min(pdist)) if len(pdist) else np.nan,
                    "pearson_embedding_vs_true_abs_diff": true_rel["pearson"],
                    "spearman_embedding_vs_true_abs_diff": true_rel["spearman"],
                    "pearson_embedding_vs_pred_abs_diff": pred_rel["pearson"],
                    "spearman_embedding_vs_pred_abs_diff": pred_rel["spearman"],
                    "prediction_true_pearson": pm["pearson"],
                    "prediction_true_spearman": pm["spearman"],
                    "prediction_true_kendall_tau_b": pm["kendall_tau_b"],
                    "prediction_mae": pm["mae"],
                    "prediction_rmse": pm["rmse"],
                })
    return pd.DataFrame(rows)


def _cross_distance_matrix(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if metric == "cosine":
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + EPS)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + EPS)
        return 1.0 - an @ bn.T
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def archive_candidate_relation_by_generation(records: Sequence[RunRecord], metric: str) -> pd.DataFrame:
    """Novelty/location of global and incumbent pools relative to the current archive."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        for gen in range(n_generations(rec.logs)):
            cand = _candidate_generation_arrays(rec, gen)
            arc = _archive_generation_arrays(rec, gen)
            if cand is None or arc is None:
                continue
            za = np.asarray(arc["emb"], dtype=float)
            ok_a = np.all(np.isfinite(za), axis=1)
            za = za[ok_a]
            if len(za) < 2 or za.shape[1] == 0:
                continue
            daa = _cross_distance_matrix(za, za, metric)
            np.fill_diagonal(daa, np.inf)
            archive_nn = np.min(daa, axis=1)
            archive_baseline = float(np.median(archive_nn[np.isfinite(archive_nn)]))

            for scope in ("global", "incumbent"):
                m = _pool_mask(cand["source"], scope)
                zc = np.asarray(cand["emb"][m], dtype=float)
                yc = np.asarray(cand["truth"][m], dtype=float)
                pc = np.asarray(cand["mu"][m], dtype=float)
                ok_c = np.all(np.isfinite(zc), axis=1) & np.isfinite(yc) & np.isfinite(pc)
                zc, yc, pc = zc[ok_c], yc[ok_c], pc[ok_c]
                if len(zc) == 0 or zc.shape[1] == 0:
                    continue
                dca = _cross_distance_matrix(zc, za, metric)
                nearest = np.min(dca, axis=1)
                centroid_distance = float(_cross_distance_matrix(
                    np.mean(zc, axis=0, keepdims=True),
                    np.mean(za, axis=0, keepdims=True), metric,
                )[0, 0])
                rows.append({
                    "run": rec.run,
                    "instance": rec.instance,
                    "algorithm": rec.algorithm,
                    "trial": rec.trial,
                    "generation": gen,
                    "scope": scope,
                    "n_archive": len(za),
                    "n_candidates": len(zc),
                    "archive_nn_median": archive_baseline,
                    "candidate_to_archive_nn_mean": float(np.mean(nearest)),
                    "candidate_to_archive_nn_median": float(np.median(nearest)),
                    "candidate_to_archive_nn_min": float(np.min(nearest)),
                    "candidate_to_archive_nn_max": float(np.max(nearest)),
                    "median_novelty_ratio_to_archive": float(np.median(nearest) / (archive_baseline + EPS)),
                    "fraction_farther_than_archive_median": float(np.mean(nearest > archive_baseline)),
                    "centroid_distance_to_archive": centroid_distance,
                    "candidate_true_best": float(np.min(yc)),
                    "candidate_true_mean": float(np.mean(yc)),
                    "candidate_pred_mean": float(np.mean(pc)),
                })
    return pd.DataFrame(rows)


def _perm_key(row: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(v) for v in np.asarray(row).reshape(-1))


def _knn_overlap(z1: np.ndarray, z2: np.ndarray, metric: str, k: int = 5) -> float:
    n = len(z1)
    if n < 3:
        return np.nan
    k = max(1, min(k, n - 1))
    d1 = _cross_distance_matrix(z1, z1, metric)
    d2 = _cross_distance_matrix(z2, z2, metric)
    np.fill_diagonal(d1, np.inf)
    np.fill_diagonal(d2, np.inf)
    nn1 = np.argsort(d1, axis=1)[:, :k]
    nn2 = np.argsort(d2, axis=1)[:, :k]
    return float(np.mean([len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(nn1, nn2)]))


def _procrustes_residual(z1: np.ndarray, z2: np.ndarray) -> float:
    if z1.shape != z2.shape or len(z1) < 2:
        return np.nan
    a = z1 - np.mean(z1, axis=0, keepdims=True)
    b = z2 - np.mean(z2, axis=0, keepdims=True)
    a = a / (np.linalg.norm(a) + EPS)
    b = b / (np.linalg.norm(b) + EPS)
    try:
        u, _, vt = np.linalg.svd(a.T @ b, full_matrices=False)
        r = u @ vt
        return float(np.mean(np.linalg.norm(a @ r - b, axis=1)))
    except Exception:
        return np.nan


def archive_embedding_updates(records: Sequence[RunRecord], metric: str) -> pd.DataFrame:
    """Stability/change of common archive solutions between consecutive generations."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        previous: Optional[Dict[str, np.ndarray]] = None
        previous_gen: Optional[int] = None
        for gen in range(n_generations(rec.logs)):
            current = _archive_generation_arrays(rec, gen)
            if current is None:
                continue
            if previous is not None and previous_gen is not None:
                map_prev = {_perm_key(p): i for i, p in enumerate(previous["perm"])}
                map_cur = {_perm_key(p): i for i, p in enumerate(current["perm"])}
                common = sorted(set(map_prev) & set(map_cur))
                if len(common) >= 2:
                    ip = np.asarray([map_prev[k] for k in common], dtype=int)
                    ic = np.asarray([map_cur[k] for k in common], dtype=int)
                    z1 = np.asarray(previous["emb"][ip], dtype=float)
                    z2 = np.asarray(current["emb"][ic], dtype=float)
                    p1 = np.asarray(previous["mu"][ip], dtype=float)
                    p2 = np.asarray(current["mu"][ic], dtype=float)
                    ok = np.all(np.isfinite(z1), axis=1) & np.all(np.isfinite(z2), axis=1) & np.isfinite(p1) & np.isfinite(p2)
                    z1, z2, p1, p2 = z1[ok], z2[ok], p1[ok], p2[ok]
                    d1 = pairwise_distances(z1, metric)
                    d2 = pairwise_distances(z2, metric)
                    rows.append({
                        "run": rec.run,
                        "instance": rec.instance,
                        "algorithm": rec.algorithm,
                        "trial": rec.trial,
                        "generation_from": previous_gen,
                        "generation_to": gen,
                        "n_common_archive": len(z1),
                        "embedding_distance_matrix_pearson": pearson_corr(d1, d2),
                        "embedding_distance_matrix_spearman": spearman_corr(d1, d2),
                        "knn5_overlap": _knn_overlap(z1, z2, metric, 5),
                        "procrustes_mean_residual": _procrustes_residual(z1, z2),
                        "prediction_pearson_across_generations": pearson_corr(p1, p2),
                        "prediction_spearman_across_generations": spearman_corr(p1, p2),
                        "mean_abs_prediction_change": float(np.mean(np.abs(p2 - p1))) if len(p1) else np.nan,
                    })
            previous = current
            previous_gen = gen
    return pd.DataFrame(rows)

# =============================================================================
# Global latent-selection diagnostics
# =============================================================================

def global_selection_rows(records: Sequence[RunRecord], metric: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        detail = get_detail(rec.logs)
        ng = n_generations(rec.logs)
        for gen in range(ng):
            selected_idx = _as_1d(_list_get(detail.get("global_selected_idx"), gen), int)
            score_top_idx = _as_1d(_list_get(detail.get("global_score_top_idx"), gen), int)
            score_rank = _as_1d(_list_get(detail.get("global_selected_score_rank"), gen), float)
            rank_delta = _as_1d(_list_get(detail.get("global_selected_rank_delta"), gen), float)
            gate = _as_1d(_list_get(detail.get("global_selected_gate_dist"), gen), float)
            min_dist_arr = _as_1d(_list_get(detail.get("global_min_dist"), gen), float)
            d_inc = _as_1d(_list_get(detail.get("global_selected_dist_to_incumbent"), gen), float)
            d_prev = _as_1d(_list_get(detail.get("global_selected_dist_to_prev"), gen), float)
            d_arc = _as_1d(_list_get(detail.get("global_selected_dist_to_archive"), gen), float)
            threshold = float(min_dist_arr[0]) if len(min_dist_arr) else np.nan
            k = len(selected_idx)
            if k == 0 and len(score_rank) == 0:
                continue

            selected_set = set(int(x) for x in selected_idx.tolist())
            top_set = set(int(x) for x in score_top_idx[:k].tolist()) if k else set()
            overlap_count = len(selected_set & top_set)
            overlap_ratio = overlap_count / k if k else np.nan

            top_emb = _as_2d_or_none(_list_get(detail.get("top_pool_emb"), gen), float)
            top_cand = _as_2d_or_none(_list_get(detail.get("top_pool_cand"), gen), int)
            global_cand = _as_2d_or_none(_list_get(detail.get("global_cand"), gen), int)
            root_mask = _as_1d(_list_get(detail.get("top_pool_root_selected"), gen), bool)
            selected_div = score_top_div = np.zeros((0,), dtype=float)
            if top_emb is not None:
                selected_positions: List[int] = []
                # top_pool_root_selected includes uncertainty/fallback roots too.  Prefer
                # exact permutation matching for the global-selected indices.
                if (top_cand is not None and global_cand is not None
                        and len(top_cand) == len(top_emb) and len(selected_idx)):
                    top_lookup = {tuple(int(v) for v in row): i for i, row in enumerate(top_cand)}
                    for gi in selected_idx.tolist():
                        if 0 <= int(gi) < len(global_cand):
                            pos = top_lookup.get(tuple(int(v) for v in global_cand[int(gi)]))
                            if pos is not None:
                                selected_positions.append(int(pos))
                elif len(root_mask) == len(top_emb):
                    selected_positions = np.where(root_mask)[0].tolist()[:k]

                if selected_positions:
                    selected_emb = top_emb[np.asarray(selected_positions, dtype=int)]
                    selected_div = pairwise_distances(selected_emb, metric)
                # top_pool starts with score order; selected outside the slice are appended.
                score_top_emb = top_emb[: min(k, len(top_emb))]
                score_top_div = pairwise_distances(score_top_emb, metric)

            selected_mean_dist = float(np.mean(selected_div)) if len(selected_div) else np.nan
            selected_min_dist = float(np.min(selected_div)) if len(selected_div) else np.nan
            top_mean_dist = float(np.mean(score_top_div)) if len(score_top_div) else np.nan
            top_min_dist = float(np.min(score_top_div)) if len(score_top_div) else np.nan
            finite_gate = gate[np.isfinite(gate)]
            gate_margin = finite_gate - threshold if np.isfinite(threshold) else np.zeros((0,), dtype=float)

            rows.append({
                "run": rec.run,
                "instance": rec.instance,
                "algorithm": rec.algorithm,
                "trial": rec.trial,
                "generation": gen,
                "k_global": k,
                "score_topk_overlap_count": overlap_count,
                "score_topk_overlap_ratio": overlap_ratio,
                "fraction_selected_outside_score_topk": 1.0 - overlap_ratio if np.isfinite(overlap_ratio) else np.nan,
                "latent_changed_selection": bool(k > 0 and overlap_count < k),
                "mean_selected_score_rank_0based": float(np.nanmean(score_rank)) if len(score_rank) else np.nan,
                "max_selected_score_rank_0based": float(np.nanmax(score_rank)) if len(score_rank) else np.nan,
                "mean_rank_delta": float(np.nanmean(rank_delta)) if len(rank_delta) else np.nan,
                "max_rank_delta": float(np.nanmax(rank_delta)) if len(rank_delta) else np.nan,
                "latent_min_dist_threshold": threshold,
                "mean_gate_distance": float(np.mean(finite_gate)) if len(finite_gate) else np.nan,
                "min_gate_distance": float(np.min(finite_gate)) if len(finite_gate) else np.nan,
                "mean_gate_margin": float(np.mean(gate_margin)) if len(gate_margin) else np.nan,
                "fraction_selected_below_threshold": float(np.mean(gate_margin < -EPS)) if len(gate_margin) else np.nan,
                "mean_distance_to_incumbent": float(np.nanmean(d_inc)) if len(d_inc) else np.nan,
                "mean_distance_to_previous_selected": float(np.nanmean(d_prev[np.isfinite(d_prev)])) if np.any(np.isfinite(d_prev)) else np.nan,
                "mean_distance_to_archive": float(np.nanmean(d_arc)) if np.any(np.isfinite(d_arc)) else np.nan,
                "selected_mean_pairwise_distance": selected_mean_dist,
                "selected_min_pairwise_distance": selected_min_dist,
                "score_topk_mean_pairwise_distance": top_mean_dist,
                "score_topk_min_pairwise_distance": top_min_dist,
                "mean_pairwise_diversity_gain": selected_mean_dist - top_mean_dist,
                "min_pairwise_diversity_gain": selected_min_dist - top_min_dist,
            })
    return pd.DataFrame(rows)


def merge_global_outcomes(global_diag: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if global_diag.empty:
        return global_diag
    if selected.empty:
        out = global_diag.copy()
        out["global_true_eval"] = 0
        out["global_best_update"] = 0
        out["global_total_improvement"] = 0.0
        return out

    outcome = (
        selected[selected["source_group"] == "global"]
        .groupby(["run", "generation"], as_index=False)
        .agg(
            global_true_eval=("label", "size"),
            global_best_update=("is_best_update", "sum"),
            global_total_improvement=("improvement", "sum"),
            global_best_label=("label", "min"),
        )
    )
    out = global_diag.merge(outcome, on=["run", "generation"], how="left")
    for col in ("global_true_eval", "global_best_update", "global_total_improvement"):
        out[col] = out[col].fillna(0)
    return out


# =============================================================================
# Surrogate accuracy
# =============================================================================

def surrogate_prediction_rows(records: Sequence[RunRecord], selected: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    # Definition 1: final true-evaluated candidates by search origin.
    if not selected.empty:
        origin = selected[np.isfinite(selected["mu"]) & np.isfinite(selected["label"])].copy()
        origin["definition"] = "selected_origin"
        origin["scope"] = origin["source_group"]
        frames.append(origin[[
            "run", "instance", "algorithm", "trial", "generation", "definition", "scope",
            "source", "label", "mu", "sigma", "is_best_update", "improvement",
        ]])

    # Definition 2: exact arrays logged for the global/local candidate pools.
    pool_rows: List[Dict[str, Any]] = []
    for rec in records:
        detail = get_detail(rec.logs)
        for gen in range(n_generations(rec.logs)):
            for scope, prefix in (("global_pool", "global"), ("local_pool", "local")):
                mu = _as_1d(_list_get(detail.get(f"{prefix}_mu"), gen), float)
                y = _as_1d(_list_get(detail.get(f"{prefix}_label"), gen), float)
                sigma = _as_1d(_list_get(detail.get(f"{prefix}_sigma"), gen), float)
                source = _as_1d(_list_get(detail.get(f"{prefix}_source"), gen), object)
                n = min(len(mu), len(y))
                for i in np.where(np.isfinite(y[:n]) & np.isfinite(mu[:n]))[0].tolist():
                    pool_rows.append({
                        "run": rec.run,
                        "instance": rec.instance,
                        "algorithm": rec.algorithm,
                        "trial": rec.trial,
                        "generation": gen,
                        "definition": "exact_logged_pool",
                        "scope": scope,
                        "source": str(source[i]) if i < len(source) else scope,
                        "label": float(y[i]),
                        "mu": float(mu[i]),
                        "sigma": float(sigma[i]) if i < len(sigma) else np.nan,
                        "is_best_update": np.nan,
                        "improvement": np.nan,
                    })
    if pool_rows:
        frames.append(pd.DataFrame(pool_rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def accuracy_metrics(pred_rows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if pred_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    def summarize(g: pd.DataFrame) -> pd.Series:
        y = g["label"].to_numpy(float)
        p = g["mu"].to_numpy(float)
        err = p - y
        ystd = float(np.std(y))
        yrange = float(np.max(y) - np.min(y)) if len(y) else np.nan
        return pd.Series({
            "n": len(g),
            "mae": float(np.mean(np.abs(err))),
            "rmse": _safe_rmse(err),
            "bias": float(np.mean(err)),
            "nrmse_by_std": _safe_rmse(err) / (ystd + EPS),
            "nrmse_by_range": _safe_rmse(err) / (yrange + EPS),
            "pearson": pearson_corr(p, y),
            "spearman": spearman_corr(p, y),
            "kendall_tau_b": kendall_tau_b(p, y),
            "mean_sigma": float(np.nanmean(g["sigma"])) if np.any(np.isfinite(g["sigma"])) else np.nan,
        })

    group_cols = [
        "run",
        "instance",
        "algorithm",
        "trial",
        "definition",
        "scope",
    ]
    metric_cols = ["label", "mu", "sigma"]

    by_run = (
        pred_rows.groupby(group_cols, dropna=False)[metric_cols]
        .apply(summarize)
        .reset_index()
    )
    overall = (
        by_run.groupby(["instance", "algorithm", "definition", "scope"], as_index=False)
        .agg(
            n_trials=("run", "nunique"),
            total_n=("n", "sum"),
            mean_n=("n", "mean"),
            mean_mae=("mae", "mean"),
            median_mae=("mae", "median"),
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            mean_bias=("bias", "mean"),
            mean_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            mean_kendall_tau_b=("kendall_tau_b", "mean"),
            mean_nrmse_by_std=("nrmse_by_std", "mean"),
        )
    )
    return by_run, overall



# =============================================================================
# FE-timing and global/local role analyses
# =============================================================================

CORE_UPDATE_SOURCES: Tuple[str, str] = ("global", "incumbent")


def _add_fe_position_columns(selected: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Add absolute/normalized FE positions and phase labels to true evaluations."""
    if selected.empty:
        return selected.copy()
    d = selected.copy()
    search_budget = d["total_fe"].to_numpy(float) - d["initial_fe"].to_numpy(float)
    search_fe = d["fe"].to_numpy(float) - d["initial_fe"].to_numpy(float)
    d["search_budget"] = search_budget
    d["search_fe"] = search_fe
    d["search_progress"] = np.divide(
        search_fe,
        search_budget,
        out=np.full(len(d), np.nan, dtype=float),
        where=np.isfinite(search_budget) & (search_budget > 0),
    )
    d["search_progress"] = np.clip(d["search_progress"].to_numpy(float), 0.0, 1.0)
    total_fe = d["total_fe"].to_numpy(float)
    d["overall_fe_progress"] = np.divide(
        d["fe"].to_numpy(float),
        total_fe,
        out=np.full(len(d), np.nan, dtype=float),
        where=np.isfinite(total_fe) & (total_fe > 0),
    )

    n_bins = max(2, int(n_bins))
    p = np.nan_to_num(d["search_progress"].to_numpy(float), nan=0.0)
    bin_idx = np.ceil(p * n_bins).astype(int) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    d["fe_bin_index"] = bin_idx
    labels = [f"{int(round(100*i/n_bins)):02d}-{int(round(100*(i+1)/n_bins)):02d}%" for i in range(n_bins)]
    d["fe_bin"] = [labels[int(i)] for i in bin_idx]
    d["fe_phase"] = np.where(
        p <= 1.0 / 3.0,
        "early",
        np.where(p <= 2.0 / 3.0, "middle", "late"),
    )
    return d


def best_update_timing_analysis(
    selected: pd.DataFrame,
    n_bins: int = 5,
    stagnation_fraction: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return FE-positioned evaluations, update events, per-run FE bins,
    FE-bin summary, early/middle/late contribution, and source indicators.
    """
    d = _add_fe_position_columns(selected, n_bins=n_bins)
    if d.empty:
        empty = pd.DataFrame()
        return d, empty, empty, empty, empty, empty

    d_core = d[d["source_group"].isin(CORE_UPDATE_SOURCES)].copy()
    events = d_core[d_core["is_best_update"].astype(bool)].copy()
    if not events.empty:
        event_parts: List[pd.DataFrame] = []
        for _, g0 in events.groupby("run", sort=False):
            g = g0.sort_values("fe").copy()
            fe = g["fe"].to_numpy(float)
            search_fe = g["search_fe"].to_numpy(float)
            gaps = np.empty(len(g), dtype=float)
            if len(g):
                gaps[0] = search_fe[0]
            if len(g) > 1:
                gaps[1:] = np.diff(fe)
            budget = g["search_budget"].to_numpy(float)
            gap_fraction = np.divide(
                gaps,
                budget,
                out=np.full(len(g), np.nan, dtype=float),
                where=np.isfinite(budget) & (budget > 0),
            )
            g["update_number_in_run"] = np.arange(1, len(g) + 1, dtype=int)
            g["gap_since_previous_update_fe"] = gaps
            g["gap_fraction_of_search"] = gap_fraction
            g["previous_update_source"] = g["source_group"].shift(1).fillna("none")
            g["source_changed_from_previous_update"] = (
                g["source_group"].astype(str) != g["previous_update_source"].astype(str)
            )
            g["is_stagnation_break"] = gap_fraction >= float(stagnation_fraction)
            g["is_largest_update_in_run"] = False
            g["is_final_update_in_run"] = False
            g["is_first_update_in_run"] = False
            if len(g):
                g.loc[g["improvement"].astype(float).idxmax(), "is_largest_update_in_run"] = True
                g.loc[g.index[-1], "is_final_update_in_run"] = True
                g.loc[g.index[0], "is_first_update_in_run"] = True
            event_parts.append(g)
        events = pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()

    run_meta = d_core[[
        "run", "instance", "algorithm", "trial", "total_fe", "initial_fe", "search_budget"
    ]].drop_duplicates("run")
    sources = list(CORE_UPDATE_SOURCES)
    bin_labels = [
        f"{int(round(100*i/max(2, int(n_bins)))):02d}-{int(round(100*(i+1)/max(2, int(n_bins)))):02d}%"
        for i in range(max(2, int(n_bins)))
    ]
    grid_rows: List[Dict[str, Any]] = []
    for _, meta in run_meta.iterrows():
        for src_name in sources:
            for bi, bl in enumerate(bin_labels):
                grid_rows.append({
                    **meta.to_dict(),
                    "source_group": src_name,
                    "fe_bin_index": bi,
                    "fe_bin": bl,
                })
    grid = pd.DataFrame(grid_rows)
    grouped = (
        d_core.groupby(["run", "source_group", "fe_bin_index", "fe_bin"], as_index=False)
        .agg(
            n_true_eval=("label", "size"),
            n_best_update=("is_best_update", "sum"),
            total_improvement=("improvement", "sum"),
        )
    )
    by_run_bin = grid.merge(
        grouped,
        on=["run", "source_group", "fe_bin_index", "fe_bin"],
        how="left",
    ) if not grid.empty else grouped
    for col in ("n_true_eval", "n_best_update", "total_improvement"):
        if col in by_run_bin:
            by_run_bin[col] = by_run_bin[col].fillna(0.0)
    if not by_run_bin.empty:
        by_run_bin["update_rate"] = (
            by_run_bin["n_best_update"] / by_run_bin["n_true_eval"].replace(0, np.nan)
        )
        run_update_total = by_run_bin.groupby("run")["n_best_update"].transform("sum")
        run_improvement_total = by_run_bin.groupby("run")["total_improvement"].transform("sum")
        by_run_bin["update_share_within_run"] = (
            by_run_bin["n_best_update"] / run_update_total.replace(0, np.nan)
        )
        by_run_bin["improvement_share_within_run"] = (
            by_run_bin["total_improvement"] / run_improvement_total.replace(0, np.nan)
        )

    if by_run_bin.empty:
        bin_summary = pd.DataFrame()
    else:
        bin_summary = (
            by_run_bin.groupby(
                ["instance", "algorithm", "source_group", "fe_bin_index", "fe_bin"],
                as_index=False,
            )
            .agg(
                n_trials=("run", "nunique"),
                mean_true_eval=("n_true_eval", "mean"),
                total_true_eval=("n_true_eval", "sum"),
                mean_best_update=("n_best_update", "mean"),
                median_best_update=("n_best_update", "median"),
                total_best_update=("n_best_update", "sum"),
                mean_total_improvement=("total_improvement", "mean"),
                total_improvement=("total_improvement", "sum"),
                mean_update_share=("update_share_within_run", "mean"),
                mean_improvement_share=("improvement_share_within_run", "mean"),
            )
        )
        bin_summary["pooled_update_rate"] = (
            bin_summary["total_best_update"] / bin_summary["total_true_eval"].replace(0, np.nan)
        )

    phase_order = ["early", "middle", "late"]
    phase_grid_rows: List[Dict[str, Any]] = []
    for _, meta in run_meta.iterrows():
        for src_name in sources:
            for phase in phase_order:
                phase_grid_rows.append({
                    **meta.to_dict(),
                    "source_group": src_name,
                    "fe_phase": phase,
                })
    phase_grid = pd.DataFrame(phase_grid_rows)
    phase_group = (
        d_core.groupby(["run", "source_group", "fe_phase"], as_index=False)
        .agg(
            n_true_eval=("label", "size"),
            n_best_update=("is_best_update", "sum"),
            total_improvement=("improvement", "sum"),
        )
    )
    by_run_phase = phase_grid.merge(
        phase_group,
        on=["run", "source_group", "fe_phase"],
        how="left",
    ) if not phase_grid.empty else phase_group
    for col in ("n_true_eval", "n_best_update", "total_improvement"):
        if col in by_run_phase:
            by_run_phase[col] = by_run_phase[col].fillna(0.0)
    if not by_run_phase.empty:
        by_run_phase["update_rate"] = (
            by_run_phase["n_best_update"] / by_run_phase["n_true_eval"].replace(0, np.nan)
        )
        phase_summary = (
            by_run_phase.groupby(["instance", "algorithm", "source_group", "fe_phase"], as_index=False)
            .agg(
                n_trials=("run", "nunique"),
                mean_true_eval=("n_true_eval", "mean"),
                total_true_eval=("n_true_eval", "sum"),
                mean_best_update=("n_best_update", "mean"),
                total_best_update=("n_best_update", "sum"),
                mean_total_improvement=("total_improvement", "mean"),
                total_improvement=("total_improvement", "sum"),
            )
        )
        phase_summary["pooled_update_rate"] = (
            phase_summary["total_best_update"] / phase_summary["total_true_eval"].replace(0, np.nan)
        )
        phase_summary["phase_order"] = phase_summary["fe_phase"].map(
            {"early": 0, "middle": 1, "late": 2}
        )
        phase_summary = phase_summary.sort_values(
            ["algorithm", "source_group", "phase_order"]
        ).drop(columns="phase_order")
    else:
        phase_summary = pd.DataFrame()

    indicator_rows: List[Dict[str, Any]] = []
    algorithms = sorted(d_core["algorithm"].astype(str).unique())
    for alg in algorithms:
        alg_eval = d_core[d_core["algorithm"] == alg]
        alg_events = events[events["algorithm"] == alg] if not events.empty else pd.DataFrame()
        total_updates_alg = float(alg_events.shape[0]) if not alg_events.empty else 0.0
        total_improvement_alg = float(alg_events["improvement"].sum()) if not alg_events.empty else 0.0
        for src_name in sources:
            ev = alg_events[alg_events["source_group"] == src_name] if not alg_events.empty else pd.DataFrame()
            ev_all = alg_eval[alg_eval["source_group"] == src_name]
            indicator_rows.append({
                "instance": str(alg_eval["instance"].iloc[0]) if len(alg_eval) else "",
                "algorithm": alg,
                "source_group": src_name,
                "n_trials": int(alg_eval["run"].nunique()),
                "total_true_evaluations": int(len(ev_all)),
                "total_best_updates": int(len(ev)),
                "pooled_update_rate": float(len(ev) / len(ev_all)) if len(ev_all) else np.nan,
                "total_improvement": float(ev["improvement"].sum()) if len(ev) else 0.0,
                "update_share": float(len(ev) / total_updates_alg) if total_updates_alg > 0 else np.nan,
                "improvement_share": float(ev["improvement"].sum() / total_improvement_alg) if total_improvement_alg > 0 else np.nan,
                "median_update_progress": float(ev["search_progress"].median()) if len(ev) else np.nan,
                "mean_update_progress": float(ev["search_progress"].mean()) if len(ev) else np.nan,
                "early_update_count": int((ev["fe_phase"] == "early").sum()) if len(ev) else 0,
                "middle_update_count": int((ev["fe_phase"] == "middle").sum()) if len(ev) else 0,
                "late_update_count": int((ev["fe_phase"] == "late").sum()) if len(ev) else 0,
                "first_update_count": int(ev["is_first_update_in_run"].sum()) if len(ev) else 0,
                "largest_update_count": int(ev["is_largest_update_in_run"].sum()) if len(ev) else 0,
                "final_update_count": int(ev["is_final_update_in_run"].sum()) if len(ev) else 0,
                "stagnation_break_count": int(ev["is_stagnation_break"].sum()) if len(ev) else 0,
                "runs_with_any_update": int(ev["run"].nunique()) if len(ev) else 0,
            })
    indicators = pd.DataFrame(indicator_rows)
    return d, events, by_run_bin, bin_summary, phase_summary, indicators


def global_incumbent_followup_sequences(events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Measure incumbent updates occurring after each global update and before the next global update."""
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for _, g0 in events.groupby("run", sort=False):
        g = g0.sort_values("fe").copy()
        global_events = g[g["source_group"] == "global"]
        for _, gev in global_events.iterrows():
            later_global = global_events[global_events["fe"] > gev["fe"]]
            next_global_fe = float(later_global["fe"].iloc[0]) if len(later_global) else float("inf")
            follow = g[
                (g["source_group"] == "incumbent")
                & (g["fe"] > gev["fe"])
                & (g["fe"] < next_global_fe)
            ].sort_values("fe")
            rows.append({
                "run": gev["run"],
                "instance": gev["instance"],
                "algorithm": gev["algorithm"],
                "trial": gev["trial"],
                "global_update_fe": float(gev["fe"]),
                "global_update_search_progress": float(gev["search_progress"]),
                "global_improvement": float(gev["improvement"]),
                "next_global_update_fe": next_global_fe if np.isfinite(next_global_fe) else np.nan,
                "n_following_incumbent_updates_before_next_global": int(len(follow)),
                "following_incumbent_improvement": float(follow["improvement"].sum()) if len(follow) else 0.0,
                "has_incumbent_followup": bool(len(follow) > 0),
                "first_incumbent_followup_gap_fe": (
                    float(follow["fe"].iloc[0] - gev["fe"]) if len(follow) else np.nan
                ),
                "last_incumbent_followup_fe": float(follow["fe"].iloc[-1]) if len(follow) else np.nan,
                "global_was_final_update_in_run": bool(gev.get("is_final_update_in_run", False)),
            })
    seq = pd.DataFrame(rows)
    if seq.empty:
        return seq, pd.DataFrame()
    summary = (
        seq.groupby(["instance", "algorithm"], as_index=False)
        .agg(
            n_global_update_events=("global_update_fe", "size"),
            n_global_updates_with_incumbent_followup=("has_incumbent_followup", "sum"),
            followup_fraction=("has_incumbent_followup", "mean"),
            mean_following_incumbent_updates=("n_following_incumbent_updates_before_next_global", "mean"),
            median_following_incumbent_updates=("n_following_incumbent_updates_before_next_global", "median"),
            total_following_incumbent_improvement=("following_incumbent_improvement", "sum"),
            mean_following_incumbent_improvement=("following_incumbent_improvement", "mean"),
            median_first_followup_gap_fe=("first_incumbent_followup_gap_fe", "median"),
        )
    )
    return seq, summary


def candidate_pool_improvement_opportunity(
    records: Sequence[RunRecord],
    selected_with_fe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Quantify whether each full candidate pool contained a true improvement and
    whether the corresponding source actually realized it through true evaluation.
    """
    rows: List[Dict[str, Any]] = []
    for rec in records:
        run_selected = selected_with_fe[selected_with_fe["run"] == rec.run]
        for gen in range(n_generations(rec.logs)):
            data = _candidate_generation_arrays(rec, gen)
            eval_gen = run_selected[run_selected["generation"] == gen].sort_values("fe")
            if data is None or eval_gen.empty:
                continue
            best_before = float(eval_gen["best_before"].iloc[0])
            if not np.isfinite(best_before):
                continue
            for scope in CORE_UPDATE_SOURCES:
                mask = _pool_mask(data["source"], scope)
                mask &= np.isfinite(data["truth"])
                truth = data["truth"][mask]
                src_eval = eval_gen[eval_gen["source_group"] == scope]
                if len(truth):
                    improving = truth < best_before - EPS
                    pool_best = float(np.min(truth))
                    oracle_improvement = max(0.0, best_before - pool_best)
                    n_improving = int(np.sum(improving))
                    fraction_improving = float(np.mean(improving))
                else:
                    pool_best = np.nan
                    oracle_improvement = np.nan
                    n_improving = 0
                    fraction_improving = np.nan
                realized_improvement = float(src_eval["improvement"].sum()) if len(src_eval) else 0.0
                realized_updates = int(src_eval["is_best_update"].sum()) if len(src_eval) else 0
                rows.append({
                    "run": rec.run,
                    "instance": rec.instance,
                    "algorithm": rec.algorithm,
                    "trial": rec.trial,
                    "generation": gen,
                    "generation_first_fe": float(eval_gen["fe"].min()),
                    "generation_search_progress": float(eval_gen["search_progress"].min()),
                    "scope": scope,
                    "best_before_generation": best_before,
                    "n_candidates": int(len(truth)),
                    "n_improving_candidates": n_improving,
                    "fraction_improving_candidates": fraction_improving,
                    "pool_has_improving_candidate": bool(n_improving > 0),
                    "pool_true_best": pool_best,
                    "pool_oracle_improvement": oracle_improvement,
                    "n_true_evaluated_from_source": int(len(src_eval)),
                    "realized_best_update": bool(realized_updates > 0),
                    "n_realized_best_updates": realized_updates,
                    "realized_improvement": realized_improvement,
                    "opportunity_captured": bool((n_improving > 0) and (realized_updates > 0)),
                    "oracle_improvement_capture_ratio": (
                        realized_improvement / oracle_improvement
                        if np.isfinite(oracle_improvement) and oracle_improvement > EPS
                        else np.nan
                    ),
                    "missed_oracle_improvement": (
                        max(0.0, oracle_improvement - realized_improvement)
                        if np.isfinite(oracle_improvement)
                        else np.nan
                    ),
                })
    by_gen = pd.DataFrame(rows)
    if by_gen.empty:
        return by_gen, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    summary = (
        by_gen.groupby(["instance", "algorithm", "scope"], as_index=False)
        .agg(
            n_runs=("run", "nunique"),
            n_generations=("generation", "size"),
            generations_with_improvement_opportunity=("pool_has_improving_candidate", "sum"),
            opportunity_rate=("pool_has_improving_candidate", "mean"),
            mean_improving_candidate_fraction=("fraction_improving_candidates", "mean"),
            mean_oracle_improvement=("pool_oracle_improvement", "mean"),
            total_oracle_improvement=("pool_oracle_improvement", "sum"),
            generations_with_realized_update=("realized_best_update", "sum"),
            total_realized_improvement=("realized_improvement", "sum"),
            captured_opportunities=("opportunity_captured", "sum"),
            mean_capture_ratio=("oracle_improvement_capture_ratio", "mean"),
            total_missed_oracle_improvement=("missed_oracle_improvement", "sum"),
        )
    )
    summary["opportunity_capture_rate"] = (
        summary["captured_opportunities"]
        / summary["generations_with_improvement_opportunity"].replace(0, np.nan)
    )

    pivot_cols = [
        "pool_has_improving_candidate", "pool_oracle_improvement",
        "realized_best_update", "realized_improvement",
    ]
    comparison = by_gen.pivot_table(
        index=["run", "instance", "algorithm", "trial", "generation", "generation_search_progress"],
        columns="scope",
        values=pivot_cols,
        aggfunc="first",
    )
    comparison.columns = [f"{metric}_{scope}" for metric, scope in comparison.columns]
    comparison = comparison.reset_index()
    for col in (
        "pool_has_improving_candidate_global", "pool_has_improving_candidate_incumbent",
        "realized_best_update_global", "realized_best_update_incumbent",
    ):
        if col not in comparison:
            comparison[col] = False
        comparison[col] = comparison[col].fillna(False).astype(bool)
    comparison["only_global_has_opportunity"] = (
        comparison["pool_has_improving_candidate_global"]
        & ~comparison["pool_has_improving_candidate_incumbent"]
    )
    comparison["only_incumbent_has_opportunity"] = (
        comparison["pool_has_improving_candidate_incumbent"]
        & ~comparison["pool_has_improving_candidate_global"]
    )
    comparison["both_have_opportunity"] = (
        comparison["pool_has_improving_candidate_global"]
        & comparison["pool_has_improving_candidate_incumbent"]
    )
    comparison["neither_has_opportunity"] = (
        ~comparison["pool_has_improving_candidate_global"]
        & ~comparison["pool_has_improving_candidate_incumbent"]
    )
    comp_summary = (
        comparison.groupby(["instance", "algorithm"], as_index=False)
        .agg(
            n_generations=("generation", "size"),
            only_global_opportunity_count=("only_global_has_opportunity", "sum"),
            only_incumbent_opportunity_count=("only_incumbent_has_opportunity", "sum"),
            both_opportunity_count=("both_have_opportunity", "sum"),
            neither_opportunity_count=("neither_has_opportunity", "sum"),
            global_realized_update_count=("realized_best_update_global", "sum"),
            incumbent_realized_update_count=("realized_best_update_incumbent", "sum"),
        )
    )
    comp_summary["only_global_opportunity_rate"] = (
        comp_summary["only_global_opportunity_count"] / comp_summary["n_generations"].replace(0, np.nan)
    )
    comp_summary["only_incumbent_opportunity_rate"] = (
        comp_summary["only_incumbent_opportunity_count"] / comp_summary["n_generations"].replace(0, np.nan)
    )
    return by_gen, summary, comparison, comp_summary

# =============================================================================
# Plot helpers
# =============================================================================

def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def _algorithm_positions(algorithms: Sequence[str], groups: Sequence[str]) -> Tuple[Dict[Tuple[str, str], float], np.ndarray]:
    width = 0.8 / max(1, len(groups))
    positions: Dict[Tuple[str, str], float] = {}
    centers = np.arange(len(algorithms), dtype=float)
    for ai, alg in enumerate(algorithms):
        for gi, group in enumerate(groups):
            positions[(alg, group)] = centers[ai] - 0.4 + width / 2 + gi * width
    return positions, centers


def plot_update_counts(by_run: pd.DataFrame, records: Sequence[RunRecord], outdir: Path) -> None:
    if by_run.empty:
        return
    scopes = [s for s in ("global", "incumbent") if s in set(by_run["source_group"])]
    if not scopes:
        return
    algorithms = sorted(set(by_run["algorithm"].astype(str)))
    pos, centers = _algorithm_positions(algorithms, scopes)
    width = 0.8 / len(scopes)

    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(algorithms)), 5.2))
    rng = np.random.default_rng(0)
    for scope in scopes:
        means = []
        xs = []
        for alg in algorithms:
            vals = by_run[(by_run["algorithm"] == alg) & (by_run["source_group"] == scope)]["n_best_update"].to_numpy(float)
            means.append(float(np.mean(vals)) if len(vals) else 0.0)
            x = pos[(alg, scope)]
            xs.append(x)
            if len(vals):
                jitter = rng.uniform(-width * 0.18, width * 0.18, size=len(vals))
                ax.scatter(np.full(len(vals), x) + jitter, vals, s=22, alpha=0.75, zorder=3)
        ax.bar(xs, means, width=width * 0.82, alpha=0.55, label=scope)

    max_fe = max((r.total_fe for r in records), default=0)
    if max_fe > 0:
        ax.set_ylim(0, max_fe + max(1, 0.03 * max_fe))
    ax.set_xticks(centers)
    ax.set_xticklabels(algorithms, rotation=25, ha="right")
    ax.set_ylabel("True evaluations that improved the incumbent (per trial)")
    ax.set_title("Best-update count by search origin\nBars: trial mean; dots: individual trials")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "best_update_count_global_vs_incumbent_per_trial.png", dpi=220)
    plt.close(fig)


def plot_update_rate(by_run: pd.DataFrame, outdir: Path) -> None:
    if by_run.empty:
        return
    scopes = [s for s in ("global", "incumbent") if s in set(by_run["source_group"])]
    algorithms = sorted(set(by_run["algorithm"].astype(str)))
    if not scopes or not algorithms:
        return
    pos, centers = _algorithm_positions(algorithms, scopes)
    width = 0.8 / len(scopes)
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(algorithms)), 5.0))
    for scope in scopes:
        xs, means = [], []
        for alg in algorithms:
            vals = by_run[(by_run["algorithm"] == alg) & (by_run["source_group"] == scope)]["update_rate"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            xs.append(pos[(alg, scope)])
            means.append(float(np.mean(vals)) if len(vals) else 0.0)
        ax.bar(xs, means, width=width * 0.82, alpha=0.7, label=scope)
    ax.set_xticks(centers)
    ax.set_xticklabels(algorithms, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Best updates / true evaluations")
    ax.set_title("Update efficiency by search origin")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "best_update_rate_global_vs_incumbent.png", dpi=220)
    plt.close(fig)


def plot_total_improvement(by_run: pd.DataFrame, outdir: Path) -> None:
    if by_run.empty:
        return
    scopes = [s for s in ("global", "incumbent") if s in set(by_run["source_group"])]
    algorithms = sorted(set(by_run["algorithm"].astype(str)))
    if not scopes:
        return
    pos, centers = _algorithm_positions(algorithms, scopes)
    width = 0.8 / len(scopes)
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(algorithms)), 5.0))
    for scope in scopes:
        xs, means = [], []
        for alg in algorithms:
            vals = by_run[(by_run["algorithm"] == alg) & (by_run["source_group"] == scope)]["total_improvement"].to_numpy(float)
            xs.append(pos[(alg, scope)])
            means.append(float(np.mean(vals)) if len(vals) else 0.0)
        ax.bar(xs, means, width=width * 0.82, alpha=0.7, label=scope)
    ax.set_xticks(centers)
    ax.set_xticklabels(algorithms, rotation=25, ha="right")
    ax.set_ylabel("Total objective improvement per trial")
    ax.set_title("Improvement magnitude contribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "total_improvement_global_vs_incumbent.png", dpi=220)
    plt.close(fig)


def plot_final_best(records: Sequence[RunRecord], outdir: Path) -> None:
    rows = []
    for r in records:
        val = r.best_val
        if val is None and r.history_best is not None and len(r.history_best):
            val = float(r.history_best[-1])
        if val is not None and np.isfinite(val):
            rows.append({"algorithm": r.algorithm, "trial": r.trial, "best": val})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    algorithms = sorted(set(df["algorithm"]))
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(algorithms)), 5.0))
    data = [df[df["algorithm"] == alg]["best"].to_numpy(float) for alg in algorithms]
    ax.boxplot(data, labels=algorithms, showmeans=True)
    rng = np.random.default_rng(1)
    for i, vals in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(vals)), vals, s=20, alpha=0.7)
    ax.set_ylabel("Final best objective (lower is better)")
    ax.set_title("Final optimization performance")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "final_best_objective_by_algorithm.png", dpi=220)
    plt.close(fig)


def _binned_curve(df: pd.DataFrame, x: str, y: str, bins: int = 12) -> pd.DataFrame:
    if len(df) < 4:
        return pd.DataFrame()
    q = min(bins, max(2, df[x].nunique()))
    try:
        work = df.copy()
        work["bin"] = pd.qcut(work[x], q=q, duplicates="drop")
        return work.groupby("bin", observed=True, as_index=False).agg(
            x_mean=(x, "mean"), y_median=(y, "median"), y_mean=(y, "mean"), n=(y, "size")
        )
    except Exception:
        return pd.DataFrame()


def plot_embedding_relation(pairs: pd.DataFrame, outdir: Path, max_points: int) -> None:
    if pairs.empty:
        return
    algorithms = sorted(set(pairs["algorithm"].astype(str)))

    # Per-algorithm scatter: less visually misleading than mixing latent spaces.
    for alg in algorithms:
        g = pairs[pairs["algorithm"] == alg].copy()
        if len(g) > max_points:
            g = g.sample(max_points, random_state=0)
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        ax.scatter(g["embedding_distance"], g["objective_abs_diff_within_generation_normalized"], s=9, alpha=0.22)
        curve = _binned_curve(g, "embedding_distance", "objective_abs_diff_within_generation_normalized")
        if not curve.empty:
            ax.plot(curve["x_mean"], curve["y_median"], marker="o", linewidth=2, label="binned median")
            ax.legend()
        rho = spearman_corr(g["embedding_distance"], g["objective_abs_diff_within_generation_normalized"])
        ax.set_xlabel("Embedding distance")
        ax.set_ylabel("|objective difference| / generation objective range")
        ax.set_title(f"Embedding distance vs objective difference\n{alg}, Spearman={rho:.3f}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(outdir / f"embedding_distance_vs_objective_{_safe_name(alg)}.png", dpi=220)
        plt.close(fig)

    # Combined binned curves for algorithm comparison.
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    drew = False
    for alg in algorithms:
        curve = _binned_curve(
            pairs[pairs["algorithm"] == alg],
            "embedding_distance",
            "objective_abs_diff_within_generation_normalized",
        )
        if curve.empty:
            continue
        ax.plot(curve["x_mean"], curve["y_median"], marker="o", linewidth=1.8, label=alg)
        drew = True
    if drew:
        ax.set_xlabel("Embedding distance")
        ax.set_ylabel("Median normalized objective difference")
        ax.set_title("Embedding/objective relationship by algorithm")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(outdir / "embedding_objective_binned_comparison.png", dpi=220)
    plt.close(fig)


def plot_global_selection(global_df: pd.DataFrame, outdir: Path) -> None:
    if global_df.empty:
        return
    algorithms = sorted(set(global_df["algorithm"].astype(str)))

    metrics = [
        ("fraction_selected_outside_score_topk", "Fraction selected outside score top-k", "global_selection_changed_fraction.png"),
        ("mean_pairwise_diversity_gain", "Mean pairwise-distance gain over score top-k", "global_selection_diversity_gain.png"),
        ("mean_rank_delta", "Mean acquisition-rank penalty", "global_selection_rank_penalty.png"),
        ("fraction_selected_below_threshold", "Fraction selected below latent threshold", "global_selection_threshold_fallback_fraction.png"),
    ]
    for col, ylabel, filename in metrics:
        if col not in global_df or not np.any(np.isfinite(global_df[col])):
            continue
        fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(algorithms)), 5.0))
        data = [global_df[global_df["algorithm"] == alg][col].dropna().to_numpy(float) for alg in algorithms]
        if not any(len(v) for v in data):
            plt.close(fig)
            continue
        ax.boxplot(data, labels=algorithms, showmeans=True)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by algorithm")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=220)
        plt.close(fig)

    # Rank/diversity trade-off.
    valid = global_df[np.isfinite(global_df["mean_rank_delta"]) & np.isfinite(global_df["mean_pairwise_diversity_gain"])]
    if not valid.empty:
        fig, ax = plt.subplots(figsize=(6.7, 5.3))
        for alg, g in valid.groupby("algorithm"):
            ax.scatter(g["mean_rank_delta"], g["mean_pairwise_diversity_gain"], s=24, alpha=0.45, label=str(alg))
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Acquisition-rank penalty")
        ax.set_ylabel("Embedding-diversity gain over score top-k")
        ax.set_title("Global selection: score/diversity trade-off")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(outdir / "global_selection_rank_diversity_tradeoff.png", dpi=220)
        plt.close(fig)

    # Coarse outcome association per generation.
    if "global_total_improvement" in global_df:
        valid = global_df[np.isfinite(global_df["mean_pairwise_diversity_gain"])]
        if not valid.empty:
            fig, ax = plt.subplots(figsize=(6.7, 5.3))
            for alg, g in valid.groupby("algorithm"):
                ax.scatter(g["mean_pairwise_diversity_gain"], g["global_total_improvement"], s=25, alpha=0.45, label=str(alg))
            ax.set_xlabel("Embedding-diversity gain over score top-k")
            ax.set_ylabel("Global-origin improvement in same generation")
            ax.set_title("Diversity change vs observed improvement\n(association, not a counterfactual test)")
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(outdir / "global_diversity_gain_vs_improvement.png", dpi=220)
            plt.close(fig)


def plot_accuracy(by_run: pd.DataFrame, pred_rows: pd.DataFrame, outdir: Path) -> None:
    if by_run.empty:
        return

    for definition, scopes in (
        ("selected_origin", ["global", "incumbent"]),
        ("exact_logged_pool", ["global_pool", "local_pool"]),
    ):
        d = by_run[(by_run["definition"] == definition) & (by_run["scope"].isin(scopes))]
        if d.empty:
            continue
        algorithms = sorted(set(d["algorithm"].astype(str)))
        present_scopes = [s for s in scopes if s in set(d["scope"])]
        pos, centers = _algorithm_positions(algorithms, present_scopes)
        width = 0.8 / len(present_scopes)

        for metric, ylabel in (("mae", "MAE"), ("rmse", "RMSE"), ("spearman", "Spearman rank correlation"), ("kendall_tau_b", "Kendall tau-b")):
            fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(algorithms)), 5.0))
            for scope in present_scopes:
                xs, means = [], []
                for alg in algorithms:
                    vals = d[(d["algorithm"] == alg) & (d["scope"] == scope)][metric].to_numpy(float)
                    vals = vals[np.isfinite(vals)]
                    xs.append(pos[(alg, scope)])
                    means.append(float(np.mean(vals)) if len(vals) else np.nan)
                ax.bar(xs, means, width=width * 0.82, alpha=0.7, label=scope)
            ax.set_xticks(centers)
            ax.set_xticklabels(algorithms, rotation=25, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Surrogate {ylabel}: {definition}")
            ax.legend()
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(outdir / f"surrogate_{metric}_{definition}.png", dpi=220)
            plt.close(fig)

    # True vs predicted, selected-origin definition.
    d = pred_rows[(pred_rows["definition"] == "selected_origin") & pred_rows["scope"].isin(["global", "incumbent"])]
    for alg, g_alg in d.groupby("algorithm"):
        if g_alg.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        for scope, g in g_alg.groupby("scope"):
            ax.scatter(g["label"], g["mu"], s=18, alpha=0.45, label=str(scope))
        vals = np.concatenate([g_alg["label"].to_numpy(float), g_alg["mu"].to_numpy(float)])
        vals = vals[np.isfinite(vals)]
        if len(vals):
            lo, hi = float(np.min(vals)), float(np.max(vals))
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set_xlabel("True objective")
        ax.set_ylabel("Predicted mean (mu)")
        ax.set_title(f"Surrogate prediction by origin: {alg}")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(outdir / f"surrogate_true_vs_pred_{_safe_name(str(alg))}.png", dpi=220)
        plt.close(fig)


def _sample_projection_indices(valid_idx: np.ndarray, priority_idx: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    valid_idx = np.asarray(valid_idx, dtype=int)
    priority_idx = np.asarray(priority_idx, dtype=int)
    priority_idx = np.asarray([i for i in priority_idx if i in set(valid_idx.tolist())], dtype=int)
    priority_idx = np.unique(priority_idx)
    if len(valid_idx) <= max_points:
        return valid_idx
    remaining = np.asarray([i for i in valid_idx if i not in set(priority_idx.tolist())], dtype=int)
    keep = max(0, max_points - len(priority_idx))
    rng = np.random.default_rng(seed)
    if len(remaining) > keep:
        remaining = rng.choice(remaining, size=keep, replace=False)
    return np.unique(np.concatenate([priority_idx, remaining]))


def plot_embedding_projections(
    records: Sequence[RunRecord],
    outdir: Path,
    method: str,
    generation_specs: Sequence[str],
    max_runs_per_algorithm: int,
    max_points: int,
) -> None:
    """Same candidate coordinates: left=prediction, right=true value for every candidate."""
    if method == "none":
        return
    chosen_count: Dict[str, int] = {}
    for rec_i, rec in enumerate(sorted(records, key=lambda r: (r.algorithm, r.trial))):
        used = chosen_count.get(rec.algorithm, 0)
        if used >= max_runs_per_algorithm:
            continue
        generated_any = False
        for gen in parse_generation_specs(generation_specs, n_generations(rec.logs)):
            data = _candidate_generation_arrays(rec, gen)
            if data is None or data["emb"].ndim != 2 or data["emb"].shape[1] == 0:
                continue
            valid = np.all(np.isfinite(data["emb"]), axis=1) & np.isfinite(data["mu"]) & np.isfinite(data["truth"])
            valid_idx = np.where(valid)[0]
            selected_idx = np.where(data["selected"] & valid)[0]
            if len(valid_idx) < 2:
                continue
            idx = _sample_projection_indices(valid_idx, selected_idx, max_points, rec_i * 1000 + gen)
            coord = _project_2d(data["emb"][idx], method, rec_i * 1000 + gen)
            pos = {int(v): i for i, v in enumerate(idx.tolist())}
            selected_in = np.asarray([i for i in selected_idx if int(i) in pos], dtype=int)
            selected_pos = np.asarray([pos[int(i)] for i in selected_in], dtype=int)
            all_values = np.concatenate([data["mu"][idx], data["truth"][idx]])
            all_values = all_values[np.isfinite(all_values)]
            vmin = float(np.min(all_values)) if len(all_values) else None
            vmax = float(np.max(all_values)) if len(all_values) else None

            fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
            sc1 = axes[0].scatter(coord[:, 0], coord[:, 1], c=data["mu"][idx], s=14, alpha=0.78, vmin=vmin, vmax=vmax)
            sc2 = axes[1].scatter(coord[:, 0], coord[:, 1], c=data["truth"][idx], s=14, alpha=0.78, vmin=vmin, vmax=vmax)
            if len(selected_pos):
                for ax in axes:
                    ax.scatter(coord[selected_pos, 0], coord[selected_pos, 1], facecolors="none", edgecolors="black", s=68, linewidths=1.2)
            fig.colorbar(sc1, ax=axes[0], label="Predicted objective (mu)")
            fig.colorbar(sc2, ax=axes[1], label="True objective (all candidates)")
            axes[0].set_title("All candidates colored by surrogate prediction")
            axes[1].set_title("Same candidates colored by true objective")
            for ax in axes:
                ax.set_xlabel(f"{method.upper()}-1")
                ax.set_ylabel(f"{method.upper()}-2")
            fig.suptitle(f"Candidate embedding: {rec.algorithm}, {rec.trial}, generation {gen}")
            fig.tight_layout()
            fig.savefig(outdir / f"embedding_2d_{_safe_name(rec.algorithm)}_{_safe_name(rec.trial)}_gen{gen:04d}_{method}.png", dpi=220)
            plt.close(fig)
            generated_any = True
        if generated_any:
            chosen_count[rec.algorithm] = used + 1


def plot_archive_candidate_projections(
    records: Sequence[RunRecord],
    outdir: Path,
    method: str,
    generation_specs: Sequence[str],
    max_runs_per_algorithm: int,
    max_points: int,
) -> None:
    """Archive and all candidates in one current-surrogate embedding space."""
    if method == "none":
        return
    chosen_count: Dict[str, int] = {}
    for rec_i, rec in enumerate(sorted(records, key=lambda r: (r.algorithm, r.trial))):
        used = chosen_count.get(rec.algorithm, 0)
        if used >= max_runs_per_algorithm:
            continue
        generated_any = False
        for gen in parse_generation_specs(generation_specs, n_generations(rec.logs)):
            cand = _candidate_generation_arrays(rec, gen)
            arc = _archive_generation_arrays(rec, gen)
            if cand is None or arc is None:
                continue
            if cand["emb"].shape[1] == 0 or arc["emb"].shape[1] == 0:
                continue
            va = np.all(np.isfinite(arc["emb"]), axis=1) & np.isfinite(arc["truth"]) & np.isfinite(arc["mu"])
            vc = np.all(np.isfinite(cand["emb"]), axis=1) & np.isfinite(cand["truth"]) & np.isfinite(cand["mu"])
            ia = np.where(va)[0]
            ic = np.where(vc)[0]
            if len(ia) + len(ic) < 2:
                continue
            # Reserve at least half of the budget for candidates when both sets are large.
            max_a = max(1, max_points // 2)
            max_c = max(1, max_points - min(len(ia), max_a))
            ia = _sample_projection_indices(ia, np.zeros(0, dtype=int), max_a, 100000 + rec_i * 1000 + gen)
            sel_c = np.where(cand["selected"] & vc)[0]
            ic = _sample_projection_indices(ic, sel_c, max_c, 200000 + rec_i * 1000 + gen)
            z = np.vstack([arc["emb"][ia], cand["emb"][ic]])
            coord = _project_2d(z, method, 300000 + rec_i * 1000 + gen)
            na = len(ia)
            ca, cc = coord[:na], coord[na:]
            source_c = np.asarray([str(v).lower() for v in cand["source"][ic]], dtype=object)

            true_values = np.concatenate([arc["truth"][ia], cand["truth"][ic]])
            pred_values = np.concatenate([arc["mu"][ia], cand["mu"][ic]])
            both = np.concatenate([true_values, pred_values])
            both = both[np.isfinite(both)]
            vmin = float(np.min(both)) if len(both) else None
            vmax = float(np.max(both)) if len(both) else None

            fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.3))
            # True-value panel.
            a0 = axes[0].scatter(ca[:, 0], ca[:, 1], c=arc["truth"][ia], marker="o", s=18, alpha=0.55, vmin=vmin, vmax=vmax, label="archive")
            axes[0].scatter(cc[:, 0], cc[:, 1], c=cand["truth"][ic], marker="x", s=20, alpha=0.68, vmin=vmin, vmax=vmax, label="candidates")
            fig.colorbar(a0, ax=axes[0], label="True objective")
            axes[0].set_title("Archive and candidates: true values")

            # Prediction panel.
            a1 = axes[1].scatter(ca[:, 0], ca[:, 1], c=arc["mu"][ia], marker="o", s=18, alpha=0.55, vmin=vmin, vmax=vmax, label="archive")
            axes[1].scatter(cc[:, 0], cc[:, 1], c=cand["mu"][ic], marker="x", s=20, alpha=0.68, vmin=vmin, vmax=vmax, label="candidates")
            fig.colorbar(a1, ax=axes[1], label="Predicted objective (mu)")
            axes[1].set_title("Archive and candidates: predictions")

            # Source/location panel.
            axes[2].scatter(ca[:, 0], ca[:, 1], c="lightgray", marker="o", s=16, alpha=0.45, label="archive")
            styles = {
                "global_pool": ("tab:orange", "x", "global candidates"),
                "incumbent": ("tab:red", "s", "incumbent candidates"),
                "global": ("tab:blue", "^", "global-root local candidates"),
                "fallback": ("tab:purple", "D", "fallback candidates"),
                "uncertainty": ("tab:green", "v", "uncertainty candidates"),
            }
            for s in sorted(set(source_c.tolist())):
                m = source_c == s
                color, marker, label = styles.get(s, ("tab:gray", "X", s))
                axes[2].scatter(cc[m, 0], cc[m, 1], c=color, marker=marker, s=28, alpha=0.72, label=label)
            axes[2].set_title("Archive/candidate location by source")
            for ax in axes:
                ax.set_xlabel(f"{method.upper()}-1")
                ax.set_ylabel(f"{method.upper()}-2")
                ax.legend(loc="best", fontsize=7)
            fig.suptitle(f"Archive + candidate embedding: {rec.algorithm}, {rec.trial}, generation {gen}")
            fig.tight_layout()
            fig.savefig(outdir / f"embedding_archive_candidates_{_safe_name(rec.algorithm)}_{_safe_name(rec.trial)}_gen{gen:04d}_{method}.png", dpi=220)
            plt.close(fig)
            generated_any = True
        if generated_any:
            chosen_count[rec.algorithm] = used + 1


def plot_candidate_pool_metric(by_gen: pd.DataFrame, metric: str, ylabel: str, filename: str, outdir: Path) -> None:
    if by_gen.empty or metric not in by_gen:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    drew = False
    for (alg, scope), g in by_gen.groupby(["algorithm", "scope"]):
        agg = g.groupby("generation", as_index=False).agg(mean=(metric, "mean"), std=(metric, "std"))
        if agg.empty:
            continue
        x = agg["generation"].to_numpy(float)
        y = agg["mean"].to_numpy(float)
        sd = agg["std"].fillna(0.0).to_numpy(float)
        ax.plot(x, y, marker="o", linewidth=1.7, label=f"{alg} | {scope}")
        ax.fill_between(x, y - sd, y + sd, alpha=0.12)
        drew = True
    if drew:
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by candidate pool")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=220)
    plt.close(fig)


def plot_embedding_structure_curves(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty:
        return
    specs = [
        ("spearman_embedding_vs_true_abs_diff", "Spearman: embedding distance vs true-value difference", "embedding_true_structure_by_generation.png"),
        ("spearman_embedding_vs_pred_abs_diff", "Spearman: embedding distance vs predicted-value difference", "embedding_pred_structure_by_generation.png"),
        ("prediction_true_spearman", "Prediction/true Spearman in embedding scopes", "embedding_scope_prediction_accuracy_by_generation.png"),
    ]
    for metric, ylabel, filename in specs:
        plot_candidate_pool_metric(df, metric, ylabel, filename, outdir)


def plot_archive_candidate_novelty(df: pd.DataFrame, outdir: Path) -> None:
    plot_candidate_pool_metric(
        df,
        "median_novelty_ratio_to_archive",
        "Median candidate/archive novelty ratio",
        "candidate_archive_novelty_by_generation.png",
        outdir,
    )


def plot_archive_embedding_updates(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    metrics = [
        ("embedding_distance_matrix_spearman", "Distance-matrix Spearman"),
        ("knn5_overlap", "5-NN overlap"),
        ("procrustes_mean_residual", "Procrustes residual"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for alg, g in df.groupby("algorithm"):
            agg = g.groupby("generation_to", as_index=False)[metric].mean()
            ax.plot(agg["generation_to"], agg[metric], marker="o", label=str(alg))
        ax.set_xlabel("Generation")
        ax.set_title(title)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Stability/change metric")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Archive embedding update between consecutive generations")
    fig.tight_layout()
    fig.savefig(outdir / "archive_embedding_update_stability.png", dpi=220)
    plt.close(fig)


def best_update_total_table(by_run: pd.DataFrame) -> pd.DataFrame:
    if by_run.empty:
        return pd.DataFrame()
    d = by_run[by_run["source_group"].isin(["global", "incumbent"])].copy()
    if d.empty:
        return d
    return (
        d.groupby(["instance", "algorithm", "source_group"], as_index=False)
        .agg(
            n_trials=("run", "nunique"),
            total_true_evaluations=("n_true_eval", "sum"),
            total_best_updates=("n_best_update", "sum"),
            mean_best_updates_per_trial=("n_best_update", "mean"),
            median_best_updates_per_trial=("n_best_update", "median"),
            total_improvement=("total_improvement", "sum"),
            mean_improvement_per_trial=("total_improvement", "mean"),
        )
        .assign(pooled_update_rate=lambda x: x["total_best_updates"] / x["total_true_evaluations"].replace(0, np.nan))
    )


def plot_best_update_totals(total_df: pd.DataFrame, outdir: Path) -> None:
    if total_df.empty:
        return
    algorithms = sorted(total_df["algorithm"].unique())
    scopes = [s for s in ("global", "incumbent") if s in set(total_df["source_group"])]
    pos, centers = _algorithm_positions(algorithms, scopes)
    width = 0.8 / max(1, len(scopes))
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(algorithms)), 5.1))
    for scope in scopes:
        xs, vals = [], []
        for alg in algorithms:
            row = total_df[(total_df["algorithm"] == alg) & (total_df["source_group"] == scope)]
            xs.append(pos[(alg, scope)])
            vals.append(float(row["total_best_updates"].iloc[0]) if len(row) else 0.0)
        ax.bar(xs, vals, width=width * 0.82, alpha=0.72, label=scope)
    ax.set_xticks(centers)
    ax.set_xticklabels(algorithms, rotation=25, ha="right")
    ax.set_ylabel("Total best-improving true evaluations across all trials")
    ax.set_title("Total incumbent-best updates by evaluation source")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "best_update_total_global_vs_incumbent.png", dpi=220)
    plt.close(fig)




def _event_marker(source: str) -> str:
    return "o" if str(source) == "global" else "s"


def plot_best_update_event_timelines(events: pd.DataFrame, outdir: Path) -> None:
    if events.empty:
        return
    for alg, d0 in events.groupby("algorithm"):
        d = d0.sort_values(["trial", "fe"])
        trials = sorted(d["trial"].astype(str).unique())
        ymap = {trial: i for i, trial in enumerate(trials)}
        max_imp = float(d["improvement"].max()) if len(d) else 0.0
        sizes = 38.0 + 170.0 * np.sqrt(
            np.clip(d["improvement"].to_numpy(float) / (max_imp + EPS), 0.0, 1.0)
        )
        d = d.copy()
        d["_size"] = sizes
        d["_y"] = d["trial"].astype(str).map(ymap).astype(float)

        for xcol, xlabel, suffix in (
            ("fe", "Function evaluations (absolute FE)", "absolute_fe"),
            ("search_progress", "Search progress after initial design", "normalized"),
        ):
            fig, ax = plt.subplots(figsize=(9.2, max(4.5, 0.36 * len(trials) + 2.4)))
            for source in CORE_UPDATE_SOURCES:
                g = d[d["source_group"] == source]
                if g.empty:
                    continue
                ax.scatter(
                    g[xcol], g["_y"], s=g["_size"], marker=_event_marker(source),
                    alpha=0.78, label=source,
                )
            ax.set_yticks(np.arange(len(trials), dtype=float))
            ax.set_yticklabels(trials)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Trial")
            ax.set_title(f"Best-update events over FE: {alg}\nmarker size = objective improvement")
            ax.grid(alpha=0.22)
            ax.legend()
            if xcol == "search_progress":
                ax.set_xlim(0, 1.02)
            fig.tight_layout()
            fig.savefig(outdir / f"best_update_event_timeline_{suffix}_{_safe_name(str(alg))}.png", dpi=220)
            plt.close(fig)


def plot_best_update_by_fe_bins(bin_summary: pd.DataFrame, outdir: Path) -> None:
    if bin_summary.empty:
        return
    for alg, d in bin_summary.groupby("algorithm"):
        bin_order = sorted(d["fe_bin_index"].dropna().astype(int).unique())
        labels = [
            str(d[d["fe_bin_index"] == bi]["fe_bin"].iloc[0]) for bi in bin_order
        ]
        x = np.arange(len(bin_order), dtype=float)
        sources = [s for s in CORE_UPDATE_SOURCES if s in set(d["source_group"])]
        width = 0.8 / max(1, len(sources))
        specs = (
            ("mean_best_update", "Mean best updates per trial", "counts"),
            ("mean_total_improvement", "Mean objective improvement per trial", "improvement"),
            ("pooled_update_rate", "Best updates / true evaluations", "rate"),
        )
        for metric, ylabel, suffix in specs:
            fig, ax = plt.subplots(figsize=(8.4, 5.0))
            for si, source in enumerate(sources):
                vals: List[float] = []
                for bi in bin_order:
                    row = d[(d["source_group"] == source) & (d["fe_bin_index"] == bi)]
                    vals.append(float(row[metric].iloc[0]) if len(row) else 0.0)
                xpos = x - 0.4 + width / 2 + si * width
                ax.bar(xpos, vals, width=width * 0.84, alpha=0.75, label=source)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Search-FE phase")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Global/incumbent contribution over search progress: {alg}")
            if metric == "pooled_update_rate":
                ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.24)
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / f"best_update_by_fe_phase_{suffix}_{_safe_name(str(alg))}.png", dpi=220)
            plt.close(fig)


def plot_cumulative_update_contribution(
    selected_with_fe: pd.DataFrame,
    events: pd.DataFrame,
    outdir: Path,
) -> None:
    if selected_with_fe.empty:
        return
    grid = np.linspace(0.0, 1.0, 101)
    for alg, meta in selected_with_fe.groupby("algorithm"):
        runs = sorted(meta["run"].astype(str).unique())
        if not runs:
            continue
        for value_kind, ylabel, suffix in (
            ("count", "Cumulative best-update count", "updates"),
            ("improvement", "Cumulative objective improvement", "improvement"),
        ):
            fig, ax = plt.subplots(figsize=(7.8, 5.2))
            for source in CORE_UPDATE_SOURCES:
                curves: List[np.ndarray] = []
                for run in runs:
                    ev = events[
                        (events["run"] == run)
                        & (events["source_group"] == source)
                    ].sort_values("search_progress") if not events.empty else pd.DataFrame()
                    if ev.empty:
                        curves.append(np.zeros_like(grid))
                        continue
                    pos = ev["search_progress"].to_numpy(float)
                    values = (
                        np.ones(len(ev), dtype=float)
                        if value_kind == "count"
                        else ev["improvement"].to_numpy(float)
                    )
                    cum_values = np.cumsum(values)
                    idx = np.searchsorted(pos, grid, side="right") - 1
                    curve = np.where(idx >= 0, cum_values[np.clip(idx, 0, len(cum_values)-1)], 0.0)
                    curves.append(curve.astype(float))
                arr = np.vstack(curves)
                mean = np.mean(arr, axis=0)
                std = np.std(arr, axis=0)
                ax.plot(grid, mean, linewidth=2.0, label=source)
                ax.fill_between(grid, mean - std, mean + std, alpha=0.16)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Search progress after initial design")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Cumulative global/incumbent contribution: {alg}\nmean ± std across trials")
            ax.grid(alpha=0.22)
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / f"cumulative_{suffix}_by_fe_{_safe_name(str(alg))}.png", dpi=220)
            plt.close(fig)


def plot_candidate_pool_improvement_opportunity(by_gen: pd.DataFrame, outdir: Path) -> None:
    if by_gen.empty:
        return
    for alg, d0 in by_gen.groupby("algorithm"):
        for metric, ylabel, suffix in (
            ("pool_oracle_improvement", "Oracle improvement available in pool", "oracle_improvement"),
            ("fraction_improving_candidates", "Fraction of candidates improving current best", "improving_fraction"),
        ):
            fig, ax = plt.subplots(figsize=(8.0, 5.0))
            for source in CORE_UPDATE_SOURCES:
                d = d0[d0["scope"] == source]
                if d.empty:
                    continue
                agg = (
                    d.groupby("generation", as_index=False)[metric]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                x = agg["generation"].to_numpy(float)
                mean = agg["mean"].to_numpy(float)
                std = agg["std"].fillna(0.0).to_numpy(float)
                ax.plot(x, mean, marker="o", linewidth=1.8, label=source)
                ax.fill_between(x, mean - std, mean + std, alpha=0.15)
            ax.set_xlabel("Generation")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Candidate-pool improvement opportunity: {alg}")
            if metric == "fraction_improving_candidates":
                ax.set_ylim(0, 1)
            ax.grid(alpha=0.22)
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / f"candidate_pool_{suffix}_by_generation_{_safe_name(str(alg))}.png", dpi=220)
            plt.close(fig)

        # Oracle vs realized improvement, source separated.
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), squeeze=False)
        for ax, source in zip(axes[0], CORE_UPDATE_SOURCES):
            d = d0[d0["scope"] == source]
            if d.empty:
                ax.set_visible(False)
                continue
            agg = d.groupby("generation", as_index=False).agg(
                oracle=("pool_oracle_improvement", "mean"),
                realized=("realized_improvement", "mean"),
            )
            ax.plot(agg["generation"], agg["oracle"], linewidth=2, label="oracle available")
            ax.plot(agg["generation"], agg["realized"], linewidth=2, label="realized")
            ax.set_title(source)
            ax.set_xlabel("Generation")
            ax.set_ylabel("Mean objective improvement")
            ax.grid(alpha=0.22)
            ax.legend(fontsize=8)
        fig.suptitle(f"Available vs realized improvement: {alg}")
        fig.tight_layout()
        fig.savefig(outdir / f"candidate_pool_oracle_vs_realized_improvement_{_safe_name(str(alg))}.png", dpi=220)
        plt.close(fig)

# =============================================================================
# Output orchestration
# =============================================================================

def final_performance_table(records: Sequence[RunRecord]) -> pd.DataFrame:
    rows = []
    for rec in records:
        val = rec.best_val
        if val is None and rec.history_best is not None and len(rec.history_best):
            val = float(rec.history_best[-1])
        rows.append({
            "run": rec.run,
            "instance": rec.instance,
            "algorithm": rec.algorithm,
            "trial": rec.trial,
            "total_fe": rec.total_fe,
            "final_best": val,
        })
    return pd.DataFrame(rows)


def sanity_table(records: Sequence[RunRecord], selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in records:
        g = selected[selected["run"] == rec.run]
        alignment_max = float(g["archive_alignment_error"].max()) if not g.empty and np.any(np.isfinite(g["archive_alignment_error"])) else np.nan
        n_updates = int(g["is_best_update"].sum()) if not g.empty else 0
        rows.append({
            "run": rec.run,
            "instance": rec.instance,
            "algorithm": rec.algorithm,
            "trial": rec.trial,
            "total_fe": rec.total_fe,
            "logged_post_initial_true_eval": len(g),
            "estimated_initial_fe": (rec.total_fe - len(g)) if rec.total_fe else np.nan,
            "n_best_updates_logged": n_updates,
            "updates_le_total_fe": bool(n_updates <= rec.total_fe) if rec.total_fe else True,
            "max_archive_alignment_error": alignment_max,
            "n_generations": n_generations(rec.logs),
            "has_embeddings": any(_as_2d_or_none(x, float) is not None for x in rec.logs.get("emb", []) if x is not None),
        })
    return pd.DataFrame(rows)


def write_instance_outputs(
    records: Sequence[RunRecord],
    outdir: Path,
    metric: str,
    max_pairs_per_generation: int,
    max_scatter_points: int,
    projection: str,
    projection_generations: Sequence[str],
    projection_max_runs: int,
    projection_max_points: int,
    fe_bins: int,
    stagnation_fraction: float,
    no_plots: bool,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    selected_frames = [selected_rows(r) for r in records]
    selected = pd.concat([x for x in selected_frames if not x.empty], ignore_index=True) if any(not x.empty for x in selected_frames) else pd.DataFrame()
    selected.to_csv(outdir / "selected_true_evaluations.csv", index=False)

    by_run, update_overall = update_summaries(selected, records)
    by_run.to_csv(outdir / "best_update_by_run_and_source.csv", index=False)
    update_overall.to_csv(outdir / "best_update_summary_by_algorithm_and_source.csv", index=False)

    # FE-position analysis of global/incumbent evaluations and best updates.
    selected_with_fe, update_events, timing_by_run_bin, timing_bin_summary, phase_summary, importance_indicators = (
        best_update_timing_analysis(
            selected,
            n_bins=fe_bins,
            stagnation_fraction=stagnation_fraction,
        )
    )
    selected_with_fe.to_csv(outdir / "selected_true_evaluations_with_fe_position.csv", index=False)
    update_events.to_csv(outdir / "best_update_events_with_fe_position.csv", index=False)
    timing_by_run_bin.to_csv(outdir / "best_update_timing_by_run_fe_bin.csv", index=False)
    timing_bin_summary.to_csv(outdir / "best_update_timing_summary_by_fe_bin.csv", index=False)
    phase_summary.to_csv(outdir / "best_update_contribution_by_phase.csv", index=False)
    importance_indicators.to_csv(outdir / "global_local_importance_indicators.csv", index=False)

    followup_sequences, followup_summary = global_incumbent_followup_sequences(update_events)
    followup_sequences.to_csv(outdir / "global_update_incumbent_followup_sequences.csv", index=False)
    followup_summary.to_csv(outdir / "global_update_incumbent_followup_summary.csv", index=False)

    final_df = final_performance_table(records)
    final_df.to_csv(outdir / "final_performance_by_run.csv", index=False)

    sanity = sanity_table(records, selected)
    sanity.to_csv(outdir / "log_sanity_check.csv", index=False)

    pairs, pair_summary = embedding_objective_pairs(records, metric, max_pairs_per_generation)
    pairs.to_csv(outdir / "embedding_distance_objective_pairs_within_generation.csv", index=False)
    pair_summary.to_csv(outdir / "embedding_distance_objective_correlation_by_run.csv", index=False)
    if not pair_summary.empty:
        pair_alg = (
            pair_summary.groupby(["instance", "algorithm"], as_index=False)
            .agg(
                n_trials=("run", "nunique"),
                mean_spearman_raw=("spearman_raw", "mean"),
                median_spearman_raw=("spearman_raw", "median"),
                mean_spearman_generation_normalized=("spearman_generation_normalized", "mean"),
                median_spearman_generation_normalized=("spearman_generation_normalized", "median"),
                total_pairs=("n_pairs", "sum"),
            )
        )
    else:
        pair_alg = pd.DataFrame()
    pair_alg.to_csv(outdir / "embedding_distance_objective_correlation_by_algorithm.csv", index=False)

    global_df = global_selection_rows(records, metric)
    global_df = merge_global_outcomes(global_df, selected)
    global_df.to_csv(outdir / "global_latent_selection_by_generation.csv", index=False)
    if not global_df.empty:
        global_summary = (
            global_df.groupby(["instance", "algorithm"], as_index=False)
            .agg(
                n_runs=("run", "nunique"),
                n_generations=("generation", "size"),
                changed_selection_fraction=("latent_changed_selection", "mean"),
                mean_fraction_outside_score_topk=("fraction_selected_outside_score_topk", "mean"),
                mean_score_rank_penalty=("mean_rank_delta", "mean"),
                mean_pairwise_diversity_gain=("mean_pairwise_diversity_gain", "mean"),
                mean_min_pairwise_diversity_gain=("min_pairwise_diversity_gain", "mean"),
                mean_threshold_fallback_fraction=("fraction_selected_below_threshold", "mean"),
                total_global_updates=("global_best_update", "sum"),
                total_global_improvement=("global_total_improvement", "sum"),
            )
        )
    else:
        global_summary = pd.DataFrame()
    global_summary.to_csv(outdir / "global_latent_selection_summary_by_algorithm.csv", index=False)

    pred_rows = surrogate_prediction_rows(records, selected)
    pred_rows.to_csv(outdir / "surrogate_prediction_rows.csv", index=False)
    acc_by_run, acc_overall = accuracy_metrics(pred_rows)
    acc_by_run.to_csv(outdir / "surrogate_accuracy_by_run_scope.csv", index=False)
    acc_overall.to_csv(outdir / "surrogate_accuracy_summary_by_algorithm_scope.csv", index=False)

    # Full-pool surrogate accuracy: all global_pool and incumbent candidates have cand_true.
    pool_by_gen = candidate_pool_accuracy_by_generation(records)
    pool_by_gen.to_csv(outdir / "candidate_pool_accuracy_by_generation.csv", index=False)
    pool_by_run, pool_summary = summarize_candidate_pool_accuracy(pool_by_gen)
    pool_by_run.to_csv(outdir / "candidate_pool_accuracy_by_run_scope.csv", index=False)
    pool_summary.to_csv(outdir / "candidate_pool_accuracy_summary_by_algorithm_scope.csv", index=False)

    # Does each global/incumbent pool contain a true improvement, and was it captured?
    opportunity_by_gen, opportunity_summary, opportunity_comparison, opportunity_comparison_summary = (
        candidate_pool_improvement_opportunity(records, selected_with_fe)
    )
    opportunity_by_gen.to_csv(outdir / "candidate_pool_improvement_opportunity_by_generation.csv", index=False)
    opportunity_summary.to_csv(outdir / "candidate_pool_improvement_opportunity_summary.csv", index=False)
    opportunity_comparison.to_csv(outdir / "candidate_pool_opportunity_comparison_by_generation.csv", index=False)
    opportunity_comparison_summary.to_csv(outdir / "candidate_pool_opportunity_comparison_summary.csv", index=False)

    # Embedding structure in archive/global/incumbent/all-candidate sets.
    emb_structure = embedding_structure_by_generation(records, metric, max_pairs_per_generation)
    emb_structure.to_csv(outdir / "embedding_structure_by_generation.csv", index=False)
    arc_cand_relation = archive_candidate_relation_by_generation(records, metric)
    arc_cand_relation.to_csv(outdir / "archive_candidate_embedding_relation_by_generation.csv", index=False)
    arc_updates = archive_embedding_updates(records, metric)
    arc_updates.to_csv(outdir / "archive_embedding_updates_between_generations.csv", index=False)

    # Explicit totals across trials, in addition to per-trial means/dots.
    update_totals = best_update_total_table(by_run)
    update_totals.to_csv(outdir / "best_update_total_global_vs_incumbent.csv", index=False)

    # Attribute the last and the largest update in each trial.
    attribution_rows = []
    if not selected.empty:
        for run, g in selected.groupby("run"):
            updates = g[g["is_best_update"]].sort_values("fe")
            base = g.iloc[0]
            attribution_rows.append({
                "run": run,
                "instance": base["instance"],
                "algorithm": base["algorithm"],
                "trial": base["trial"],
                "n_updates": len(updates),
                "last_update_source": updates.iloc[-1]["source_group"] if len(updates) else "none",
                "last_update_fe": updates.iloc[-1]["fe"] if len(updates) else np.nan,
                "largest_update_source": updates.loc[updates["improvement"].idxmax(), "source_group"] if len(updates) else "none",
                "largest_update_size": updates["improvement"].max() if len(updates) else 0.0,
            })
    pd.DataFrame(attribution_rows).to_csv(outdir / "best_update_source_attribution_by_run.csv", index=False)

    summary = {
        "instance": records[0].instance if records else "",
        "algorithms": sorted(set(r.algorithm for r in records)),
        "n_runs": len(records),
        "distance_metric": metric,
        "important_interpretation": {
            "best_update_count": "Per-trial number of true evaluations that reduced the incumbent objective.",
            "global_source": "Final true-evaluated candidate produced from a global-selected root neighborhood.",
            "incumbent_source": "Final true-evaluated candidate produced directly around the current true incumbent.",
            "global_selection_counterfactual": "Selection logs show whether latent filtering changed candidates and diversity. Objective benefit requires a separate score-top-k ablation because unselected candidates have no true labels.",
            "embedding_pairing": "Pairwise distances are computed only within a generation because the surrogate and embedding coordinates may change after retraining.",
            "fe_timing": "Search progress excludes the initial design because those evaluations do not have global/incumbent sources.",
            "stagnation_break": f"An update is marked as a stagnation break when the gap since the previous best update is at least {float(stagnation_fraction):.3f} of the post-initial search budget.",
            "global_followup": "Incumbent follow-up is a temporal association after a global update, not proof that both updates belong to the same basin unless candidate lineage is logged.",
            "pool_opportunity": "Pool oracle improvement uses cand_true for every generated candidate and is analysis-only; it is not counted as search FE.",
        },
        "sanity_failures": sanity[
            (~sanity["updates_le_total_fe"]) |
            (np.isfinite(sanity["max_archive_alignment_error"]) & (sanity["max_archive_alignment_error"] > 1e-8))
        ].to_dict(orient="records") if not sanity.empty else [],
    }
    with open(outdir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not no_plots:
        plot_update_counts(by_run, records, outdir)
        plot_update_rate(by_run, outdir)
        plot_total_improvement(by_run, outdir)
        plot_final_best(records, outdir)
        plot_embedding_relation(pairs, outdir, max_scatter_points)
        plot_global_selection(global_df, outdir)
        plot_accuracy(acc_by_run, pred_rows, outdir)
        plot_candidate_pool_metric(pool_by_gen, "kendall_tau_b", "Full-pool Kendall tau-b", "candidate_pool_kendall_by_generation.png", outdir)
        plot_candidate_pool_metric(pool_by_gen, "spearman", "Full-pool Spearman correlation", "candidate_pool_spearman_by_generation.png", outdir)
        plot_candidate_pool_metric(pool_by_gen, "mae", "Full-pool MAE", "candidate_pool_mae_by_generation.png", outdir)
        plot_candidate_pool_metric(pool_by_gen, "top10_overlap", "Predicted/true top-10% overlap", "candidate_pool_top10_overlap_by_generation.png", outdir)
        plot_embedding_structure_curves(emb_structure, outdir)
        plot_archive_candidate_novelty(arc_cand_relation, outdir)
        plot_archive_embedding_updates(arc_updates, outdir)
        plot_best_update_totals(update_totals, outdir)
        plot_best_update_event_timelines(update_events, outdir)
        plot_best_update_by_fe_bins(timing_bin_summary, outdir)
        plot_cumulative_update_contribution(selected_with_fe, update_events, outdir)
        plot_candidate_pool_improvement_opportunity(opportunity_by_gen, outdir)
        plot_embedding_projections(
            records=records,
            outdir=outdir,
            method=projection,
            generation_specs=projection_generations,
            max_runs_per_algorithm=projection_max_runs,
            max_points=projection_max_points,
        )
        plot_archive_candidate_projections(
            records=records,
            outdir=outdir,
            method=projection,
            generation_specs=projection_generations,
            max_runs_per_algorithm=projection_max_runs,
            max_points=projection_max_points,
        )

    print(f"Saved: {outdir}")
    print("  - selected_true_evaluations.csv")
    print("  - best_update_by_run_and_source.csv")
    print("  - global_latent_selection_by_generation.csv")
    print("  - surrogate_accuracy_summary_by_algorithm_scope.csv")
    print("  - candidate_pool_accuracy_by_generation.csv")
    print("  - embedding_structure_by_generation.csv")
    print("  - archive_candidate_embedding_relation_by_generation.csv")
    print("  - archive_embedding_updates_between_generations.csv")
    print("  - best_update_total_global_vs_incumbent.csv")
    print("  - best_update_events_with_fe_position.csv")
    print("  - best_update_timing_summary_by_fe_bin.csv")
    print("  - global_local_importance_indicators.csv")
    print("  - global_update_incumbent_followup_summary.csv")
    print("  - candidate_pool_improvement_opportunity_by_generation.csv")
    print("  - candidate_pool_opportunity_comparison_summary.csv")
    print("  - embedding_distance_objective_correlation_by_algorithm.csv")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs="*", default=None, help="Files, directories, or glob patterns.")
    parser.add_argument("--base-input", default="E:\\2325_yamaguchi\\_results_TEVC_20260709\\", help="Root containing INSTANCE/ALGORITHM/trial_XX/result.npz.")
    parser.add_argument("--instance", nargs="*", default=["N-econ36"], help="Instance names under --base-input.")
    parser.add_argument("--algorithm", nargs="*", default=["lat2way_sage_mean_knn_density_0.2_2"], help="Algorithm directory names or glob patterns under each instance.")
    parser.add_argument("--outdir", default="E:\\2325_yamaguchi\\_results_TEVC_20260709\\_lat_ana", help="Output root. One subdirectory is created per instance.")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--max-pairs-per-generation", type=int, default=5000)
    parser.add_argument("--max-scatter-points", type=int, default=40000)
    parser.add_argument("--projection", choices=["pca", "umap", "tsne", "none"], default="umap")
    parser.add_argument("--projection-generations", nargs="+", default=["first,middle,last"], help="first,middle,last, integer indices, or comma-separated values.")
    parser.add_argument("--projection-max-runs", type=int, default=1, help="Number of trials projected per algorithm.")
    parser.add_argument("--projection-max-points", type=int, default=3000)
    parser.add_argument("--fe-bins", type=int, default=5, help="Number of equal-width post-initial FE bins for timing analysis.")
    parser.add_argument("--stagnation-fraction", type=float, default=0.20, help="Gap fraction of post-initial search budget used to flag stagnation-breaking updates.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(f"[analyze_latent_2way_logs] version={ANALYSIS_SCRIPT_VERSION}")

    specs = discover_files(args.input, args.base_input, args.instance, args.algorithm)
    if not specs:
        raise FileNotFoundError("No result files found. Check --base-input/--instance/--algorithm or --input.")

    records: List[RunRecord] = []
    failed: List[Tuple[str, str]] = []
    for spec in specs:
        try:
            records.append(load_run(spec))
        except Exception as e:
            failed.append((str(spec.path), str(e)))
            warnings.warn(f"Skipped {spec.path}: {e}")
    if not records:
        raise RuntimeError("All files failed to load.")

    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)
    for instance, recs in sorted(pd.Series(records).groupby(lambda i: records[i].instance)):
        write_instance_outputs(
            records=list(recs),
            outdir=outroot / _safe_name(str(instance)),
            metric=args.metric,
            max_pairs_per_generation=max(1, int(args.max_pairs_per_generation)),
            max_scatter_points=max(100, int(args.max_scatter_points)),
            projection=args.projection,
            projection_generations=args.projection_generations,
            projection_max_runs=max(1, int(args.projection_max_runs)),
            projection_max_points=max(50, int(args.projection_max_points)),
            fe_bins=max(2, int(args.fe_bins)),
            stagnation_fraction=min(1.0, max(0.0, float(args.stagnation_fraction))),
            no_plots=bool(args.no_plots),
        )

    if failed:
        pd.DataFrame(failed, columns=["path", "error"]).to_csv(outroot / "failed_files.csv", index=False)


if __name__ == "__main__":
    main()
