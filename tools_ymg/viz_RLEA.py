#!/usr/bin/env python3
"""Visualize and compare permutation search trajectories from result.npz logs.

The expected experiment layout matches scripts/main.py:

    ROOT / INSTANCE / METHOD / trial_XX / result.npz

Each result.npz is expected to contain archive_perm and archive_fx. All selected
methods/trials are embedded together so every panel shares one coordinate system.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


INSTANCE_CHOICES = [
    "burma14", "ulysses22", "fri26", "bayg29", "swiss42", "att48", "berlin52",
    "br17", "ftv33", "ftv35", "ftv38", "p43", "ry48p", "ft53",
    "N-pal11", "N-pal19", "N-pal23", "N-pal27", "N-econ36", "N-p40-01",
    "N-p44-01", "N-be75np",
]

Perm = Sequence[int]
DistanceFunc = Callable[[Perm, Perm], float]
Row = Dict[str, Any]


# =============================================================================
# Permutation distances
# =============================================================================

def hamming_distance(p1: Perm, p2: Perm) -> float:
    return float(np.sum(np.asarray(p1) != np.asarray(p2)))


def undirected_adjacency_distance(p1: Perm, p2: Perm, cyclic: bool = True) -> float:
    def edge_set(p: Perm) -> set[frozenset[int]]:
        edges = {frozenset((int(p[i]), int(p[i + 1]))) for i in range(len(p) - 1)}
        if cyclic and len(p) > 1:
            edges.add(frozenset((int(p[-1]), int(p[0]))))
        return edges

    return float(len(edge_set(p1).symmetric_difference(edge_set(p2))) / 2.0)


def directed_adjacency_distance(p1: Perm, p2: Perm, cyclic: bool = True) -> float:
    def edge_set(p: Perm) -> set[Tuple[int, int]]:
        edges = {(int(p[i]), int(p[i + 1])) for i in range(len(p) - 1)}
        if cyclic and len(p) > 1:
            edges.add((int(p[-1]), int(p[0])))
        return edges

    return float(len(edge_set(p1).symmetric_difference(edge_set(p2))) / 2.0)


def kendall_tau_distance(p1: Perm, p2: Perm) -> float:
    pos1 = {int(item): i for i, item in enumerate(p1)}
    pos2 = {int(item): i for i, item in enumerate(p2)}
    items = [int(v) for v in p1]
    distance = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if (pos1[a] < pos1[b]) != (pos2[a] < pos2[b]):
                distance += 1
    return float(distance)


def interchange_distance(p1: Perm, p2: Perm) -> float:
    """Minimum number of arbitrary swaps needed to transform p1 into p2."""
    p1 = [int(v) for v in p1]
    p2 = [int(v) for v in p2]
    pos2 = {v: i for i, v in enumerate(p2)}
    perm = [pos2[v] for v in p1]
    seen = [False] * len(perm)
    cycles = 0
    for i in range(len(perm)):
        if seen[i]:
            continue
        cycles += 1
        j = i
        while not seen[j]:
            seen[j] = True
            j = perm[j]
    return float(len(perm) - cycles)


def get_distance_func(name: str, problem: str, cyclic: bool) -> Tuple[str, DistanceFunc]:
    if name == "auto":
        p = problem.upper()
        if p == "TSP":
            name = "adjacency_undirected"
        elif p == "ATSP":
            name = "adjacency_directed"
        elif p == "LOP":
            name = "kendall"
        else:
            name = "hamming"

    if name == "hamming":
        return name, hamming_distance
    if name == "kendall":
        return name, kendall_tau_distance
    if name == "interchange":
        return name, interchange_distance
    if name == "adjacency_undirected":
        return name, lambda a, b: undirected_adjacency_distance(a, b, cyclic=cyclic)
    if name == "adjacency_directed":
        return name, lambda a, b: directed_adjacency_distance(a, b, cyclic=cyclic)
    raise ValueError(f"Unknown distance: {name}")


# =============================================================================
# Loading result.npz trajectories
# =============================================================================

def safe_load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def unwrap_object(x: Any) -> Any:
    if isinstance(x, np.ndarray) and x.shape == () and x.dtype == object:
        return x.item()
    return x


def flatten_archive_perm(archive_perm: Any) -> np.ndarray:
    arr = unwrap_object(archive_perm)
    if arr is None:
        return np.zeros((0, 0), dtype=np.int32)
    raw = np.asarray(arr)
    if raw.dtype == object:
        rows: List[np.ndarray] = []
        for item in raw:
            item = unwrap_object(item)
            sub = np.asarray(item, dtype=np.int32)
            if sub.size == 0:
                continue
            if sub.ndim == 1:
                rows.append(sub.reshape(1, -1))
            elif sub.ndim == 2:
                rows.append(sub)
        if not rows:
            return np.zeros((0, 0), dtype=np.int32)
        return np.vstack(rows).astype(np.int32)
    arr2 = np.asarray(arr, dtype=np.int32)
    if arr2.ndim == 1:
        return arr2.reshape(1, -1)
    if arr2.ndim == 2:
        return arr2
    return arr2.reshape(-1, arr2.shape[-1]).astype(np.int32)


def flatten_archive_fx(archive_fx: Any) -> np.ndarray:
    arr = unwrap_object(archive_fx)
    if arr is None:
        return np.zeros((0,), dtype=np.float64)
    raw = np.asarray(arr)
    if raw.dtype == object:
        vals: List[float] = []
        for item in raw:
            item = unwrap_object(item)
            try:
                vals.extend(np.asarray(item, dtype=np.float64).reshape(-1).tolist())
            except Exception:
                try:
                    vals.append(float(item))
                except Exception:
                    pass
        return np.asarray(vals, dtype=np.float64)
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def load_meta(method_dir: Path) -> Dict[str, Any]:
    meta_path = method_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def infer_problem(method_dirs: Sequence[Path], default: str) -> str:
    if default != "auto":
        return default
    for method_dir in method_dirs:
        problem = str(load_meta(method_dir).get("problem", "")).upper()
        if problem:
            return problem
    return "TSP"


def collect_method_dirs(root: Path, ins_name: str) -> List[Path]:
    ins_dir = root / ins_name
    if not ins_dir.exists():
        raise FileNotFoundError(f"Instance folder not found: {ins_dir}")
    return [p for p in ins_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]


def match_methods(method_dirs: Sequence[Path], includes: Sequence[str], excludes: Sequence[str]) -> List[Path]:
    if includes:
        inc = [s.lower() for s in includes]
        selected = [p for p in method_dirs if any(s in p.name.lower() for s in inc)]
    else:
        selected = list(method_dirs)
    if excludes:
        exc = [s.lower().replace("*", "") for s in excludes]
        selected = [p for p in selected if not any(s and s in p.name.lower() for s in exc)]
    return sorted(selected, key=lambda p: p.name)


def parse_trials(items: Sequence[str]) -> Optional[set[int]]:
    if not items or any(str(x).lower() == "all" for x in items):
        return None
    out: set[int] = set()
    for item in items:
        s = str(item).strip()
        if not s:
            continue
        if "-" in s:
            a, b = s.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(s))
    return out


def parse_trial_id(trial_dir: Path) -> int:
    m = re.search(r"trial_(\d+)", trial_dir.name)
    if m:
        return int(m.group(1))
    return -1


def parse_legend_map(items: Sequence[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for s in items:
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            out.append((k, v))
    return out


def legend_name(method_name: str, legend_map: Sequence[Tuple[str, str]]) -> str:
    for k, v in legend_map:
        if method_name == k:
            return v
    for k, v in sorted(legend_map, key=lambda item: len(item[0]), reverse=True):
        if k in method_name:
            return v
    return method_name


def load_trajectories(
    method_dirs: Sequence[Path],
    trial_filter: Optional[set[int]],
    max_eval: Optional[int],
    eval_stride: int,
    max_points_per_method: int,
    legend_map: Sequence[Tuple[str, str]],
) -> List[Row]:
    rows: List[Row] = []
    eval_stride = max(1, int(eval_stride))

    for method_dir in method_dirs:
        method_rows_start = len(rows)
        trial_dirs = sorted(
            [p for p in method_dir.iterdir() if p.is_dir() and p.name.startswith("trial_")],
            key=parse_trial_id,
        )
        for trial_dir in trial_dirs:
            trial_id = parse_trial_id(trial_dir)
            if trial_filter is not None and trial_id not in trial_filter:
                continue
            npz_path = trial_dir / "result.npz"
            if not npz_path.exists():
                continue
            try:
                data = safe_load_npz(npz_path)
                perms = flatten_archive_perm(data.get("archive_perm"))
                fx = flatten_archive_fx(data.get("archive_fx"))
            except Exception as e:
                print(f"[WARN] failed to load {npz_path}: {e}")
                continue
            n = min(len(perms), len(fx))
            if max_eval is not None:
                n = min(n, int(max_eval))
            if n <= 0:
                continue
            for i in range(0, n, eval_stride):
                rows.append({
                    "instance": method_dir.parent.name,
                    "method": method_dir.name,
                    "display_method": legend_name(method_dir.name, legend_map),
                    "trial": int(trial_id),
                    "fe": int(i + 1),
                    "objective": float(fx[i]),
                    "permutation": " ".join(str(int(v)) for v in perms[i]),
                    "result_path": str(npz_path),
                })

        if max_points_per_method > 0:
            method_rows = rows[method_rows_start:]
            if len(method_rows) > max_points_per_method:
                keep_idx = np.linspace(0, len(method_rows) - 1, max_points_per_method, dtype=int)
                del rows[method_rows_start:]
                rows.extend([method_rows[int(i)] for i in keep_idx])

    return rows


# =============================================================================
# Embedding and plotting
# =============================================================================

def build_distance_matrix(permutations: Sequence[Perm], distance_func: DistanceFunc) -> np.ndarray:
    n = len(permutations)
    total_pairs = n * (n - 1) // 2
    dmat = np.zeros((n, n), dtype=np.float64)
    print(f"[INFO] building pairwise distance matrix: n={n}, pairs={total_pairs}")

    try:
        from tqdm import tqdm
        row_iter = tqdm(range(n), total=n, desc="distance rows", unit="row")
    except Exception:
        row_iter = range(n)

    done_pairs = 0
    for i in row_iter:
        row_pairs = n - i - 1
        for j in range(i + 1, n):
            d = distance_func(permutations[i], permutations[j])
            dmat[i, j] = d
            dmat[j, i] = d
        done_pairs += row_pairs
        if hasattr(row_iter, "set_postfix"):
            row_iter.set_postfix(pairs=f"{done_pairs}/{total_pairs}")
        elif i % 100 == 0 and n >= 300:
            print(f"  distance row {i}/{n}, pairs={done_pairs}/{total_pairs}")
    return dmat


def run_mds(distance_matrix: np.ndarray, random_state: int) -> Tuple[np.ndarray, float, str]:
    try:
        from sklearn.manifold import MDS

        kwargs: Dict[str, Any] = dict(
            n_components=2,
            dissimilarity="precomputed",
            random_state=random_state,
            n_init=10,
            max_iter=1000,
        )
        try:
            mds = MDS(normalized_stress="auto", **kwargs)
        except TypeError:
            mds = MDS(**kwargs)
        coords = mds.fit_transform(distance_matrix)
        return coords, float(mds.stress_), "metric_mds"
    except Exception as e:
        print(f"[WARN] sklearn MDS unavailable ({e}); using classical MDS fallback")
        d2 = np.square(distance_matrix.astype(np.float64))
        n = d2.shape[0]
        eye = np.eye(n)
        ones = np.ones((n, n), dtype=np.float64) / max(1, n)
        jmat = eye - ones
        bmat = -0.5 * jmat @ d2 @ jmat
        eigvals, eigvecs = np.linalg.eigh(bmat)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order[:2]]
        eigvecs = eigvecs[:, order[:2]]
        eigvals = np.maximum(eigvals, 0.0)
        coords = eigvecs * np.sqrt(eigvals)
        fitted = np.sqrt(np.maximum(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2), 0.0))
        stress = float(np.sum((distance_matrix - fitted) ** 2))
        return coords, stress, "classical_mds"


def write_csv(path: Path, rows: Sequence[Row], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> List[Row]:
    rows: List[Row] = []
    int_fields = {"trial", "fe", "is_initial", "fe_start", "fe_end", "n_points"}
    float_fields = {"objective", "x", "y", "centroid_x", "centroid_y", "best_objective_in_bin"}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: Row = {}
            for key, value in raw.items():
                if value is None:
                    row[key] = value
                    continue
                value = value.strip()
                if key in int_fields and value:
                    row[key] = int(float(value))
                elif key in float_fields and value:
                    row[key] = float(value)
                else:
                    row[key] = value
            rows.append(row)
    return rows


def filter_saved_rows(
    rows: Sequence[Row],
    includes: Sequence[str],
    excludes: Sequence[str],
    trial_filter: Optional[set[int]],
    max_eval: Optional[int],
    eval_stride: int,
    max_points_per_method: int,
    legend_map: Sequence[Tuple[str, str]],
) -> List[Row]:
    inc = [s.lower() for s in includes]
    exc = [s.lower().replace("*", "") for s in excludes]
    eval_stride = max(1, int(eval_stride))
    out: List[Row] = []
    for row in rows:
        method = str(row.get("method", ""))
        method_l = method.lower()
        if inc and not any(s in method_l for s in inc):
            continue
        if exc and any(s and s in method_l for s in exc):
            continue
        trial = int(row["trial"])
        fe = int(row["fe"])
        if trial_filter is not None and trial not in trial_filter:
            continue
        if max_eval is not None and fe > int(max_eval):
            continue
        if (fe - 1) % eval_stride != 0:
            continue
        row = dict(row)
        row["display_method"] = legend_name(method, legend_map)
        row["is_initial"] = int(row.get("is_initial", 0))
        out.append(row)

    if max_points_per_method > 0:
        limited: List[Row] = []
        for method in sorted({str(r["method"]) for r in out}):
            method_rows = [r for r in out if str(r["method"]) == method]
            if len(method_rows) > max_points_per_method:
                keep_idx = np.linspace(0, len(method_rows) - 1, max_points_per_method, dtype=int)
                method_rows = [method_rows[int(i)] for i in keep_idx]
            limited.extend(method_rows)
        out = limited
    return out


def filter_centroid_rows(
    rows: Sequence[Row],
    display_methods: Sequence[str],
    trial_filter: Optional[set[int]],
) -> List[Row]:
    display_set = set(display_methods)
    out: List[Row] = []
    for row in rows:
        if str(row.get("display_method", "")) not in display_set:
            continue
        if trial_filter is not None and int(row["trial"]) not in trial_filter:
            continue
        out.append(row)
    return out


def find_saved_centroid_csv(out_dir: Path, ins_name: str, distance_name: str, centroid_step: int) -> Optional[Path]:
    candidates = [
        out_dir / f"{ins_name}_searchspace_{distance_name}_centroid_flow_color_step{centroid_step}.csv",
        out_dir / f"{ins_name}_searchspace_{distance_name}_centroid_flow_step{centroid_step}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_saved_mds_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def select_temporal_rows(rows: Sequence[Row], cutoff: int, mode: str, window: int) -> List[Row]:
    cutoff = int(cutoff)
    if mode == "window":
        lo = max(1, cutoff - max(1, int(window)) + 1)
        return [r for r in rows if lo <= int(r["fe"]) <= cutoff]
    return [r for r in rows if int(r["fe"]) <= cutoff]


def build_time_cutoffs(rows: Sequence[Row], step: int) -> List[int]:
    max_fe = max(int(r["fe"]) for r in rows)
    step = max(1, int(step))
    cutoffs = list(range(step, max_fe + 1, step))
    if not cutoffs or cutoffs[-1] != max_fe:
        cutoffs.append(max_fe)
    return cutoffs


def make_summary(rows: Sequence[Row]) -> List[Row]:
    groups: Dict[Tuple[str, str, int], List[Row]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["display_method"]), int(row["trial"]))
        groups.setdefault(key, []).append(row)
    out: List[Row] = []
    for (method, display_method, trial), items in sorted(groups.items()):
        objectives = [float(r["objective"]) for r in items]
        fes = [int(r["fe"]) for r in items]
        out.append({
            "method": method,
            "display_method": display_method,
            "trial": trial,
            "n_points": len(items),
            "first_fe": min(fes),
            "last_fe": max(fes),
            "best_objective": min(objectives),
        })
    return out


def build_display_method_order(
    rows: Sequence[Row],
    includes: Sequence[str],
    legend_map: Sequence[Tuple[str, str]],
) -> List[str]:
    existing = list(dict.fromkeys(str(r["display_method"]) for r in rows))
    if not includes:
        return existing

    out: List[str] = []
    for include in includes:
        pattern = str(include).lower().replace("*", "")
        if not pattern:
            continue
        for row in rows:
            method = str(row.get("method", ""))
            if pattern in method.lower():
                display = legend_name(method, legend_map)
                if display not in out:
                    out.append(display)
    for display in existing:
        if display not in out:
            out.append(display)
    return out


def apply_outer_axis_labels(ax: Any, ax_idx: int, ncols: int, nrows: int, xlabel: str, ylabel: str) -> None:
    is_left = (ax_idx % ncols) == 0
    is_bottom = (ax_idx // ncols) == (nrows - 1)
    ax.set_xlabel(xlabel if is_bottom else "")
    ax.set_ylabel(ylabel if is_left else "")
    ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)


def canonical_permutation_key(perm_text: Any, equivalence: str) -> Tuple[int, ...]:
    vals = tuple(int(x) for x in str(perm_text).split())
    if not vals:
        return vals
    equivalence = str(equivalence).lower()
    if equivalence == "exact":
        return vals
    variants = [vals]
    if equivalence in ("cyclic", "cyclic_reverse"):
        variants = [vals[i:] + vals[:i] for i in range(len(vals))]
    if equivalence == "cyclic_reverse":
        rev = tuple(reversed(vals))
        variants.extend(rev[i:] + rev[:i] for i in range(len(rev)))
    return min(variants)


def aggregate_duplicate_points(
    rows: Sequence[Row],
    duplicate_equivalence: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    groups: Dict[Tuple[int, ...], List[Row]] = {}
    for row in rows:
        key = canonical_permutation_key(row.get("permutation", ""), duplicate_equivalence)
        groups.setdefault(key, []).append(row)

    out_rows: List[Row] = []
    counts: List[int] = []
    for items in groups.values():
        representative = min(items, key=lambda r: int(r["fe"]))
        representative = dict(representative)
        representative["objective"] = min(float(r["objective"]) for r in items)
        out_rows.append(representative)
        counts.append(len(items))

    out_rows.sort(key=lambda r: int(r["fe"]))
    count_by_key = {
        canonical_permutation_key(row.get("permutation", ""), duplicate_equivalence): count
        for row, count in zip([min(items, key=lambda r: int(r["fe"])) for items in groups.values()], counts)
    }
    counts = [count_by_key[canonical_permutation_key(row.get("permutation", ""), duplicate_equivalence)] for row in out_rows]
    return (
        np.asarray([float(r["x"]) for r in out_rows], dtype=float),
        np.asarray([float(r["y"]) for r in out_rows], dtype=float),
        np.asarray([float(r["objective"]) for r in out_rows], dtype=float),
        np.asarray([int(r["fe"]) for r in out_rows], dtype=int),
        np.asarray(counts, dtype=float),
    )


def add_figure_legend(fig: Any, axes: Sequence[Any], fontsize: int = 8) -> None:
    dedup: Dict[str, Any] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and label not in dedup:
                dedup[label] = handle
    if dedup:
        ncol = min(4, max(1, len(dedup)))
        fig.legend(
            dedup.values(),
            dedup.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=ncol,
            frameon=True,
            fontsize=fontsize,
            markerscale=1.15,
            borderpad=0.55,
            labelspacing=0.45,
            handletextpad=0.55,
        )


def save_plots(
    rows: Sequence[Row],
    out_dir: Path,
    ins_name: str,
    distance_name: str,
    fmt: str,
    cmap: str,
    show_arrows: bool,
    n_init: int,
    arrow_every: int,
    arrow_color: str,
    arrow_lw: float,
    arrow_scale: float,
    show_lines: bool,
    suffix: str = "",
    title_suffix: str = "",
    extent_rows: Optional[Sequence[Row]] = None,
    temporal_encoding: str = "none",
    temporal_min_alpha: float = 0.18,
    temporal_size_min: float = 14.0,
    temporal_size_max: float = 58.0,
    centroid_rows: Optional[Sequence[Row]] = None,
    scale_duplicate_points: bool = False,
    duplicate_size_factor: float = 1.0,
    duplicate_equivalence: str = "cyclic_reverse",
    method_order: Optional[Sequence[str]] = None,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.patheffects as pe

    if not rows:
        raise ValueError("No rows to plot.")
    extent_source = list(extent_rows) if extent_rows is not None else list(rows)
    existing_methods = list(dict.fromkeys(str(r["display_method"]) for r in extent_source))
    if method_order is None:
        methods = existing_methods
    else:
        methods = [m for m in method_order if m in existing_methods]
        methods.extend(m for m in existing_methods if m not in methods)
    n_methods = len(methods)
    ncols = min(3, max(1, n_methods))
    nrows = int(math.ceil(n_methods / ncols))
    fig, axes_arr = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.7 * nrows), sharex=True, sharey=True)
    axes = np.asarray(axes_arr).reshape(-1)

    all_x = np.asarray([float(r["x"]) for r in extent_source], dtype=float)
    all_y = np.asarray([float(r["y"]) for r in extent_source], dtype=float)
    vmin = min(float(r["objective"]) for r in extent_source)
    vmax = max(float(r["objective"]) for r in extent_source)
    fe_min = min(int(r["fe"]) for r in extent_source)
    fe_max = max(int(r["fe"]) for r in extent_source)
    temporal_encoding = str(temporal_encoding).lower()
    encode_alpha = temporal_encoding in ("alpha", "both")
    encode_size = temporal_encoding in ("size", "both")
    obj_norm = plt.Normalize(vmin=vmin, vmax=vmax)
    obj_cmap = plt.get_cmap(cmap)

    def temporal_t(fe_values: np.ndarray) -> np.ndarray:
        denom = max(1, fe_max - fe_min)
        return np.clip((fe_values.astype(float) - fe_min) / denom, 0.0, 1.0)

    def temporal_alpha(fe_values: np.ndarray, default_alpha: float) -> np.ndarray | float:
        if not encode_alpha:
            return default_alpha
        t = temporal_t(fe_values)
        return float(temporal_min_alpha) + (1.0 - float(temporal_min_alpha)) * t

    def temporal_size(fe_values: np.ndarray, default_size: float) -> np.ndarray | float:
        if not encode_size:
            return default_size
        t = temporal_t(fe_values)
        return float(temporal_size_min) + (float(temporal_size_max) - float(temporal_size_min)) * t

    def objective_colors(values: np.ndarray, fe_values: np.ndarray, default_alpha: float) -> np.ndarray:
        rgba = obj_cmap(obj_norm(values))
        if encode_alpha:
            rgba[:, 3] = temporal_alpha(fe_values, default_alpha)
        else:
            rgba[:, 3] = default_alpha
        return rgba

    centroid_by_method: Dict[str, List[Row]] = {}
    if centroid_rows:
        for row in centroid_rows:
            centroid_by_method.setdefault(str(row["display_method"]), []).append(row)
    flow_colors = plt.get_cmap("tab10")

    def overlay_centroid_flow(ax: Any, method: str) -> None:
        method_centroids = centroid_by_method.get(method, [])
        for trial_pos, trial in enumerate(sorted({int(r["trial"]) for r in method_centroids})):
            trial_rows = sorted(
                [r for r in method_centroids if int(r["trial"]) == trial],
                key=lambda r: int(r["fe_start"]),
            )
            if not trial_rows:
                continue
            cx = np.asarray([float(r["centroid_x"]) for r in trial_rows], dtype=float)
            cy = np.asarray([float(r["centroid_y"]) for r in trial_rows], dtype=float)
            color = flow_colors(trial_pos % 10)
            ax.plot(cx, cy, color=color, lw=2.4, alpha=0.9, zorder=7, label=f"Centroid flow trial {trial}")
            ax.scatter(cx, cy, s=86, facecolor=color, edgecolor="white", linewidth=1.4, alpha=0.98, zorder=8)
            if len(cx) > 0:
                ax.scatter(cx[0], cy[0], marker="o", s=92, facecolor="none", edgecolor=color, linewidth=2.0, zorder=9)
                ax.scatter(cx[-1], cy[-1], marker="*", s=170, facecolor=color, edgecolor="black", linewidth=0.6, zorder=9)
            for i in range(len(cx) - 1):
                ax.annotate(
                    "",
                    xy=(cx[i + 1], cy[i + 1]),
                    xytext=(cx[i], cy[i]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color,
                        lw=2.7,
                        alpha=0.92,
                        mutation_scale=18,
                        shrinkA=7,
                        shrinkB=7,
                    ),
                    zorder=10,
                )
            for i in range(len(cx)):
                label = ax.text(
                    cx[i],
                    cy[i],
                    str(i + 1),
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                    zorder=12,
                )
                label.set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])

    scatters = []
    n_init = max(0, int(n_init))
    arrow_every = max(1, int(arrow_every))
    for ax_idx, (ax, method) in enumerate(zip(axes, methods)):
        init_label_done = False
        init_best_label_done = False
        search_label_done = False
        best_label_done = False
        method_rows = [r for r in rows if str(r["display_method"]) == method]
        for trial in sorted({int(r["trial"]) for r in method_rows}):
            trial_rows = sorted([r for r in method_rows if int(r["trial"]) == trial], key=lambda r: int(r["fe"]))
            x = np.asarray([float(r["x"]) for r in trial_rows], dtype=float)
            y = np.asarray([float(r["y"]) for r in trial_rows], dtype=float)
            obj = np.asarray([float(r["objective"]) for r in trial_rows], dtype=float)
            if len(x) == 0:
                continue

            fe = np.asarray([int(r["fe"]) for r in trial_rows], dtype=int)
            init_mask = fe <= n_init
            search_mask = ~init_mask

            x_search_raw = x[search_mask]
            y_search_raw = y[search_mask]
            search_points = np.column_stack([x_search_raw, y_search_raw])
            if scale_duplicate_points:
                search_rows = [r for r in trial_rows if int(r["fe"]) > n_init]
                x_search, y_search, obj_search, fe_search, search_dup_counts = aggregate_duplicate_points(
                    search_rows, duplicate_equivalence
                )
            else:
                x_search = x_search_raw
                y_search = y_search_raw
                obj_search = obj[search_mask]
                fe_search = fe[search_mask]
                search_dup_counts = np.ones(len(x_search), dtype=float)

            if show_lines and len(search_points) > 1:
                segments = np.stack([search_points[:-1], search_points[1:]], axis=1)
                ax.add_collection(LineCollection(segments, colors="0.18", linewidth=1.45, alpha=0.55, zorder=1))

            if len(x_search) > 0:
                search_sizes = np.asarray(temporal_size(fe_search, 30.0), dtype=float)
                search_sizes = search_sizes * np.maximum(1.0, float(duplicate_size_factor)) * search_dup_counts
                sc = ax.scatter(
                    x_search,
                    y_search,
                    c=objective_colors(obj_search, fe_search, 0.82),
                    s=search_sizes,
                    edgecolor="black",
                    linewidth=0.25,
                    zorder=2,
                    label="Search evaluations" if not search_label_done else None,
                )
                search_label_done = True
                scatters.append(sc)

            if np.any(init_mask):
                if scale_duplicate_points:
                    init_rows = [r for r in trial_rows if int(r["fe"]) <= n_init]
                    x_init, y_init, obj_init, fe_init, init_dup_counts = aggregate_duplicate_points(
                        init_rows, duplicate_equivalence
                    )
                else:
                    x_init = x[init_mask]
                    y_init = y[init_mask]
                    fe_init = fe[init_mask]
                    obj_init = obj[init_mask]
                    init_dup_counts = np.ones(len(x_init), dtype=float)
                init_sizes = np.asarray(temporal_size(fe_init, 34.0), dtype=float)
                init_sizes = init_sizes * np.maximum(1.0, float(duplicate_size_factor)) * init_dup_counts
                init_sc = ax.scatter(
                    x_init,
                    y_init,
                    c=objective_colors(obj_init, fe_init, 0.92),
                    marker="s",
                    s=init_sizes,
                    edgecolor="black",
                    linewidth=0.45,
                    zorder=4,
                    label="Initial solutions" if not init_label_done else None,
                )
                init_label_done = True
                if not scatters:
                    scatters.append(init_sc)

                init_indices = np.where(init_mask)[0]
                init_best_idx = int(init_indices[int(np.argmin(obj[init_mask]))])
                ax.scatter(
                    x[init_best_idx],
                    y[init_best_idx],
                    marker="D",
                    s=72,
                    facecolor="none",
                    edgecolor="red",
                    linewidth=1.7,
                    zorder=6,
                    label="Best initial" if not init_best_label_done else None,
                )
                init_best_label_done = True

            best_idx = int(np.argmin(obj))
            ax.scatter(
                x[best_idx],
                y[best_idx],
                marker="*",
                s=180,
                c="red",
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
                label="Best found" if not best_label_done else None,
            )
            best_label_done = True

            if show_arrows and len(search_points) > 1:
                for i in range(0, len(search_points) - 1, arrow_every):
                    ax.annotate(
                        "",
                        xy=(x_search[i + 1], y_search[i + 1]),
                        xytext=(x_search[i], y_search[i]),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=arrow_color,
                            alpha=0.78,
                            lw=arrow_lw,
                            mutation_scale=arrow_scale,
                            shrinkA=2.0,
                            shrinkB=2.0,
                        ),
                        zorder=3,
                    )
        overlay_centroid_flow(ax, method)
        ax.set_title(method)
        apply_outer_axis_labels(ax, ax_idx, ncols, nrows, "MDS dimension 1st", "MDS dimension 2nd")
        ax.grid(True, alpha=0.25)

    for ax in axes[n_methods:]:
        ax.axis("off")

    margin_x = 0.08 * (float(all_x.max()) - float(all_x.min()) + 1e-9)
    margin_y = 0.08 * (float(all_y.max()) - float(all_y.min()) + 1e-9)
    for ax in axes[:n_methods]:
        ax.set_xlim(float(all_x.min()) - margin_x, float(all_x.max()) + margin_x)
        ax.set_ylim(float(all_y.min()) - margin_y, float(all_y.max()) + margin_y)

    add_figure_legend(fig, axes[:n_methods], fontsize=10)

    if scatters:
        colorbar_mappable = plt.cm.ScalarMappable(norm=obj_norm, cmap=obj_cmap)
        colorbar_mappable.set_array([])
        cbar = fig.colorbar(colorbar_mappable, ax=axes[:n_methods].tolist(), fraction=0.025, pad=0.05)
        cbar.set_label("Objective value (lower is better)")

    # fig.suptitle(f"{ins_name}: search trajectories by {distance_name} distance{title_suffix}", fontsize=13, y=0.995)
    fig.suptitle(f"{ins_name}: search trajectories", fontsize=13, y=0.995)
    fig.subplots_adjust(top=0.88, right=0.87)
    fig_path = out_dir / f"{ins_name}_searchspace_{distance_name}{suffix}.{fmt}"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_centroid_flow_plot(
    rows: Sequence[Row],
    out_dir: Path,
    ins_name: str,
    distance_name: str,
    fmt: str,
    n_init: int,
    centroid_step: int,
    show_initial: bool,
    suffix: str = "_centroid_flow",
    background_cmap: str = "Greys_r",
    background_alpha: float = 0.18,
    method_order: Optional[Sequence[str]] = None,
) -> Tuple[Path, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("No rows to plot.")

    centroid_step = max(1, int(centroid_step))
    n_init = max(0, int(n_init))
    existing_methods = list(dict.fromkeys(str(r["display_method"]) for r in rows))
    if method_order is None:
        methods = existing_methods
    else:
        methods = [m for m in method_order if m in existing_methods]
        methods.extend(m for m in existing_methods if m not in methods)
    n_methods = len(methods)
    ncols = min(3, max(1, n_methods))
    nrows = int(math.ceil(n_methods / ncols))
    fig, axes_arr = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.7 * nrows), sharex=True, sharey=True)
    axes = np.asarray(axes_arr).reshape(-1)

    all_x = np.asarray([float(r["x"]) for r in rows], dtype=float)
    all_y = np.asarray([float(r["y"]) for r in rows], dtype=float)
    all_obj = np.asarray([float(r["objective"]) for r in rows], dtype=float)
    vmin = float(np.min(all_obj))
    vmax = float(np.max(all_obj))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(background_cmap)
    flow_colors = plt.get_cmap("tab10")
    centroid_rows: List[Row] = []

    for ax_idx, (ax, method) in enumerate(zip(axes, methods)):
        method_rows = [r for r in rows if str(r["display_method"]) == method]
        x_bg = np.asarray([float(r["x"]) for r in method_rows], dtype=float)
        y_bg = np.asarray([float(r["y"]) for r in method_rows], dtype=float)
        obj_bg = np.asarray([float(r["objective"]) for r in method_rows], dtype=float)
        fe_bg = np.asarray([int(r["fe"]) for r in method_rows], dtype=int)

        ax.scatter(
            x_bg,
            y_bg,
            c=obj_bg,
            cmap=cmap,
            norm=norm,
            s=22,
            alpha=float(background_alpha),
            edgecolor="none",
            zorder=1,
            label="Evaluated solutions",
        )

        if show_initial and np.any(fe_bg <= n_init):
            init_mask = fe_bg <= n_init
            ax.scatter(
                x_bg[init_mask],
                y_bg[init_mask],
                marker="s",
                s=26,
                facecolor="none",
                edgecolor="0.15",
                linewidth=0.45,
                alpha=0.45,
                zorder=2,
                label="Initial solutions",
            )

        for trial_pos, trial in enumerate(sorted({int(r["trial"]) for r in method_rows})):
            trial_rows = sorted([r for r in method_rows if int(r["trial"]) == trial], key=lambda r: int(r["fe"]))
            if not trial_rows:
                continue
            max_fe_trial = max(int(r["fe"]) for r in trial_rows)
            first_centroid_fe = 1 if show_initial else n_init + 1
            starts = list(range(first_centroid_fe, max_fe_trial + 1, centroid_step))
            centroids: List[Tuple[float, float, int, int, int]] = []
            for start in starts:
                end = start + centroid_step - 1
                bin_rows = [r for r in trial_rows if start <= int(r["fe"]) <= end]
                if not bin_rows:
                    continue
                cx = float(np.mean([float(r["x"]) for r in bin_rows]))
                cy = float(np.mean([float(r["y"]) for r in bin_rows]))
                count = int(len(bin_rows))
                best_obj = float(min(float(r["objective"]) for r in bin_rows))
                centroids.append((cx, cy, start, end, count))
                centroid_rows.append({
                    "method": str(trial_rows[0]["method"]),
                    "display_method": method,
                    "trial": int(trial),
                    "fe_start": int(start),
                    "fe_end": int(end),
                    "n_points": count,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "best_objective_in_bin": best_obj,
                })

            if not centroids:
                continue
            color = flow_colors(trial_pos % 10)
            cx = np.asarray([c[0] for c in centroids], dtype=float)
            cy = np.asarray([c[1] for c in centroids], dtype=float)
            labels = [f"{c[2]}-{c[3]}" for c in centroids]

            ax.plot(cx, cy, color=color, lw=2.3, alpha=0.9, zorder=4, label=f"Centroid flow trial {trial}")
            ax.scatter(cx, cy, marker="o", s=70, facecolor="white", edgecolor=color, linewidth=1.8, zorder=5)
            ax.scatter(cx[0], cy[0], marker="s", s=95, facecolor="white", edgecolor=color, linewidth=2.0, zorder=6)
            ax.scatter(cx[-1], cy[-1], marker="*", s=170, facecolor=color, edgecolor="black", linewidth=0.6, zorder=7)

            for i in range(len(centroids) - 1):
                ax.annotate(
                    "",
                    xy=(cx[i + 1], cy[i + 1]),
                    xytext=(cx[i], cy[i]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color,
                        lw=2.6,
                        alpha=0.9,
                        mutation_scale=18,
                        shrinkA=7,
                        shrinkB=7,
                    ),
                    zorder=6,
                )
            for i, label in enumerate(labels):
                ax.text(cx[i], cy[i], str(i + 1), ha="center", va="center", fontsize=7, color="black", zorder=8)

        ax.set_title(method)
        apply_outer_axis_labels(ax, ax_idx, ncols, nrows, "MDS dimension 1st", "MDS dimension 2nd")
        ax.grid(True, alpha=0.25)

    for ax in axes[n_methods:]:
        ax.axis("off")

    margin_x = 0.08 * (float(all_x.max()) - float(all_x.min()) + 1e-9)
    margin_y = 0.08 * (float(all_y.max()) - float(all_y.min()) + 1e-9)
    for ax in axes[:n_methods]:
        ax.set_xlim(float(all_x.min()) - margin_x, float(all_x.max()) + margin_x)
        ax.set_ylim(float(all_y.min()) - margin_y, float(all_y.max()) + margin_y)

    add_figure_legend(fig, axes[:n_methods], fontsize=10)

    cbar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_mappable.set_array([])
    cbar = fig.colorbar(cbar_mappable, ax=axes[:n_methods].tolist(), fraction=0.025, pad=0.1)
    cbar.set_label("Objective value (background, lower is better)")

    fig.suptitle(f"{ins_name}: centroid flow every {centroid_step} FE by {distance_name} distance", fontsize=13, y=0.995)
    fig.subplots_adjust(top=0.88, right=0.87)
    fig_path = out_dir / f"{ins_name}_searchspace_{distance_name}{suffix}_step{centroid_step}.{fmt}"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = out_dir / f"{ins_name}_searchspace_{distance_name}{suffix}_step{centroid_step}.csv"
    write_csv(
        csv_path,
        centroid_rows,
        ["method", "display_method", "trial", "fe_start", "fe_end", "n_points", "centroid_x", "centroid_y", "best_objective_in_bin"],
    )
    return fig_path, csv_path
# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize permutation search spaces from result.npz logs")
    ap.add_argument("--root", default="E:\\2325_yamaguchi\\_results_TEVC_20260709\\", help="Result root folder, same as --save_path in scripts/main.py")
    ap.add_argument("--ins-name", default="N-p40-01", choices=INSTANCE_CHOICES, help="Instance folder under --root")
    ap.add_argument("--include", nargs="*", default=["umm_fixed","fatrls_init100_fixed","rflos_fixed","gbdtma_gbdt_mean_fixed_fin","lat2way_sage_mean_knn_density_1_mu_10_50","lat2way_sage_mean_knn_density_1_mu_10_50_BB"], help="Method name substrings to include")
    # ap.add_argument("--include", nargs="*", default=["umm_fixed","rflos_fixed","gbdtma_gbdt_mean_fixed_fin","fatrls_init100_fixed","lat2way_sage_mean_knn_density_1_mu_10_50_BB","lat2way_sage_mean_knn_density_1_mu_10_50","lat2way_sage_mean_knn_density_1_1_50","lat2way_sage_mean_knn_density_1_5_50","lat2way_sage_mean_knn_density_1_20_50","lat2way_rf_mean_knn_density_1_mu_10_50","lat2way_gbdt_mean_knn_density_1_mu_10_50"], help="Method name substrings to include")
    ap.add_argument("--exclude", nargs="*", default=["*_whole"], help="Method name substrings to exclude")
    ap.add_argument("--problem", choices=["auto", "TSP", "ATSP", "LOP", "QAP", "PFSP"], default="auto", help="Used by --distance auto")
    ap.add_argument("--distance", choices=["auto", "hamming", "kendall", "interchange", "adjacency_undirected", "adjacency_directed"], default="auto", help="Permutation distance for embedding")
    ap.add_argument("--acyclic", action="store_true", help="Do not connect last-first edge for adjacency distances")
    ap.add_argument("--trials", nargs="*", default=["0"], help="Trial ids, ranges like 0-4, or all")
    ap.add_argument("--max-eval", type=int, default=300, help="Maximum FE per trial to include; use 0 for all")
    ap.add_argument("--eval-stride", type=int, default=1, help="Keep every Nth evaluated solution")
    ap.add_argument("--max-points-per-method", type=int, default=0, help="Evenly downsample each method after loading; 0 disables")
    ap.add_argument("--random-state", type=int, default=42, help="MDS random state")
    ap.add_argument("--out-dir", default=None, help="Output folder; default: <root>/<ins-name>/_searchspace")
    ap.add_argument("--format", choices=["png", "pdf", "svg"], default="pdf", help="Figure format")
    ap.add_argument("--cmap", default="viridis_r", help="Matplotlib colormap for objective values")
    ap.add_argument("--n-init", type=int, default=100, help="Number of initial evaluated solutions per trial; plotted as square markers and excluded from arrows")
    ap.add_argument("--no-arrows", default=True, action="store_true", help="Skip trajectory arrows")
    ap.add_argument("--no-lines", default=True, action="store_true", help="Skip trajectory connecting lines")
    ap.add_argument("--arrow-every", type=int, default=10, help="Draw one arrow every N search steps after --n-init")
    ap.add_argument("--arrow-color", default="#d62728", help="Arrow color")
    ap.add_argument("--arrow-lw", type=float, default=1.8, help="Arrow line width")
    ap.add_argument("--arrow-scale", type=float, default=14.0, help="Arrow head size")
    ap.add_argument("--no-plot", action="store_true", help="Only write CSV files; matplotlib is not required")
    ap.add_argument("--scale-duplicate-points", action="store_true", help="Aggregate equivalent permutations and scale point size by duplicate count")
    ap.add_argument("--duplicate-size-factor", type=float, default=0.8, help="Extra size multiplier N for duplicate-scaled points; final size is base_size * N * duplicate_count")
    ap.add_argument("--duplicate-equivalence", choices=["exact", "cyclic", "cyclic_reverse"], default="cyclic_reverse", help="Permutation equivalence used by --scale-duplicate-points")
    ap.add_argument("--use-saved", dest="use_saved", default=True, action="store_true", help="Reuse saved trajectory/centroid CSV coordinates when available")
    ap.add_argument("--recompute-embedding", dest="use_saved", action="store_false", help="Ignore saved CSV coordinates and recompute the MDS embedding")
    ap.add_argument("--combined-centroid-flow", default=False, action="store_true", help="Save a normal trajectory plot with centroid-flow arrows overlaid")
    ap.add_argument("--time-slices",default=False ,action="store_true", help="Save temporal snapshot figures at regular FE intervals")
    ap.add_argument("--time-step", type=int, default=50, help="FE interval for --time-slices")
    ap.add_argument("--time-mode", choices=["cumulative", "window"], default="cumulative", help="Temporal snapshot mode: cumulative FE<=t or sliding window")
    ap.add_argument("--time-window", type=int, default=50, help="Window size for --time-mode window")
    ap.add_argument("--time-no-lines", action="store_true", help="Disable connecting lines only in temporal snapshot figures")
    ap.add_argument("--time-no-arrows", action="store_true", help="Disable arrows only in temporal snapshot figures")
    ap.add_argument("--embed-time",default=False ,action="store_true", help="Save additional single-plot figures that encode FE/time into point alpha, size, or both")
    ap.add_argument("--embed-time-by", nargs="+", choices=["alpha", "color", "size", "both"], default=["alpha", "color", "size","both"], help="Temporal encodings to generate when --embed-time is set")
    ap.add_argument("--time-min-alpha", type=float, default=0.18, help="Minimum alpha for the oldest points in --embed-time alpha/both figures")
    ap.add_argument("--time-size-min", type=float, default=14.0, help="Minimum point size for the oldest points in --embed-time size/both figures")
    ap.add_argument("--time-size-max", type=float, default=58.0, help="Maximum point size for the newest points in --embed-time size/both figures")
    ap.add_argument("--centroid-flow", default=True, action="store_true", help="Save an additional plot that connects centroids of each FE interval with thick arrows")
    ap.add_argument("--centroid-step", type=int, default=25, help="FE interval width used by --centroid-flow")
    ap.add_argument("--centroid-hide-initial",default=True ,action="store_true", help="Do not outline initial solutions in the centroid-flow background")
    ap.add_argument("--centroid-color-alpha", type=float, default=0.55, help="Background point alpha for the additional color centroid-flow figure")
    ap.add_argument(
        "--legend-map",
        nargs="*",
        default=[
            "lat2way_sage_mean_knn_density_1_mu_10_50=RLEA",
            "lat2way_sage_mean_knn_density_1_1_50=RLEA_1",
            "lat2way_sage_mean_knn_density_1_5_50=RLEA_5",
            "lat2way_sage_mean_knn_density_1_20_50=RLEA_20",
            "lat2way_sage_mean_knn_density_1_mu_10_50_BB=RLEA-BB",
            "lat2way_rf_mean_knn_density_1_mu_10_50=RLEA-RF",
            "lat2way_gbdt_mean_knn_density_1_mu_10_50=RLEA-GBDT",
            "gbdtma_gbdt_mean_fixed_fin=GBDTMA",
            "gbdtma_sage_mean_fixed_fin=RLEA-old",
            "fatrls_init100_fixed=FAT-RLS",
            "fatrls_fixed=FAT-RLS",
            "umm_fixed=UMM",
            "rflos_fixed=RFLoS",
        ],
        help="Legend rename rules in key=value form",
    )
    args = ap.parse_args()

    root = Path(args.root)
    cwd = Path.cwd()
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif (cwd / f"{args.ins_name}_searchspace_trajectory.csv").exists():
        out_dir = cwd
    else:
        out_dir = root / args.ins_name / "_searchspace"
    out_dir.mkdir(parents=True, exist_ok=True)

    legend_map = parse_legend_map(args.legend_map)
    trial_filter = parse_trials(args.trials)
    max_eval = None if args.max_eval <= 0 else args.max_eval
    stress_json = out_dir / f"{args.ins_name}_searchspace_mds.json"
    saved_metadata = load_saved_mds_metadata(stress_json)
    problem = str(saved_metadata.get("problem") or (args.problem if args.problem != "auto" else "TSP"))
    if args.distance == "auto" and saved_metadata.get("distance"):
        distance_name = str(saved_metadata["distance"])
        distance_func: Optional[DistanceFunc] = None
    else:
        distance_name, distance_func = get_distance_func(args.distance, problem, cyclic=not args.acyclic)

    traj_csv = out_dir / f"{args.ins_name}_searchspace_trajectory.csv"
    use_saved_rows = bool(args.use_saved and traj_csv.exists())
    if use_saved_rows:
        print(f"[INFO] reusing saved coordinates -> {traj_csv}")
        rows = filter_saved_rows(
            read_csv_rows(traj_csv),
            includes=args.include,
            excludes=args.exclude,
            trial_filter=trial_filter,
            max_eval=max_eval,
            eval_stride=args.eval_stride,
            max_points_per_method=args.max_points_per_method,
            legend_map=legend_map,
        )
        if not rows:
            raise SystemExit("No rows remained after filtering the saved trajectory CSV.")
        summary_rows = make_summary(rows)
        summary_csv = out_dir / f"{args.ins_name}_searchspace_summary.csv"
        write_csv(
            summary_csv,
            summary_rows,
            ["method", "display_method", "trial", "n_points", "first_fe", "last_fe", "best_objective"],
        )
        embedding_method = str(saved_metadata.get("embedding_method", "saved_csv"))
        stress = float(saved_metadata.get("mds_stress", "nan")) if saved_metadata else float("nan")
    else:
        method_dirs_all = collect_method_dirs(root, args.ins_name)
        method_dirs = match_methods(method_dirs_all, args.include, args.exclude)
        if not method_dirs:
            raise SystemExit("No method folders matched. Check --include/--exclude.")

        problem = infer_problem(method_dirs, args.problem)
        distance_name, distance_func = get_distance_func(args.distance, problem, cyclic=not args.acyclic)
        print("[INFO] selected methods:")
        for pth in method_dirs:
            print(f"  - {pth.name}")
        print(f"[INFO] problem={problem}, distance={distance_name}, trials={args.trials}")

        rows = load_trajectories(
            method_dirs=method_dirs,
            trial_filter=trial_filter,
            max_eval=max_eval,
            eval_stride=args.eval_stride,
            max_points_per_method=args.max_points_per_method,
            legend_map=legend_map,
        )
        if not rows:
            raise SystemExit("No valid archive_perm/archive_fx trajectories found.")

        permutations = [[int(x) for x in str(row["permutation"]).split()] for row in rows]
        unique_perms: List[List[int]] = []
        perm_to_unique_idx: Dict[Tuple[int, ...], int] = {}
        row_to_unique_idx: List[int] = []
        for perm in permutations:
            key = tuple(int(v) for v in perm)
            if key not in perm_to_unique_idx:
                perm_to_unique_idx[key] = len(unique_perms)
                unique_perms.append(list(key))
            row_to_unique_idx.append(perm_to_unique_idx[key])

        if distance_func is None:
            raise RuntimeError("distance_func unexpectedly missing while recomputing embedding")
        print(f"[INFO] embedding {len(unique_perms)} unique evaluated solutions for {len(rows)} logged points")
        dmat = build_distance_matrix(unique_perms, distance_func)
        print("[INFO] running MDS embedding")
        unique_coords, stress, embedding_method = run_mds(dmat, random_state=args.random_state)
        print("[INFO] MDS embedding finished")
        for row, unique_idx in zip(rows, row_to_unique_idx):
            coord = unique_coords[int(unique_idx)]
            row["x"] = float(coord[0])
            row["y"] = float(coord[1])
            row["is_initial"] = int(int(row["fe"]) <= int(args.n_init))

        write_csv(
            traj_csv,
            rows,
            ["instance", "method", "display_method", "trial", "fe", "is_initial", "objective", "x", "y", "permutation", "result_path"],
        )
        summary_rows = make_summary(rows)
        summary_csv = out_dir / f"{args.ins_name}_searchspace_summary.csv"
        write_csv(
            summary_csv,
            summary_rows,
            ["method", "display_method", "trial", "n_points", "first_fe", "last_fe", "best_objective"],
        )
        stress_json.write_text(
            json.dumps(
                {
                    "instance": args.ins_name,
                    "problem": problem,
                    "distance": distance_name,
                    "n_points": int(len(rows)),
                    "n_unique_permutations": int(len(unique_perms)),
                    "n_init": int(args.n_init),
                    "embedding_method": embedding_method,
                    "mds_stress": float(stress),
                    "methods": [pth.name for pth in method_dirs],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    method_order = build_display_method_order(rows, args.include, legend_map)

    print(f"[OK] trajectory csv -> {traj_csv}")
    print(f"[OK] summary csv -> {summary_csv}")
    print(f"[OK] mds metadata -> {stress_json}")
    print(f"[OK] {embedding_method} raw stress = {stress:.6g}")

    if not args.no_plot:
        fig_path = save_plots(
            rows=rows,
            out_dir=out_dir,
            ins_name=args.ins_name,
            distance_name=distance_name,
            fmt=args.format,
            cmap=args.cmap,
            show_arrows=not args.no_arrows,
            n_init=args.n_init,
            arrow_every=args.arrow_every,
            arrow_color=args.arrow_color,
            arrow_lw=args.arrow_lw,
            arrow_scale=args.arrow_scale,
            show_lines=not args.no_lines,
            suffix="_duplicate_scaled" if args.scale_duplicate_points else "",
            title_suffix=" (duplicate-scaled)" if args.scale_duplicate_points else "",
            scale_duplicate_points=args.scale_duplicate_points,
            duplicate_size_factor=args.duplicate_size_factor,
            duplicate_equivalence=args.duplicate_equivalence,
            method_order=method_order,
        )
        print(f"[OK] figure -> {fig_path}")

        if args.combined_centroid_flow:
            centroid_csv = find_saved_centroid_csv(out_dir, args.ins_name, distance_name, args.centroid_step)
            if centroid_csv is not None:
                raw_centroid_rows = read_csv_rows(centroid_csv)
                if args.centroid_hide_initial:
                    raw_centroid_rows = [r for r in raw_centroid_rows if int(r["fe_start"]) > int(args.n_init)]
                for row in raw_centroid_rows:
                    if row.get("method"):
                        row["display_method"] = legend_name(str(row["method"]), legend_map)
                centroid_rows = filter_centroid_rows(
                    raw_centroid_rows,
                    display_methods=list(dict.fromkeys(str(r["display_method"]) for r in rows)),
                    trial_filter=trial_filter,
                )
                combined_fig_path = save_plots(
                    rows=rows,
                    out_dir=out_dir,
                    ins_name=args.ins_name,
                    distance_name=distance_name,
                    fmt=args.format,
                    cmap=args.cmap,
                    show_arrows=not args.no_arrows,
                    n_init=args.n_init,
                    arrow_every=args.arrow_every,
                    arrow_color=args.arrow_color,
                    arrow_lw=args.arrow_lw,
                    arrow_scale=args.arrow_scale,
                    show_lines=not args.no_lines,
                    suffix=f"_with_centroid_flow_step{args.centroid_step}",
                    title_suffix=f" + centroid flow step {args.centroid_step}",
                    centroid_rows=centroid_rows,
                    scale_duplicate_points=args.scale_duplicate_points,
                    duplicate_size_factor=args.duplicate_size_factor,
                    duplicate_equivalence=args.duplicate_equivalence,
                    method_order=method_order,
                )
                print(f"[OK] combined centroid-flow figure -> {combined_fig_path}")
            else:
                print(f"[WARN] centroid CSV not found for combined plot: step={args.centroid_step}")

        if args.embed_time:
            seen_encodings: set[str] = set()
            for encoding in args.embed_time_by:
                encoding = "alpha" if encoding == "color" else encoding
                if encoding in seen_encodings:
                    continue
                seen_encodings.add(encoding)
                time_fig_path = save_plots(
                    rows=rows,
                    out_dir=out_dir,
                    ins_name=args.ins_name,
                    distance_name=distance_name,
                    fmt=args.format,
                    cmap=args.cmap,
                    show_arrows=not args.no_arrows,
                    n_init=args.n_init,
                    arrow_every=args.arrow_every,
                    arrow_color=args.arrow_color,
                    arrow_lw=args.arrow_lw,
                    arrow_scale=args.arrow_scale,
                    show_lines=not args.no_lines,
                    suffix=f"_time_{encoding}",
                    title_suffix=f" (time encoded by {encoding})",
                    temporal_encoding=encoding,
                    temporal_min_alpha=args.time_min_alpha,
                    temporal_size_min=args.time_size_min,
                    temporal_size_max=args.time_size_max,
                    scale_duplicate_points=args.scale_duplicate_points,
                    duplicate_size_factor=args.duplicate_size_factor,
                    duplicate_equivalence=args.duplicate_equivalence,
                    method_order=method_order,
                )
                print(f"[OK] time-encoded figure ({encoding}) -> {time_fig_path}")

        if args.centroid_flow:
            centroid_fig_path, centroid_csv_path = save_centroid_flow_plot(
                rows=rows,
                out_dir=out_dir,
                ins_name=args.ins_name,
                distance_name=distance_name,
                fmt=args.format,
                n_init=args.n_init,
                centroid_step=args.centroid_step,
                show_initial=not args.centroid_hide_initial,
                method_order=method_order,
            )
            print(f"[OK] centroid-flow figure -> {centroid_fig_path}")
            print(f"[OK] centroid-flow csv -> {centroid_csv_path}")

            centroid_color_fig_path, centroid_color_csv_path = save_centroid_flow_plot(
                rows=rows,
                out_dir=out_dir,
                ins_name=args.ins_name,
                distance_name=distance_name,
                fmt=args.format,
                n_init=args.n_init,
                centroid_step=args.centroid_step,
                show_initial=not args.centroid_hide_initial,
                suffix="_centroid_flow_color",
                background_cmap=args.cmap,
                background_alpha=args.centroid_color_alpha,
                method_order=method_order,
            )
            print(f"[OK] centroid-flow color figure -> {centroid_color_fig_path}")
            print(f"[OK] centroid-flow color csv -> {centroid_color_csv_path}")

        if args.time_slices:
            frame_dir = out_dir / f"{args.ins_name}_searchspace_{distance_name}_frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            cutoffs = build_time_cutoffs(rows, args.time_step)
            print(f"[INFO] saving {len(cutoffs)} temporal frames -> {frame_dir}")
            for cutoff in cutoffs:
                frame_rows = select_temporal_rows(rows, cutoff, args.time_mode, args.time_window)
                if not frame_rows:
                    continue
                if args.time_mode == "window":
                    lo = max(1, int(cutoff) - max(1, int(args.time_window)) + 1)
                    title_suffix = f" (FE {lo}-{int(cutoff)})"
                    suffix = f"_{args.time_mode}_{lo:04d}_{int(cutoff):04d}"
                else:
                    title_suffix = f" (FE <= {int(cutoff)})"
                    suffix = f"_{args.time_mode}_{int(cutoff):04d}"
                frame_path = save_plots(
                    rows=frame_rows,
                    out_dir=frame_dir,
                    ins_name=args.ins_name,
                    distance_name=distance_name,
                    fmt=args.format,
                    cmap=args.cmap,
                    show_arrows=(not args.no_arrows) and (not args.time_no_arrows),
                    n_init=args.n_init,
                    arrow_every=args.arrow_every,
                    arrow_color=args.arrow_color,
                    arrow_lw=args.arrow_lw,
                    arrow_scale=args.arrow_scale,
                    show_lines=(not args.no_lines) and (not args.time_no_lines),
                    suffix=suffix,
                    title_suffix=title_suffix,
                    extent_rows=rows,
                    temporal_min_alpha=args.time_min_alpha,
                    temporal_size_min=args.time_size_min,
                    temporal_size_max=args.time_size_max,
                    scale_duplicate_points=args.scale_duplicate_points,
                    duplicate_size_factor=args.duplicate_size_factor,
                    duplicate_equivalence=args.duplicate_equivalence,
                    method_order=method_order,
                )
                print(f"[OK] frame FE={int(cutoff)} -> {frame_path}")


if __name__ == "__main__":
    main()
