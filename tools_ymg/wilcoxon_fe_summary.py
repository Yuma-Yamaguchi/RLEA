import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from tqdm import tqdm

def _nan_best_so_far(values: np.ndarray, direction: str) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values
    out = np.empty_like(values)
    best = np.nan
    for i, v in enumerate(values):
        if np.isnan(v):
            out[i] = best
            continue
        if np.isnan(best):
            best = v
        elif direction == "max":
            best = max(best, v)
        else:
            best = min(best, v)
        out[i] = best
    return out

def _safe_load_npz(path, keys=None):
    try:
        with np.load(path, allow_pickle=True) as d:
            if keys is None:
                return {k: d[k] for k in d.files}

            return {
                k: d[k]
                for k in keys
                if k in d.files
            }

    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return {}

def _best_so_far(values: np.ndarray, direction: str) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values
    if direction == "max":
        return np.maximum.accumulate(values)
    return np.minimum.accumulate(values)

def _flatten_archive_fx(archive_fx: np.ndarray) -> np.ndarray:
    arr = np.asarray(archive_fx)
    if arr.dtype == object:
        flat_list: List[float] = []
        for item in arr:
            if isinstance(item, (list, tuple, np.ndarray)):
                flat_list.extend(np.asarray(item, dtype=float).reshape(-1).tolist())
            else:
                try:
                    flat_list.append(float(item))
                except Exception:
                    continue
        return np.asarray(flat_list, dtype=float)
    return arr.astype(float).reshape(-1)

def _infer_direction(method_dir: Path, user_direction: str) -> str:
    if user_direction in ("min", "max"):
        return user_direction
    meta_path = method_dir / "meta.json"
    if not meta_path.exists():
        return "min"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return "min"
    _ = str(meta.get("problem", "")).upper()
    return "min"

def _value_at_eval(history: np.ndarray, eval_axis: np.ndarray, eval_point: int, fallback: str) -> float:
    if history.size == 0 or eval_axis.size == 0:
        return np.nan
    idx = np.searchsorted(eval_axis, eval_point, side="right") - 1
    if idx < 0:
        return float(history[0]) if fallback == "last" else np.nan
    if idx < history.size:
        return float(history[idx])
    return float(history[-1]) if fallback == "last" else np.nan

def _load_histories(method_dir: Path, direction: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    trial_dirs = sorted([p for p in method_dir.iterdir() if p.is_dir() and p.name.startswith("trial_")])
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for td in trial_dirs:
        try:
            trial_id = int(td.name.split("_", 1)[1])
        except Exception:
            continue
        npz_path = td / "result.npz"
        if not npz_path.exists():
            continue
        try:
            data = _safe_load_npz(npz_path, ["archive_fx","history"])
        except Exception:
            continue
        archive_fx = data.get("archive_fx", None)
        if archive_fx is not None and np.asarray(archive_fx).size > 0:
            flat_fx = _flatten_archive_fx(archive_fx)
            if flat_fx.size > 0:
                hist = _best_so_far(flat_fx, direction)
                axis = np.arange(1, hist.size + 1, dtype=int)
                out[trial_id] = (axis, hist)
                continue
        history = data.get("history", None)
        if history is not None and np.asarray(history).size > 0:
            hist = np.asarray(history, dtype=float).reshape(-1)
            axis = np.arange(1, hist.size + 1, dtype=int)
            out[trial_id] = (axis, hist)
    return out

def _load_kendall_histories(method_dir: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    trial_dirs = sorted([p for p in method_dir.iterdir() if p.is_dir() and p.name.startswith("trial_")])
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for td in trial_dirs:
        try:
            trial_id = int(td.name.split("_", 1)[1])
        except Exception:
            continue
        npz_path = td / "result.npz"
        if not npz_path.exists():
            continue
        try:
            data = _safe_load_npz(npz_path, ["test_kendall_tau","train_sizes"])
        except Exception:
            continue
        tau = data.get("test_kendall_tau", None)
        train_sizes = data.get("train_sizes", None)
        if tau is None or train_sizes is None:
            continue
        hist = np.asarray(tau, dtype=float).reshape(-1)
        axis = np.asarray(train_sizes, dtype=int).reshape(-1)
        n = min(hist.size, axis.size)
        if n == 0:
            continue
        hist = hist[:n]
        axis = axis[:n]
        order = np.argsort(axis, kind="stable")
        axis = axis[order]
        hist = hist[order]
        out[trial_id] = (axis, hist)
    return out

def _load_metric_histories(
    method_dir: Path,
    direction: str,
    target_metric: str,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], str]:
    if target_metric == "performance":
        return _load_histories(method_dir, direction), "fe"
    if target_metric == "kendall_tau":
        return _load_kendall_histories(method_dir), "train_size"
    raise ValueError(f"Unsupported target_metric: {target_metric}")

def _resolve_method_dir(ins_dir: Path, method_name: str, match_mode: str) -> Optional[Path]:
    cand = [p for p in ins_dir.iterdir() if p.is_dir()]
    if match_mode == "exact":
        hit = [p for p in cand if p.name == method_name]
    else:
        key = method_name.lower()
        hit = [p for p in cand if key in p.name.lower()]
    if len(hit) == 1:
        return hit[0]
    return None

def _paired_values_at_eval(
    base_hist: Dict[int, Tuple[np.ndarray, np.ndarray]],
    comp_hist: Dict[int, Tuple[np.ndarray, np.ndarray]],
    eval_point: int,
    fallback: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = sorted(set(base_hist.keys()) & set(comp_hist.keys()))
    bvals: List[float] = []
    cvals: List[float] = []
    tids: List[int] = []
    for tid in ids:
        bax, b = base_hist[tid]
        cax, c = comp_hist[tid]
        be = _value_at_eval(b, bax, eval_point, fallback)
        ce = _value_at_eval(c, cax, eval_point, fallback)
        if np.isnan(be) or np.isnan(ce):
            continue
        bvals.append(be)
        cvals.append(ce)
        tids.append(tid)
    return np.asarray(bvals, dtype=float), np.asarray(cvals, dtype=float), np.asarray(tids, dtype=int)

def main() -> None:
    ap = argparse.ArgumentParser(description="Wilcoxon signed-rank test at specific FE across instances")
    ap.add_argument("--root", default="", help="result root")
    ap.add_argument("--instances", nargs="+", default=["burma14","ulysses22","fri26","bayg29","swiss42","att48","berlin52", "br17","ftv33","ftv35","ftv38","p43","ry48p","ft53", "N-pal11","N-pal19","N-pal23","N-pal27","N-econ36","N-p40-01","N-be75np"], help="instance names", choices=["burma14","ulysses22","fri26","bayg29","swiss42","att48","berlin52", "br17","ftv33","ftv35","ftv38","p43","ry48p","ft53", "N-pal11","N-pal19","N-pal23","N-pal27","N-econ36","N-p40-01","N-p44-01","N-be75np"])
    ap.add_argument("--methods", nargs="+", default=["RLEA", "gbdtma", "fatrls", "umm", "rflos"], help="algorithm names to compare")
    ap.add_argument("--baseline", default="RLEA", help="baseline algorithm name")
    ap.add_argument("--eval-points", nargs="+", type=int, default=[200,300,400,500,600], help="FE points for performance mode or train_sizes for kendall_tau mode", choices=[200,300,400,500,600])
    ap.add_argument("--target-metric", choices=["performance", "kendall_tau"], default="performance",
                    help="compare objective performance at FE or test Kendall's tau at train_size")
    ap.add_argument("--direction", choices=["auto", "min", "max"], default="auto")
    ap.add_argument("--alternative", choices=["two-sided", "less", "greater"], default="two-sided")
    ap.add_argument("--zero-method", choices=["wilcox", "pratt", "zsplit"], default="wilcox")
    ap.add_argument("--eval-fallback", choices=["last", "nan"], default="last")
    ap.add_argument("--match-mode", choices=["exact", "contains"], default="exact")
    ap.add_argument("--out-csv", default=None, help="output csv path")
    args = ap.parse_args()

    root = Path(args.root)
    default_name = "wilcoxon_fe_summary_all.csv" if args.target_metric == "performance" else "wilcoxon_kendall_summary_all.csv"
    out_csv = Path(args.out_csv) if args.out_csv else (root / default_name)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    pooled: Dict[Tuple[str, int], List[float]] = {}

    methods = [m for m in args.methods if m != args.baseline]
    if not methods:
        raise SystemExit("No comparison methods remain after removing baseline.")

    for ins in tqdm(args.instances, desc="Processing instances"):
        ins_dir = root / ins
        if not ins_dir.exists():
            continue
        base_dir = _resolve_method_dir(ins_dir, args.baseline, args.match_mode)
        if base_dir is None:
            continue
        direction = _infer_direction(base_dir, args.direction)
        base_hist, axis_kind = _load_metric_histories(base_dir, direction, args.target_metric)
        if not base_hist:
            continue

        for method in methods:
            comp_dir = _resolve_method_dir(ins_dir, method, args.match_mode)
            if comp_dir is None:
                continue
            comp_hist, _ = _load_metric_histories(comp_dir, direction, args.target_metric)
            if not comp_hist:
                continue

            for fe in args.eval_points:
                b, c, tids = _paired_values_at_eval(base_hist, comp_hist, fe, args.eval_fallback)
                n = int(b.size)
                if n == 0:
                    rows.append(
                        {
                            "instance": ins,
                            "eval": int(fe),
                            "baseline": args.baseline,
                            "method": method,
                            "target_metric": args.target_metric,
                            "axis_kind": axis_kind,
                            "n_pairs": 0,
                            "direction": direction,
                            "alternative": args.alternative,
                            "statistic": np.nan,
                            "pvalue": np.nan,
                            "baseline_mean": np.nan,
                            "method_mean": np.nan,
                            "delta_mean": np.nan,
                            "baseline_median": np.nan,
                            "method_median": np.nan,
                            "delta_median": np.nan,
                        }
                    )
                    continue

                diff = c - b
                try:
                    w = wilcoxon(
                        c,
                        b,
                        alternative=args.alternative,
                        zero_method=args.zero_method,
                        mode="auto",
                    )
                    stat = float(w.statistic)
                    pval = float(w.pvalue)
                except Exception:
                    stat = np.nan
                    pval = np.nan

                rows.append(
                    {
                        "instance": ins,
                        "eval": int(fe),
                        "baseline": args.baseline,
                        "method": method,
                        "target_metric": args.target_metric,
                        "axis_kind": axis_kind,
                        "n_pairs": n,
                        "direction": direction,
                        "alternative": args.alternative,
                        "statistic": stat,
                        "pvalue": pval,
                        "baseline_mean": float(np.nanmean(b)),
                        "method_mean": float(np.nanmean(c)),
                        "delta_mean": float(np.nanmean(diff)),
                        "baseline_median": float(np.nanmedian(b)),
                        "method_median": float(np.nanmedian(c)),
                        "delta_median": float(np.nanmedian(diff)),
                        "paired_trial_ids": ",".join(map(str, tids.tolist())),
                    }
                )

                pooled.setdefault((method, int(fe)), []).extend(diff[~np.isnan(diff)].tolist())

    for (method, fe), diffs in sorted(pooled.items(), key=lambda x: (x[0][0], x[0][1])):
        arr = np.asarray(diffs, dtype=float)
        if arr.size == 0:
            continue
        try:
            w = wilcoxon(
                arr,
                alternative=args.alternative,
                zero_method=args.zero_method,
                mode="auto",
            )
            stat = float(w.statistic)
            pval = float(w.pvalue)
        except Exception:
            stat = np.nan
            pval = np.nan
        rows.append(
            {
                "instance": "__ALL__",
                "eval": int(fe),
                "baseline": args.baseline,
                "method": method,
                "target_metric": args.target_metric,
                "axis_kind": "fe" if args.target_metric == "performance" else "train_size",
                "n_pairs": int(arr.size),
                "direction": args.direction,
                "alternative": args.alternative,
                "statistic": stat,
                "pvalue": pval,
                "baseline_mean": np.nan,
                "method_mean": np.nan,
                "delta_mean": float(np.nanmean(arr)),
                "baseline_median": np.nan,
                "method_median": np.nan,
                "delta_median": float(np.nanmedian(arr)),
                "paired_trial_ids": "",
            }
        )

    if not rows:
        raise SystemExit("No valid comparisons found. Check names/match mode and data paths.")

    df = pd.DataFrame(rows)
    instance_order = {ins: i for i, ins in enumerate(args.instances)}
    df["_instance_order"] = df["instance"].map(instance_order).fillna(len(instance_order)).astype(int)
    df = df.sort_values(["_instance_order", "eval", "method"], kind="stable").drop(columns=["_instance_order"])
    df.to_csv(out_csv, index=False)
    print(f"[OK] wilcoxon summary -> {out_csv}")


if __name__ == "__main__":
    main()
