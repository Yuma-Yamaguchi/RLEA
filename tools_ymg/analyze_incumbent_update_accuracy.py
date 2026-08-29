from __future__ import annotations

import argparse
import concurrent.futures
import csv
import glob
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


def _unwrap_object(x: Any) -> Any:
    if isinstance(x, np.ndarray) and x.shape == () and x.dtype == object:
        return x.item()
    return x


def _normalize_loaded(x: Any) -> Any:
    x = _unwrap_object(x)
    if isinstance(x, np.ndarray) and x.dtype == object:
        return [_unwrap_object(v) for v in x.tolist()]
    return x


def _as_1d(x: Any, dtype: Any) -> np.ndarray:
    if x is None:
        return np.zeros((0,), dtype=dtype)
    try:
        return np.asarray(x, dtype=dtype).reshape(-1)
    except Exception:
        return np.zeros((0,), dtype=dtype)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def _mean_or_nan(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _std_or_nan(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.std(vals, ddof=1)) if len(vals) >= 2 else float("nan")


def infer_instance_algorithm(path: Path, root: Optional[Path]) -> tuple[str, str, str]:
    trial = path.parent.name
    if root is not None:
        try:
            rel = path.relative_to(root)
            parts = rel.parts
            if len(parts) >= 4:
                return parts[0], parts[1], trial
        except ValueError:
            pass
    algorithm = path.parents[1].name if len(path.parents) >= 2 else ""
    instance = path.parents[2].name if len(path.parents) >= 3 else ""
    return instance, algorithm, trial


def infer_surrogate(algorithm: str) -> str:
    m = re.search(r"RLEA_([^_]+)_", algorithm)
    if m:
        return m.group(1)
    m = re.search(r"(?:^|_)(rf|gbdt)(?:_|$)", algorithm)
    return m.group(1) if m else algorithm


def load_meta(algorithm_dir: Path) -> Dict[str, Any]:
    meta_path = algorithm_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_result_files(
    root: Optional[Path],
    instances: Optional[Sequence[str]],
    algorithms: Optional[Sequence[str]],
    inputs: Optional[Sequence[str]],
) -> List[Path]:
    files: List[Path] = []
    if inputs:
        for item in inputs:
            if any(ch in item for ch in "*?[]"):
                files.extend(Path(p) for p in glob.glob(item))
            else:
                files.append(Path(item))
    if root is not None:
        if instances:
            instance_dirs = [root / ins for ins in instances]
        else:
            instance_dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")]

        for instance_dir in instance_dirs:
            if not instance_dir.exists() or not instance_dir.is_dir():
                continue
            if algorithms:
                algo_dirs: Iterable[Path] = [instance_dir / alg for alg in algorithms]
            else:
                algo_dirs = [p for p in instance_dir.iterdir() if p.is_dir() and p.name.startswith("RLEA_")]
            for algo_dir in algo_dirs:
                if algo_dir.exists():
                    files.extend(sorted(algo_dir.glob("trial_*/result.npz")))
    return sorted({p for p in files if p.name == "result.npz" and p.exists()})


def make_metric_row(
    instance: str,
    algorithm: str,
    surrogate: str,
    trial: str,
    step: Optional[int],
    total: int,
    correct: int,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
) -> Dict[str, Any]:
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    bal_vals = [v for v in (recall, specificity) if not math.isnan(v)]
    return {
        "instance": instance,
        "algorithm": algorithm,
        "surrogate": surrogate,
        "trial": trial,
        "step": step,
        "n_comparisons": int(total),
        "correct": int(correct),
        "accuracy": _safe_div(correct, total),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision_update": _safe_div(tp, tp + fp),
        "recall_update": recall,
        "specificity_no_update": specificity,
        "f1_update": _safe_div(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": float(np.mean(bal_vals)) if bal_vals else float("nan"),
        "mcc": _safe_div(
            tp * tn - fp * fn,
            math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
        ),
    }


def summarize_one_result(path: Path, root: Optional[Path], include_steps: bool, max_fe: Optional[int]) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    instance, algorithm, trial = infer_instance_algorithm(path, root)
    meta = load_meta(path.parents[1])
    surrogate = str(meta.get("RLEA_surrogate") or infer_surrogate(algorithm))

    z = np.load(path, allow_pickle=True)
    if "incumbent_internal" not in z.files:
        raise KeyError(f"{path} has no incumbent_internal log")

    inc_logs = _normalize_loaded(z["incumbent_internal"])
    history_len = int(len(z["history"])) if "history" in z.files else 0
    fe_offset = max(0, history_len - len(inc_logs)) if history_len else 0
    total = correct = tp = fp = tn = fn = 0
    n_true_update = n_sur_update = 0
    missing_truth_gens: List[int] = []
    by_step: Dict[int, Dict[str, int]] = {}

    for gen_idx, raw in enumerate(inc_logs):
        fe_at_generation = fe_offset + gen_idx + 1
        if max_fe is not None and fe_at_generation > int(max_fe):
            break
        diag = _unwrap_object(raw)
        if not isinstance(diag, Mapping):
            continue

        scalar_count = int(diag.get("comparison_count", 0) or 0)
        has_scalar_confusion = all(k in diag for k in ("comparison_tp", "comparison_fp", "comparison_tn", "comparison_fn"))

        if has_scalar_confusion:
            cur_tp = int(diag.get("comparison_tp", 0) or 0)
            cur_fp = int(diag.get("comparison_fp", 0) or 0)
            cur_tn = int(diag.get("comparison_tn", 0) or 0)
            cur_fn = int(diag.get("comparison_fn", 0) or 0)
            n = cur_tp + cur_fp + cur_tn + cur_fn
            cur_correct = cur_tp + cur_tn
            if scalar_count and n != scalar_count:
                n = scalar_count
        else:
            surrogate_update = _as_1d(diag.get("comparison_surrogate_update"), bool)
            true_update = _as_1d(diag.get("comparison_true_update"), bool)
            decision_correct = _as_1d(diag.get("comparison_decision_correct"), bool)
            n = int(min(len(surrogate_update), len(true_update), len(decision_correct)))
            if n == 0:
                if scalar_count > 0:
                    missing_truth_gens.append(int(diag.get("generation", gen_idx)))
                continue
            surrogate_update = surrogate_update[:n]
            true_update = true_update[:n]
            decision_correct = decision_correct[:n]
            cur_correct = int(np.sum(decision_correct))
            cur_tp = int(np.sum(surrogate_update & true_update))
            cur_fp = int(np.sum(surrogate_update & ~true_update))
            cur_tn = int(np.sum(~surrogate_update & ~true_update))
            cur_fn = int(np.sum(~surrogate_update & true_update))

        if n == 0:
            continue

        total += n
        correct += cur_correct
        tp += cur_tp
        fp += cur_fp
        tn += cur_tn
        fn += cur_fn
        n_true_update += cur_tp + cur_fn
        n_sur_update += cur_tp + cur_fp

        if not include_steps:
            continue

        steps = _as_1d(diag.get("comparison_step"), int)
        surrogate_update = _as_1d(diag.get("comparison_surrogate_update"), bool)
        true_update = _as_1d(diag.get("comparison_true_update"), bool)
        n_step = int(min(len(steps), len(surrogate_update), len(true_update)))
        if n_step == 0:
            continue
        steps = steps[:n_step]
        surrogate_update = surrogate_update[:n_step]
        true_update = true_update[:n_step]
        step_values = np.unique(steps)
        for step in step_values:
            mask = steps == step
            d = by_step.setdefault(int(step), {"total": 0, "correct": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0})
            su = surrogate_update[mask]
            tu = true_update[mask]
            st_tp = int(np.sum(su & tu))
            st_fp = int(np.sum(su & ~tu))
            st_tn = int(np.sum(~su & ~tu))
            st_fn = int(np.sum(~su & tu))
            d["total"] += int(np.sum(mask))
            d["correct"] += st_tp + st_tn
            d["tp"] += st_tp
            d["fp"] += st_fp
            d["tn"] += st_tn
            d["fn"] += st_fn

    step_rows = [
        make_metric_row(instance, algorithm, surrogate, trial, step, d["total"], d["correct"], d["tp"], d["fp"], d["tn"], d["fn"])
        for step, d in sorted(by_step.items())
    ]
    row = make_metric_row(instance, algorithm, surrogate, trial, None, total, correct, tp, fp, tn, fn)
    row["n_generations"] = int(len(inc_logs))
    row["fe_offset"] = int(fe_offset)
    row["max_fe"] = "" if max_fe is None else int(max_fe)
    row["n_true_update"] = int(n_true_update)
    row["n_surrogate_update"] = int(n_sur_update)
    row["truth_update_rate"] = _safe_div(n_true_update, total)
    row["surrogate_update_rate"] = _safe_div(n_sur_update, total)
    row["missing_truth_generations"] = len(missing_truth_gens)
    row["path"] = str(path)
    return row, step_rows


def _group_rows(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[tuple[Any, ...], List[Dict[str, Any]]]:
    out: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        out.setdefault(key, []).append(row)
    return out


def aggregate_trials(rows_in: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metric_cols = [
        "accuracy",
        "precision_update",
        "recall_update",
        "specificity_no_update",
        "f1_update",
        "balanced_accuracy",
        "mcc",
        "truth_update_rate",
        "surrogate_update_rate",
    ]
    count_cols = ["n_comparisons", "correct", "tp", "fp", "tn", "fn", "n_true_update", "n_surrogate_update"]
    for (instance, algorithm, surrogate), group in _group_rows(rows_in, ["instance", "algorithm", "surrogate"]).items():
        row: Dict[str, Any] = {
            "instance": instance,
            "algorithm": algorithm,
            "surrogate": surrogate,
            "n_trials": len(group),
        }
        for col in count_cols:
            row[f"{col}_sum"] = int(sum(int(g.get(col, 0) or 0) for g in group))
        row["micro_accuracy"] = _safe_div(row["correct_sum"], row["n_comparisons_sum"])
        row["micro_precision_update"] = _safe_div(row["tp_sum"], row["tp_sum"] + row["fp_sum"])
        row["micro_recall_update"] = _safe_div(row["tp_sum"], row["tp_sum"] + row["fn_sum"])
        row["micro_specificity_no_update"] = _safe_div(row["tn_sum"], row["tn_sum"] + row["fp_sum"])
        row["micro_f1_update"] = _safe_div(2 * row["tp_sum"], 2 * row["tp_sum"] + row["fp_sum"] + row["fn_sum"])
        for col in metric_cols:
            row[f"{col}_mean"] = _mean_or_nan([float(g.get(col, float("nan"))) for g in group])
            row[f"{col}_std"] = _std_or_nan([float(g.get(col, float("nan"))) for g in group])
        rows.append(row)
    return sorted(rows, key=lambda r: (str(r["instance"]), str(r["surrogate"]), str(r["algorithm"])))


def aggregate_steps(rows_in: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (instance, algorithm, surrogate, step), group in _group_rows(rows_in, ["instance", "algorithm", "surrogate", "step"]).items():
        tp = int(sum(int(g.get("tp", 0) or 0) for g in group))
        fp = int(sum(int(g.get("fp", 0) or 0) for g in group))
        tn = int(sum(int(g.get("tn", 0) or 0) for g in group))
        fn = int(sum(int(g.get("fn", 0) or 0) for g in group))
        total = int(sum(int(g.get("n_comparisons", 0) or 0) for g in group))
        correct = int(sum(int(g.get("correct", 0) or 0) for g in group))
        row = make_metric_row(str(instance), str(algorithm), str(surrogate), "ALL", int(step), total, correct, tp, fp, tn, fn)
        row["n_trials"] = len({g.get("trial") for g in group})
        rows.append(row)
    return sorted(rows, key=lambda r: (str(r["instance"]), str(r["surrogate"]), str(r["algorithm"]), int(r["step"])))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def display_name(row: Mapping[str, Any]) -> str:
    surrogate = str(row.get("surrogate", ""))
    algorithm = str(row.get("algorithm", ""))
    if algorithm.endswith("_BB") or "_BB" in algorithm:
        return "RLEA-BB"
    if surrogate == "rf":
        return "RLEA-RF"
    if surrogate == "gbdt":
        return "RLEA-GBDT"
    return surrogate or algorithm


def average_ranks(values_by_col: Mapping[str, float], higher_is_better: bool = True) -> Dict[str, float]:
    valid = [(k, float(v)) for k, v in values_by_col.items() if not math.isnan(float(v))]
    ranks = {k: float("nan") for k in values_by_col}
    if not valid:
        return ranks
    valid.sort(key=lambda kv: kv[1], reverse=higher_is_better)
    pos = 0
    while pos < len(valid):
        end = pos + 1
        while end < len(valid) and valid[end][1] == valid[pos][1]:
            end += 1
        rank = (pos + 1 + end) / 2.0
        for key, _ in valid[pos:end]:
            ranks[key] = rank
        pos = end
    return ranks


def table_metric_to_trial_metric(metric: str) -> str:
    metric = str(metric)
    if metric.endswith("_mean") or metric.endswith("_std"):
        return metric.rsplit("_", 1)[0]
    if metric.startswith("micro_"):
        return metric[len("micro_"):]
    if metric.endswith("_sum"):
        return metric[:-4]
    return metric


def _normal_two_sided_p_from_z(z: float) -> float:
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


def _fallback_paired_pvalue(diffs: Sequence[float]) -> float:
    vals = [float(d) for d in diffs if not math.isnan(float(d))]
    if len(vals) < 2:
        return float("nan")
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    if std == 0.0:
        return 1.0 if mean == 0.0 else 0.0
    z = mean / (std / math.sqrt(len(vals)))
    return _normal_two_sided_p_from_z(z)


def _fallback_unpaired_pvalue(a: Sequence[float], b: Sequence[float]) -> float:
    av = np.asarray([float(x) for x in a if not math.isnan(float(x))], dtype=float)
    bv = np.asarray([float(x) for x in b if not math.isnan(float(x))], dtype=float)
    if len(av) < 2 or len(bv) < 2:
        return float("nan")
    va = float(np.var(av, ddof=1))
    vb = float(np.var(bv, ddof=1))
    se = math.sqrt(va / len(av) + vb / len(bv))
    if se == 0.0:
        return 1.0 if float(np.mean(av)) == float(np.mean(bv)) else 0.0
    z = (float(np.mean(av)) - float(np.mean(bv))) / se
    return _normal_two_sided_p_from_z(z)


def significance_symbol(
    values: Sequence[float],
    baseline_values: Sequence[float],
    alpha: float,
    higher_is_better: bool = True,
) -> Tuple[str, float]:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    base = [float(v) for v in baseline_values if not math.isnan(float(v))]
    if len(vals) == 0 or len(base) == 0:
        return "", float("nan")
    mean_delta = float(np.mean(vals) - np.mean(base))
    try:
        from scipy import stats
        if len(vals) == len(base) and len(vals) >= 2:
            diffs = np.asarray(vals, dtype=float) - np.asarray(base, dtype=float)
            if np.allclose(diffs, 0.0):
                p_value = 1.0
            else:
                p_value = float(stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
        elif len(vals) >= 2 and len(base) >= 2:
            p_value = float(stats.mannwhitneyu(vals, base, alternative="two-sided").pvalue)
        else:
            p_value = float("nan")
    except Exception:
        if len(vals) == len(base) and len(vals) >= 2:
            p_value = _fallback_paired_pvalue(np.asarray(vals, dtype=float) - np.asarray(base, dtype=float))
        else:
            p_value = _fallback_unpaired_pvalue(vals, base)
    if math.isnan(p_value) or p_value >= float(alpha):
        return "~", p_value
    if higher_is_better:
        return ("+" if mean_delta > 0 else "-"), p_value
    return ("+" if mean_delta < 0 else "-"), p_value


def collect_trial_metric_values(
    trial_rows: Sequence[Dict[str, Any]],
    metric: str,
) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    by_key: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for row in trial_rows:
        if metric not in row:
            continue
        try:
            value = float(row.get(metric, float("nan")))
        except Exception:
            continue
        if math.isnan(value):
            continue
        key = (str(row.get("instance", "")), str(row.get("algorithm", "")), display_name(row))
        by_key.setdefault(key, {})[str(row.get("trial", ""))] = value
    return by_key


def resolve_baseline_row(
    rows: Sequence[Dict[str, Any]],
    baseline: str,
) -> Optional[Dict[str, Any]]:
    baseline = str(baseline)
    for row in rows:
        if str(row.get("algorithm", "")) == baseline:
            return row
    for row in rows:
        if display_name(row) == baseline:
            return row
    return None


def make_instance_table(
    summary_rows: Sequence[Dict[str, Any]],
    trial_rows: Sequence[Dict[str, Any]],
    metric: str,
    baseline_algorithm: Optional[str],
    alpha: float,
) -> List[Dict[str, Any]]:
    instances = sorted({str(r.get("instance", "")) for r in summary_rows})
    col_order: List[str] = []
    by_instance: Dict[str, Dict[str, float]] = {}
    sig_by_instance: Dict[str, Dict[str, str]] = {}
    rank_sums: Dict[str, float] = {}
    rank_counts: Dict[str, int] = {}
    trial_metric = table_metric_to_trial_metric(metric)
    trial_values = collect_trial_metric_values(trial_rows, trial_metric)

    for inst in instances:
        vals: Dict[str, float] = {}
        sigs: Dict[str, str] = {}
        inst_rows = [r for r in summary_rows if str(r.get("instance", "")) == inst]
        baseline_row = resolve_baseline_row(inst_rows, baseline_algorithm or "") if baseline_algorithm else None
        baseline_trials: Dict[str, float] = {}
        if baseline_row is not None:
            baseline_key = (inst, str(baseline_row.get("algorithm", "")), display_name(baseline_row))
            baseline_trials = trial_values.get(baseline_key, {})
        for row in inst_rows:
            col = display_name(row)
            if col not in col_order:
                col_order.append(col)
            vals[col] = float(row.get(metric, float("nan")))
            if baseline_row is not None:
                if str(row.get("algorithm", "")) == str(baseline_row.get("algorithm", "")):
                    sigs[col] = "~"
                else:
                    key = (inst, str(row.get("algorithm", "")), col)
                    other_trials = trial_values.get(key, {})
                    common_trials = sorted(set(other_trials) & set(baseline_trials))
                    if len(common_trials) >= 2:
                        other_vals = [other_trials[t] for t in common_trials]
                        base_vals = [baseline_trials[t] for t in common_trials]
                    else:
                        other_vals = list(other_trials.values())
                        base_vals = list(baseline_trials.values())
                    sigs[col], _ = significance_symbol(other_vals, base_vals, alpha, higher_is_better=True)
        by_instance[inst] = vals
        sig_by_instance[inst] = sigs
        ranks = average_ranks(vals, higher_is_better=True)
        for col, rank in ranks.items():
            if math.isnan(rank):
                continue
            rank_sums[col] = rank_sums.get(col, 0.0) + rank
            rank_counts[col] = rank_counts.get(col, 0) + 1

    preferred = ["RLEA-RF", "RLEA-GBDT", "RLEA", "RLEA-BB"]
    col_order = [c for c in preferred if c in col_order] + [c for c in col_order if c not in preferred]

    table: List[Dict[str, Any]] = []
    for inst in instances:
        row: Dict[str, Any] = {"instance": inst}
        for col in col_order:
            val = by_instance[inst].get(col, float("nan"))
            if math.isnan(val):
                row[col] = ""
            else:
                sig = sig_by_instance.get(inst, {}).get(col, "")
                row[col] = f"{val:.6g}" if not sig else f"{val:.6g} {sig}"
        table.append(row)

    rank_row: Dict[str, Any] = {"instance": "avg_rank"}
    for col in col_order:
        rank_row[col] = "" if rank_counts.get(col, 0) == 0 else f"{rank_sums[col] / rank_counts[col]:.6g}"
    table.append(rank_row)
    return table


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze true-value accuracy of incumbent update decisions.")
    ap.add_argument("--root", default="")
    ap.add_argument("--instance", default=None, help="Single instance name such as att48. If omitted, scan all instances.")
    ap.add_argument("--instances", nargs="+", default=["burma14","ulysses22","fri26","bayg29","swiss42","att48","berlin52", "br17","ftv33","ftv35","ftv38","p43","ry48p","ft53", "N-pal11","N-pal19","N-pal23","N-pal27","N-econ36","N-p40-01","N-p44-01","N-be75np"], help="instance names", choices=["burma14","ulysses22","fri26","bayg29","swiss42","att48","berlin52", "br17","ftv33","ftv35","ftv38","p43","ry48p","ft53", "N-pal11","N-pal19","N-pal23","N-pal27","N-econ36","N-p40-01","N-be75np"])
    ap.add_argument("--algorithm", nargs="*", default=["RLEA"], help="Algorithm directory names to include under every selected instance.")
    ap.add_argument("--input", nargs="*", default=None, help="Explicit result.npz paths or glob patterns.")
    ap.add_argument("--out-dir", type=Path, default="_incumbent_update_accuracy", help="Directory to write CSV outputs.")
    ap.add_argument("--require-truth", action="store_true", help="Skip runs without comparison truth logs.")
    ap.add_argument("--verbose", action="store_true", help="Print each loaded result.npz path.")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    ap.add_argument("--no-step", action="store_true", help="Skip step-wise aggregation for faster summary/trial CSV generation.")
    ap.add_argument("--jobs", type=int, default=1, help="Number of result.npz files to load in parallel.")
    ap.add_argument("--max-fe", type=int, default=300, help="Aggregate only incumbent comparison logs whose corresponding FE is <= this value.")
    ap.add_argument("--table-metric", default="balanced_accuracy_mean", help="Summary metric used in the instance table.")
    ap.add_argument("--table-baseline-algorithm", default="RLEA", help="Algorithm directory name or displayed table column used as the baseline for p<alpha significance marks.")
    ap.add_argument("--table-alpha", type=float, default=0.05, help="Significance threshold for table marks: +, ~, -.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root) if args.root is not None else None
    instances = args.instances if args.instances else ([args.instance] if args.instance else None)
    files = discover_result_files(root, instances, args.algorithm, args.input)
    if not files:
        raise SystemExit("No result.npz files found.")
    if args.verbose:
        print(f"discovered_files: {len(files)}")

    trial_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    skipped: List[str] = []

    def handle_result(path: Path) -> tuple[Path, Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
        try:
            row, rows_by_step = summarize_one_result(path, root, include_steps=not args.no_step, max_fe=args.max_fe)
        except Exception as e:
            return path, None, [], str(e)
        if args.require_truth and row["n_comparisons"] == 0:
            return path, None, [], "no comparison truth logs"
        return path, row, rows_by_step, None

    jobs = max(1, int(args.jobs))
    if jobs == 1:
        iterator = tqdm(
            files,
            total=len(files),
            desc="Loading result.npz",
            unit="run",
            disable=args.no_progress,
        )
        for path in iterator:
            if args.verbose:
                tqdm.write(str(path))
            path, row, rows_by_step, err = handle_result(path)
            if err is not None:
                skipped.append(f"{path}: {err}")
                continue
            assert row is not None
            trial_rows.append(row)
            step_rows.extend(rows_by_step)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(handle_result, path) for path in files]
            iterator = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Loading result.npz x{jobs}",
                unit="run",
                disable=args.no_progress,
            )
            for fut in iterator:
                path, row, rows_by_step, err = fut.result()
                if args.verbose:
                    tqdm.write(str(path))
                if err is not None:
                    skipped.append(f"{path}: {err}")
                    continue
                assert row is not None
                trial_rows.append(row)
                step_rows.extend(rows_by_step)

    if not trial_rows:
        raise SystemExit("No usable result.npz files found. Did these runs include comparison_true_update logs?")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trial_rows = sorted(trial_rows, key=lambda r: (str(r["instance"]), str(r["surrogate"]), str(r["algorithm"]), str(r["trial"])))
    step_rows = sorted(step_rows, key=lambda r: (str(r["instance"]), str(r["surrogate"]), str(r["algorithm"]), int(r["step"]), str(r["trial"])))
    summary_rows = aggregate_trials(trial_rows)
    step_summary_rows = aggregate_steps(step_rows) if not args.no_step else []
    table_rows = make_instance_table(summary_rows, trial_rows, args.table_metric, args.table_baseline_algorithm, args.table_alpha)

    trial_path = args.out_dir / "incumbent_update_accuracy_trials.csv"
    summary_path = args.out_dir / "incumbent_update_accuracy_summary.csv"
    step_path = args.out_dir / "incumbent_update_accuracy_by_step.csv"
    table_path = args.out_dir / "incumbent_update_accuracy_table.csv"
    write_csv(trial_path, trial_rows)
    write_csv(summary_path, summary_rows)
    if not args.no_step:
        write_csv(step_path, step_summary_rows)
    write_csv(table_path, table_rows)

    print(f"loaded_runs: {len(trial_rows)}")
    print(f"summary: {summary_path}")
    print(f"trials:  {trial_path}")
    if not args.no_step:
        print(f"steps:   {step_path}")
    print(f"table:   {table_path}")
    if skipped:
        print(f"skipped: {len(skipped)}")
        for msg in skipped[:10]:
            print(f"  {msg}")
    cols = [
        "instance",
        "surrogate",
        "n_trials",
        "n_comparisons_sum",
        "micro_accuracy",
        "micro_precision_update",
        "micro_recall_update",
        "micro_specificity_no_update",
    ]
    print(" ".join(cols))
    for row in summary_rows:
        print(" ".join(str(row.get(col, "")) for col in cols))


if __name__ == "__main__":
    main()
