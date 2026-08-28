import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Use a non-interactive backend for saving figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

# def _safe_load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
#     with np.load(npz_path, allow_pickle=True) as d:
#         return {k: d[k] for k in d.files}


def _best_so_far(values: np.ndarray, direction: str) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values
    if direction == "max":
        return np.maximum.accumulate(values)
    return np.minimum.accumulate(values)


def _flatten_archive_fx(archive_fx: np.ndarray) -> np.ndarray:
    """
    archive_fx may be nested like [generations][batch] (object array).
    Flatten to 1D evaluation sequence in order.
    """
    arr = np.asarray(archive_fx)
    if arr.dtype == object:
        flat_list: List[float] = []
        for item in arr:
            if isinstance(item, (list, tuple, np.ndarray)):
                sub = np.asarray(item, dtype=float).reshape(-1)
                flat_list.extend(sub.tolist())
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
    problem = str(meta.get("problem", "")).upper()
    if problem == "LOP":
        return "min"
    return "min"


def _collect_method_dirs(root: Path, ins_name: str) -> List[Path]:
    ins_dir = root / ins_name
    if not ins_dir.exists():
        raise FileNotFoundError(f"Instance folder not found: {ins_dir}")
    return [p for p in ins_dir.iterdir() if p.is_dir()]


def _match_methods(method_dirs: List[Path], includes: List[str], excludes: List[str]) -> List[Path]:
    if not includes:
        selected = method_dirs
    else:
        inc = [s.lower() for s in includes]
        selected = [p for p in method_dirs if any(s in p.name.lower() for s in inc)]
    if excludes:
        exc = [s.lower() for s in excludes]
        selected = [p for p in selected if not any(s in p.name.lower() for s in exc)]
    return selected


def _load_histories(method_dir: Path, direction: str) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    trial_dirs = sorted([p for p in method_dir.iterdir() if p.is_dir() and p.name.startswith("trial_")])
    histories: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for td in trial_dirs:
        try:
            trial_id = int(td.name.split("_", 1)[1])
        except Exception:
            trial_id = len(histories)
        npz_path = td / "result.npz"
        if not npz_path.exists():
            continue
        try:
            data = _safe_load_npz(npz_path, keys=["archive_fx", "history"])
        except Exception:
            continue
        archive_fx = data.get("archive_fx", None)
        if archive_fx is not None and np.asarray(archive_fx).size > 0:
            flat_fx = _flatten_archive_fx(archive_fx)
            if flat_fx.size == 0:
                continue
            h = _best_so_far(flat_fx, direction)
            evals = np.arange(1, h.size + 1, dtype=int)
            histories.append((trial_id, evals, h))
            continue
        history = data.get("history", None)
        if history is not None and np.asarray(history).size > 0:
            h = np.asarray(history, dtype=float).reshape(-1)
            evals = np.arange(1, h.size + 1, dtype=int)
            histories.append((trial_id, evals, h))
    return histories


def _pad_to_matrix(histories: List[Tuple[int, np.ndarray, np.ndarray]]) -> np.ndarray:
    if not histories:
        return np.empty((0, 0), dtype=float)
    max_len = max(h.size for _, _, h in histories)
    mat = np.full((len(histories), max_len), np.nan, dtype=float)
    for i, (_, _, h) in enumerate(histories):
        mat[i, : h.size] = h
    return mat


def _summarize_curves(histories: List[Tuple[int, np.ndarray, np.ndarray]]) -> Dict[str, np.ndarray]:
    mat = _pad_to_matrix(histories)
    if mat.size == 0:
        return {"mean": np.array([]), "std": np.array([]), "median": np.array([]), "n": np.array([])}
    return {
        "mean": np.nanmean(mat, axis=0),
        "std": np.nanstd(mat, axis=0),
        "median": np.nanmedian(mat, axis=0),
        "n": np.sum(~np.isnan(mat), axis=0),
    }


def _value_at_eval(history: np.ndarray, eval_axis: np.ndarray, eval_point: int, fallback: str) -> float:
    if history.size == 0 or eval_axis.size == 0:
        return np.nan
    # find rightmost eval_axis <= eval_point
    idx = np.searchsorted(eval_axis, eval_point, side="right") - 1
    if idx < 0:
        return float(history[0]) if fallback == "last" else np.nan
    if idx < history.size:
        return float(history[idx])
    return float(history[-1]) if fallback == "last" else np.nan


def _parse_legend_map(items: List[str]) -> List[Tuple[str, str]]:
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


def _legend_name(method_name: str, legend_map: List[Tuple[str, str]]) -> str:
    for k, v in legend_map:
        if method_name == k or k in method_name:
            return v
    return method_name


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize permutation optimization results")
    ap.add_argument("--root", default="E:\\2325_yamaguchi\\_results_TEVC_20260709\\", help="Result root folder (same as --save_path in main.py)")
    ap.add_argument("--ins-name", default="br17", help="Instance name (folder under root)", choices=["burma14","ulysses22","fri26","bayg29","swiss42","att48","berlin52", "br17","ftv33","ftv35","ftv38","p43","ry48p","ft53", "N-pal11","N-pal19","N-pal23","N-pal27","N-econ36","N-p40-01","N-p44-01","N-be75np"])
    # ap.add_argument("--include", nargs="*", default=["lat2way_sage_mean_knn_density_0.2_1_noLoc","lat2way_sage_mean_knn_density_0.2_2","lat2way_sage_mean_knn_density_0.2_3","lat2way_sage_mean_knn_density_0.2_4","lat2way_sage_mean_nn_density_1","lat2way_sage_mean_nn_density_2_0","lat2way_oracle_mean_best","fatrls_init100_fixed","umm_fixed","rflos_fixed","gbdtma_sage_mean_fixed_fin","gbdtma_gbdt_mean_fixed_fin"], help="Method name substrings to include")
    # ap.add_argument("--include", nargs="*", default=["lat2way_sage_mean_knn_density_1_1_50","lat2way_sage_mean_knn_density_1_5_50","lat2way_sage_mean_knn_density_1_10_50","lat2way_sage_mean_knn_density_1_20_50","lat2way_sage_mean_nn_density_1","fatrls_init100_fixed","gbdtma_gbdt_mean_fixed_fin"], help="Method name substrings to include")
    ap.add_argument("--include", nargs="*", default=["umm_fixed","fatrls_init100_fixed","rflos_fixed","gbdtma_gbdt_mean_fixed_fin","lat2way_sage_mean_knn_density_1_mu_10_50_BB","lat2way_sage_mean_knn_density_1_mu_10_50","lat2way_rf_mean_knn_density_1_mu_10_50","lat2way_gbdt_mean_knn_density_1_mu_10_50"], help="Method name substrings to include")
    # ap.add_argument("--include", nargs="*", default=["lat2way_sage_mean_knn_density_0.2_1_noLoc","lat2way_sage_mean_knn_density_0.2_2","lat2way_sage_mean_knn_density_0.2_3","lat2way_sage_mean_knn_density_0.2_4","lat2way_sage_mean_nn_density_1","fatrls_init100_fixed","gbdtma_gbdt_mean_fixed_fin","lat2way_sage_mean_knn_density_0.2_2_fixed"], help="Method name substrings to include")
    ap.add_argument("--exclude", nargs="*", default=["gbdtma_sage_mean_fixed_fin_BB","*bo*","*避難*","*_0", "*_whole","evals"], help="Method name substrings to exclude")
    ap.add_argument("--direction", choices=["auto", "min", "max"], default="auto", help="Objective direction")
    ap.add_argument("--eval-points", nargs="+", type=int, default=[10, 50, 100, 200, 300, 400, 500, 600],
                    help="Evaluation counts to compare (1-based)")
    ap.add_argument("--eval-fallback", choices=["last", "nan"], default="last",
                    help="If eval_point is out of eval axis range")
    ap.add_argument("--out-dir", default=None, help="Output folder (default: <root>/<ins_name>/_summary)")
    ap.add_argument("--curve-format", choices=["png", "pdf", "svg"], default="pdf")
    # ap.add_argument(
    #     "--legend-map",
    #     nargs="*",
    #     default=["lat2way_sage_mean_knn_density_0.2_2=Latent-2Ways_density_2","lat2way_sage_mean_knn_density_0.2_3=Latent-2Ways_density_3","lat2way_sage_mean_knn_density_0.2_4=Latent-2Ways_density_4","lat2way_sage_mean_knn_density_0.2_1_noLoc=Latent-2Ways_density_global","lat2way_oracle_mean_best=Latent-2Ways_oracle","lat2way_sage_mean_nn_density_1=Latent-2Ways_best","lat2way_sage_mean_knn_density_0.2_2_fixed=Latent-2ways_density-2_fixed","lat2way_sage_mean_nn_density_2_0=Latent-2Ways_density_2-0","lat2way_sage_mean_knn_promisingFar_2_0=Latent-2Ways_knn_promisingFar_2_0","lat2way_sage_mean_knn_promisingFar_2=Latent-2Ways_knn_promisingFar_2","lat2way_sage_mean_insert_rev=Latent-2Ways_rev","fatrls_init100_fixed=FAT-RLS-Init100","fatrls_fixed=FAT-RLS","umm_fixed=UMM","rflos_fixed=RFLoS","gbdtma_sage_mean_fixed_fin_BB=RLEA-BB","gbdtma_sage_mean_fixed_fin=RLEA","gbdtma_sage_mean_fixed_fin=Proposal_lcb","gbdtma_gbdt_mean_fixed_fin=GBDTMA","umm_fixed=UMM","rflos_fixed=RFLoS"],
    #     help="Legend rename rules in 'key=value' form. If key is contained in method name, rename to value.",
    # )
    ap.add_argument(
        "--legend-map",
        nargs="*",
        default=["lat2way_rf_mean_knn_density_1_mu_10_50=RLEA-rf","lat2way_gbdt_mean_knn_density_1_mu_10_50=RLEA-gbdt","lat2way_sage_mean_knn_density_1_1_50=Lat_EA_sig1_pop50","lat2way_sage_mean_knn_density_1_5_50=Lat_EA_sig5_pop50","lat2way_sage_mean_knn_density_1_mu_10_50_BB=RLEA_BB","lat2way_sage_mean_knn_density_1_mu_10_50=RLEA","lat2way_sage_mean_knn_density_1_20_50=Lat_EA_sig20_pop50","lat2way_sage_mean_knn_density_1_5_100=Lat_EA_sig5_pop100","lat2way_sage_mean_knn_density_1_10_100=Lat_EA_sig10_pop100","lat2way_sage_mean_knn_density_1_20_100=Lat_EA_sig20_pop100","lat2way_oracle_mean_best=Latent-2Ways_oracle","lat2way_sage_mean_nn_density_1=Latent-2Ways_best","fatrls_init100_fixed=FAT-RLS","fatrls_fixed=FAT-RLS","umm_fixed=UMM","rflos_fixed=RFLoS","gbdtma_sage_mean_fixed_fin_BB=RLEA-BB","gbdtma_sage_mean_fixed_fin=RLEA","gbdtma_sage_mean_fixed_fin=Proposal_lcb","gbdtma_gbdt_mean_fixed_fin=GBDTMA","umm_fixed=UMM","rflos_fixed=RFLoS"],
        help="Legend rename rules in 'key=value' form. If key is contained in method name, rename to value.",
    )
    ap.add_argument("--no-plot", action="store_true", help="Skip plotting curves")
    args = ap.parse_args()
    legend_map = _parse_legend_map(args.legend_map)

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else (root / args.ins_name / "_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    method_dirs = _collect_method_dirs(root, args.ins_name)
    method_dirs = _match_methods(method_dirs, args.include, args.exclude)
    if not method_dirs:
        raise SystemExit("No method folders matched. Check --include/--exclude.")

    curve_rows = []
    value_rows = []
    summary_rows = []
    method_items = []

    # First pass: load histories and find global max eval length
    global_max_len = 0
    for method_dir in method_dirs:
        direction = _infer_direction(method_dir, args.direction)
        histories = _load_histories(method_dir, direction)
        if not histories:
            continue
        max_len = max(h.size for _, _, h in histories)
        global_max_len = max(global_max_len, max_len)
        summary = _summarize_curves(histories)
        method_items.append((method_dir, direction, histories, summary))

    if global_max_len == 0:
        raise SystemExit("No valid histories found in selected methods.")

    global_eval_axis = np.arange(1, global_max_len + 1, dtype=int)

    fig, ax = plt.subplots(figsize=(5, 6))
    # fig, ax = plt.subplots(figsize=(10, 6))
    for method_dir, direction, histories, summary in method_items:
        method_name = method_dir.name
        display_name = _legend_name(method_name, legend_map)

        # curve data for csv aligned to global axis
        for i in range(global_max_len):
            in_range = i < summary["mean"].size
            curve_rows.append({
                "method": method_name,
                "eval": int(global_eval_axis[i]),
                "mean": float(summary["mean"][i]) if in_range else np.nan,
                "std": float(summary["std"][i]) if in_range else np.nan,
                "median": float(summary["median"][i]) if in_range else np.nan,
                "n": int(summary["n"][i]) if in_range else 0,
            })

        # plot (pad with NaN to align on global x)
        if not args.no_plot and summary["mean"].size > 0:
            y = np.full(global_max_len, np.nan, dtype=float)
            y[: summary["mean"].size] = summary["mean"]
            y_lo = np.full(global_max_len, np.nan, dtype=float)
            y_hi = np.full(global_max_len, np.nan, dtype=float)
            y_lo[: summary["mean"].size] = summary["mean"] - summary["std"]
            y_hi[: summary["mean"].size] = summary["mean"] + summary["std"]
            safe_label = display_name if not display_name.startswith("_") else f"method{display_name}"
            ax.plot(global_eval_axis, y, label=safe_label)
            ax.fill_between(global_eval_axis, y_lo, y_hi, alpha=0.2)
            plt.legend()

        # values at eval points (per trial axis is 1..len)
        for trial_id, eval_axis, h in histories:
            for e in args.eval_points:
                value_rows.append({
                    "method": method_name,
                    "trial": int(trial_id),
                    "eval": int(e),
                    "value": _value_at_eval(h, eval_axis, e, args.eval_fallback),
                })

        # summary at eval points
        for e in args.eval_points:
            vals = [row["value"] for row in value_rows if row["method"] == method_name and row["eval"] == e]
            arr = np.asarray(vals, dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                continue
            summary_rows.append({
                "method": method_name,
                "eval": int(e),
                "n": int(arr.size),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "direction": direction,
            })

    # Save CSVs
    curve_csv = out_dir / "curve_stats.csv"
    pd.DataFrame(curve_rows).to_csv(curve_csv, index=False)

    values_csv = out_dir / "values_at_evals.csv"
    pd.DataFrame(value_rows).to_csv(values_csv, index=False)

    summary_csv = out_dir / "summary_at_evals.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    # PGFPlots用CSV
    pgf_dir = out_dir / "pgf"
    pgf_dir.mkdir(exist_ok=True)

    curve_df = pd.DataFrame(curve_rows)

    for method_name, df_method in curve_df.groupby("method"):
        display_name = _legend_name(method_name, legend_map)

        df_method = df_method.copy()
        df_method["lower"] = df_method["mean"] - df_method["std"]
        df_method["upper"] = df_method["mean"] + df_method["std"]

        # ファイル名として安全な名前にする
        safe_name = (
            display_name
            .replace(" ", "_")
            .replace("/", "_")
        )

        df_method[
            ["eval", "mean", "std", "lower", "upper"]
        ].to_csv(
            pgf_dir / f"{safe_name}.csv",
            index=False
        )

    # Save plot
    if not args.no_plot: #! legend inside graph
        ax.set_xlabel("Evaluation Count", fontsize=14)
        ax.set_ylabel("Best-So-Far Value", fontsize=14)
        ax.set_xlim(0, 300)
        ax.tick_params(axis="both", labelsize=14)
        # ax.set_title(f"Optimization Curves: {args.ins_name}")
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", frameon=True,fontsize=14)
        fig_path = out_dir / f"{args.ins_name}_long.{args.curve_format}"
        fig.tight_layout()
        fig.savefig(fig_path)

    # if not args.no_plot: #! legend side
    #     ax.set_xlabel("Evaluation Count", fontsize=14)
    #     ax.set_ylabel("Best-So-Far Value", fontsize=14)
    #     ax.set_xlim(0, 600)
    #     ax.tick_params(axis="both", labelsize=14)
    #     # ax.set_title(f"Optimization Curves: {args.ins_name}")
    #     ax.grid(True)
    #     handles, labels = ax.get_legend_handles_labels()
    #     if handles:
    #         # ax.legend(
    #         #     loc="center left",
    #         #     bbox_to_anchor=(1.02, 0.5),
    #         #     frameon=True,
    #         #     fontsize=14,
    #         #     borderaxespad=0,
    #         # )
    #         fig.legend(
    #             handles,
    #             labels,
    #             loc="center left",
    #             bbox_to_anchor=(0.72, 0.5),
    #             frameon=True,
    #             fontsize=13,
    #             borderaxespad=0.0,
    #         )
    #     fig_path = out_dir / f"{args.ins_name}_long_test.{args.curve_format}"
    #     # fig.tight_layout(rect=[0, 0, 0.78, 1])
    #     fig.subplots_adjust(
    #         left=0.10,
    #         right=0.69,
    #         bottom=0.14,
    #         top=0.96,
    #     )
    #     fig.savefig(fig_path, bbox_inches="tight")


    print(f"[OK] curve stats -> {curve_csv}")
    print(f"[OK] values at evals -> {values_csv}")
    print(f"[OK] summary at evals -> {summary_csv}")
    if not args.no_plot:
        print(f"[OK] curve plot -> {fig_path}")


if __name__ == "__main__":
    main()
