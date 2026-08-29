from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

Result = Dict[str, Any]
EvalFn = Callable[[List[int]], float]


def reset_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def tsp_eval_factory(dist: np.ndarray) -> EvalFn:
    def evaluate(perm: List[int]) -> float:
        total = 0.0
        for i in range(len(perm) - 1):
            total += float(dist[perm[i]][perm[i + 1]])
        total += float(dist[perm[-1]][perm[0]])
        return total

    return evaluate


def lo_eval_factory(weights: np.ndarray) -> EvalFn:
    def evaluate(perm: List[int]) -> float:
        total = 0.0
        for i in range(1, len(perm)):
            for j in range(i):
                total += float(weights[perm[i]][perm[j]])
        return total

    return evaluate


def pack_result(
    best_perm: List[int],
    best_val: float,
    eval_perms: List[List[int]],
    eval_vals: List[float],
    history_best: List[float],
    logs: Dict[str, Any] | None = None,
) -> Result:
    logs = logs or {}
    result: Result = {
        "best_perm": np.asarray(best_perm, dtype=np.int32),
        "best_fx": np.float64(best_val),
        "history": np.asarray(history_best, dtype=np.float64),
        "archive_perm": np.asarray(eval_perms, dtype=np.int32),
        "archive_fx": np.asarray(eval_vals, dtype=np.float64),
    }
    for key, value in logs.items():
        dtype = np.float64 if key in {"metric", "eval_count", "elapsed_time", "step_time"} else object
        result[key] = np.asarray(value, dtype=dtype)
    return result


def run_fatrls(dim: int, true_eval_fn: EvalFn, args: argparse.Namespace, seed: int) -> Result:
    from fatrls import fat_rls

    reset_seed(seed)
    return pack_result(
        *fat_rls(
            dim=dim,
            true_eval_fn=true_eval_fn,
            budget=args.max_evals,
            move_type="insertion",
            seed=seed,
            tabu_mode="item",
            init_count=args.fatrls_init_count,
        )
    )


def run_umm(dim: int, true_eval_fn: EvalFn, args: argparse.Namespace, seed: int) -> Result:
    from umm import umm

    reset_seed(seed)
    return pack_result(
        *umm(
            dim=dim,
            true_eval_fn=true_eval_fn,
            budget=args.max_evals,
            seed=seed,
            m_ini=args.umm_m_ini,
            ratio_samples_learn=args.umm_ratio_samples_learn,
            weight_mass_learn=args.umm_weight_mass_learn,
        )
    )


def run_rflos(dim: int, true_eval_fn: EvalFn, args: argparse.Namespace, seed: int) -> Result:
    from RFLoS import RFLoSConfig, rflos

    reset_seed(seed)
    return pack_result(
        *rflos(
            dim=dim,
            true_eval_fn=true_eval_fn,
            budget=args.max_evals,
            cfg=RFLoSConfig(
                k=args.rflos_k,
                wmax=args.rflos_wmax,
                elr=args.rflos_elr,
                pop_size=args.pop_size,
            ),
            seed=seed,
        )
    )


def run_gbdtma(dim: int, true_eval_fn: EvalFn, args: argparse.Namespace, seed: int) -> Result:
    from gbdtma import GBDTMAConfig, gbdtma

    reset_seed(seed)
    return pack_result(
        *gbdtma(
            dim=dim,
            true_eval_fn=true_eval_fn,
            budget=args.max_evals,
            cfg=GBDTMAConfig(pop_size=args.pop_size),
            seed=seed,
        )
    )


def run_rlea(dim: int, true_eval_fn: EvalFn, args: argparse.Namespace, seed: int) -> Result:
    from RLEA import RLEAConfig, rlea

    reset_seed(seed)
    return pack_result(
        *rlea(
            dim=dim,
            true_eval_fn=true_eval_fn,
            budget=args.max_evals,
            cfg=RLEAConfig(
                K=args.RLEA_K,
                fit_per=args.RLEA_fit_per,
                pop_size=args.pop_size,
                incumbent_pool_size=args.incumbent_pool_size,
                incumbent_sigma=args.RLEA_incumbent_sigma,
            ),
            seed=seed,
        )
    )


ALGO_RUNNERS = {
    "RLEA": run_rlea,
    "rflos": run_rflos,
    "gbdtma": run_gbdtma,
    "umm": run_umm,
    "fatrls": run_fatrls,
}


def load_problem(args: argparse.Namespace) -> tuple[int, EvalFn, str]:
    if args.prob_name == "TSP":
        from data.tsplib import get_tsp_instance as tsp_get_instance

        dist, dim = tsp_get_instance.get_tsp_dat(args.ins_name)
        return int(dim), tsp_eval_factory(dist), "TSP"
    if args.prob_name == "LOP":
        from data.lolib import get_lo_instance as lo_get_instance

        weights, dim = lo_get_instance.get_lolib_dat(args.ins_name)
        return int(dim), lo_eval_factory(weights), "LOP"
    if args.prob_name == "ATSP":
        from data.atsplib import get_atsp_instance as atsp_get_instance

        dist, dim = atsp_get_instance.get_atsp_dat(args.ins_name)
        return int(dim), tsp_eval_factory(dist), "ATSP"
    raise ValueError(f"unknown problem: {args.prob_name}")

def save_trial(trial_dir: Path, result: Result) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trial_dir / "result.npz", **result)
    with open(trial_dir / "result.txt", "w", encoding="utf-8") as file:
        file.write(f"best_fx: {float(result['best_fx'])}\n")
        file.write("best_perm: " + " ".join(map(str, result["best_perm"].tolist())) + "\n")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Permutation optimization experiments")
    parser.add_argument("--ins_name", type=str, default="N-pal27")
    parser.add_argument("--prob_name", type=str, default="LOP", choices=["TSP", "LOP", "ATSP"])
    parser.add_argument("--algo", type=str, default="RLEA", choices=list(ALGO_RUNNERS.keys()))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--save_path", type=str, default="results")
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--max_evals", type=int, default=600)
    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--fatrls_init_count", type=int, default=100)
    parser.add_argument("--umm_m_ini", type=int, default=100)
    parser.add_argument("--umm_ratio_samples_learn", type=float, default=0.25)
    parser.add_argument("--umm_weight_mass_learn", type=float, default=0.9)
    parser.add_argument("--rflos_k", type=int, default=5)
    parser.add_argument("--rflos_wmax", type=int, default=10)
    parser.add_argument("--rflos_elr", type=float, default=0.2)
    parser.add_argument("--RLEA_K", type=int, default=1)
    parser.add_argument("--RLEA_fit_per", type=int, default=5)
    parser.add_argument("--RLEA_incumbent_sigma", type=int, default=10)
    parser.add_argument("--incumbent_pool_size", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dim, true_eval_fn, problem = load_problem(args)
    out_root = Path(args.save_path) / args.ins_name / args.algo
    out_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "instance": args.ins_name,
        "problem": problem,
        "dim": dim,
        "algo": args.algo,
        "max_evals": args.max_evals,
        "trials": args.trials,
        "seed0": args.seed0,
    }
    with open(out_root / "meta.json", "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)

    runner = ALGO_RUNNERS[args.algo]
    print(f"[MODE={args.algo}] instance={args.ins_name} problem={problem} dim={dim}")
    for trial in range(args.trials):
        seed = args.seed0 + trial
        print(f"[Trial {trial + 1}/{args.trials}] seed={seed}")
        result = runner(dim, true_eval_fn, args, seed)
        save_trial(out_root / f"trial_{trial:02d}", result)
        print(f"best_fx={float(result['best_fx']):.6f}")
    print(f"[DONE] results saved in {out_root}")


if __name__ == "__main__":
    main()
