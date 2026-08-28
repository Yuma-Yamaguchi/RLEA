# main.py (refactored)
import argparse
import os
import json
import numpy as np
import random
import torch, sys
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from surrogatePermOptim import SurrogatePermBO
from fatrls import fat_rls  # FAT-RLS 縺ｮ繧ｳ繧｢髢｢謨ｰ
from umm import umm
from gbdtma import gbdtma  # GBDTMA 縺ｮ繝ｩ繝・ヱ
from latent_2way import latent_2ways  # Latent 2-ways 縺ｮ繝ｩ繝・ヱ
from RFLoS import rflos, RFLoSConfig
from gpso import gpso
from fatrls_gnn_surrogate import fatrls_with_gnn_surrogate, SurFATRLSConfig

# 繝・・繧ｿ繝ｭ繝ｼ繝
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.qaplib import get_instance as qap_get_instance
from data.tsplib import get_tsp_instance as tsp_get_instance
from data.lolib import get_lo_instance as lo_get_instance
from data.pfsplib import get_pfsp_instance as pfsp_get_instance
from data.atsplib import get_atsp_instance as atsp_get_instance

def reset_seed(seed: int = 0) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


# ===== 隧穂ｾ｡髢｢謨ｰ繝輔ぃ繧ｯ繝医Μ =====

def tsp_eval_factory(dist: np.ndarray):
    dim = dist.shape[0]

    def f(perm: List[int]) -> float:
        s = 0.0
        for i in range(dim - 1):
            s += dist[perm[i]][perm[i + 1]]
        s += dist[perm[-1]][perm[0]]
        return float(s)

    return f


def qap_eval_factory(F: np.ndarray, D: np.ndarray):
    dim = F.shape[0]

    def f(perm: List[int]) -> float:
        tot = 0.0
        for i in range(dim):
            for j in range(dim):
                tot += F[i][j] * D[perm[i]][perm[j]]
        return float(tot)

    return f


def lo_eval_factory(W: np.ndarray):
    """
    LOP: 荳倶ｸ芽ｧ偵・蜥鯉ｼ域怙螟ｧ蛹厄ｼ峨ｒ縺昴・縺ｾ縺ｾ霑斐☆
    """
    n = W.shape[0]

    def f(perm: List[int]) -> float:
        s = 0.0
        for i in range(1, n):
            ii = perm[i]
            for j in range(i):
                jj = perm[j]
                s += float(W[ii][jj])
        return float(s)

    return f

def pfsp_eval_factory(P: np.ndarray):
    """
    PFSP 縺ｮ makespan・域怙蟆丞喧・峨ｒ霑斐☆縲・
    P : (n_jobs, n_machines)
    """
    P = np.asarray(P, dtype=np.int64)
    n_jobs, n_machines = P.shape

    def f(perm: List[int]) -> float:
        machines_ct = np.zeros(n_machines, dtype=np.int64)
        for job in perm:
            prev_finish = 0
            p_row = P[job]
            for m in range(n_machines):
                start = machines_ct[m] if machines_ct[m] > prev_finish else prev_finish
                finish = start + p_row[m]
                machines_ct[m] = finish
                prev_finish = finish
        return float(machines_ct[-1])

    return f


# ===== 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝縺斐→縺ｮ縲・ trial 螳溯｡後Λ繝・ヱ縲・=====

def run_algo_bo(
    dim: int,
    true_eval_fn,
    problem: str,
    dist_for_2opt: np.ndarray,
    args,
    surrogate_kind: str,
    seed: int,
) -> Dict[str, Any]:
    """Run SurrogatePermBO for one trial and return an npz-ready dict."""
    reset_seed(seed)

    bo = SurrogatePermBO(
        dim=dim,
        true_eval_fn=true_eval_fn,
        problem=problem,
        surrogate_kind=surrogate_kind,
        pop_size=args.pop_size,
        n_init_true=args.n_init_true,
        batch_true=args.batch_true,
        candidate_nums=args.candidate_nums,
        optimizer=args.optimizer,
        int_gen=args.int_gen,
        kappa=args.kappa,
        acq=args.acq,
        xi=args.xi,
        max_queries=args.max_queries,
        max_eval=args.max_evals,
        gnn_epochs=args.gnn_epochs,
        seed=seed,
        parent_selection=args.bo_parent_selection,
        elite_rate=args.bo_elite_rate,
        logger=None,
    )

    # use_2opt = bool(args.use_2opt) and (problem == "TSP") and (dist_for_2opt is not None)

    # SurrogatePermBO.run 縺ｯ cand/label/pred/sigma 縺ｾ縺ｧ霑斐＠縺ｦ縺上ｋ
    best_perm, best_fx, history, cand, label, pred, sigma, archive_perm, archive_fx = bo.run(
        # dist_mtx_for_2opt=dist_for_2opt if use_2opt else None
    )

    return {
        "best_perm": np.array(best_perm, dtype=np.int32),
        "best_fx": np.float64(best_fx),
        "history": np.array(history, dtype=np.float64),
        "cand": np.array(cand, dtype=object),
        "label": np.array(label, dtype=object),
        "pred": np.array(pred, dtype=object),
        "sigma": np.array(sigma, dtype=object),
        "archive_perm": np.array(archive_perm, dtype=np.int32),
        "archive_fx": np.array(archive_fx, dtype=np.float64),
    }

def run_algo_fatrls(
    dim: int,
    true_eval_fn,
    problem: str,
    dist_for_2opt: np.ndarray,
    args,
    surrogate_kind: str,
    seed: int,
) -> Dict[str, Any]:
    """Run plain FAT-RLS. surrogate_kind is ignored."""
    reset_seed(seed)

    # if problem == "TSP":
    #     tabu_mode = "move"
    #     move_type = "invert"
    # elif problem == "ATSP":
    #     tabu_mode = "move"
    #     move_type = "invert"
    # elif problem == "QAP":
    #     tabu_mode = "move"
    #     move_type = "swap"
    # elif problem == "LOP":
    #     tabu_mode = "move"
    #     move_type = "swap"
    # elif problem == "PFSP":
    #     tabu_mode = "item"
    #     move_type = "insert"

    tabu_mode = "item"
    move_type = "insertion"

    best_perm, best_val, eval_perms, eval_vals, history_best = fat_rls(
        dim=dim,
        true_eval_fn=true_eval_fn,
        budget=args.max_evals,
        tabu_mode=tabu_mode,
        move_type=move_type,
        seed=seed,
        init_count=args.fatrls_init_count,
    )

    return {
        "best_perm": np.array(best_perm, dtype=np.int32),
        "best_fx": np.float64(best_val),
        "history": np.array(history_best, dtype=np.float64),
        "archive_perm": np.array(eval_perms, dtype=np.int32),
        "archive_fx": np.array(eval_vals, dtype=np.float64),
    }

def run_algo_umm(
    dim: int,
    true_eval_fn,
    problem: str,
    dist_for_2opt: np.ndarray,
    args,
    surrogate_kind: str,
    seed: int,
) -> Dict[str, Any]:
    """UMM (single trajectory)."""
    reset_seed(seed)

    best_perm, best_val, eval_perms, eval_vals, history_best = umm(
        dim=dim,
        true_eval_fn=true_eval_fn,
        budget=args.max_evals,
        seed=seed,
        m_ini=args.umm_m_ini,
        ratio_samples_learn=args.umm_ratio_samples_learn,
        weight_mass_learn=args.umm_weight_mass_learn,
    )

    return {
        "best_perm": np.array(best_perm, dtype=np.int32),
        "best_fx": np.float64(best_val),
        "history": np.array(history_best, dtype=np.float64),
        "archive_perm": np.array(eval_perms, dtype=np.int32),
        "archive_fx": np.array(eval_vals, dtype=np.float64),
    }

def run_algo_rflos(
    dim: int,
    true_eval_fn,
    problem: str,
    dist_for_2opt: np.ndarray,
    args,
    surrogate_kind: str,
    seed: int,
) -> Dict[str, Any]:
    """RFLoS (single trajectory) with gbdtma-compatible outputs."""
    reset_seed(seed)

    cfg = RFLoSConfig(
        k=args.rflos_k,
        wmax=args.rflos_wmax,
        elr=args.rflos_elr,
        pop_size=args.rflos_pop_size,
    )

    best_perm, best_val, eval_perms, eval_vals, history_best, logs = rflos(
        dim=dim,
        true_eval_fn=true_eval_fn,
        budget=args.max_evals,
        cfg=cfg,
        seed=seed,
    )

    return {
        "best_perm": np.array(best_perm, dtype=np.int32),
        "best_fx": np.float64(best_val),
        "history": np.array(history_best, dtype=np.float64),
        "cand": np.array(logs.get("cand", []), dtype=object),
        "label": np.array(logs.get("label", []), dtype=object),
        "pred": np.array(logs.get("pred", []), dtype=object),
        "sigma": np.array(logs.get("sigma", []), dtype=object),
        "archive_perm": np.array(eval_perms, dtype=np.int32),
        "archive_fx": np.array(eval_vals, dtype=np.float64),
        "metric": np.array(logs.get("metric", []), dtype=np.float64),
    }

def run_algo_gbdtma(
    dim: int,
    true_eval_fn,
    problem: str,
    dist_for_2opt: np.ndarray,
    args,
    surrogate_kind: str,
    seed: int,
) -> Dict[str, Any]:
    """Run GBDTMA with the selected surrogate."""
    reset_seed(seed)

    from gbdtma import GBDTMAConfig
    cfg = GBDTMAConfig(
        surrogate_kind=surrogate_kind,
        gnn_epochs=args.gbdtma_gnn_epochs,
        batch_size=args.gbdtma_batch_size,
        acq=args.gbdtma_acq,
        kappa=args.gbdtma_kappa,
        problem=problem,
    )

    best_perm, best_val, eval_perms, eval_vals, history_best, logs = gbdtma(
        dim=dim,
        true_eval_fn=true_eval_fn,
        budget=args.max_evals,
        cfg=cfg,
        seed=seed,
    )

    return {
        "best_perm": np.array(best_perm, dtype=np.int32),
        "best_fx": np.float64(best_val),
        "history": np.array(history_best, dtype=np.float64),
        "cand": np.array(logs["cand"], dtype=object),
        "label": np.array(logs["label"], dtype=object),
        "pred": np.array(logs["pred"], dtype=object),
        "sigma": np.array(logs.get("sigma", []), dtype=object),
        "archive_perm": np.array(eval_perms, dtype=np.int32),
        "archive_fx": np.array(eval_vals, dtype=np.float64),
    }

def run_algo_latent_2ways(
    dim: int,
    true_eval_fn,
    problem: str,
    dist_for_2opt: np.ndarray,
    args,
    surrogate_kind: str,
    seed: int,
) -> Dict[str, Any]:
    """Latent 2-ways (single trajectory)."""
    reset_seed(seed)

    from latent_2way import Latent2WaysConfig
    cfg = Latent2WaysConfig(
        K= args.latent_2ways_K,
        fit_per=args.latent_2ways_fit_per,
        surrogate_kind=surrogate_kind,
        gnn_epochs=args.latent_2ways_gnn_epochs,
        batch_size=args.latent_2ways_batch_size,
        acq=args.latent_2ways_acq,
        kappa=args.latent_2ways_kappa,
        problem=problem,
        min_dist_type =args.latent_2ways_min_dist_type,
        latent_min_dist_q = args.latent_2ways_latent_min_dist_q,
        incumbent_sigma=args.latent_2ways_incumbent_sigma,
        incumbent_pool_size=args.incumbent_pool_size,
        mu_plus_lambda=args.mu_plus_lambda,
    )

    best_perm, best_val, eval_perms, eval_vals, history_best, logs = latent_2ways(
        dim=dim,
        true_eval_fn=true_eval_fn,
        budget=args.max_evals,
        cfg=cfg,
        seed=seed,
    )

    return {
        "best_perm": np.array(best_perm, dtype=np.int32),
        "best_fx": np.float64(best_val),
        "history": np.array(history_best, dtype=np.float64),
        "archive_perm": np.array(eval_perms, dtype=np.int32),
        "archive_fx": np.array(eval_vals, dtype=np.float64),
        "eval_count": np.array(logs.get("eval_count", []), dtype=np.int64),
        "elapsed_time": np.array(logs.get("elapsed_time", []), dtype=np.float64),
        "step_time": np.array(logs.get("step_time", []), dtype=np.float64),
        "cand": np.array(logs.get("cand", []), dtype=object),
        "label": np.array(logs.get("label", []), dtype=object),
        "pred": np.array(logs.get("pred", []), dtype=object),
        "sigma": np.array(logs.get("sigma", []), dtype=object),
        "source": np.array(logs.get("source", []), dtype=object),
        "mu": np.array(logs.get("mu", []), dtype=object),
        "emb": np.array(logs.get("emb", []), dtype=object),
        "selected_mask": np.array(logs.get("selected_mask", []), dtype=object),
        "selected_order": np.array(logs.get("selected_order", []), dtype=object),
        "selected_eval_source": np.array(logs.get("selected_eval_source", []), dtype=object),
        "incumbent_internal": np.array(logs.get("incumbent_internal", []), dtype=object),
        "detail": np.array(logs.get("detail", []), dtype=object),
    }

# def run_algo_gpso()

def run_algo_fatrls_gnn(
    dim: int,
    true_eval_fn,
    problem: str,
    dist_for_2opt: np.ndarray,
    args,
    surrogate_kind: str,
    seed: int,
) -> Dict[str, Any]:
    """FAT-RLS + surrogate ranking (single-point true evaluation)."""
    reset_seed(seed)

    cfg = SurFATRLSConfig(
        cand_per_gen=args.sur_fat_cand_per_gen,
        evals_per_gen=1,
        warmup=args.sur_fat_warmup,
        retrain_every=args.sur_fat_retrain_every,
        acq=args.acq,
        kappa=args.kappa,
        xi=args.xi,
        move_type=args.sur_fat_move_type,
        tabu_mode=args.sur_fat_tabu_mode,
        local_train_mode=args.sur_fat_local_mode,
        local_train_k=args.sur_fat_local_k,
        log_all_labels=args.sur_fat_log_all_labels,
        surrogate_kind=surrogate_kind,
        problem=problem,
        gnn_epochs=args.gnn_epochs,
        batch_size=args.batch_size,
    )

    best_perm, best_val, eval_perms, eval_vals, history_best, logs = fatrls_with_gnn_surrogate(
        dim=dim,
        true_eval_fn=true_eval_fn,
        budget=args.max_evals,
        cfg=cfg,
        seed=seed,
    )

    return {
        "best_perm": np.array(best_perm, dtype=np.int32),
        "best_fx": np.float64(best_val),
        "history": np.array(history_best, dtype=np.float64),
        "cand": np.array(logs.get("cand", []), dtype=object),
        "label": np.array(logs.get("label", []), dtype=object),
        "pred": np.array(logs.get("pred", []), dtype=object),
        "sigma": np.array(logs.get("sigma", []), dtype=object),
        "archive_perm": np.array(eval_perms, dtype=np.int32),
        "archive_fx": np.array(eval_vals, dtype=np.float64),
    }

def run_algo_cego(
    dim: int,
    true_eval_fn,
    problem: str,
    dist_for_2opt,
    args,
    surrogate_kind: str,
    seed: int,
):
    """
    CEGO runner (minimization).
    Returns dict with the same schema as fatrls/gbdtma runners.
    """
    import numpy as np
    from cego import cego, CEGOConfig

    # CEGO縺ｯ蜀・Κ縺ｧKriging+EI+EA繧貞屓縺吶・縺ｧ surrogate_kind 縺ｯ蝓ｺ譛ｬ辟｡隕悶〒OK
    cfg = CEGOConfig(
        # initial design size (true evals)
        eval_init=getattr(args, "cego_eval_init", 20),

        # distance: "interchange" | "hamming" | "kendall"
        distance=getattr(args, "cego_distance", "interchange"),

        # kriging
        nugget=getattr(args, "cego_nugget", 1e-10),
        log_theta_min=getattr(args, "cego_log_theta_min", -10.0),
        log_theta_max=getattr(args, "cego_log_theta_max", 10.0),

        # duplicate fallback / max-min design
        creation_retries=getattr(args, "cego_creation_retries", 100),

        # EA (EI optimizer)
        ea_popsize=getattr(args, "cego_ea_popsize", 50),
        ea_generations=getattr(args, "cego_ea_generations", 200),
        ea_budget=getattr(args, "cego_ea_budget", 1000),
        ea_tournament_size=getattr(args, "cego_ea_tourn_size", 2),
        ea_tournament_p=getattr(args, "cego_ea_tourn_p", 0.9),
        ea_recomb_rate=getattr(args, "cego_ea_recomb_rate", 1.0),
        ea_mut_rate=getattr(args, "cego_ea_mut_rate", 1.0),
    )

    best_perm, best_fx, eval_perms, eval_vals, history_best, logs = cego(
        dim=dim,
        true_eval_fn=true_eval_fn,
        budget=args.max_evals,
        cfg=cfg,
        seed=seed,
    )

    # logs: cand/pred/label 縺ｯ荳紋ｻ｣・・uter iteration・峨＃縺ｨ縺ｫ object 驟榊・縺ｧ菫晏ｭ・
    return {
        "best_perm": np.array(best_perm, dtype=np.int32),
        "best_fx": np.float64(best_fx),
        "history": np.array(history_best, dtype=np.float64),
        "archive_perm": np.array(eval_perms, dtype=np.int32),
        "archive_fx": np.array(eval_vals, dtype=np.float64),

        # unified logs
        "cand": np.array(logs.get("cand", []), dtype=object),
        "pred": np.array(logs.get("pred", []), dtype=object),
        "label": np.array(logs.get("label", []), dtype=object),

        # optional extras (縺ゅｌ縺ｰ菫晏ｭ・
        "ei": np.array(logs.get("ei", []), dtype=object) if "ei" in logs else np.array([], dtype=object),
        "theta": np.array(logs.get("theta", []), dtype=np.float64) if "theta" in logs else np.array([], dtype=np.float64),
        "picked": np.array(logs.get("picked", []), dtype=object) if "picked" in logs else np.array([], dtype=object),
    }

# 縺薙％縺ｫ莉雁ｾ・gbdtma / UMM 繧定ｶｳ縺励※縺・￥
ALGO_RUNNERS = {
    "bo": run_algo_bo,
    "fatrls_gnn": run_algo_fatrls_gnn,
    "fatrls": run_algo_fatrls,
    "umm": run_algo_umm,
    "rflos": run_algo_rflos,
    "gbdtma": run_algo_gbdtma,
    "cego": run_algo_cego,
    "latent_2ways": run_algo_latent_2ways
}

def main():
    parser = argparse.ArgumentParser(description="Permutation optimization experiments")

    # 蝠城｡瑚ｨｭ螳・
    # parser.add_argument("--ins_name", type=str, default="ATSP-D30", help="Instance name (TSPLIB/QAPLIB/LOLIB/PFSP)")
    parser.add_argument("--ins_name", type=str, default=["N-pal27"], help="Instance name (TSPLIB/QAPLIB/LOLIB/PFSP)",choices=["burma14","ulysses22","fri26","bayg29","swiss42","att48","berlin52", "br17","ftv33","ftv35","ftv38","p43","ry48p","ft53", "N-pal11","N-pal19","N-pal23","N-pal27","N-econ36","N-p40-01","N-be75np",  "N-pal19","N-N-pal27","N-econ36","N-p44-01"])
    parser.add_argument("--prob_name", type=str, default="LOP", choices=["QAP", "TSP", "LOP", "PFSP", "ATSP"], help="Problem name")

    # 螳滄ｨ楢ｨｭ螳・
    parser.add_argument("--algo", type=str, default="latent_2ways", choices=list(ALGO_RUNNERS.keys()), help="which algorithm to run")
    parser.add_argument("--trials", type=int, default=30, help="number of trials")
    parser.add_argument("--save_path",
    default="E:\\2325_yamaguchi\\_results_TEVC_20260709\\", help="output root folder")
    parser.add_argument("--seed0", type=int, default=0, help="initial seed; trial t uses seed0+t")

    # Surrogate BO
    parser.add_argument("--batch_size", type=int, default=128, help="(GNN) batch size")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rf"],
        help="model names (for algo=bo only); each will be run separately",
    )
    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--n_init_true", type=int, default=100, help="initial true-evaluated samples")
    parser.add_argument("--batch_true", type=int, default=5, help="#candidates to actually evaluate per iteration")
    parser.add_argument("--candidate_nums", type=int, default=100, help="#children per iteration")
    parser.add_argument("--int_gen", type=int, default=20, help="internal optimizer generations")
    parser.add_argument("--optimizer", type=str, default="ga", choices=["ga", "tabu"], help="inner optimizer")
    parser.add_argument("--kappa", type=float, default=2.0, help="UCB/LCB parameter")
    parser.add_argument("--acq", type=str, default="mean", choices=["lcb", "ucb", "ei", "pi", "mean"], help="acquisition function (minimization)")
    parser.add_argument("--xi", type=float, default=0.0, help="exploration parameter for EI/PI")
    parser.add_argument("--bo_parent_selection", type=str, default="random",
                        choices=["random", "fitness"],
                        help="bo: GA parent selection (random like GBDTMA, or fitness by true eval)")
    parser.add_argument("--bo_elite_rate", type=float, default=0.2,
                        help="bo: elite rate for pop update (like GBDTMA)")
    parser.add_argument("--max_queries", type=int, default=30, help="kept for compatibility")
    parser.add_argument("--max_evals", type=int, default=600, help="true evaluation budget")
    parser.add_argument("--fatrls_init_count", type=int, default=100,
                        help="FAT-RLS: number of random initial true evaluations before scheduled moves")
    parser.add_argument("--umm_m_ini", type=int, default=100, help="UMM: initial sample size")
    parser.add_argument("--umm_ratio_samples_learn", type=float, default=0.25, help="UMM: ratio of samples for rho fitting")
    parser.add_argument("--umm_weight_mass_learn", type=float, default=0.9, help="UMM: target cumulative mass for rho fitting")
    parser.add_argument("--rflos_k", type=int, default=5, help="RFLoS: number of selected candidates per generation")
    parser.add_argument("--rflos_wmax", type=int, default=10, help="RFLoS: local walk length")
    parser.add_argument("--rflos_elr", type=float, default=0.2, help="RFLoS: elite rate for population update")
    parser.add_argument("--rflos_pop_size", type=int, default=100, help="RFLoS: population size")

    parser.add_argument("--latent_2ways_surrogate", nargs="+", default=["sage"],
                        choices=["gbdt","rf","gcn","gat","sage","gin","jkgcn","permformer","rand","nn","oracle"],
                        help="latent_2ways: surrogate type")
    parser.add_argument("--latent_2ways_gnn_epochs", type=int, default=300, help="latent_2ways: GNN epochs")
    parser.add_argument("--latent_2ways_batch_size", type=int, default=128, help="latent_2ways: GNN batch size")
    parser.add_argument("--latent_2ways_acq", type=str, default="mean", choices=["mean","lcb"],
                        help="latent_2ways: scoring method (mean or lcb)")
    parser.add_argument("--latent_2ways_kappa", type=float, default=2.0, help="latent_2ways: LCB kappa")
    parser.add_argument("--latent_2ways_min_dist_type", type=str, default="knn_density", choices=["candidate", "pop", "min_both","knn_density","knn_promisingFar"],
                        help="latent_2ways: minimum distance type")
    parser.add_argument("--latent_2ways_K", type = int, default = 1, help = "the total number K to evaluate in each generation")
    parser.add_argument("--latent_2ways_fit_per", type = int, default = 5, help = "the fit rate per FE")
    parser.add_argument("--latent_2ways_latent_min_dist_q", type = float, default = 0.2, help = "the quantile for latent space distance to select the minimum distance")
    parser.add_argument("--latent_2ways_incumbent_sigma", type=int, default=10,
                        help="latent_2ways: insert parent-child refinement rounds for incumbent_k; 1 keeps previous behavior")
    parser.add_argument("--incumbent_pool_size", type=int, default=50,
                        help="latent_2ways: number of top candidates to keep in the incumbent pool")
    parser.add_argument("--mu_plus_lambda", type=bool, default=False, help="whether EA will be mu+lambda or not")

    parser.add_argument("--gnn_epochs", type=int, default=300, help="GNN epochs (if used)")
    parser.add_argument("--gbdtma_surrogate", nargs="+", default=["rf","gbdt"],
                        choices=["gbdt","rf","gcn","gat","sage","gin","jkgcn","permformer","rand","nn","oracle"],
                        help="gbdtma: surrogate type")
    parser.add_argument("--gbdtma_gnn_epochs", type=int, default=300, help="gbdtma: GNN epochs")
    parser.add_argument("--gbdtma_batch_size", type=int, default=128, help="gbdtma: GNN batch size")
    parser.add_argument("--gbdtma_acq", type=str, default="mean", choices=["mean","lcb"],
                        help="gbdtma: scoring method (mean or lcb)")
    parser.add_argument("--gbdtma_kappa", type=float, default=2.0, help="gbdtma: LCB kappa")
    parser.add_argument("--sur_fat_cand_per_gen", type=int, default=100, help="fatrls_gnn: surrogate-scored candidates per generation")
    parser.add_argument("--sur_fat_warmup", type=int, default=20, help="fatrls_gnn: true eval warmup before surrogate")
    parser.add_argument("--sur_fat_retrain_every", type=int, default=10, help="fatrls_gnn: surrogate retrain interval (generation)")
    parser.add_argument("--sur_fat_move_type", type=str, default="insertion", choices=["insertion", "swap", "invert"],
                        help="fatrls_gnn: neighborhood move type")
    parser.add_argument("--sur_fat_tabu_mode", type=str, default="move", choices=["item", "move"],
                        help="fatrls_gnn: tabu mode")
    parser.add_argument("--sur_fat_local_mode", type=str, default="all",
                        choices=["all", "edge", "kendall", "hamming"],
                        help="fatrls_gnn: surrogate training data selection")
    parser.add_argument("--sur_fat_local_k", type=int, default=200,
                        help="fatrls_gnn: #points used when local mode != all")
    parser.add_argument("--sur_fat_log_all_labels", action="store_true",
                        help="fatrls_gnn: log true labels for all candidates (no effect on search)")

    # --- CEGO options ---
    parser.add_argument("--cego_eval_init", type=int, default=20)
    parser.add_argument("--cego_distance", type=str, default="interchange",
                        choices=["interchange", "hamming", "kendall"])

    parser.add_argument("--cego_nugget", type=float, default=1e-10)
    parser.add_argument("--cego_log_theta_min", type=float, default=-10.0)
    parser.add_argument("--cego_log_theta_max", type=float, default=10.0)

    parser.add_argument("--cego_creation_retries", type=int, default=100)

    parser.add_argument("--cego_ea_popsize", type=int, default=50)
    parser.add_argument("--cego_ea_generations", type=int, default=200)
    parser.add_argument("--cego_ea_budget", type=int, default=1000)
    parser.add_argument("--cego_ea_tourn_size", type=int, default=2)
    parser.add_argument("--cego_ea_tourn_p", type=float, default=0.9)
    parser.add_argument("--cego_ea_recomb_rate", type=float, default=1.0)
    parser.add_argument("--cego_ea_mut_rate", type=float, default=1.0)
    # parser.add_argument("--use_2opt", action="store_true", help="TSP only: enable 2-opt local search")

    args = parser.parse_args()

    for ins_name in args.ins_name:
        # ===== 蝠城｡後う繝ｳ繧ｹ繧ｿ繝ｳ繧ｹ縺ｮ隱ｭ縺ｿ霎ｼ縺ｿ =====
        if args.prob_name == "TSP":
            dist, dim = tsp_get_instance.get_tsp_dat(ins_name)
            true_eval_fn = tsp_eval_factory(dist)
            dist_for_2opt = dist
            problem = "TSP"
        elif args.prob_name == "QAP":
            F, D, dim = qap_get_instance.get_dat(ins_name)
            true_eval_fn = qap_eval_factory(F, D)
            dist_for_2opt = None
            problem = "QAP"
        elif args.prob_name == "LOP":
            W, dim = lo_get_instance.get_lolib_dat(ins_name)
            true_eval_fn = lo_eval_factory(W)
            dist_for_2opt = None
            problem = "LOP"
        elif args.prob_name == "PFSP":
            p_times, n_jobs, n_machines = pfsp_get_instance.get_pfsp_dat(ins_name)
            dim = n_jobs
            true_eval_fn = pfsp_eval_factory(p_times)
            dist_for_2opt = None
            problem = "PFSP"
        elif args.prob_name == "ATSP":
            dist, dim = atsp_get_instance.get_atsp_dat(ins_name)
            true_eval_fn = tsp_eval_factory(dist)
            dist_for_2opt = dist
            problem = "ATSP"
        else:
            raise ValueError(f"Unknown problem name: {args.prob_name}")

        root = Path(args.save_path)
        root.mkdir(parents=True, exist_ok=True)

        runner = ALGO_RUNNERS[args.algo]

        # ===== Surrogate BO: models 縺ｮ謨ｰ縺縺大屓縺・=====
        if args.algo == "bo":
            for surrogate_kind in args.models:
                out_root = root / ins_name / f"bo_{surrogate_kind}_{args.acq}_{args.optimizer}_{args.int_gen}_{args.batch_true}_{args.candidate_nums}"
                out_root.mkdir(parents=True, exist_ok=True)

                meta = {
                    "instance": ins_name,
                    "problem": args.prob_name,
                    "dim": int(dim),
                    "algo": "bo",
                    "surrogate": surrogate_kind,
                    "pop_size": args.pop_size,
                    "n_init_true": args.n_init_true,
                    "batch_true": args.batch_true,
                    "candidate_nums": args.candidate_nums,
                    "int_gen": args.int_gen,
                    "optimizer": args.optimizer,
                    "kappa": args.kappa,
                    "acq": args.acq,
                    "xi": args.xi,
                    "bo_parent_selection": args.bo_parent_selection,
                    "bo_elite_rate": args.bo_elite_rate,
                    "max_queries": args.max_queries,
                    "max_evals": args.max_evals,
                    "gnn_epochs": args.gnn_epochs,
                    "seed0": args.seed0,
                }
                with open(out_root / "meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                print(f"[MODE=bo] surrogate={surrogate_kind}")

                # for t in range(args.trials):
                for t in range(16,23):
                    seed = args.seed0 + t
                    trial_dir = out_root / f"trial_{t:02d}"
                    trial_dir.mkdir(parents=True, exist_ok=True)

                    print(f"  [Trial {t}/{args.trials}] seed={seed}")

                    result = runner(
                        dim=dim,
                        true_eval_fn=true_eval_fn,
                        problem=problem,
                        dist_for_2opt=dist_for_2opt,
                        args=args,
                        surrogate_kind=surrogate_kind,
                        seed=seed,
                    )

                    np.savez_compressed(trial_dir / "result.npz", **result)

                    best_fx = float(result["best_fx"])
                    best_perm = result["best_perm"].tolist()

                    with open(trial_dir / "result.txt", "w", encoding="utf-8") as f:
                        f.write(f"best_fx: {best_fx}\n")
                        f.write("best_perm: " + " ".join(map(str, best_perm)) + "\n")

                    print(f"    -> best_fx={best_fx:.6f}")

                print(f"[DONE] Surrogate={surrogate_kind} results saved in {out_root}")

        # ===== FAT-RLS + Surrogate: models 縺ｮ謨ｰ縺縺大屓縺・=====
        elif args.algo == "fatrls_gnn":
            for surrogate_kind in args.models:
                out_root = root / ins_name / (
                    f"fatrls_gnn_{surrogate_kind}_inv_{args.acq}_cand{args.sur_fat_cand_per_gen}"
                    f"_warm{args.sur_fat_warmup}_rt{args.sur_fat_retrain_every}"
                )
                out_root.mkdir(parents=True, exist_ok=True)

                meta = {
                    "instance": ins_name,
                    "problem": args.prob_name,
                    "dim": int(dim),
                    "algo": "fatrls_gnn",
                    "surrogate": surrogate_kind,
                    "sur_fat_cand_per_gen": args.sur_fat_cand_per_gen,
                    "sur_fat_warmup": args.sur_fat_warmup,
                    "sur_fat_retrain_every": args.sur_fat_retrain_every,
                    "sur_fat_move_type": args.sur_fat_move_type,
                    "sur_fat_tabu_mode": args.sur_fat_tabu_mode,
                    "sur_fat_local_mode": args.sur_fat_local_mode,
                    "sur_fat_local_k": args.sur_fat_local_k,
                    "sur_fat_log_all_labels": bool(args.sur_fat_log_all_labels),
                    "acq": args.acq,
                    "kappa": args.kappa,
                    "xi": args.xi,
                    "max_evals": args.max_evals,
                    "gnn_epochs": args.gnn_epochs,
                    "seed0": args.seed0,
                }
                with open(out_root / "meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                print(f"[MODE=fatrls_gnn] surrogate={surrogate_kind}")

                # for t in range(args.trials):
                for t in range(0,3):
                    seed = args.seed0 + t
                    trial_dir = out_root / f"trial_{t:02d}"
                    trial_dir.mkdir(parents=True, exist_ok=True)

                    print(f"  [Trial {t}/{args.trials}] seed={seed}")

                    result = runner(
                        dim=dim,
                        true_eval_fn=true_eval_fn,
                        problem=problem,
                        dist_for_2opt=dist_for_2opt,
                        args=args,
                        surrogate_kind=surrogate_kind,
                        seed=seed,
                    )

                    np.savez_compressed(trial_dir / "result.npz", **result)

                    best_fx = float(result["best_fx"])
                    best_perm = result["best_perm"].tolist()

                    with open(trial_dir / "result.txt", "w", encoding="utf-8") as f:
                        f.write(f"best_fx: {best_fx}\n")
                        f.write("best_perm: " + " ".join(map(str, best_perm)) + "\n")

                    print(f"    -> best_fx={best_fx:.6f}")

                print(f"[DONE] Surrogate={surrogate_kind} results saved in {out_root}")

        # ===== GBDTMA: surrogate list =====
        elif args.algo == "gbdtma":
            for surrogate_kind in args.gbdtma_surrogate:
                out_root = root / ins_name / f"gbdtma_{surrogate_kind}_{args.gbdtma_acq}_fixed_fin"
                out_root.mkdir(parents=True, exist_ok=True)

                meta = {
                    "instance": ins_name,
                    "problem": args.prob_name,
                    "dim": int(dim),
                    "algo": "gbdtma",
                    "gbdtma_surrogate": surrogate_kind,
                    "gbdtma_gnn_epochs": args.gbdtma_gnn_epochs,
                    "gbdtma_batch_size": args.gbdtma_batch_size,
                    "gbdtma_acq": args.gbdtma_acq,
                    "gbdtma_kappa": args.gbdtma_kappa,
                    "gbdtma_problem": problem,
                    "max_evals": args.max_evals,
                    "seed0": args.seed0,
                }
                with open(out_root / "meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                print(f"[MODE=gbdtma] surrogate={surrogate_kind}")

                for t in range(12,15):
                # for t in range(args.trials):
                    seed = args.seed0 + t
                    trial_dir = out_root / f"trial_{t:02d}"
                    trial_dir.mkdir(parents=True, exist_ok=True)

                    print(f"  [Trial {t}/{args.trials}] seed={seed}")

                    result = runner(
                        dim=dim,
                        true_eval_fn=true_eval_fn,
                        problem=problem,
                        dist_for_2opt=dist_for_2opt,
                        args=args,
                        surrogate_kind=surrogate_kind,
                        seed=seed,
                    )

                    np.savez_compressed(trial_dir / "result.npz", **result)

                    best_fx = float(result["best_fx"])
                    best_perm = result["best_perm"].tolist()

                    with open(trial_dir / "result.txt", "w", encoding="utf-8") as f:
                        f.write(f"best_fx: {best_fx}\n")
                        f.write("best_perm: " + " ".join(map(str, best_perm)) + "\n")

                    print(f"    -> best_fx={best_fx:.6f}")

                print(f"[DONE] Surrogate={surrogate_kind} results saved in {out_root}")


        # ===== Latent2Ways: surrogate list =====
        elif args.algo == "latent_2ways":
            for surrogate_kind in args.latent_2ways_surrogate:
                # out_root = root / ins_name / f"lat2way_{surrogate_kind}_{args.latent_2ways_acq}_{args.latent_2ways_min_dist_type}_{args.latent_2ways_latent_min_dist_q}_{args.latent_2ways_K}_{args.latent_2ways_incumbent_sigma}_{args.incumbent_pool_size}"
                if args.mu_plus_lambda:
                    mu_plus_lambda_str = "mu-la"
                else:
                    mu_plus_lambda_str = "mu"
                out_root = root / ins_name / f"lat2way_{surrogate_kind}_{args.latent_2ways_acq}_{args.latent_2ways_min_dist_type}_{args.latent_2ways_K}_{mu_plus_lambda_str}_{args.latent_2ways_incumbent_sigma}_{args.incumbent_pool_size}_BB"
                out_root.mkdir(parents=True, exist_ok=True)

                meta = {
                    "instance": ins_name,
                    "problem": args.prob_name,
                    "dim": int(dim),
                    "algo": "latent_2ways",
                    "latent_2ways_surrogate": surrogate_kind,
                    "latent_2ways_gnn_epochs": args.latent_2ways_gnn_epochs,
                    "latent_2ways_batch_size": args.latent_2ways_batch_size,
                    "latent_2ways_acq": args.latent_2ways_acq,
                    "latent_2ways_kappa": args.latent_2ways_kappa,
                    "latent_2ways_problem": problem,
                    "max_evals": args.max_evals,
                    "seed0": args.seed0,
                    "mu_plus_lambda": args.mu_plus_lambda,
                }
                with open(out_root / "meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                print(f"[MODE=latent_2ways] surrogate={surrogate_kind}")
                print(f"[INSTANCE={ins_name}] problem={problem} dim={dim}")

                # for t in range(0,7):
                for t in range(args.trials):
                    seed = args.seed0 + t
                    trial_dir = out_root / f"trial_{t:02d}"
                    trial_dir.mkdir(parents=True, exist_ok=True)

                    print(f"  [Trial {t}/{args.trials}] seed={seed}")

                    result = runner(
                        dim=dim,
                        true_eval_fn=true_eval_fn,
                        problem=problem,
                        dist_for_2opt=dist_for_2opt,
                        args=args,
                        surrogate_kind=surrogate_kind,
                        seed=seed,
                    )

                    np.savez_compressed(trial_dir / "result.npz", **result)

                    best_fx = float(result["best_fx"])
                    best_perm = result["best_perm"].tolist()

                    with open(trial_dir / "result.txt", "w", encoding="utf-8") as f:
                        f.write(f"best_fx: {best_fx}\n")
                        f.write("best_perm: " + " ".join(map(str, best_perm)) + "\n")

                    print(f"    -> best_fx={best_fx:.6f}")

                print(f"[DONE] Surrogate={surrogate_kind} results saved in {out_root}")

        # # ===== RFLoS: surrogate list =====
        # elif args.algo == "gbdtma":
        #     for surrogate_kind in args.gbdtma_surrogate:
        #         out_root = root / ins_name / f"gbdtma_{surrogate_kind}_{args.gbdtma_acq}_fixed"
        #         out_root.mkdir(parents=True, exist_ok=True)

        #         meta = {
        #             "instance": ins_name,
        #             "problem": args.prob_name,
        #             "dim": int(dim),
        #             "algo": "gbdtma",
        #             "gbdtma_surrogate": surrogate_kind,
        #             "gbdtma_gnn_epochs": args.gbdtma_gnn_epochs,
        #             "gbdtma_batch_size": args.gbdtma_batch_size,
        #             "gbdtma_acq": args.gbdtma_acq,
        #             "gbdtma_kappa": args.gbdtma_kappa,
        #             "gbdtma_problem": problem,
        #             "max_evals": args.max_evals,
        #             "seed0": args.seed0,
        #         }
        #         with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        #             json.dump(meta, f, ensure_ascii=False, indent=2)

        #         print(f"[MODE=gbdtma] surrogate={surrogate_kind}")

        #         for t in range(20,30):
        #         # for t in range(args.trials):
        #             seed = args.seed0 + t
        #             trial_dir = out_root / f"trial_{t:02d}"
        #             trial_dir.mkdir(parents=True, exist_ok=True)

        #             print(f"  [Trial {t}/{args.trials}] seed={seed}")

        #             result = runner(
        #                 dim=dim,
        #                 true_eval_fn=true_eval_fn,
        #                 problem=problem,
        #                 dist_for_2opt=dist_for_2opt,
        #                 args=args,
        #                 surrogate_kind=surrogate_kind,
        #                 seed=seed,
        #             )

        #             np.savez_compressed(trial_dir / "result.npz", **result)

        #             best_fx = float(result["best_fx"])
        #             best_perm = result["best_perm"].tolist()

        #             with open(trial_dir / "result.txt", "w", encoding="utf-8") as f:
        #                 f.write(f"best_fx: {best_fx}\n")
        #                 f.write("best_perm: " + " ".join(map(str, best_perm)) + "\n")

        #             print(f"    -> best_fx={best_fx:.6f}")

        #         print(f"[DONE] Surrogate={surrogate_kind} results saved in {out_root}")

        # ===== FAT-RLS / UMM 縺ｪ縺ｩ: 1 algo = 1 險ｭ螳・=====
        else:
            algo_name = args.algo
            if algo_name == "fatrls" and args.fatrls_init_count != 1:
                out_root = root / ins_name / f"{algo_name}_init{args.fatrls_init_count}_fixed"
            else:
                out_root = root / ins_name / f"{algo_name}_fixed"
            out_root.mkdir(parents=True, exist_ok=True)

            meta = {
                "instance": ins_name,
                "problem": args.prob_name,
                "dim": int(dim),
                "algo": algo_name,
                "max_evals": args.max_evals,
                "fatrls_init_count": args.fatrls_init_count if algo_name == "fatrls" else None,
                "seed0": args.seed0,
            }
            with open(out_root / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            print(f"[MODE={algo_name}]")

            # for t in range(20,30):
            for t in range(args.trials):
                seed = args.seed0 + t
                trial_dir = out_root / f"trial_{t:02d}"
                trial_dir.mkdir(parents=True, exist_ok=True)

                print(f"  [Trial {t}/{args.trials}] seed={seed}")

                result = runner(
                    dim=dim,
                    true_eval_fn=true_eval_fn,
                    problem=problem,
                    dist_for_2opt=dist_for_2opt,
                    args=args,
                    surrogate_kind="-",  # FAT-RLS 遲峨〒縺ｯ辟｡隕・
                    seed=seed,
                )

                np.savez_compressed(trial_dir / "result.npz", **result)

                best_fx = float(result["best_fx"])
                best_perm = result["best_perm"].tolist()

                with open(trial_dir / "result.txt", "w", encoding="utf-8") as f:
                    f.write(f"best_fx: {best_fx}\n")
                    f.write("best_perm: " + " ".join(map(str, best_perm)) + "\n")

                print(f"    -> best_fx={best_fx:.6f}")

            print(f"[DONE] algo={algo_name} results saved in {out_root}")


if __name__ == "__main__":
    main()


