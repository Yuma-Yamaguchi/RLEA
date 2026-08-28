# surrogate_perm_bo.py
# Permutation Optimization with Surrogate (UCB) + Evolutionary Operators
# - Supports TSP (2-opt/insert/invert) and QAP (swap-heavy)
# - Surrogate: GNNRegressor ('gcn','gat','sage','gin','jkgcn','permformer') or 'rf'
# - Call SurrogatePermBO.run() to execute the loop

from __future__ import annotations
import random, numpy as np
from typing import Callable, List, Tuple, Optional, Literal

from fatrls import fat_rls_init

import torch
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau
from sklearn.pipeline import Pipeline

from trainer_predictor import GNNRegressor

from optimizer import GAOptimizer, TabuSearchOptimizer
from tqdm import tqdm
from math import sqrt, pi, exp

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return (1.0/np.sqrt(2*np.pi)) * np.exp(-0.5 * x * x)

def _norm_cdf(x: np.ndarray) -> np.ndarray:
    # 近似でもOKだが、scipyがあるなら from scipy.stats import norm; norm.cdf を使っても可
    # ここでは安定な近似でなく簡便な方法としてtanh近似でもよいが、実運用ならscipy推奨
    # 最小限の実装として、erf を使う:
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def acq_scores_minimization(
    mu: np.ndarray,
    sigma: np.ndarray,
    best_y: float,
    kind: str = "lcb",     # "lcb" | "ucb" | "ei" | "pi" | "mean"
    kappa: float = 2.0,
    xi: float = 0.0
) -> np.ndarray:
    """
    最小化問題向け。戻り値は「大きいほど良い」スコア（pick_topk_diverseに渡せる形）。
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sig = np.maximum(sigma, 1e-12)

    if kind == "mean":
        # そのままでは小さいほど良いので、符号反転して大きいほど良いに
        return -mu

    if kind == "lcb":
        # LCB = mu - kappa*sigma を最小化 → スコアは -(LCB)
        return -(mu - kappa * sig)

    if kind == "ucb":
        # UCB = mu + kappa*sigma を最小化 → スコアは -(UCB)
        return -(mu + kappa * sig)

    # EI / PI は「改善量 = best_y - Y」を使う（最小化）
    imp = (best_y - mu) - xi
    z = imp / sig

    if kind == "ei":
        # EI = imp * Phi(z) + sigma * phi(z)
        # sigma==0 のときEI=0にする
        from scipy.stats import norm
        Phi = norm.cdf(z)
        phi = norm.pdf(z)
        ei = imp * Phi + sig * phi
        ei[sig <= 1e-12] = 0.0
        return ei

    if kind == "pi":
        from scipy.stats import norm
        Phi = norm.cdf(z)
        pi = Phi
        pi[sig <= 1e-12] = 0.0
        return pi

    raise ValueError(f"unknown acquisition kind: {kind}")

def perm_to_edge_node_feat(
    perm: torch.Tensor,
    dim: int = 1,
    make_cycle: bool = True,     # True: 巡回（最後→最初 も結ぶ）/ False: パス
    undirected: bool = True,     # True: 双方向エッジ化
    every_connect: bool = False,  # True: 全ノード間を接続（完全グラフ）
    each_connect: bool = False,  # True: QAP用 番号を明示化
    top_k: int = 1,               # 上位K個のエッジを選択
    normalize_rank: bool = True  # True: 順位を [0,1] に正規化（dim==1 のとき）
):
    """
    perm: [L]  各要素は「都市ID（ノードID）」の順列（例: [3,0,2,1]）
          -> ノードは 0..L-1 を想定、perm は訪問順を表す
    dim : ノード特徴の次元
          dim==1 なら「訪問順位（正規化可）」の1D特徴
          dim>1  なら Transformer 風の sin/cos 位置埋め込みで dim 次元を出力
    戻り値:
      edge_index: [2, E] (Long)
      node_feature: [L, dim] (Float)  ※ ノード順は 0..L-1 のID順
    """
    device = perm.device
    L = perm.numel()
    assert L >= 2, "少なくとも2ノードが必要です"
    # ---- エッジ（順列の隣接） ----
    # 訪問順に並んだノード列
    seq = perm.long()  # [L]

    # 連続ペア (perm[i], perm[i+1]) を作成
    src = []
    dst = []
    if every_connect:
        # 完全グラフ化
        for i in range(top_k):          # 上からK個を source とする
            vi = perm[i].item()
            for j in range(i + 1, L):
                vj = perm[j].item()
                # src.append(vi) # standard version
                # dst.append(vj)
                src.append(vj) #reverse version
                dst.append(vi)
        src = torch.tensor(src, dtype=torch.long, device=perm.device)
        dst = torch.tensor(dst, dtype=torch.long, device=perm.device)
    elif each_connect:
        # QAP用 各ノードを明示的に接続
        src = range(L)
        src = torch.tensor(src, dtype=torch.long, device=perm.device)
        dst = seq
    else:
        src = seq[:-1]
        dst = seq[1:]

    if make_cycle:
        # 最後→最初も結んでサイクル化
        src = torch.cat([src, seq[-1:].clone()], dim=0)  # [..., last]
        dst = torch.cat([dst, seq[:1].clone()],  dim=0)  # [..., first]

    edge_index = torch.stack([src, dst], dim=0)  # [2, E]

    if undirected:
        # 双方向化
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    edge_index = edge_index.to(device)

    # ---- ノード特徴（訪問順位の埋め込み） ----
    # 各ノードID v の訪問順位 rank[v] を求める
    # 例: perm=[3,0,2,1] -> rank[3]=0, rank[0]=1, rank[2]=2, rank[1]=3
    inv = torch.empty(L, dtype=torch.long, device=device)
    inv[seq] = torch.arange(L, device=device)  # inv[v] = そのノードの訪問順位(0..L-1)

    if dim == 1:
        node_feature = torch.zeros(L, 1, device=device)  # 何も入れない/フラット
    else:
        node_feature = torch.zeros(L, L, device=device)
        for i in range(L):
            node_feature[i, i] = 1.0

    node_feature = node_feature.to(torch.float32)

    return edge_index, node_feature

# =========================================================
# サロゲート（GNN or ランダムフォレスト）
# =========================================================

class SurrogateWrapper:
    """(mu, sigma) を返す Surrogate 統一インタフェース"""
    def __init__(
        self,
        kind: Literal['rf','gbdt','gcn','gat','sage','gin','jkgcn','permformer','rand','oracle'],
        dim: int,
        batch_size: int = 256,
        epochs: int = 200,
        make_cycle: bool = True,
        undirected: bool = True,
        every_connect: bool = False,
        each_connect: bool = False,
        top_k: int = 1,
        device: Optional[torch.device] = None,
        seed: int = 0,
        true_eval_fn: Optional[Callable[[List[int]], float]] = None
    ):
        self.kind = kind
        self.dim = dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.make_cycle = make_cycle
        self.undirected = undirected
        self.every_connect = every_connect
        self.each_connect = each_connect
        self.top_k = top_k
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if kind == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=seed, n_jobs=-1)
        elif kind == 'gbdt':
            self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=seed)
        elif kind == 'rand':
            self.model = None
            self._rng = np.random.default_rng(seed)  # 乱数用
        elif kind == 'oracle':
            self.model = true_eval_fn
        elif kind == 'nn':
            # self.model = MLPRegressor(
            #     hidden_layer_sizes=(self.dim, int(self.dim/2), int(self.dim/3)),  # 3層、各層のユニット数
            #     activation="relu",
            #     solver="adam",
            #     max_iter=500,
            #     random_state=seed
            # )
            self.model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("mlp", MLPRegressor(
                        hidden_layer_sizes=(self.dim, int(self.dim/2), int(self.dim/4)),  # 3層、各層のユニット数
                        activation="relu",
                        solver="adam",
                        max_iter=1000,
                        random_state=seed
                    ))
                ])
        else:
            self.model = GNNRegressor(kind, epochs=epochs, batch_size=batch_size, input_dim=dim, use_scaler=True,seed = seed)

    def _fe_gnn(self, perms: List[List[int]]):
        # edge_index_list, node_feature_list を作る
        e_list, x_list = [], []
        for p in perms:
            perm_t = torch.tensor(p, dtype=torch.long)
            ei, xf = perm_to_edge_node_feat(
                perm_t, dim=self.dim, make_cycle=self.make_cycle, undirected=self.undirected, every_connect=self.every_connect, each_connect=self.each_connect, top_k=self.top_k
            )
            e_list.append(ei)
            x_list.append(xf)
        return e_list, x_list

    def fit(self, perms: List[List[int]], y: List[float]):
        if self.kind == 'rf' or self.kind == 'gbdt' or self.kind =="nn":
            X = np.array(perms, dtype=np.int32)
            self.model.fit(X, np.asarray(y, dtype=np.float32))
        elif self.kind == 'rand' or self.kind == 'oracle':
            return  # 学習しない
        else:
            e_list, x_list = self._fe_gnn(perms)
            _, _, _ = self.model.fit(e_list, x_list, y)

    def predict(
        self,
        perms: List[List[int]],
        return_embedding: bool = False,
    ):
        if self.kind == 'rf':
            X = np.array(perms, dtype=np.int32)
            if hasattr(self.model, "estimators_") and isinstance(self.model.estimators_, list):
                preds = np.stack([est.predict(X) for est in self.model.estimators_], axis=0)
                mu = preds.mean(axis=0)
                sigma = preds.std(axis=0, ddof=1)
            else:
                mu = self.model.predict(X)
                sigma = np.full_like(mu, fill_value=np.std(mu))  # フォールバック
            sigma = np.maximum(sigma, 1e-9)
            result = (mu.astype(np.float32), sigma.astype(np.float32))
            return (*result, None) if return_embedding else result
        elif self.kind == 'gbdt':
            X = np.array(perms, dtype=np.int32)
            mu = self.model.predict(X)
            sigma = np.full_like(mu, fill_value=1e-6, dtype=np.float32)
            result = (mu.astype(np.float32), sigma.astype(np.float32))
            return (*result, None) if return_embedding else result
        elif self.kind == 'nn': ##sigmaは適当 もう使わない
            X = np.array(perms, dtype=np.float32)
            mu = self.model.predict(X)  # shape: (n_samples,)
            sigma = np.full_like(mu, fill_value=1e-6, dtype=np.float32)
            result = (mu.astype(np.float32), sigma.astype(np.float32))
            return (*result, None) if return_embedding else result
        elif self.kind == 'rand':
            n = len(perms)
            # ランダムにスコアリングさせるための μ/σ
            # どの獲得関数でもランダム化されるよう、両方に乱数を入れておく
            mu = self._rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float32)
            sigma = self._rng.uniform(low=0.1, high=1.0, size=n).astype(np.float32)
            result = (mu, sigma)
            return (*result, None) if return_embedding else result
        elif self.kind == 'oracle':
            mu = np.array([self.model(p) for p in perms], dtype=np.float32)
            sigma = np.full_like(mu, fill_value=1e-6, dtype=np.float32)
            result = (mu, sigma)
            return (*result, None) if return_embedding else result
        else:
            e_list, x_list = self._fe_gnn(perms)
            # mu, sigma, _, _ = self.model.pred_UCB(e_list, x_list, mc_dropout=False, T=30, kappa=2.0)
            embeddings = []
            hook = None
            if return_embedding:
                hook = self.model.nas_agent.head.register_forward_pre_hook(
                    lambda _module, inputs: embeddings.append(inputs[0].detach().cpu())
                )
            try:
                pred = self.model.pred(e_list, x_list)
            finally:
                if hook is not None:
                    hook.remove()

            mu = pred.cpu().numpy().astype(np.float32)
            sigma = np.full_like(mu, fill_value=1e-6, dtype=np.float32)
            if return_embedding:
                embedding = torch.cat(embeddings, dim=0).numpy().astype(np.float32)
                return mu, sigma, embedding
            return mu, sigma
            # return mu.cpu().numpy().astype(np.float32), sigma.cpu().numpy().astype(np.float32)

# =========================================================
# メインループ本体
# =========================================================

class SurrogatePermBO:
    """
    順列最適化の統括クラス
    - 初期ランダム -> 真評価 -> サロゲート学習 -> 進化で子生成 -> サロゲート評価 -> TopK 真評価 -> 繰り返し
    """
    def __init__(
        self,
        dim: int,
        true_eval_fn: Callable[[List[int]], float],
        problem: Literal['TSP','QAP','LOP','PFSP', 'ATSP'] = 'TSP',
        surrogate_kind: Literal['rf','gbdt','gcn','gat','sage','gin','jkgcn','permformer','rand','oracle'] = 'gin',
        pop_size: int = 100,
        n_init_true: int = 100,
        batch_true: int = 20,
        candidate_nums: int = 400,
        int_gen: int = 20,
		optimizer: str = 'ga',
        kappa: float = 2.0,
        xi = 0.0,
        acq: str = 'lcb',
        max_queries: int = 150,
        max_eval: int = 1000,
        gnn_epochs: int = 200,
        seed: int = 0,
        parent_selection: Literal['fitness','random'] = 'random',
        elite_rate: float = 0.2,
        logger=None,
    ):
        self.dim = dim
        self.true_eval_fn = true_eval_fn
        self.problem = problem
        self.pop_size = pop_size
        self.n_init_true = n_init_true
        self.batch_true = batch_true
        self.candidate_nums = candidate_nums
        self.optimizer = optimizer
        self.int_gen = int_gen
        self.kappa = kappa
        self.xi = xi
        self.acq = acq
        self.max_queries = max_queries
        self.max_eval = max_eval
        self.logger = logger
        self.seed = seed
        self.parent_selection = parent_selection
        self.elite_rate = elite_rate

        if problem == 'TSP':
            make_cycle, undirected = True, True
            every_connect, top_k = False, dim
            each_connect = False
        elif problem == 'ATSP':
            make_cycle, undirected = True, False
            every_connect, top_k = False, dim
            each_connect = False
        elif problem == 'QAP':
            make_cycle, undirected = False, False
            every_connect, top_k = False, dim
            each_connect = True
        elif problem == 'LOP':
            make_cycle, undirected = False, False
            every_connect, top_k = True, dim  # 完全グラフ化
            each_connect = False
        elif problem == 'PFSP':
            make_cycle, undirected = False, False
            every_connect, top_k = True, dim
            each_connect = False
        else:
            raise ValueError("problem は 'TSP' か 'QAP' か 'LOP' か 'PFSP' を指定してください")

        self.surrogate = SurrogateWrapper(
            kind=surrogate_kind, dim=dim, epochs=gnn_epochs, make_cycle=make_cycle, undirected=undirected, every_connect=every_connect, each_connect=each_connect, top_k=top_k, seed=seed
        )

        # set_seed(seed)

        self.archive_perm: List[List[int]] = []
        self.archive_fx:   List[float] = []

    def run(
        self
    ) -> Tuple[List[int], float, List[float]]:
        """
        戻り値: (best_perm, best_fx, history_best_fx)
        """
        # 1) 初期ランダム → 真評価
        for _ in range(self.n_init_true):
            p = list(range(self.dim))
            random.shuffle(p)
            # 初期個体段階での 2-opt は任意（黒箱なら None で通る）
            # if self.problem == 'TSP' and dist_mtx_for_2opt is not None:
            #     p = local_search_2opt(p, dist_mtx_for_2opt, max_moves=32)
            f = self.true_eval_fn(p)
            self.archive_perm.append(p)
            self.archive_fx.append(f)

        # perm, ev = fat_rls_init(
        # dim=self.dim,
        # true_eval_fn=self.true_eval_fn,
        # budget=self.max_eval,
        # move_type="invert",
        # seed=self.seed,
        # init=self.n_init_true,
        # tabu_mode="move",
        # )
        # self.archive_perm.extend(perm)
        # self.archive_fx.extend(ev)

        # return self.archive_perm

        history_best = [float(np.min(self.archive_fx))]
        cand = []
        label = []
        pred = []
        sig = []

        # 2) サロゲート初期学習
        self.surrogate.fit(self.archive_perm, self.archive_fx)

        # 3) 母集団（真評価上位）
        idx = np.argsort(self.archive_fx)
        elite_num = max(1, min(self.pop_size, int(np.floor(self.elite_rate * self.pop_size))))
        elite_idx = idx[:elite_num]
        rest_idx = idx[elite_num:]
        remain = self.pop_size - elite_num
        if remain <= 0:
            sel = elite_idx
        else:
            if rest_idx.size == 0:
                other = np.random.choice(elite_idx, size=remain, replace=True)
            else:
                replace = rest_idx.size < remain
                other = np.random.choice(rest_idx, size=remain, replace=replace)
            sel = np.concatenate([elite_idx, other])
        pop = [self.archive_perm[i][:] for i in sel]

        # 3.5) Optimizer 構築（TSPなら距離があれば2-optをON）
        # local_search = '2opt' if (self.problem == 'TSP' and dist_mtx_for_2opt is not None) else 'none'
        # 交叉と変異の初期設定（必要なら外から渡す形にもできます）
        if self.problem == 'TSP':
            crossover = 'pmx'
            # mutation_weights = {'swap':0.2,'insert':0.4,'invert':0.4}
            # mutation_weights = {'invert':0.6, 'insert':0.3, 'double_insert':0.1}
            mutation_weights = {'invert':1.0}
        elif self.problem == 'ATSP':
            crossover = 'pmx'
            mutation_weights = {'swap':1.0}
        elif self.problem == 'QAP':
            crossover = 'pmx'
            mutation_weights = {'swap':1.0}
        elif self.problem == 'LOP':
            crossover = 'pmx'
            # mutation_weights = {'swap':0.2,'insert':0.4,'invert':0.4}
            mutation_weights = {'swap':1.0}
            # mutation_weights = {'swap':0.5,'invert':0.5}
        elif self.problem == 'PFSP':
            crossover = 'pmx'
            # mutation_weights = {'swap':0.7,'insert':0.2,'invert':0.1}
            mutation_weights = {'swap':1.0}
        else:
            raise ValueError("problem は 'TSP' か 'QAP' か 'LOP' か 'PFSP' を指定してください")

        if self.optimizer == 'ga':
            sel_eval_fn = self.true_eval_fn if self.parent_selection == 'fitness' else None
            optimizer = GAOptimizer(
                dim=self.dim,
                problem=self.problem,
                true_eval_fn=sel_eval_fn,   # 親選択のトーナメントで利用（randomの場合はNone）
                crossover=crossover,
                mutation_weights=mutation_weights,
                selection_k=3,
                rng_seed=self.seed,
            )
        elif self.optimizer == 'tabu':
            optimizer = TabuSearchOptimizer(
                dim=self.dim,
                problem=self.problem,
                true_eval_fn=self.true_eval_fn,
                schedule_budget= self.max_eval,
                tabu_k= 10,
                rng_seed=self.seed,
            )
        else:
            raise ValueError(f"unknown optimizer: {self.optimizer}")

        # 4) 反復
        total_queries = len(self.archive_perm)
        # while total_queries < self.max_queries:

        with tqdm(total=self.max_eval, desc="True evaluations") as pbar:
            pbar.update((total_queries))
            while total_queries < self.max_eval:
                # 4-1) 子候補の生成
                children = optimizer.generate_children(
                    pop=pop,
                    num_children=self.candidate_nums,
                )

                # 4-2) サロゲート評価
                mu, sigma = self.surrogate.predict(children)

                cand.append(children)
                label.append([self.true_eval_fn(c) for c in children])
                pred.append(mu)
                sig.append(sigma)

                # ★ 最小化用の獲得関数：LCB = mu - kappa*sigma を最小化したい
                # pick_topk_diverse は「大きいほどよい」スコアを前提にしているため、
                # scores = -(LCB) を渡して“大きい順”に選べば LCB が小さい（＝良い）候補が選ばれる
                scores = acq_scores_minimization(
                    mu, sigma,
                    best_y=float(np.min(self.archive_fx)),
                    kind=self.acq,
                    kappa=self.kappa,
                    xi=self.xi,
                )

                children, scores = self.internal_optimizer(
                    optimizer, children, scores.tolist(), self.surrogate, gen_num= self.int_gen
                )
                # scores = -(mu - self.kappa * sigma)

                # 4-3) UCB/LCB 上位 + 多様性で b 個選抜 → 真評価
                chosen = optimizer.pick_topk_diverse(children, scores, self.batch_true)
                yB = [self.true_eval_fn(c) for c in chosen]
                self.archive_perm.extend(chosen)
                self.archive_fx.extend(yB)

                pbar.set_postfix(best=float(np.min(self.archive_fx)))
                pbar.update(len(chosen))

                total_queries = len(self.archive_perm)

                # 4-4) サロゲート再学習
                self.surrogate.fit(self.archive_perm, self.archive_fx)

                # 4-5) 次世代母集団更新（真評価に基づくエリート）
                idx = np.argsort(self.archive_fx)
                elite_num = max(1, min(self.pop_size, int(np.floor(self.elite_rate * self.pop_size))))
                elite_idx = idx[:elite_num]
                rest_idx = idx[elite_num:]
                remain = self.pop_size - elite_num
                if remain <= 0:
                    sel = elite_idx
                else:
                    if rest_idx.size == 0:
                        other = np.random.choice(elite_idx, size=remain, replace=True)
                    else:
                        replace = rest_idx.size < remain
                        other = np.random.choice(rest_idx, size=remain, replace=replace)
                    sel = np.concatenate([elite_idx, other])
                pop = [self.archive_perm[i][:] for i in sel]

                best_now = float(self.archive_fx[idx[0]])
                history_best.append(best_now)
                if self.logger:
                    self.logger.info(f"[queries={total_queries}] best={best_now:.6f}")

        best_idx = int(np.argmin(self.archive_fx))
        return self.archive_perm[best_idx], self.archive_fx[best_idx], history_best, cand, label, pred, sig, self.archive_perm, self.archive_fx

    def internal_optimizer(self, optimizer, parent_perm: List[int], parent_score: List[float], surrogate_model, gen_num: int) -> None:
        for _ in range(self.int_gen):
            children = optimizer.generate_int_children(
                pop=parent_perm,
            )
            children_mu, children_sigma = surrogate_model.predict(children)
            child_scores = acq_scores_minimization(
                children_mu, children_sigma,
                best_y=float(np.min(self.archive_fx)),
                kind=self.acq,
                kappa=self.kappa,
                xi=self.xi
            )

            for i, child in enumerate(children):
                if child_scores[i] > parent_score[i]:
                    parent_perm[i] = child
                    parent_score[i] = child_scores[i]

        return parent_perm, parent_score

# =========================================================
# 使い方（サンプル）
# =========================================================
# from data.tsplib import get_tsp_instance
# from data.qaplib import get_instance
#
# # TSP:
# dist, dim = get_tsp_instance.get_tsp_dat("pr1002")
# def tsp_eval(perm: List[int]) -> float:
#     s = 0.0
#     for i in range(dim-1): s += dist[perm[i]][perm[i+1]]
#     s += dist[perm[-1]][perm[0]]
#     return s
# bo = SurrogatePermBO(dim=dim, true_eval_fn=tsp_eval, problem='TSP',
#                      surrogate_kind='gin', pop_size=100, n_init_true=100,
#                      batch_true=20, candidate_nums=400, max_queries=150)
# best_perm, best_fx, hist = bo.run(dist_mtx_for_2opt=dist)
#
# # QAP:
# F, D, dim = get_instance.get_dat("nug20")
# def qap_eval(perm: List[int]) -> float:
#     tot = 0
#     for i in range(dim):
#         for j in range(dim):
#             tot += F[i][j] * D[perm[i]][perm[j]]
#     return float(tot)
# bo = SurrogatePermBO(dim=dim, true_eval_fn=qap_eval, problem='QAP',
#                      surrogate_kind='rf', pop_size=100, n_init_true=100,
#                      batch_true=20, candidate_nums=400, max_queries=150)
# best_perm, best_fx, hist = bo.run()
