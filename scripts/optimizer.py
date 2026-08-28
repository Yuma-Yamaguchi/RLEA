# permutation_optimizer.py
from __future__ import annotations
import random, numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Literal

# ==========
# 基本ユーティリティ
# ==========
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def kendall_tau_distance(p: List[int], q: List[int]) -> int:
    pos = {v:i for i,v in enumerate(q)}
    arr = [pos[v] for v in p]
    invs = 0
    def mergesort(a):
        nonlocal invs
        L = len(a)
        if L <= 1: return a
        m = L//2
        Ls = mergesort(a[:m]); Rs = mergesort(a[m:])
        i=j=0; out=[]
        while i<len(Ls) and j<len(Rs):
            if Ls[i] <= Rs[j]:
                out.append(Ls[i]); i+=1
            else:
                out.append(Rs[j]); j+=1; invs += len(Ls)-i
        out.extend(Ls[i:]); out.extend(Rs[j:])
        return out
    mergesort(arr)
    return invs

# ==========
# 演算子（交叉／変異／2-opt）
# ==========
def ox(parent1: List[int], parent2: List[int]) -> List[int]:
    n = len(parent1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b] = parent1[a:b]
    pos = b
    for v in parent2:
        if v not in child:
            if pos >= n: pos = 0
            child[pos] = v
            pos += 1
    return child

def pmx(parent1: List[int], parent2: List[int]) -> List[int]:
    n = len(parent1)
    a, b = sorted(random.sample(range(n), 2))
    c = [-1]*n
    c[a:b+1] = parent1[a:b+1]
    map12 = {parent1[i]: parent2[i] for i in range(a,b+1)}
    map21 = {v:k for k,v in map12.items()}
    for i in list(range(0,a))+list(range(b+1,n)):
        x = parent2[i]
        while x in c:
            x = map21.get(x, x)
            if x in c: x = map21.get(x, x)
            if x in c: break
        if x not in c:
            c[i] = x
    for i in range(n):
        if c[i] == -1:
            for v in parent2:
                if v not in c:
                    c[i] = v; break
    return c

def cx(parent1: List[int], parent2: List[int]) -> List[int]:
    n = len(parent1)
    c = [-1]*n
    used = [False]*n
    idx = 0
    while not all(used):
        if used[idx]:
            idx = used.index(False)
        start = idx
        while True:
            c[idx] = parent1[idx]
            used[idx] = True
            idx = parent1.index(parent2[idx])
            if idx == start: break
        idx = used.index(False) if not all(used) else idx
    for i in range(n):
        if c[i] == -1:
            c[i] = parent2[i]
    return c

def ux(parent1: List[int], parent2: List[int]) -> List[int]:
    n = len(parent1)
    child = [-1] * n
    taken = set()
    # ランダムに親1/親2から値を選ぶ（重複を除く）
    for i in range(n):
        if random.random() < 0.5:
            val = parent1[i]
        else:
            val = parent2[i]
        if val not in taken:
            child[i] = val
            taken.add(val)
    # 残りを親2にしたがって補完
    fill_vals = [v for v in parent2 if v not in taken]
    for i in range(n):
        if child[i] == -1:
            child[i] = fill_vals.pop(0)
    return child

def mutate_swap(p: List[int]) -> List[int]:
    n = len(p); i, j = random.sample(range(n), 2)
    c = p[:]; c[i], c[j] = c[j], c[i]; return c

def mutate_insert(p: List[int]) -> List[int]:
    n = len(p); i, j = sorted(random.sample(range(n), 2))
    c = p[:]; v = c.pop(j); c.insert(i, v); return c

def mutate_invert(p: List[int]) -> List[int]:
    n = len(p); i, j = sorted(random.sample(range(n), 2))
    c = p[:]; c[i:j] = reversed(c[i:j]); return c

def mutate_double_bridge(p: List[int]) -> List[int]:
    """
    TSPなどでよく使われる強い撹乱オペレータ。
    順列を4分割して、ブロック1↔3を入れ替える。
    """
    n = len(p)
    if n < 8:  # あまり短い場合は invert にフォールバック
        return mutate_invert(p)
    # 4つの切れ目をランダムに選び、昇順にソート
    a, b, c, d = sorted(random.sample(range(1, n - 1), 4))
    # ブロックを入れ替える
    c_new = p[:a] + p[c:d] + p[b:c] + p[a:b] + p[d:]
    return c_new

def mutate_three_cycle(p: List[int]) -> List[int]:
    """
    3つの位置をサイクル的に入れ替える突然変異。
    例: (i,j,k) = (2,5,7) → p[i]->p[j], p[j]->p[k], p[k]->p[i]
    """
    n = len(p)
    i, j, k = sorted(random.sample(range(n), 3))
    c = p[:]
    c[i], c[j], c[k] = p[k], p[i], p[j]
    return c

def mutate_scramble(p: List[int]) -> List[int]:
    n = len(p); i, j = sorted(random.sample(range(n), 2))
    c = p[:]; seg = c[i:j]; random.shuffle(seg); c[i:j] = seg; return c

def two_opt_once(perm: List[int], dist: np.ndarray) -> Tuple[List[int], float, bool]:
    n = len(perm)
    def seg(a,b): return dist[a,b]
    cur = perm; best_i=best_j=None; delta_best=0.0; improved=False
    for i in range(n-1):
        a,b = cur[i], cur[(i+1)%n]
        for j in range(i+2, n - (0 if i>0 else 1)):
            c,d = cur[j], cur[(j+1)%n]
            delta = (seg(a,c)+seg(b,d)) - (seg(a,b)+seg(c,d))
            if delta < -1e-12 and (not improved or delta < delta_best):
                improved=True; delta_best=delta; best_i, best_j=i, j
    if improved:
        newp = cur[:]
        newp[best_i+1:best_j+1] = reversed(newp[best_i+1:best_j+1])
        return newp, delta_best, True
    return cur, 0.0, False

def local_search_2opt(perm: List[int], dist: np.ndarray, max_moves: int = 32) -> List[int]:
    cur = perm[:]; moves=0
    while moves < max_moves:
        cur, delta, ok = two_opt_once(cur, dist)
        if not ok: break
        moves += 1
    return cur

# ==========
# サロゲートの型
# ==========
# surrogate.predict(perms: List[List[int]]) -> (mu: np.ndarray, sigma: np.ndarray)
SurrogatePredictFn = Callable[[List[List[int]]], Tuple[np.ndarray, np.ndarray]]

# ==========
# Optimizer クラス
# ==========
class GAOptimizer:
    """
    汎用・順列EAオプティマイザ（交叉/変異/ローカル探索を指定可能）
    - problem: 'TSP' or 'QAP'（TSPなら2-optを有効化可能）
    - crossover: 'ox' | 'pmx' | 'cx' | 'ux'
    - mutation_weights: {'swap':0.3,'insert':0.3,'invert':0.4,...}
    """
    def __init__(
        self,
        dim: int,
        problem: Literal['TSP','QAP','LOP','PFSP', 'ATSP'] = 'TSP',
        true_eval_fn: Optional[Callable[[List[int]], float]] = None,
        crossover: Literal['ox','pmx','cx', 'ux'] = 'ox',
        mutation_weights: Optional[Dict[str, float]] = None,
        selection_k: int = 3,
        rng_seed: int = 0,
    ):
        self.dim = dim
        self.problem = problem
        self.true_eval_fn = true_eval_fn
        self.crossover = crossover
        self.mutation_weights = mutation_weights or {'swap':0.3,'insert':0.3,'invert':0.4}
        self.selection_k = selection_k
        random.seed(rng_seed); np.random.seed(rng_seed)

    # ---- 親選択（トーナメント）----
    def _select_parent(self, pop: List[List[int]]) -> List[int]:
        k = min(self.selection_k, len(pop))
        cand = random.sample(pop, k)
        if self.true_eval_fn is None:
            return random.choice(cand)
        return min(cand, key=self.true_eval_fn)

    # ---- 交叉をディスパッチ ----
    def _crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        if self.crossover == 'ox':  return ox(p1, p2)
        if self.crossover == 'pmx': return pmx(p1, p2)
        if self.crossover == 'cx':  return cx(p1, p2)
        if self.crossover == 'ux':  return ux(p1, p2)
        raise ValueError(f"unknown crossover: {self.crossover}")

    # ---- 変異を重み付きランダム ----
    def _mutate(self, c: List[int]) -> List[int]:
        ops = []
        for name, w in self.mutation_weights.items():
            ops.extend([name]*max(1, int(round(w*100))))
        op = random.choice(ops) if ops else 'invert'
        if op == 'swap':     return mutate_swap(c)
        if op == 'insert':   return mutate_insert(c)
        if op == 'invert':   return mutate_invert(c)
        if op == 'double_bridge': return mutate_double_bridge(c)
        if op == 'three_cycle':  return mutate_three_cycle(c)
        if op == 'scramble': return mutate_scramble(c)
        return mutate_invert(c)

    # ---- 子を作る（交叉＋変異＋（TSPなら2-opt））----
    def _make_child(self, p1: List[int], p2: List[int]) -> List[int]:
        c = self._crossover(p1, p2)
        c = self._mutate(c)
        # if self.problem == 'TSP' and self.local_search == '2opt' and dist_mtx is not None:
        #     c = local_search_2opt(c, dist_mtx, max_moves=8)
        return c

    # ---- 候補から UCB 上位 + 多様性で b 個選抜 ----
    def pick_topk_diverse(self, candidates: List[List[int]], scores: np.ndarray, k: int) -> List[List[int]]:
        order = np.argsort(scores)[::-1]  # UCB高い順
        chosen: List[List[int]] = []
        for idx in order:
            if len(chosen) >= k: break
            pi = candidates[idx]
            # ここで距離閾値を入れたい場合は下記のように：
            # if any(kendall_tau_distance(pi, pj) < threshold for pj in chosen): continue
            chosen.append(pi)
        return chosen

    # ---- 子をまとめて生成 ----
    def generate_children(self, pop: List[List[int]], num_children: int) -> List[List[int]]:
        children = []
        while len(children) < num_children:
            p1 = self._select_parent(pop)
            p2 = self._select_parent(pop)
            c = self._make_child(p1, p2)
            children.append(c)
        return children

    def generate_int_children(self, pop: List[List[int]]) -> List[List[int]]:
        children = []
        for i, p in enumerate(pop):
            p1 = p
            r = random.randint(0, len(pop)-1)
            while r == i:
                r = random.randint(0, len(pop)-1)
            p2 = pop[r]
            c = self._make_child(p1, p2)
            children.append(c)
        return children
# =================================================================

import random
from collections import deque
from typing import Callable, Deque, Dict, List, Literal, Optional, Sequence, Tuple
import numpy as np


class TabuSearchOptimizer:
    """
    FAT-RLS 互換の挿入近傍 Tabuサーチを、"候補生成器" として使えるように実装。
    - 最小化評価関数 true_eval_fn を与える（親選択や初期基準に利用）
    - generate_children() は、pop から親を選んで挿入操作を適用し、num_children 個の候補を返す
    - pick_topk_diverse() は GAOptimizer と同じ（スコア大→良い）

    d のスケジューリング:
      p = progress in [0,1] に対し d = round( 1 + s_beta(p)*(d_ini-1) )
      s_beta(p) = 1 - 1/(1 + ((1-p)/max(p,eps))^beta)

    備考:
      - 本クラスは “真の受理判定” は行いません（外側BOループが担当）。
        そのため Tabu キューは「候補生成の連続呼び出し」に渡って維持されます。
    """

    def __init__(
        self,
        dim: int,
        problem: Literal['TSP','QAP','LOP','PFSP', 'ATSP'] = 'TSP',
        true_eval_fn: Optional[Callable[[List[int]], float]] = None,
        *,
        d_ini: int = 8,
        beta: float = 2.0,
        tabu_k: int = 8,
        selection_k: int = 3,
        schedule_budget: Optional[int] = None,  # 進捗スケジュール用の総評価回数（外側の max_eval を渡すと◎）
        rng_seed: int = 0,
    ):
        if true_eval_fn is None:
            raise ValueError("true_eval_fn is required (minimization).")
        self.dim = dim
        self.problem = problem
        self.true_eval_fn = true_eval_fn
        self.d_ini = max(1, int(d_ini))
        self.beta = float(beta)
        self.tabu_k = max(0, int(tabu_k))
        self.selection_k = max(1, int(selection_k))
        self.schedule_budget = schedule_budget or 10_000  # 適当なデフォルト
        self.calls = 0  # generate_children の呼び出し回数（進捗近似）
        self.TQ: Deque[int] = deque(maxlen=self.tabu_k if self.tabu_k > 0 else 0)
        random.seed(rng_seed); np.random.seed(rng_seed)

    # ========= ユーティリティ（GA と同じ感覚） =========
    def _select_parent(self, pop: List[List[int]]) -> List[int]:
        k = min(self.selection_k, len(pop))
        cand = random.sample(pop, k)
        if self.true_eval_fn is None:
            return random.choice(cand)
        return min(cand, key=self.true_eval_fn)

    def pick_topk_diverse(self, candidates: List[List[int]], scores: np.ndarray, k: int) -> List[List[int]]:
        """スコア大→良い とみなして上位から選抜（多様性フィルタを入れたい場合はここを拡張）。"""
        order = np.argsort(scores)[::-1]
        chosen: List[List[int]] = []
        for idx in order:
            if len(chosen) >= k:
                break
            chosen.append(candidates[idx])
        return chosen

    # ========= 進捗→d のスケジュール =========
    def _compute_d_by_progress(self, progress: float) -> int:
        """FAT-RLS の skewed-S 式で d を決定。"""
        eps = 1e-12
        p = min(max(progress, eps), 1.0 - eps)
        t = ((1.0 - p) / p) ** self.beta
        s = 1.0 - 1.0 / (1.0 + t)
        d = int(round(1.0 + s * (self.d_ini - 1.0)))
        return max(2, min(d, max(1, self.dim - 1)))

    # ========= 近傍生成（挿入 |i-j| = d, tabu 回避） =========
    @staticmethod
    def _insertion(seq: List[int], i: int, j: int) -> None:
        if i == j:
            return
        x = seq.pop(i)
        if j > i:
            j -= 1
        seq.insert(j, x)

    def _random_insertion_with_distance(
        self, base: List[int], d: int, tabu_q: Optional[Deque[int]], max_trials: int = 64
    ) -> Tuple[List[int], int, bool]:
        """|i-j|=d を満たす挿入をランダムに1回適用。戻り値: (child, moved_item, success)"""
        n = len(base)
        for _ in range(max_trials):
            i = random.randrange(n)
            j = i + d if random.random() < 0.5 else i - d
            if j < 0 or j >= n:
                continue
            moved = base[i]
            if tabu_q is not None and moved in tabu_q:
                continue
            child = base[:]
            self._insertion(child, i, j)
            return child, moved, True
        # fallback: 何も見つからなければランダム隣接 swap
        if n >= 2:
            i = random.randrange(n - 1)
            child = base[:]
            child[i], child[i + 1] = child[i + 1], child[i]
            return child, child[i], True
        return base[:], base[0] if n else -1, False

    # ========= 外側BOループに合わせたインターフェース =========
    def generate_children(
        self,
        pop: List[List[int]],
        num_children: int,
        dist_mtx: Optional[np.ndarray] = None,  # 互換性維持（未使用）
    ) -> List[List[int]]:
        """
        GAOptimizer と同じシグネチャ。
        - 親をトーナメントで選び、スケジュールで決めた距離 d の挿入を tabu 回避で当てる
        - これを num_children 回繰り返すだけ（受理判定や真評価は外側で）
        """
        self.calls += 1
        progress = min(1.0, self.calls / max(1, self.schedule_budget))
        d = self._compute_d_by_progress(progress)

        children: List[List[int]] = []
        for _ in range(num_children):
            parent = self._select_parent(pop)
            child, moved, _ok = self._random_insertion_with_distance(parent, d, self.TQ if self.tabu_k > 0 else None)
            if self.tabu_k > 0:
                self.TQ.append(moved)
            children.append(child)
        return children

    def generate_int_children(self, pop: List[List[int]]) -> List[List[int]]:
        children = []
        progress = min(1.0, self.calls / max(1, self.schedule_budget))
        d = self._compute_d_by_progress(progress)
        children: List[List[int]] = []
        for i, p in enumerate(pop):
            parent = p
            child, moved, _ok = self._random_insertion_with_distance(p, d, self.TQ if self.tabu_k > 0 else None)
            if self.tabu_k > 0:
                self.TQ.append(moved)
            children.append(child)
        return children
# =================================================================