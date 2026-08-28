import os
import glob
from typing import Tuple, Callable, List, Optional
import numpy as np

def find_lolib_file(base_dir: str, ins_name: str) -> str:
    """
    LOLIB のファイルを柔軟に探す:
    - 拡張子なし: "<base>/<ins_name>"
    - 任意拡張子や番号付き: "<base>/<ins_name>.*"
    - サブディレクトリ配下にもある場合: 再帰的に探索
    """
    # 1) 完全一致（拡張子なし）
    candidate = os.path.join(base_dir, ins_name)
    if os.path.isfile(candidate):
        return candidate

    # 2) 同ディレクトリでワイルドカード
    pats = [
        os.path.join(base_dir, f"{ins_name}.*"),
        os.path.join(base_dir, f"{ins_name}"),
    ]
    for pat in pats:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]

    # 3) 再帰的探索（サブフォルダを横断）
    for root, _, files in os.walk(base_dir):
        # 完全一致
        cand = os.path.join(root, ins_name)
        if os.path.isfile(cand):
            return cand
        # ワイルドカード
        matches = sorted(glob.glob(os.path.join(root, f"{ins_name}.*")))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"LOLIB instance not found: base_dir='{base_dir}', ins_name='{ins_name}'")


def parse_square_matrix_from_lines(lines: List[str], n: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    先頭行に n が書かれている場合と、書かれていない場合の両方に対応。
    - n が None なら、最初の非空行が int に解釈できればそれを n として採用
      できなければ、行数から n を推定し、n 行ぶんの正方行列を読む。
    - 行はスペース区切りを想定。空行はスキップ。
    """
    cleaned = [ln.strip() for ln in lines if ln.strip() != ""]
    idx = 0

    # n 不明なら最初の行で判定
    if n is None:
        try:
            n = int(cleaned[0].split()[0])
            idx = 1
        except ValueError:
            # 先頭が数値でない -> 行列本体がすぐ始まる前提
            # この場合、n は行数から推定する（ただし正方性を仮定）
            # ※ LOLIB は正方行列が基本
            n = len(cleaned)
            idx = 0

    # n 行ぶんをパース（各行は n 個）
    W = np.zeros((n, n), dtype=np.float64)
    row = 0
    while row < n and idx < len(cleaned):
        parts = cleaned[idx].replace(",", " ").split()
        # 行途中で改行されている可能性が低い前提（LOLIBは1行n要素が基本）
        if len(parts) < n:
            # もし要素が足りなければ、次の行も連結して埋める
            buf = parts[:]
            j = idx + 1
            while len(buf) < n and j < len(cleaned):
                buf += cleaned[j].replace(",", " ").split()
                j += 1
            parts = buf
            idx = j - 1  # 読み進めたところまで調整
        for col in range(n):
            W[row, col] = float(parts[col])
        row += 1
        idx += 1

    if row != n:
        raise ValueError(f"Matrix parsing failed: expected {n} rows, got {row}")

    return W, n


def get_lolib_dat(ins_name: str,
                       base_dir: str = "./data/lolib") -> Tuple[np.ndarray, int, Callable[[List[int]], float]]:
    """
    LOLIB（IO, MB, RandA1... など）インスタンスから
    - 重み行列 W（n×n）
    - 次元 n
    - 評価関数 eval_func（LOPの定義: 順列 pi に対して sum_{i<j} W[pi[i], pi[j]] を返す）
    を返す。

    Parameters
    ----------
    ins_name : str
        例: "IOA.25", "MBA.30", "RandA1.100", "N-t1d100.01" など（拡張子不要）
    base_dir : str
        ルートディレクトリ

    Returns
    -------
    W : np.ndarray
    n : int
    eval_func : Callable[[List[int]], float]
    """
    file_path = find_lolib_file(base_dir, ins_name)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 先頭行に n が書かれている前提が多いが、無い場合も吸収
    W, n = parse_square_matrix_from_lines(lines, n=None)

    # # LOP の評価関数（最大化）
    # def eval_func(pi: List[int]) -> float:
    #     """
    #     0-indexed permutation pi に対し、
    #     f(pi) = sum_{i<j} W[pi[i], pi[j]]
    #     """
    #     s = 0.0
    #     for i in range(n - 1):
    #         ii = pi[i]
    #         # ベクトル化で高速化も可能だが、まずは可読性優先
    #         for j in range(i + 1, n):
    #             s += W[ii, pi[j]]
    #     return s

    return W, n


# ===== 動作例 =====
if __name__ == "__main__":
    # 例: "./data/lolib/N-t1d100.01" など
    W, n = get_lolib_dat("N-econ36")

    print("Dimension:", n)
    print("Matrix shape:", W.shape)

    # pi = list(range(n))
    # print("Score(identity):", eval_lop(pi))
