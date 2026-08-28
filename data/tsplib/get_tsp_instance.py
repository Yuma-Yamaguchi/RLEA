import tsplib95
import numpy as np
import os

def get_tsp_dat(ins_name):
    """
    TSPLIBの.tspファイルを読み込んで距離行列と次元を返す関数。
    ins_name : 例 "berlin52", "eil51" など（拡張子不要）
    """
    INSTANCE_PATH = "./data/tsplib/instance/"
    file_name = os.path.join(INSTANCE_PATH, ins_name, f"{ins_name}.tsp")

    # 読み込み
    problem = tsplib95.load_problem(file_name)

    # 次元
    dim = problem.dimension

    # ノード番号の取得
    # 通常は 1..dim だが、get_nodes() を使うと仕様差に強い
    nodes = list(problem.get_nodes())

    # 距離行列を生成
    distance_mtx = np.zeros((dim, dim), dtype=np.float64)
    for a, i in enumerate(nodes):
        for b, j in enumerate(nodes):
            distance_mtx[a, b] = float(problem.get_weight(i, j))

    return distance_mtx, dim
    # INSTANCE_PATH = "./data/tsplib/instance/"
    # file_name = os.path.join(INSTANCE_PATH, ins_name, f"{ins_name}.tsp")

    # # --- 読み込み ---
    # problem = tsplib95.load_problem(file_name)

    # # --- 次元 ---
    # dim = problem.dimension

    # coords = {i: np.array(coord, dtype=np.float64) for i, coord in problem.node_coords.items()}

    # # --- 距離行列を生成 ---
    # distance_mtx = np.zeros((dim, dim), dtype=np.float64)
    # for i in range(1, dim + 1):
    #     for j in range(1, dim + 1):
    #         distance_mtx[i - 1, j - 1] = float(np.linalg.norm(coords[i] - coords[j]))

    # return distance_mtx, dim

if __name__ == "__main__":
    distance_mtx, dim = get_tsp_dat("TSP-D10")
    print("Distance Matrix:")
    print(distance_mtx)
    print("Dimension:", dim)