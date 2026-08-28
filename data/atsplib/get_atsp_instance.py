import tsplib95
import numpy as np
import os

def get_atsp_dat(ins_name):
    """
    TSPLIBの.atspファイルを読み込んで距離行列と次元を返す関数。
    ins_name : 例 "ATSP-D10" など（拡張子不要）
    """
    INSTANCE_PATH = "./data/atsplib/instance/"
    file_name = os.path.join(INSTANCE_PATH, ins_name, f"{ins_name}.atsp")

    # --- 読み込み ---
    problem = tsplib95.load_problem(file_name)

    # --- 次元 ---
    dim = problem.dimension

    # coords = {i: np.array(coord, dtype=np.float64) for i, coord in problem.node_coords.items()}

    nodes = list(problem.get_nodes())
    distance_mtx = np.zeros((dim, dim), dtype=np.float64)
    for a, i in enumerate(nodes):
        for b, j in enumerate(nodes):
            distance_mtx[a, b] = float(problem.get_weight(i, j))
    # # --- 距離行列を生成 ---
    # distance_mtx = np.zeros((dim, dim), dtype=np.float64)
    # for i in range(1, dim + 1):
    #     for j in range(1, dim + 1):
    #         distance_mtx[i - 1, j - 1] = float(np.linalg.norm(coords[i] - coords[j]))

    return distance_mtx, dim

if __name__ == "__main__":
    distance_mtx, dim = get_atsp_dat("ry48p")
    print("Distance Matrix:")
    print(distance_mtx)
    print("Dimension:", dim)
