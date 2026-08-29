import tsplib95
import numpy as np
import os

def get_atsp_dat(ins_name):

    INSTANCE_PATH = "./data/atsplib/instance/"
    file_name = os.path.join(INSTANCE_PATH, ins_name, f"{ins_name}.atsp")

    problem = tsplib95.load_problem(file_name)

    dim = problem.dimension
    nodes = list(problem.get_nodes())
    distance_mtx = np.zeros((dim, dim), dtype=np.float64)
    for a, i in enumerate(nodes):
        for b, j in enumerate(nodes):
            distance_mtx[a, b] = float(problem.get_weight(i, j))

    return distance_mtx, dim
