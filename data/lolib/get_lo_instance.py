import os
import glob
from typing import Tuple, Callable, List, Optional
import numpy as np

def find_lolib_file(base_dir: str, ins_name: str) -> str:
    candidate = os.path.join(base_dir, ins_name)
    if os.path.isfile(candidate):
        return candidate

    pats = [
        os.path.join(base_dir, f"{ins_name}.*"),
        os.path.join(base_dir, f"{ins_name}"),
    ]
    for pat in pats:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]

    for root, _, files in os.walk(base_dir):
        cand = os.path.join(root, ins_name)
        if os.path.isfile(cand):
            return cand
        matches = sorted(glob.glob(os.path.join(root, f"{ins_name}.*")))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"LOLIB instance not found: base_dir='{base_dir}', ins_name='{ins_name}'")


def parse_square_matrix_from_lines(lines: List[str], n: Optional[int] = None) -> Tuple[np.ndarray, int]:
    cleaned = [ln.strip() for ln in lines if ln.strip() != ""]
    idx = 0

    if n is None:
        try:
            n = int(cleaned[0].split()[0])
            idx = 1
        except ValueError:
            n = len(cleaned)
            idx = 0

    W = np.zeros((n, n), dtype=np.float64)
    row = 0
    while row < n and idx < len(cleaned):
        parts = cleaned[idx].replace(",", " ").split()
        if len(parts) < n:
            buf = parts[:]
            j = idx + 1
            while len(buf) < n and j < len(cleaned):
                buf += cleaned[j].replace(",", " ").split()
                j += 1
            parts = buf
            idx = j - 1 
        for col in range(n):
            W[row, col] = float(parts[col])
        row += 1
        idx += 1

    if row != n:
        raise ValueError(f"Matrix parsing failed: expected {n} rows, got {row}")

    return W, n


def get_lolib_dat(ins_name: str,
                       base_dir: str = "./data/lolib") -> Tuple[np.ndarray, int, Callable[[List[int]], float]]:
    file_path = find_lolib_file(base_dir, ins_name)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    W, n = parse_square_matrix_from_lines(lines, n=None)
    return W, n