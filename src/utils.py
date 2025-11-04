import os
import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from scipy.linalg import pinvh


def set_global_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(base_dir: str, *subdirs: str) -> Dict[str, Path]:
    base = Path(base_dir)
    paths: Dict[str, Path] = {}
    for sub in subdirs:
        p = base / sub
        p.mkdir(parents=True, exist_ok=True)
        paths[sub] = p
    return paths


def save_npy(path: Union[str, Path], arr: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, arr)


def save_json(path: Union[str, Path], obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv_diag(path: Union[str, Path], mats: np.ndarray) -> None:
    # mats: (N, D, D) -> save diagonal summary per sample
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "mean_diag", "std_diag"])
        for i in range(mats.shape[0]):
            d = np.diag(mats[i])
            writer.writerow([i, float(d.mean()), float(d.std())])


def pinvh_stable(sym_mat: np.ndarray, jitter: float = 0.0) -> np.ndarray:
    m = sym_mat
    if jitter > 0:
        m = m + jitter * np.eye(m.shape[0], dtype=m.dtype)
    return pinvh(m, check_finite=False)
