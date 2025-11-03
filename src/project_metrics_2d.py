import argparse
import pickle
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import networkx as nx

from utils import save_npy, set_global_seed


def _all_pairs_shortest_path_matrix(G: nx.Graph) -> np.ndarray:
    """Compute all-pairs shortest path matrix"""
    nodes = list(G.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    D = np.full((N, N), np.inf, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    for src in nodes:
        lengths = nx.single_source_dijkstra_path_length(G, src, weight="weight")
        i = index[src]
        for dst, d in lengths.items():
            j = index[dst]
            D[i, j] = d
            D[j, i] = d

    # handle disconnected pairs by filling with max finite
    finite_vals = D[np.isfinite(D)]
    if finite_vals.size == 0:
        raise RuntimeError("Graph has no finite paths.")
    max_val = float(finite_vals.max())
    D[~np.isfinite(D)] = max_val * 1.1
    return D


def run(config_path: str) -> None:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_global_seed(int(cfg["seed"]))

    base_dir = Path(cfg["artifacts_dir"])
    # Create metrics_2d directory
    metrics_2d_dir = base_dir / "metrics_2d"
    metrics_2d_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    eta = np.load(base_dir / cfg["embeddings_dir"] / "eta.npy")  # (N,D)
    G_theta = np.load(base_dir / cfg["metrics_dir"] / "G_theta.npy")  # (N,D,D)
    
    # Load Fisher metric graph for MDS
    G_fisher = pickle.load(open(base_dir / cfg["graphs_dir"] / "gpt2.gpickle", 'rb'))
    
    # Compute MDS embedding
    D = _all_pairs_shortest_path_matrix(G_fisher)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=int(cfg["seed"]), max_iter=1000, eps=1e-9)
    X_mds = mds.fit_transform(D)
    
    # Project eta to 2D using PCA
    eta_mean = eta.mean(axis=0)
    eta_centered = eta - eta_mean
    pca = PCA(n_components=2)
    eta_2d = pca.fit_transform(eta_centered)
    
    # Project Fisher metrics to 2D
    # P: projection matrix from D-dim to 2D
    P = pca.components_.T  # (D, 2)
    
    # Project each G_theta[i] from DxD to 2x2
    N = G_theta.shape[0]
    G_2d = np.zeros((N, 2, 2), dtype=np.float32)
    for i in range(N):
        G_2d[i] = P.T @ G_theta[i] @ P
    
    # Save 2D metrics
    out_path = metrics_2d_dir / "G_theta_2d.npy"
    save_npy(out_path, G_2d)
    
    print("2D Fisher metrics saved:", out_path)
    print("Shape:", G_2d.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    args = parser.parse_args()
    run(args.config)

