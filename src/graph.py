import argparse
import pickle
from pathlib import Path
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors

from utils import save_npy


def run(config_path: str) -> None:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg["artifacts_dir"])
    eta = np.load(base / cfg["embeddings_dir"] / "eta.npy")  # (N,D)
    G = np.load(base / cfg["metrics_dir"] / "G_theta.npy")   # (N,D,D)

    N, D = eta.shape
    k = int(cfg["knn_k"])  # neighbors per node

    # initial kNN by Euclidean in eta space
    nn = NearestNeighbors(n_neighbors=min(k + 1, N), metric="euclidean")
    nn.fit(eta)
    dists, inds = nn.kneighbors(eta)  # include self at index 0

    G_nx = nx.Graph()
    for i in range(N):
        G_nx.add_node(i)

    for i in range(N):
        for j in inds[i][1:]:  # skip self
            if G_nx.has_edge(i, int(j)):
                continue
            v = eta[i] - eta[int(j)]  # (D,)
            G_avg = 0.5 * (G[i] + G[int(j)])  # (D,D)
            w_ij = float(np.sqrt(max(1e-12, v @ (G_avg @ v))))
            G_nx.add_edge(i, int(j), weight=w_ij)

    out_path = base / cfg["graphs_dir"] / "gpt2.gpickle"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(G_nx, f)

    print("Graph saved:", out_path)
    print("Nodes:", G_nx.number_of_nodes(), "Edges:", G_nx.number_of_edges())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    args = parser.parse_args()
    run(args.config)
