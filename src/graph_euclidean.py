import argparse
import pickle
from pathlib import Path
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors


def run(config_path: str) -> None:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg["artifacts_dir"])
    eta = np.load(base / cfg["embeddings_dir"] / "eta.npy")  # (N,D)

    N, D = eta.shape
    k = int(cfg["knn_k"])  # neighbors per node

    # kNN by Euclidean distance in eta space
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
            # Weight: Euclidean distance
            idx = list(inds[i]).index(j)
            w_ij = float(dists[i][idx])
            G_nx.add_edge(i, int(j), weight=w_ij)

    out_path = base / cfg["graphs_dir"] / "gpt2_euclidean.gpickle"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(G_nx, f)

    print("Euclidean graph saved:", out_path)
    print("Nodes:", G_nx.number_of_nodes(), "Edges:", G_nx.number_of_edges())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    args = parser.parse_args()
    run(args.config)

