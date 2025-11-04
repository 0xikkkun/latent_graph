import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def run(config_path: str) -> None:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg["artifacts_dir"])
    eta = np.load(base / cfg["embeddings_dir"] / "eta.npy")  # (N,D)

    N, D = eta.shape
    k = int(cfg["knn_k"])  # neighbors per node

    # GPU使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eta_gpu = torch.from_numpy(eta).float().to(device)  # (N,D)

    # kNN by Euclidean distance in eta space (CPUでsklearnを使用)
    nn = NearestNeighbors(n_neighbors=min(k + 1, N), metric="euclidean")
    nn.fit(eta)
    dists, inds = nn.kneighbors(eta)  # include self at index 0

    G_nx = nx.Graph()
    for i in range(N):
        G_nx.add_node(i)

    # ユークリッド距離をGPUで計算
    for i in tqdm(range(N), desc="Computing Euclidean distances", unit="node"):
        for j in inds[i][1:]:  # skip self
            j = int(j)
            if G_nx.has_edge(i, j):
                continue
            # GPU上でユークリッド距離計算
            v = eta_gpu[i] - eta_gpu[j]  # (D,)
            w_ij = torch.norm(v, p=2)  # L2ノルム
            G_nx.add_edge(i, j, weight=w_ij.item())

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

