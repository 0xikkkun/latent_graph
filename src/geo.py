import argparse
import pickle
import json
from pathlib import Path
import numpy as np
import networkx as nx
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def _all_pairs_shortest_path_matrix(G: nx.Graph) -> np.ndarray:
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

    base = Path(cfg["artifacts_dir"])
    graph_path = base / cfg["graphs_dir"] / "gpt2.gpickle"
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)

    # Load metadata with labels
    meta_path = base / cfg["embeddings_dir"] / "meta.json"
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    labels = meta.get("labels", [])
    unique_labels = meta.get("unique_labels", [])
    label_counts = meta.get("label_counts", {})

    D = _all_pairs_shortest_path_matrix(G)

    # MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=int(cfg["seed"]))
    X = mds.fit_transform(D)

    # save distance matrix and plot
    dist_out = base / cfg["geodesic_dir"] / "gpt2_dist.npy"
    dist_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(dist_out, D)

    plot_out = base / cfg["plots_dir"] / "gpt2_isomap_mds.png"
    plot_out.parent.mkdir(parents=True, exist_ok=True)

    # Create color mapping for labels
    if len(unique_labels) > 1:
        # Create color map
        colors = cm.tab10(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        point_colors = [label_to_color[label] for label in labels]
        
        plt.figure(figsize=(12, 10))
        
        # Draw edges first (so they appear behind nodes)
        for edge in G.edges():
            i, j = edge
            if i < len(X) and j < len(X):  # Ensure indices are valid
                plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 
                        'k-', alpha=0.3, linewidth=0.5)
        
        # Draw nodes with labels
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(X[mask, 0], X[mask, 1], s=30, c=[label_to_color[label]], 
                       alpha=0.8, label=f"{label} ({label_counts.get(label, 0)})", 
                       edgecolors='black', linewidth=0.5)
        
        plt.title("GPT-2 Latent Graph (MDS) with Edges - Labeled by Data Source")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    else:
        # Fallback to single color if only one label
        plt.figure(figsize=(8, 6))
        
        # Draw edges first
        for edge in G.edges():
            i, j = edge
            if i < len(X) and j < len(X):
                plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 
                        'k-', alpha=0.3, linewidth=0.5)
        
        # Draw nodes
        plt.scatter(X[:, 0], X[:, 1], s=30, c="tab:blue", alpha=0.9, 
                   edgecolors='black', linewidth=0.5)
        plt.title("GPT-2 Latent Graph (MDS) with Edges")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.tight_layout()

    plt.savefig(plot_out, dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved:", dist_out)
    print("Saved:", plot_out)
    print(f"Labels found: {unique_labels}")
    print(f"Label distribution: {label_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    args = parser.parse_args()
    run(args.config)
