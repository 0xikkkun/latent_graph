import argparse
import pickle
import json
from pathlib import Path
import numpy as np
import networkx as nx
from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import cosine_distances
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
    
    # Load graph for MDS
    graph_path = base / cfg["graphs_dir"] / "gpt2.gpickle"
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)

    # Load eta for t-SNE
    eta_path = base / cfg["embeddings_dir"] / "eta.npy"
    eta = np.load(eta_path)

    # Load metadata with labels
    meta_path = base / cfg["embeddings_dir"] / "meta.json"
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    labels = meta.get("labels", [])
    unique_labels = meta.get("unique_labels", [])
    label_counts = meta.get("label_counts", {})

    # MDS
    D = _all_pairs_shortest_path_matrix(G)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=int(cfg["seed"]))
    X_mds = mds.fit_transform(D)

    # t-SNE with cosine distance
    cosine_dist = cosine_distances(eta)
    tsne = TSNE(n_components=2, metric="precomputed", init="random", random_state=int(cfg["seed"]))
    X_tsne = tsne.fit_transform(cosine_dist)

    # Create comparison plot
    comparison_out = base / cfg["plots_dir"] / "gpt2_comparison_mds_tsne.png"
    comparison_out.parent.mkdir(parents=True, exist_ok=True)

    # Create color mapping for labels
    if len(unique_labels) > 1:
        # Create color map
        colors = cm.tab10(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        # Create comparison plot
        plt.figure(figsize=(20, 8))
        
        # MDS subplot
        plt.subplot(1, 2, 1)
        
        # Draw edges first (so they appear behind nodes)
        for edge in G.edges():
            i, j = edge
            if i < len(X_mds) and j < len(X_mds):  # Ensure indices are valid
                plt.plot([X_mds[i, 0], X_mds[j, 0]], [X_mds[i, 1], X_mds[j, 1]], 
                        'k-', alpha=0.3, linewidth=0.5)
        
        # Draw nodes with labels
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(X_mds[mask, 0], X_mds[mask, 1], s=30, c=[label_to_color[label]], 
                       alpha=0.8, label=f"{label} ({label_counts.get(label, 0)})", 
                       edgecolors='black', linewidth=0.5)
        
        plt.title("GPT-2 Latent Graph (MDS) with Edges")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # t-SNE subplot
        plt.subplot(1, 2, 2)
        
        # Draw nodes with labels (no edges for t-SNE)
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=30, c=[label_to_color[label]], 
                       alpha=0.8, label=f"{label} ({label_counts.get(label, 0)})", 
                       edgecolors='black', linewidth=0.5)
        
        plt.title("GPT-2 Latent Space (t-SNE with Cosine Distance)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
    else:
        # Fallback to single color if only one label
        plt.figure(figsize=(20, 8))
        
        # MDS subplot
        plt.subplot(1, 2, 1)
        
        # Draw edges first
        for edge in G.edges():
            i, j = edge
            if i < len(X_mds) and j < len(X_mds):
                plt.plot([X_mds[i, 0], X_mds[j, 0]], [X_mds[i, 1], X_mds[j, 1]], 
                        'k-', alpha=0.3, linewidth=0.5)
        
        # Draw nodes
        plt.scatter(X_mds[:, 0], X_mds[:, 1], s=30, c="tab:blue", alpha=0.9, 
                   edgecolors='black', linewidth=0.5)
        plt.title("GPT-2 Latent Graph (MDS) with Edges")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        
        # t-SNE subplot
        plt.subplot(1, 2, 2)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=30, c="tab:blue", alpha=0.9, 
                   edgecolors='black', linewidth=0.5)
        plt.title("GPT-2 Latent Space (t-SNE with Cosine Distance)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        
        plt.tight_layout()

    plt.savefig(comparison_out, dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved:", comparison_out)
    print(f"Labels found: {unique_labels}")
    print(f"Label distribution: {label_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    args = parser.parse_args()
    run(args.config)
