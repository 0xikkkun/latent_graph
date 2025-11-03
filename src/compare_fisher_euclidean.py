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
    
    # Load Fisher-metric graph for MDS
    graph_path = base / cfg["graphs_dir"] / "gpt2.gpickle"
    with open(graph_path, 'rb') as f:
        G_fisher = pickle.load(f)

    # Load Euclidean graph for MDS
    graph_euclidean_path = base / cfg["graphs_dir"] / "gpt2_euclidean.gpickle"
    with open(graph_euclidean_path, 'rb') as f:
        G_euclidean = pickle.load(f)

    # Load metadata with labels
    meta_path = base / cfg["embeddings_dir"] / "meta.json"
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    labels = meta.get("labels", [])
    unique_labels = meta.get("unique_labels", [])
    label_counts = meta.get("label_counts", {})

    # MDS with Fisher metric
    D_fisher = _all_pairs_shortest_path_matrix(G_fisher)
    mds_fisher = MDS(n_components=2, dissimilarity="precomputed", random_state=int(cfg["seed"]), max_iter=1000, eps=1e-9)
    X_mds_fisher = mds_fisher.fit_transform(D_fisher)

    # MDS with Euclidean distance
    D_euclidean = _all_pairs_shortest_path_matrix(G_euclidean)
    mds_euclidean = MDS(n_components=2, dissimilarity="precomputed", random_state=int(cfg["seed"]), max_iter=1000, eps=1e-9)
    X_mds_euclidean = mds_euclidean.fit_transform(D_euclidean)

    # Create comparison plot
    comparison_out = base / cfg["plots_dir"] / f"gpt2_comparison_fisher_vs_euclidean_ns{cfg['num_samples']}_nd{cfg['num_datasets']}_k{cfg['knn_k']}.png"
    comparison_out.parent.mkdir(parents=True, exist_ok=True)
    
    # Load 2D Fisher metrics
    metrics_2d_path = base / "metrics_2d" / "G_theta_2d.npy"
    G_2d = None
    if metrics_2d_path.exists():
        G_2d = np.load(metrics_2d_path)
        print(f"Loaded G_2d with shape: {G_2d.shape}")
    else:
        print("Warning: G_theta_2d.npy not found. Ellipses will not be displayed.")

    # Create color mapping for labels
    if len(unique_labels) > 1:
        # Create color map
        colors = cm.tab10(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        # Create comparison plot
        plt.figure(figsize=(20, 8))
        
        # Add parameter info as suptitle
        params_text = f"Parameters: num_samples={cfg['num_samples']}, num_datasets={cfg['num_datasets']}, knn_k={cfg['knn_k']}"
        plt.suptitle(params_text, fontsize=12, y=0.98)
        
        # Fisher metric MDS subplot
        plt.subplot(1, 2, 1)
        
        # Draw edges first (so they appear behind nodes)
        for edge in G_fisher.edges():
            i, j = edge
            if i < len(X_mds_fisher) and j < len(X_mds_fisher):
                plt.plot([X_mds_fisher[i, 0], X_mds_fisher[j, 0]], 
                        [X_mds_fisher[i, 1], X_mds_fisher[j, 1]], 
                        'k-', alpha=0.3, linewidth=0.5)
        
        # Draw nodes with labels
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(X_mds_fisher[mask, 0], X_mds_fisher[mask, 1], s=30, 
                       c=[label_to_color[label]], alpha=0.8, 
                       label=f"{label} ({label_counts.get(label, 0)})", 
                       edgecolors='black', linewidth=0.5)
        
        plt.title("GPT-2 Latent Graph (MDS with Fisher Metric)")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Euclidean MDS subplot
        plt.subplot(1, 2, 2)
        
        # Draw edges
        for edge in G_euclidean.edges():
            i, j = edge
            if i < len(X_mds_euclidean) and j < len(X_mds_euclidean):
                plt.plot([X_mds_euclidean[i, 0], X_mds_euclidean[j, 0]], 
                        [X_mds_euclidean[i, 1], X_mds_euclidean[j, 1]], 
                        'k-', alpha=0.3, linewidth=0.5)
        
        # Draw nodes with labels
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(X_mds_euclidean[mask, 0], X_mds_euclidean[mask, 1], s=30, 
                       c=[label_to_color[label]], alpha=0.8, 
                       label=f"{label} ({label_counts.get(label, 0)})", 
                       edgecolors='black', linewidth=0.5)
        
        plt.title("GPT-2 Latent Graph (MDS with Euclidean Distance)")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
    else:
        # Fallback to single color if only one label
        plt.figure(figsize=(20, 8))
        
        # Fisher metric MDS subplot
        plt.subplot(1, 2, 1)
        
        # Draw edges first
        for edge in G_fisher.edges():
            i, j = edge
            if i < len(X_mds_fisher) and j < len(X_mds_fisher):
                plt.plot([X_mds_fisher[i, 0], X_mds_fisher[j, 0]], 
                        [X_mds_fisher[i, 1], X_mds_fisher[j, 1]], 
                        'k-', alpha=0.3, linewidth=0.5)
        
        # Draw nodes
        plt.scatter(X_mds_fisher[:, 0], X_mds_fisher[:, 1], s=30, c="tab:blue", 
                   alpha=0.9, edgecolors='black', linewidth=0.5)
        plt.title("GPT-2 Latent Graph (MDS with Fisher Metric)")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        
        # Euclidean MDS subplot
        plt.subplot(1, 2, 2)
        
        # Draw edges
        for edge in G_euclidean.edges():
            i, j = edge
            if i < len(X_mds_euclidean) and j < len(X_mds_euclidean):
                plt.plot([X_mds_euclidean[i, 0], X_mds_euclidean[j, 0]], 
                        [X_mds_euclidean[i, 1], X_mds_euclidean[j, 1]], 
                        'k-', alpha=0.3, linewidth=0.5)
        
        # Draw nodes
        plt.scatter(X_mds_euclidean[:, 0], X_mds_euclidean[:, 1], s=30, c="tab:blue", 
                   alpha=0.9, edgecolors='black', linewidth=0.5)
        plt.title("GPT-2 Latent Graph (MDS with Euclidean Distance)")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        
        plt.tight_layout()

    plt.savefig(comparison_out, dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved:", comparison_out)
    print(f"Labels found: {unique_labels}")
    print(f"Label distribution: {label_counts}")
    
    # Also save Fisher metric plot with ellipses only
    fisher_out = base / cfg["plots_dir"] / f"gpt2_fisher_metric_ns{cfg['num_samples']}_nd{cfg['num_datasets']}_k{cfg['knn_k']}.png"
    if len(unique_labels) > 1 and G_2d is not None:
        plt.figure(figsize=(12, 10))
        
        # Add parameter info as suptitle
        params_text = f"Parameters: num_samples={cfg['num_samples']}, num_datasets={cfg['num_datasets']}, knn_k={cfg['knn_k']}"
        plt.suptitle(params_text, fontsize=12, y=0.98)
        
        # Draw ellipses FIRST (behind everything else)
        theta = np.linspace(0, 2*np.pi, 50)
        
        # Compute scale factor to make ellipses visible
        # Find the typical range of MDS coordinates
        mds_range = np.max(X_mds_fisher) - np.min(X_mds_fisher)
        # Find the typical size of Fisher metric eigenvalues
        all_eigenvals = []
        for i in range(len(X_mds_fisher)):
            G_i = G_2d[i]
            eigenvals, _ = np.linalg.eigh(G_i)
            eigenvals = np.maximum(eigenvals, 1e-10)
            all_eigenvals.extend(np.sqrt(eigenvals))
        typical_ellipse_size = np.median(all_eigenvals)
        
        # Scale factor: make ellipse about 2-3% of the MDS range (smaller)
        scale = (mds_range * 0.025) / max(typical_ellipse_size, 1e-6)
        print(f"Ellipse scale factor: {scale:.3f}")
        
        for i in range(len(X_mds_fisher)):
            G_i = G_2d[i]
            eigenvals, eigenvecs = np.linalg.eigh(G_i)
            eigenvals = np.maximum(eigenvals, 1e-10)
            idx = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            a = np.sqrt(eigenvals[0]) * scale
            b = np.sqrt(eigenvals[1]) * scale
            angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
            
            x_ellipse = a * np.cos(theta)
            y_ellipse = b * np.sin(theta)
            
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            x_rot = x_ellipse * cos_angle - y_ellipse * sin_angle
            y_rot = x_ellipse * sin_angle + y_ellipse * cos_angle
            
            x_final = x_rot + X_mds_fisher[i, 0]
            y_final = y_rot + X_mds_fisher[i, 1]
            
            label = labels[i]
            color = label_to_color[label]
            # Fill ellipse with color, then draw border
            plt.fill(x_final, y_final, color=color, alpha=0.2)
            plt.plot(x_final, y_final, color=color, alpha=0.6, linewidth=0.5)
        
        # Draw edges
        for edge in G_fisher.edges():
            i, j = edge
            if i < len(X_mds_fisher) and j < len(X_mds_fisher):
                plt.plot([X_mds_fisher[i, 0], X_mds_fisher[j, 0]], 
                        [X_mds_fisher[i, 1], X_mds_fisher[j, 1]], 
                        'k-', alpha=0.3, linewidth=0.5)
        
        # Draw nodes with labels
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(X_mds_fisher[mask, 0], X_mds_fisher[mask, 1], s=30, 
                       c=[label_to_color[label]], alpha=0.8, 
                       label=f"{label} ({label_counts.get(label, 0)})", 
                       edgecolors='black', linewidth=0.5)
        
        plt.title("GPT-2 Latent Graph (MDS with Fisher Metric)")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(fisher_out, dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved:", fisher_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    args = parser.parse_args()
    run(args.config)

