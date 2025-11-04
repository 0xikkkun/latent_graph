import argparse
import pickle
import json
from pathlib import Path
import numpy as np
import networkx as nx
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


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


def run(config_path: str, show_curvature: bool = False, show_labels: bool = True) -> None:
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

    # Calculate total samples and num_datasets from num_datasets and samples_per_dataset
    # This is used for file naming and parameter display
    if "num_datasets" in cfg and "samples_per_dataset" in cfg:
        num_datasets = int(cfg["num_datasets"])
        samples_per_dataset = int(cfg["samples_per_dataset"])
        total_samples = num_datasets * samples_per_dataset
    elif "dataset_samples" in cfg and cfg["dataset_samples"]:
        # Fallback for backward compatibility (old config files with dataset_samples)
        total_samples = sum(cfg["dataset_samples"].values())
        num_datasets = len(cfg["dataset_samples"])
    else:
        # Fallback for backward compatibility (old config files)
        total_samples = cfg.get("num_samples", 100)
        num_datasets = cfg.get("num_datasets", 10)
    
    # Create comparison plot
    # Save to plot_latest directory
    plot_latest_dir = base / "plot_latest"
    plot_latest_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_out = plot_latest_dir / f"gpt2_comparison_fisher_vs_euclidean_ns{total_samples}_nd{num_datasets}_k{cfg['knn_k']}.png"
    
    # Load 2D Fisher metrics
    metrics_2d_path = base / "metrics_2d" / "G_theta_2d.npy"
    G_2d = None
    if metrics_2d_path.exists():
        G_2d = np.load(metrics_2d_path)
        print(f"Loaded G_2d with shape: {G_2d.shape}")
    else:
        print("Warning: G_theta_2d.npy not found. Ellipses will not be displayed.")
    
    # Load scalar curvature for coloring (load early so it's available for both plots)
    curvature_2d_path = base / cfg["curvature_2d_dir"] / "scalar_curvature_2d.npy"
    kappa_2d = None
    if curvature_2d_path.exists():
        kappa_2d = np.load(curvature_2d_path)
        print(f"Loaded scalar curvature with shape: {kappa_2d.shape}")
    else:
        print("Warning: scalar_curvature_2d.npy not found. Run curvature_2d.py first.")
        if show_curvature:
            print("Note: --show-curvature is enabled but curvature data is not available.")

    # Create color mapping for labels
    if len(unique_labels) > 1:
        # Create color map
        colors = cm.tab10(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        # Create comparison plot
        plt.figure(figsize=(20, 8))
        
        # Add parameter info as suptitle
        # Calculate total samples from dataset_samples if available
        if "dataset_samples" in cfg and cfg["dataset_samples"]:
            total_samples = sum(cfg["dataset_samples"].values())
            num_datasets = len(cfg["dataset_samples"])
            params_text = f"Parameters: total_samples={total_samples}, num_datasets={num_datasets}, knn_k={cfg['knn_k']}"
        else:
            # Fallback for backward compatibility
            total_samples = cfg.get("num_samples", 100)
            num_datasets = cfg.get("num_datasets", 10)
            params_text = f"Parameters: total_samples={total_samples}, num_datasets={num_datasets}, knn_k={cfg['knn_k']}"
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
        
        # Draw nodes with labels (or curvature coloring if available)
        if kappa_2d is not None and show_curvature:
            # Curvature coloring: red (negative/low) to blue (positive/high)
            # Use percentiles to set scale, outliers will be shown in black
            kappa_p2_5 = np.percentile(kappa_2d, 2.5)
            kappa_p97_5 = np.percentile(kappa_2d, 97.5)
            kappa_vmin = kappa_p2_5
            kappa_vmax = kappa_p97_5
            
            normal_mask = (kappa_2d >= kappa_vmin) & (kappa_2d <= kappa_vmax)
            outlier_low_mask = kappa_2d < kappa_vmin  # 小さい外れ値
            outlier_high_mask = kappa_2d > kappa_vmax  # 大きい外れ値
            
            # Plot normal values with colormap
            if np.any(normal_mask):
                scatter = plt.scatter(X_mds_fisher[normal_mask, 0], X_mds_fisher[normal_mask, 1], s=50,
                                   c=kappa_2d[normal_mask], cmap='RdYlBu_r', alpha=0.9,
                                   edgecolors='black', linewidth=0.8, vmin=kappa_vmin, vmax=kappa_vmax)
                cbar = plt.colorbar(scatter, label='Scalar Curvature')
                cbar.set_label('Scalar Curvature', rotation=270, labelpad=20)
            
            # Plot small outliers in black
            if np.any(outlier_low_mask):
                plt.scatter(X_mds_fisher[outlier_low_mask, 0], X_mds_fisher[outlier_low_mask, 1], s=50,
                           c='black', alpha=0.9, edgecolors='black', linewidth=0.8)
            
            # Plot large outliers in white
            if np.any(outlier_high_mask):
                plt.scatter(X_mds_fisher[outlier_high_mask, 0], X_mds_fisher[outlier_high_mask, 1], s=50,
                           c='white', alpha=0.9, edgecolors='black', linewidth=0.8)
        elif show_labels:
            # Label-based coloring
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                plt.scatter(X_mds_fisher[mask, 0], X_mds_fisher[mask, 1], s=30, 
                           c=[label_to_color[label]], alpha=0.8, 
                           label=f"{label} ({label_counts.get(label, 0)})", 
                           edgecolors='black', linewidth=0.5)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # No coloring - use single color
            plt.scatter(X_mds_fisher[:, 0], X_mds_fisher[:, 1], s=30, 
                       c='gray', alpha=0.8, 
                       edgecolors='black', linewidth=0.5)
        
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
    # Save to plot_latest directory (reuse total_samples and num_datasets calculated above)
    fisher_out = plot_latest_dir / f"gpt2_fisher_metric_ns{total_samples}_nd{num_datasets}_k{cfg['knn_k']}.png"
    if len(unique_labels) > 1 and G_2d is not None:
        plt.figure(figsize=(12, 10))
        
        # Add parameter info as suptitle (reuse total_samples and num_datasets calculated above)
        params_text = f"Parameters: total_samples={total_samples}, num_datasets={num_datasets}, knn_k={cfg['knn_k']}"
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
            # Ellipse color: always use label color (or gray), not curvature color
            if show_labels:
                color = label_to_color[label]
                plt.fill(x_final, y_final, color=color, alpha=0.2)
                plt.plot(x_final, y_final, color=color, alpha=0.6, linewidth=0.5)
            else:
                # Gray ellipses when labels are not shown
                plt.fill(x_final, y_final, color='gray', alpha=0.1)
                plt.plot(x_final, y_final, color='gray', alpha=0.3, linewidth=0.5)
        
        # Draw edges
        for edge in G_fisher.edges():
            i, j = edge
            if i < len(X_mds_fisher) and j < len(X_mds_fisher):
                plt.plot([X_mds_fisher[i, 0], X_mds_fisher[j, 0]], 
                        [X_mds_fisher[i, 1], X_mds_fisher[j, 1]], 
                        'k-', alpha=0.3, linewidth=0.5)
        
        # Draw nodes with labels (or curvature coloring if available)
        if kappa_2d is not None and show_curvature:
            # Curvature coloring: red (negative/low) to blue (positive/high)
            # Use percentiles to set scale, outliers will be shown in black
            kappa_p2_5 = np.percentile(kappa_2d, 2.5)
            kappa_p97_5 = np.percentile(kappa_2d, 97.5)
            kappa_vmin = kappa_p2_5
            kappa_vmax = kappa_p97_5
            
            normal_mask = (kappa_2d >= kappa_vmin) & (kappa_2d <= kappa_vmax)
            outlier_low_mask = kappa_2d < kappa_vmin  # 小さい外れ値
            outlier_high_mask = kappa_2d > kappa_vmax  # 大きい外れ値
            
            # Plot normal values with colormap
            if np.any(normal_mask):
                scatter = plt.scatter(X_mds_fisher[normal_mask, 0], X_mds_fisher[normal_mask, 1], s=50,
                                   c=kappa_2d[normal_mask], cmap='RdYlBu_r', alpha=0.9,
                                   edgecolors='black', linewidth=0.8, vmin=kappa_vmin, vmax=kappa_vmax)
                cbar = plt.colorbar(scatter, label='Scalar Curvature')
                cbar.set_label('Scalar Curvature', rotation=270, labelpad=20)
            
            # Plot small outliers in black
            if np.any(outlier_low_mask):
                plt.scatter(X_mds_fisher[outlier_low_mask, 0], X_mds_fisher[outlier_low_mask, 1], s=50,
                           c='black', alpha=0.9, edgecolors='black', linewidth=0.8)
            
            # Plot large outliers in white
            if np.any(outlier_high_mask):
                plt.scatter(X_mds_fisher[outlier_high_mask, 0], X_mds_fisher[outlier_high_mask, 1], s=50,
                           c='white', alpha=0.9, edgecolors='black', linewidth=0.8)
        elif show_labels:
            # Label-based coloring
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                plt.scatter(X_mds_fisher[mask, 0], X_mds_fisher[mask, 1], s=30, 
                           c=[label_to_color[label]], alpha=0.8, 
                           label=f"{label} ({label_counts.get(label, 0)})", 
                           edgecolors='black', linewidth=0.5)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # No coloring - use single color
            plt.scatter(X_mds_fisher[:, 0], X_mds_fisher[:, 1], s=30, 
                       c='gray', alpha=0.8, 
                       edgecolors='black', linewidth=0.5)
        
        plt.title("GPT-2 Latent Graph (MDS with Fisher Metric)")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.tight_layout()
        
        plt.savefig(fisher_out, dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved:", fisher_out)
        
        # Also save curvature-only plot if curvature is available and shown
        if kappa_2d is not None and show_curvature:
            # Save curvature plot to plot_latest directory (reuse total_samples and num_datasets calculated above)
            curvature_out = plot_latest_dir / f"gpt2_curvature_ns{total_samples}_nd{num_datasets}_k{cfg['knn_k']}.png"
            plt.figure(figsize=(12, 10))
            
            # Add parameter info as suptitle (reuse total_samples and num_datasets calculated above)
            params_text = f"Parameters: total_samples={total_samples}, num_datasets={num_datasets}, knn_k={cfg['knn_k']}"
            plt.suptitle(params_text, fontsize=12, y=0.98)
            
            # Draw edges first
            for edge in G_fisher.edges():
                i, j = edge
                if i < len(X_mds_fisher) and j < len(X_mds_fisher):
                    plt.plot([X_mds_fisher[i, 0], X_mds_fisher[j, 0]], 
                            [X_mds_fisher[i, 1], X_mds_fisher[j, 1]], 
                            'k-', alpha=0.3, linewidth=0.5)
            
            # Curvature coloring with outlier handling
            kappa_p2_5 = np.percentile(kappa_2d, 2.5)
            kappa_p97_5 = np.percentile(kappa_2d, 97.5)
            kappa_vmin = kappa_p2_5
            kappa_vmax = kappa_p97_5
            
            normal_mask = (kappa_2d >= kappa_vmin) & (kappa_2d <= kappa_vmax)
            outlier_low_mask = kappa_2d < kappa_vmin  # 小さい外れ値
            outlier_high_mask = kappa_2d > kappa_vmax  # 大きい外れ値
            
            # Plot normal values with colormap
            if np.any(normal_mask):
                scatter = plt.scatter(X_mds_fisher[normal_mask, 0], X_mds_fisher[normal_mask, 1], s=50,
                                   c=kappa_2d[normal_mask], cmap='RdYlBu_r', alpha=0.9,
                                   edgecolors='black', linewidth=0.8, vmin=kappa_vmin, vmax=kappa_vmax)
                cbar = plt.colorbar(scatter, label='Scalar Curvature')
                cbar.set_label('Scalar Curvature', rotation=270, labelpad=20)
            
            # Plot small outliers in black
            if np.any(outlier_low_mask):
                plt.scatter(X_mds_fisher[outlier_low_mask, 0], X_mds_fisher[outlier_low_mask, 1], s=50,
                           c='black', alpha=0.9, edgecolors='black', linewidth=0.8, label='Outliers (low)')
            
            # Plot large outliers in white
            if np.any(outlier_high_mask):
                plt.scatter(X_mds_fisher[outlier_high_mask, 0], X_mds_fisher[outlier_high_mask, 1], s=50,
                           c='white', alpha=0.9, edgecolors='black', linewidth=0.8, label='Outliers (high)')
            
            plt.title("GPT-2 Latent Graph - Scalar Curvature Visualization")
            plt.xlabel("Dim 1")
            plt.ylabel("Dim 2")
            if np.any(outlier_low_mask) or np.any(outlier_high_mask):
                plt.legend()
            plt.tight_layout()
            
            plt.savefig(curvature_out, dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved curvature plot:", curvature_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    parser.add_argument("--show-curvature", action="store_true", help="Show curvature as color")
    args = parser.parse_args()
    run(args.config, show_curvature=args.show_curvature)

