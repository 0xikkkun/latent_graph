import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def run(config_path: str) -> None:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg["artifacts_dir"])
    
    # Load eta embeddings
    eta_path = base / cfg["embeddings_dir"] / "eta.npy"
    eta = np.load(eta_path)

    # Load metadata with labels
    meta_path = base / cfg["embeddings_dir"] / "meta.json"
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    labels = meta.get("labels", [])
    unique_labels = meta.get("unique_labels", [])
    label_counts = meta.get("label_counts", {})

    # t-SNE with cosine distance
    cosine_dist = cosine_distances(eta)
    tsne = TSNE(n_components=2, metric="precomputed", init="random", random_state=int(cfg["seed"]))
    X = tsne.fit_transform(cosine_dist)

    # Create output directory
    plot_out = base / cfg["plots_dir"] / "gpt2_tsne.png"
    plot_out.parent.mkdir(parents=True, exist_ok=True)

    # Create color mapping for labels
    if len(unique_labels) > 1:
        # Create color map
        colors = cm.tab10(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        plt.figure(figsize=(12, 10))
        
        # Draw nodes with labels
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(X[mask, 0], X[mask, 1], s=30, c=[label_to_color[label]], 
                       alpha=0.8, label=f"{label} ({label_counts.get(label, 0)})", 
                       edgecolors='black', linewidth=0.5)
        
        plt.title("GPT-2 Latent Space (t-SNE with Cosine Distance) - Labeled by Data Source")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    else:
        # Fallback to single color if only one label
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], s=30, c="tab:blue", alpha=0.9, 
                   edgecolors='black', linewidth=0.5)
        plt.title("GPT-2 Latent Space (t-SNE with Cosine Distance)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()

    plt.savefig(plot_out, dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved:", plot_out)
    print(f"Labels found: {unique_labels}")
    print(f"Label distribution: {label_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    args = parser.parse_args()
    run(args.config)
