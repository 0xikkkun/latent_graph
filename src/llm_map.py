"""
Phase 2 skeleton: Multiple LLM graph comparison.
- Compute Graph Edit Distance (GED) matrix between graphs
- MDS embedding for LLM map
Implementation TBD.
"""

import argparse
from pathlib import Path
import numpy as np
from sklearn.manifold import MDS


def run(graph_paths, out_dir: str, seed: int = 42):
    # Placeholder: loads per-model geodesic distances, then computes inter-model distances
    # TODO: replace with GED between graphs
    models = []
    feats = []
    for p in graph_paths:
        p = Path(p)
        models.append(p.stem)
        # naive feature: mean shortest-path (placeholder)
        D = np.load(p)
        feats.append(float(np.mean(D)))
    feats = np.array(feats)[:, None]

    # pairwise L2 distance on placeholder feature
    M = len(models)
    dist = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(M):
            dist[i, j] = float(abs(feats[i, 0] - feats[j, 0]))

    X = MDS(n_components=2, dissimilarity="precomputed", random_state=seed).fit_transform(dist)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ged_matrix.npy", dist)
    np.save(out / "llm_map_mds.npy", X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", nargs="+", help="Paths to per-model geodesic distance .npy files")
    parser.add_argument("--out_dir", type=str, default="artifacts_gpt2/llm_map")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.graphs, args.out_dir, args.seed)
