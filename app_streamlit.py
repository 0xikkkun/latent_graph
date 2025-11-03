import streamlit as st
import yaml
import subprocess
import json
import pickle
from pathlib import Path
import numpy as np
import networkx as nx
from sklearn.manifold import MDS
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(page_title="LLM Latent Space Visualization", layout="wide")

# Title
st.title("🎯 LLM潜在空間の幾何学的分析")


# Sidebar for parameters
st.sidebar.header("📝 パラメータ設定")

num_samples = st.sidebar.slider("num_samples", min_value=10, max_value=500, value=100, step=10)
num_datasets = st.sidebar.slider("num_datasets", min_value=1, max_value=12, value=10, step=1)
knn_k = st.sidebar.slider("knn_k", min_value=3, max_value=20, value=10, step=1)

dataset_options = ["multi_source", "newsgroup", "JeanKaddour/minipile"]
dataset_name = st.sidebar.selectbox("dataset.name", dataset_options, index=0)

run_button = st.sidebar.button("🚀 パイプライン実行", type="primary")

# Display options
st.sidebar.header("🎨 表示オプション")
show_edges = st.sidebar.checkbox("エッジを表示", value=True)
show_ellipses = st.sidebar.checkbox("楕円を表示", value=True)


def update_config(num_samples, num_datasets, knn_k, dataset_name):
    """Update config.yaml with new parameters"""
    config_path = Path("src/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    config["num_samples"] = num_samples
    config["num_datasets"] = num_datasets
    config["knn_k"] = knn_k
    config["dataset"]["name"] = dataset_name
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return config


def run_pipeline(progress_bar, progress_text):
    """Run the complete pipeline"""
    scripts = [
        ("1/5: データ抽出中...", "src/extract.py"),
        ("2/5: Fisherグラフ構築中...", "src/graph.py"),
        ("3/5: Euclideanグラフ構築中...", "src/graph_euclidean.py"),
        ("4/5: Fisher計量2D射影中...", "src/project_metrics_2d.py"),
        ("5/5: 可視化生成中...", "src/compare_fisher_euclidean.py"),
    ]
    
    for progress_msg, script in scripts:
        progress_text.text(progress_msg)
        try:
            result = subprocess.run(
                ["python", script, "--config", "src/config.yaml"],
                capture_output=True,
                text=True,
                check=True
            )
            progress_bar.progress(scripts.index((progress_msg, script)) + 1)
        except subprocess.CalledProcessError as e:
            st.error(f"Error running {script}: {e.stderr}")
            return False
    
    return True


def load_data(config):
    """Load results from artifacts directory"""
    base = Path(config["artifacts_dir"])
    
    # Load graphs
    fisher_graph_path = base / config["graphs_dir"] / "gpt2.gpickle"
    euclidean_graph_path = base / config["graphs_dir"] / "gpt2_euclidean.gpickle"
    
    with open(fisher_graph_path, 'rb') as f:
        G_fisher = pickle.load(f)
    with open(euclidean_graph_path, 'rb') as f:
        G_euclidean = pickle.load(f)
    
    # Load metadata
    meta_path = base / config["embeddings_dir"] / "meta.json"
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Load texts from meta.json
    labels = meta.get("labels", [])
    unique_labels = meta.get("unique_labels", [])
    label_counts = meta.get("label_counts", {})
    texts = meta.get("texts", [])
    
    # Load 2D metrics
    metrics_2d_path = base / "metrics_2d" / "G_theta_2d.npy"
    G_2d = None
    if metrics_2d_path.exists():
        G_2d = np.load(metrics_2d_path)
    
    return G_fisher, G_euclidean, labels, unique_labels, label_counts, texts, G_2d


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


def compute_mds(G: nx.Graph, seed: int = 42):
    """Compute MDS embedding from graph"""
    D = _all_pairs_shortest_path_matrix(G)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed, max_iter=1000, eps=1e-9)
    X = mds.fit_transform(D)
    return X


def add_ellipses_to_plot(fig, X_mds, G_2d, labels, unique_labels, row=1, col=1, alpha=0.15):
    """Add Fisher metric ellipses to the plot"""
    import plotly.colors as pc
    colors = pc.qualitative.Set3[:len(unique_labels)]
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Number of points on ellipse
    n_ellipse_points = 50
    theta = np.linspace(0, 2*np.pi, n_ellipse_points)
    
    # Compute scale factor to make ellipses visible
    mds_range = np.max(X_mds) - np.min(X_mds)
    all_eigenvals = []
    for i in range(len(X_mds)):
        G_i = G_2d[i]
        eigenvals, _ = np.linalg.eigh(G_i)
        eigenvals = np.maximum(eigenvals, 1e-10)
        all_eigenvals.extend(np.sqrt(eigenvals))
    typical_ellipse_size = np.median(all_eigenvals)
    scale = (mds_range * 0.025) / max(typical_ellipse_size, 1e-6)
    
    for i in range(len(X_mds)):
        # Get 2D Fisher metric for this point
        G_i = G_2d[i]
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(G_i)
        # Ensure eigenvalues are positive
        eigenvals = np.maximum(eigenvals, 1e-10)
        # Sort by eigenvalues
        idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Scale by eigenvalues
        a = np.sqrt(eigenvals[0]) * scale  # major axis
        b = np.sqrt(eigenvals[1]) * scale  # minor axis
        
        # Rotation angle
        angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
        
        # Create ellipse in parametric form
        x_ellipse = a * np.cos(theta)
        y_ellipse = b * np.sin(theta)
        
        # Rotate
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        x_rot = x_ellipse * cos_angle - y_ellipse * sin_angle
        y_rot = x_ellipse * sin_angle + y_ellipse * cos_angle
        
        # Translate to point position
        x_final = x_rot + X_mds[i, 0]
        y_final = y_rot + X_mds[i, 1]
        
        # Get color for this label
        label = labels[i]
        color = label_to_color[label]
        # Add alpha for fill and border
        import re
        if color.startswith('rgb'):
            match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color)
            if match:
                r, g, b = match.groups()
                color_fill = f'rgba({r},{g},{b},0.2)'
                color_border = f'rgba({r},{g},{b},0.6)'
            else:
                color_fill = color
                color_border = color
        else:
            color_fill = color
            color_border = color
        
        # Add ellipse (filled)
        fig.add_trace(
            go.Scatter(
                x=x_final,
                y=y_final,
                mode='lines',
                fill='toself',
                fillcolor=color_fill,
                line=dict(color=color_border, width=0.5),
                hoverinfo='skip',
                showlegend=False
            ),
            row=row, col=col
        )


def create_distance_scatter_plot(D_fisher, D_euclidean, labels, unique_labels):
    """Create scatter plot of Fisher vs Euclidean distances"""
    import plotly.colors as pc
    colors = pc.qualitative.Set3[:len(unique_labels)]
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Get all pairwise distances (upper triangle to avoid duplicates)
    N = len(D_fisher)
    fisher_dists = []
    euclidean_dists = []
    point_labels = []
    
    for i in range(N):
        for j in range(i + 1, N):
            fisher_dists.append(D_fisher[i, j])
            euclidean_dists.append(D_euclidean[i, j])
            # Create label pair
            label_pair = f"{labels[i]} - {labels[j]}"
            point_labels.append(label_pair)
    
    # Normalize by mean
    mean_fisher = np.mean(fisher_dists)
    mean_euclidean = np.mean(euclidean_dists)
    
    fisher_dists_norm = [d / mean_fisher for d in fisher_dists]
    euclidean_dists_norm = [d / mean_euclidean for d in euclidean_dists]
    
    fig = go.Figure()
    
    # Plot all points
    fig.add_trace(go.Scatter(
        x=fisher_dists_norm,
        y=euclidean_dists_norm,
        mode='markers',
        marker=dict(size=4, color='rgba(0,0,0,0.3)'),
        text=point_labels,
        hovertemplate='Fisher (normalized): %{x:.4f}<br>Euclidean (normalized): %{y:.4f}<br>%{text}<extra></extra>',
        name='All pairs',
        showlegend=True
    ))
    
    # Add diagonal reference line
    max_val = max(max(fisher_dists_norm), max(euclidean_dists_norm))
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color='red', width=1, dash='dash'),
        name='y=x reference',
        showlegend=True
    ))
    
    fig.update_layout(
        title="Fisher Distance vs Euclidean Distance (Normalized by Mean)",
        xaxis_title="Fisher Distance (Normalized)",
        yaxis_title="Euclidean Distance (Normalized)",
        height=800,
        width=800,
        hovermode='closest',
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    return fig


def create_plotly_figure(X_fisher, X_euclidean, G_fisher, G_euclidean, labels, unique_labels, label_counts, G_2d=None, show_edges=True, show_ellipses=True):
    """Create Plotly comparison figure"""
    # Define colors for labels
    import plotly.colors as pc
    colors = pc.qualitative.Set3[:len(unique_labels)]
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("GPT-2 Latent Graph (MDS with Fisher Metric)", 
                       "GPT-2 Latent Graph (MDS with Euclidean Distance)"),
        horizontal_spacing=0.1
    )
    
    # Add Fisher metric ellipses FIRST (behind everything else)
    if G_2d is not None and show_ellipses:
        add_ellipses_to_plot(fig, X_fisher, G_2d, labels, unique_labels, row=1, col=1, alpha=0.15)
    
    # Plot Fisher metric graph
    if show_edges:
        for edge in G_fisher.edges():
            i, j = edge
            if i < len(X_fisher) and j < len(X_fisher):
                fig.add_trace(
                    go.Scatter(
                        x=[X_fisher[i, 0], X_fisher[j, 0]],
                        y=[X_fisher[i, 1], X_fisher[j, 1]],
                        mode='lines',
                        line=dict(color='rgba(0,0,0,0.3)', width=0.5),
                        hoverinfo='skip',
                        showlegend=False
                    ),
                    row=1, col=1
                )
    
    # Plot Euclidean graph
    if show_edges:
        for edge in G_euclidean.edges():
            i, j = edge
            if i < len(X_euclidean) and j < len(X_euclidean):
                fig.add_trace(
                    go.Scatter(
                        x=[X_euclidean[i, 0], X_euclidean[j, 0]],
                        y=[X_euclidean[i, 1], X_euclidean[j, 1]],
                        mode='lines',
                        line=dict(color='rgba(0,0,0,0.3)', width=0.5),
                        hoverinfo='skip',
                        showlegend=False
                    ),
                    row=1, col=2
                )
    
    # Plot nodes for Fisher metric
    for label in unique_labels:
        mask = np.array(labels) == label
        texts_preview = []
        for i in np.where(mask)[0]:
            if "texts" in st.session_state and i < len(st.session_state.texts):
                text = st.session_state.texts[i]
                texts_preview.append(text[:100] + "..." if len(text) > 500 else text)
            else:
                texts_preview.append("")
        
        hover_texts = [f"{label}<br>Text: {text}" for text in texts_preview]
        
        fig.add_trace(
            go.Scatter(
                x=X_fisher[mask, 0],
                y=X_fisher[mask, 1],
                mode='markers',
                marker=dict(size=8, color=label_to_color[label], line=dict(width=0.5, color='black')),
                name=f"{label} ({label_counts.get(label, 0)})",
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>',
                legendgroup="group1",
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Plot nodes for Euclidean
    for label in unique_labels:
        mask = np.array(labels) == label
        texts_preview = []
        for i in np.where(mask)[0]:
            if "texts" in st.session_state and i < len(st.session_state.texts):
                text = st.session_state.texts[i]
                texts_preview.append(text[:500] + "..." if len(text) > 500 else text)
            else:
                texts_preview.append("")
        
        hover_texts = [f"{label}<br>Text: {text}" for text in texts_preview]
        
        fig.add_trace(
            go.Scatter(
                x=X_euclidean[mask, 0],
                y=X_euclidean[mask, 1],
                mode='markers',
                marker=dict(size=8, color=label_to_color[label], line=dict(width=0.5, color='black')),
                name=f"{label} ({label_counts.get(label, 0)})",
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>',
                legendgroup="group2",
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=1.01)
    )
    
    fig.update_xaxes(title_text="Dim 1", row=1, col=1)
    fig.update_yaxes(title_text="Dim 2", row=1, col=1, scaleanchor="x1", scaleratio=1)
    fig.update_xaxes(title_text="Dim 1", row=1, col=2)
    fig.update_yaxes(title_text="Dim 2", row=1, col=2, scaleanchor="x2", scaleratio=1)
    
    return fig


def create_fisher_metric_plot(X_fisher, G_fisher, labels, unique_labels, label_counts, G_2d=None, show_edges=True, show_ellipses=True):
    """Create single Fisher metric plot with ellipses"""
    import plotly.colors as pc
    colors = pc.qualitative.Set3[:len(unique_labels)]
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Create single subplot
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("GPT-2 Latent Graph (MDS with Fisher Metric)"),
        horizontal_spacing=0.1
    )
    
    # Add Fisher metric ellipses FIRST (behind everything else)
    if G_2d is not None and show_ellipses:
        add_ellipses_to_plot(fig, X_fisher, G_2d, labels, unique_labels, row=1, col=1, alpha=0.15)
    
    # Plot Fisher metric graph
    if show_edges:
        for edge in G_fisher.edges():
            i, j = edge
            if i < len(X_fisher) and j < len(X_fisher):
                fig.add_trace(
                    go.Scatter(
                        x=[X_fisher[i, 0], X_fisher[j, 0]],
                        y=[X_fisher[i, 1], X_fisher[j, 1]],
                        mode='lines',
                        line=dict(color='rgba(0,0,0,0.3)', width=0.5),
                        hoverinfo='skip',
                        showlegend=False
                    ),
                    row=1, col=1
                )
    
    # Plot nodes for Fisher metric
    for label in unique_labels:
        mask = np.array(labels) == label
        texts_preview = []
        for i in np.where(mask)[0]:
            if "texts" in st.session_state and i < len(st.session_state.texts):
                text = st.session_state.texts[i]
                texts_preview.append(text[:500] + "..." if len(text) > 500 else text)
            else:
                texts_preview.append("")
        
        hover_texts = [f"{label}<br>Text: {text}" for text in texts_preview]
        
        fig.add_trace(
            go.Scatter(
                x=X_fisher[mask, 0],
                y=X_fisher[mask, 1],
                mode='markers',
                marker=dict(size=8, color=label_to_color[label], line=dict(width=0.5, color='black')),
                name=f"{label} ({label_counts.get(label, 0)})",
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>',
                legendgroup="group1",
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=1.01)
    )
    
    fig.update_xaxes(title_text="Dim 1", row=1, col=1)
    fig.update_yaxes(title_text="Dim 2", row=1, col=1, scaleanchor="x", scaleratio=1)
    
    return fig


# Main logic
if run_button:
    # Update config
    config = update_config(num_samples, num_datasets, knn_k, dataset_name)
    st.sidebar.success("Config updated!")
    
    # Progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # Run pipeline
    success = run_pipeline(progress_bar, progress_text)
    
    if success:
        progress_text.text("✅ 完了！")
        
        # Load and visualize
        try:
            G_fisher, G_euclidean, labels, unique_labels, label_counts, texts, G_2d = load_data(config)
            
            # Store texts in session state for plotly
            st.session_state.texts = texts
            
            # Compute MDS
            X_fisher = compute_mds(G_fisher, seed=config["seed"])
            X_euclidean = compute_mds(G_euclidean, seed=config["seed"])
            
            # Create comparison plot
            fig = create_plotly_figure(X_fisher, X_euclidean, G_fisher, G_euclidean, 
                                     labels, unique_labels, label_counts, G_2d, show_edges, show_ellipses)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create distance scatter plot
            D_fisher = _all_pairs_shortest_path_matrix(G_fisher)
            D_euclidean = _all_pairs_shortest_path_matrix(G_euclidean)
            
            scatter_fig = create_distance_scatter_plot(D_fisher, D_euclidean, labels, unique_labels)
            st.plotly_chart(scatter_fig, use_container_width=False)
            
            # Create Fisher metric plot with ellipses
            if G_2d is not None:
                st.subheader("🎯 Fisher Metric Visualization with Ellipses")
                fisher_fig = create_fisher_metric_plot(X_fisher, G_fisher, labels, unique_labels, label_counts, G_2d, show_edges, show_ellipses)
                st.plotly_chart(fisher_fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading/visualizing results: {e}")
    else:
        st.error("Pipeline execution failed. Please check the error messages above.")


# Try to load existing results
try:
    with open("src/config.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    
    base = Path(default_config["artifacts_dir"])
    fisher_graph_path = base / default_config["graphs_dir"] / "gpt2.gpickle"
    euclidean_graph_path = base / default_config["graphs_dir"] / "gpt2_euclidean.gpickle"
    meta_path = base / default_config["embeddings_dir"] / "meta.json"
    
    if fisher_graph_path.exists() and euclidean_graph_path.exists() and meta_path.exists():
        st.info("📊 既存の結果が見つかりました。可視化を読み込んでいます...")
        
        # Load data
        G_fisher, G_euclidean, labels, unique_labels, label_counts, texts, G_2d = load_data(default_config)
        
        # Store texts in session state for plotly
        st.session_state.texts = texts
        
        # Warn if texts are missing
        if not texts:
            st.warning("⚠️ 既存のデータにテキスト情報が含まれていません。新しいデータでパイプラインを実行してください。")
        
        # Compute MDS
        X_fisher = compute_mds(G_fisher, seed=default_config["seed"])
        X_euclidean = compute_mds(G_euclidean, seed=default_config["seed"])
        
        # Create comparison plot
        fig = create_plotly_figure(X_fisher, X_euclidean, G_fisher, G_euclidean, 
                                 labels, unique_labels, label_counts, G_2d, show_edges, show_ellipses)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create distance scatter plot
        D_fisher = _all_pairs_shortest_path_matrix(G_fisher)
        D_euclidean = _all_pairs_shortest_path_matrix(G_euclidean)
        
        scatter_fig = create_distance_scatter_plot(D_fisher, D_euclidean, labels, unique_labels)
        st.plotly_chart(scatter_fig, use_container_width=False)
        
        # Create Fisher metric plot with ellipses
        if G_2d is not None:
            st.subheader("🎯 Fisher Metric Visualization with Ellipses")
            fisher_fig = create_fisher_metric_plot(X_fisher, G_fisher, labels, unique_labels, label_counts, G_2d, show_edges, show_ellipses)
            st.plotly_chart(fisher_fig, use_container_width=True)
        
except Exception as e:
    if "run_button" not in locals():
        st.info("👈 左側のパラメータを設定して「パイプライン実行」ボタンを押してください。")

