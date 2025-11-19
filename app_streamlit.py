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

# Load current config to get default values
config_path = Path("src/config.yaml")
current_config = {}
if config_path.exists():
    with open(config_path, "r") as f:
        current_config = yaml.safe_load(f) or {}

# Dataset selection
dataset_options = ["multi_source", "newsgroup", "JeanKaddour/minipile"]
current_dataset = current_config.get("dataset", {}).get("name", "multi_source")
dataset_index = dataset_options.index(current_dataset) if current_dataset in dataset_options else 0
dataset_name = st.sidebar.selectbox("dataset.name", dataset_options, index=dataset_index)

# Dataset configuration (only for multi_source and minipile)
num_datasets = None
samples_per_dataset = None
if dataset_name in ["multi_source", "JeanKaddour/minipile"]:
    num_datasets = st.sidebar.slider(
        "num_datasets",
        min_value=1,
        max_value=12,
        value=current_config.get("num_datasets", 10),
        step=1
    )
    samples_per_dataset = st.sidebar.slider(
        "samples_per_dataset",
        min_value=1,
        max_value=100,
        value=current_config.get("samples_per_dataset", 10),
        step=1
    )
    total = num_datasets * samples_per_dataset
    st.sidebar.info(f"合計サンプル数: {total}")

knn_k = st.sidebar.slider("knn_k", min_value=3, max_value=20, value=current_config.get("knn_k", 10), step=1)

run_button = st.sidebar.button("🚀 パイプライン実行", type="primary")

# Display options
st.sidebar.header("🎨 表示オプション")
show_edges = st.sidebar.checkbox("エッジを表示", value=True)
show_ellipses = st.sidebar.checkbox("楕円を表示", value=True)
show_labels = st.sidebar.checkbox("ラベルを色で表示", value=True)
show_curvature = st.sidebar.checkbox("曲率を色で表示", value=False)

# Export options
st.sidebar.header("📤 エクスポート")
export_html_button = st.sidebar.button("📄 HTMLとしてエクスポート", type="secondary")


def export_plots_to_html(config, output_dir="html_export", show_edges=True, show_ellipses=True, show_labels=True, show_curvature=False):
    """Export all Plotly plots to HTML files"""
    from datetime import datetime
    import plotly.io as pio
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Load data
        G_fisher, G_euclidean, labels, unique_labels, label_counts, texts, G_2d, kappa_2d = load_data(config)
        
        # Compute MDS
        X_fisher = compute_mds(G_fisher, seed=config["seed"])
        X_euclidean = compute_mds(G_euclidean, seed=config["seed"])
        
        # Create all plots
        fig_comparison = create_plotly_figure(X_fisher, X_euclidean, G_fisher, G_euclidean, 
                                             labels, unique_labels, label_counts, G_2d, kappa_2d, 
                                             show_edges, show_ellipses, show_labels, show_curvature)
        
        D_fisher = _all_pairs_shortest_path_matrix(G_fisher)
        D_euclidean = _all_pairs_shortest_path_matrix(G_euclidean)
        scatter_fig = create_distance_scatter_plot(D_fisher, D_euclidean, labels, unique_labels)
        
        # Export comparison plot
        comparison_html = output_path / f"comparison_{timestamp}.html"
        fig_comparison.write_html(str(comparison_html))
        
        # Export scatter plot
        scatter_html = output_path / f"distance_scatter_{timestamp}.html"
        scatter_fig.write_html(str(scatter_html))
        
        # Export Fisher metric plot if available
        if G_2d is not None:
            fisher_fig = create_fisher_metric_plot(X_fisher, G_fisher, labels, unique_labels, label_counts, 
                                                  G_2d, kappa_2d, show_edges, show_ellipses, show_labels, show_curvature)
            fisher_html = output_path / f"fisher_metric_{timestamp}.html"
            fisher_fig.write_html(str(fisher_html))
        
        # Export histogram if available
        if kappa_2d is not None:
            import plotly.colors as pc
            colors = pc.qualitative.Set3[:len(unique_labels)]
            label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
            
            fig_hist = go.Figure()
            for label in unique_labels:
                mask = np.array(labels) == label
                kappa_label = kappa_2d[mask]
                fig_hist.add_trace(go.Histogram(
                    x=kappa_label,
                    nbinsx=50,
                    name=f"{label} ({label_counts.get(label, 0)})",
                    marker_color=label_to_color[label],
                    opacity=1.0,
                    marker_line=dict(color='white', width=0.5),
                    hovertemplate=f'Label: {label}<br>Range: %{{x}}<br>Count: %{{y}}<extra></extra>'
                ))
            
            kappa_mean = np.mean(kappa_2d)
            kappa_variance = np.var(kappa_2d)
            
            fig_hist.update_layout(
                title=dict(text="Scalar Curvature Distribution", font=dict(size=18), x=0.5, xanchor='center'),
                xaxis=dict(title="Scalar Curvature (κ)", showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                yaxis=dict(title="Frequency", showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                height=500,
                barmode='stack',
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=True,
                annotations=[
                    dict(text=f"Mean: {kappa_mean:.4f}, Variance: {kappa_variance:.4f}",
                         xref="paper", yref="paper", x=0.5, y=1.03, showarrow=False)
                ]
            )
            
            fig_hist.add_vline(x=kappa_mean, line_dash="dash", line_color="black", line_width=1.5, opacity=0.5)
            
            hist_html = output_path / f"curvature_histogram_{timestamp}.html"
            fig_hist.write_html(str(hist_html))
        
        # Export MDS eigenvalues
        eigenvals_fisher, explained_ratio_fisher = compute_mds_eigenvalues(G_fisher)
        eigenvals_euclidean, explained_ratio_euclidean = compute_mds_eigenvalues(G_euclidean)
        
        n_show = min(20, len(eigenvals_fisher), len(eigenvals_euclidean))
        fig_eigen = make_subplots(
            rows=1, cols=2,
            subplot_titles=("MDS Eigenvalues (Fisher Metric)", "MDS Eigenvalues (Euclidean Distance)"),
            horizontal_spacing=0.15
        )
        
        fig_eigen.add_trace(
            go.Bar(x=list(range(1, n_show + 1)), y=eigenvals_fisher[:n_show], name="Fisher", marker_color='steelblue',
                  text=[f'{r:.3f}' for r in explained_ratio_fisher[:n_show]], textposition='outside'),
            row=1, col=1
        )
        
        fig_eigen.add_trace(
            go.Bar(x=list(range(1, n_show + 1)), y=eigenvals_euclidean[:n_show], name="Euclidean", marker_color='coral',
                  text=[f'{r:.3f}' for r in explained_ratio_euclidean[:n_show]], textposition='outside'),
            row=1, col=2
        )
        
        fig_eigen.update_xaxes(title_text="Dimension", row=1, col=1)
        fig_eigen.update_yaxes(title_text="Eigenvalue", row=1, col=1)
        fig_eigen.update_xaxes(title_text="Dimension", row=1, col=2)
        fig_eigen.update_yaxes(title_text="Eigenvalue", row=1, col=2)
        fig_eigen.update_layout(height=500, showlegend=False)
        
        eigen_html = output_path / f"mds_eigenvalues_{timestamp}.html"
        fig_eigen.write_html(str(eigen_html))
        
        # Create index HTML file
        index_html = output_path / f"index_{timestamp}.html"
        
        # Read Plotly HTML content and extract plot divs and scripts
        def extract_plot_content(html_file):
            """Extract the plot div and scripts from Plotly HTML file"""
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
                import re
                # Extract everything between <body> and </body>
                body_match = re.search(r'<body[^>]*>(.*?)</body>', content, re.DOTALL)
                body_content = body_match.group(1) if body_match else ""
                
                # Extract all script tags (including Plotly library)
                script_pattern = r'<script[^>]*>.*?</script>'
                scripts = re.findall(script_pattern, content, re.DOTALL)
                scripts_content = "\n".join(scripts)
                
                return body_content, scripts_content
        
        # Extract plot contents
        comparison_body, comparison_scripts = extract_plot_content(comparison_html)
        scatter_body, scatter_scripts = extract_plot_content(scatter_html)
        fisher_body, fisher_scripts = ("", "") if G_2d is None else extract_plot_content(fisher_html)
        hist_body, hist_scripts = ("", "") if kappa_2d is None else extract_plot_content(hist_html)
        eigen_body, eigen_scripts = extract_plot_content(eigen_html)
        
        # Combine all scripts (Plotly library is the same, so we only need it once)
        # Extract unique script content (avoid duplicates of Plotly library)
        all_scripts_set = set()
        all_scripts_list = []
        
        # Add scripts from each plot, avoiding duplicates
        for scripts in [comparison_scripts, scatter_scripts, fisher_scripts, hist_scripts, eigen_scripts]:
            if scripts:
                # Split by script tags and add unique ones
                import re
                script_matches = re.findall(r'<script[^>]*>.*?</script>', scripts, re.DOTALL)
                for script in script_matches:
                    # Use a hash or first 100 chars to identify duplicates
                    script_id = script[:200] if len(script) > 200 else script
                    if script_id not in all_scripts_set:
                        all_scripts_set.add(script_id)
                        all_scripts_list.append(script)
        
        all_scripts = "\n".join(all_scripts_list)
        
        with open(index_html, 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM潜在空間の幾何学的分析 - {timestamp}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .plot-container {{
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plot-wrapper {{
            width: 100%;
            min-height: 600px;
            margin: 10px 0;
        }}
        .info {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }}
    </style>
    {all_scripts}
</head>
<body>
    <h1>🎯 LLM潜在空間の幾何学的分析</h1>
    <div class="info">
        <p><strong>エクスポート日時:</strong> {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}</p>
        <p><strong>設定:</strong> {config.get('dataset', {}).get('name', 'N/A')}, k={config.get('knn_k', 'N/A')}</p>
    </div>
    
    <div class="plot-container">
        <h2>1. Fisher計量 vs ユークリッド距離の比較</h2>
        <div class="plot-wrapper">
            {comparison_body}
        </div>
    </div>
    
    <div class="plot-container">
        <h2>2. 距離散布図</h2>
        <div class="plot-wrapper">
            {scatter_body}
        </div>
    </div>
""")
            if G_2d is not None:
                f.write(f"""    <div class="plot-container">
        <h2>3. Fisher計量可視化（楕円付き）</h2>
        <div class="plot-wrapper">
            {fisher_body}
        </div>
    </div>
""")
            if kappa_2d is not None:
                f.write(f"""    <div class="plot-container">
        <h2>4. スカラー曲率ヒストグラム</h2>
        <div class="plot-wrapper">
            {hist_body}
        </div>
    </div>
""")
            f.write(f"""    <div class="plot-container">
        <h2>5. MDS固有値</h2>
        <div class="plot-wrapper">
            {eigen_body}
        </div>
    </div>
</body>
</html>""")
        
        return output_path, [
            ("比較グラフ", comparison_html),
            ("距離散布図", scatter_html),
            ("Fisher計量", fisher_html if G_2d is not None else None),
            ("曲率ヒストグラム", hist_html if kappa_2d is not None else None),
            ("MDS固有値", eigen_html),
            ("インデックス", index_html)
        ]
    except Exception as e:
        raise Exception(f"エクスポート中にエラーが発生しました: {e}")


def update_config(knn_k, dataset_name, num_datasets=None, samples_per_dataset=None):
    """Update config.yaml with new parameters"""
    config_path = Path("src/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    config["knn_k"] = knn_k
    config["dataset"]["name"] = dataset_name
    
    # Update dataset parameters if provided
    if dataset_name in ["multi_source", "JeanKaddour/minipile"]:
        if num_datasets is not None:
            config["num_datasets"] = num_datasets
        if samples_per_dataset is not None:
            config["samples_per_dataset"] = samples_per_dataset
    else:
        # Remove dataset parameters if not applicable
        config.pop("num_datasets", None)
        config.pop("samples_per_dataset", None)
    
    # Remove deprecated parameters
    config.pop("num_samples", None)
    config.pop("dataset_samples", None)
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    return config


def run_pipeline(progress_bar, progress_text):
    """Run the complete pipeline"""
    scripts = [
        ("1/6: データ抽出中...", "src/extract.py"),
        ("2/6: Fisherグラフ構築中...", "src/graph.py"),
        ("3/6: Euclideanグラフ構築中...", "src/graph_euclidean.py"),
        ("4/6: Fisher計量2D射影中...", "src/project_metrics_2d.py"),
        ("5/6: 曲率計算中...", "src/curvature_2d.py"),
        ("6/6: 可視化生成中...", "src/compare_fisher_euclidean.py"),
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
    
    # Load scalar curvature
    curvature_2d_path = base / config["curvature_2d_dir"] / "scalar_curvature_2d.npy"
    kappa_2d = None
    if curvature_2d_path.exists():
        kappa_2d = np.load(curvature_2d_path)
    
    return G_fisher, G_euclidean, labels, unique_labels, label_counts, texts, G_2d, kappa_2d


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


def compute_mds_eigenvalues(G: nx.Graph):
    """Compute MDS eigenvalues from distance matrix"""
    D = _all_pairs_shortest_path_matrix(G)
    N = D.shape[0]
    
    # Classical MDS: convert distance matrix to inner product matrix
    # Center the distance matrix
    D_sq = D ** 2
    row_means = D_sq.mean(axis=1, keepdims=True)
    col_means = D_sq.mean(axis=0, keepdims=True)
    grand_mean = D_sq.mean()
    
    # Double centering
    B = -0.5 * (D_sq - row_means - col_means + grand_mean)
    
    # Compute eigenvalues (keep original order from eigvalsh - ascending order)
    eigenvals = np.linalg.eigvalsh(B)
    eigenvals = eigenvals[eigenvals > 0]  # Keep only positive eigenvalues
    
    # Calculate explained variance ratio
    total_variance = eigenvals.sum()
    explained_ratio = eigenvals / total_variance if total_variance > 0 else eigenvals
    
    return eigenvals, explained_ratio


def add_ellipses_to_plot(fig, X_mds, G_2d, labels, unique_labels, row=1, col=1, alpha=0.15, use_gray=False):
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
        if use_gray:
            color_fill = 'rgba(128,128,128,0.1)'
            color_border = 'rgba(128,128,128,0.3)'
        else:
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


def create_plotly_figure(X_fisher, X_euclidean, G_fisher, G_euclidean, labels, unique_labels, label_counts, G_2d=None, kappa_2d=None, show_edges=True, show_ellipses=True, show_labels=True, show_curvature=False):
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
        indices_list = []
        for i in np.where(mask)[0]:
            if "texts" in st.session_state and i < len(st.session_state.texts):
                text = st.session_state.texts[i]
                texts_preview.append(text[:100] + "..." if len(text) > 100 else text)
            else:
                texts_preview.append("")
            indices_list.append(i)
        
        # Build hover texts with metric matrix if available and ellipses are shown
        hover_texts = []
        for idx, text in zip(indices_list, texts_preview):
            hover_str = f"{label}<br>Text: {text}"
            if show_ellipses and G_2d is not None and idx < len(G_2d):
                G_i = G_2d[idx]
                hover_str += f"<br><br>2D Metric Matrix:<br>"
                hover_str += f"[{G_i[0,0]:.4f}, {G_i[0,1]:.4f}]<br>"
                hover_str += f"[{G_i[1,0]:.4f}, {G_i[1,1]:.4f}]"
            hover_texts.append(hover_str)
        
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
                texts_preview.append(text[:100] + "..." if len(text) > 100 else text)
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


def create_fisher_metric_plot(X_fisher, G_fisher, labels, unique_labels, label_counts, G_2d=None, kappa_2d=None, show_edges=True, show_ellipses=True, show_labels=True, show_curvature=False):
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
    # Ellipses are always colored by labels (or gray), not by curvature
    if G_2d is not None and show_ellipses:
        if show_labels:
            # Use label colors for ellipses
            add_ellipses_to_plot(fig, X_fisher, G_2d, labels, unique_labels, row=1, col=1, alpha=0.15)
        else:
            # Gray ellipses when labels are not shown
            add_ellipses_to_plot(fig, X_fisher, G_2d, labels, unique_labels, row=1, col=1, alpha=0.15, use_gray=True)
    
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
    if kappa_2d is not None and show_curvature:
        # Curvature coloring with outlier handling
        # Use percentiles to set scale, outliers will be shown in black
        kappa_p2_5 = np.percentile(kappa_2d, 2.5)
        kappa_p97_5 = np.percentile(kappa_2d, 97.5)
        kappa_vmin = kappa_p2_5
        kappa_vmax = kappa_p97_5
        
        # Create masks for normal values and outliers
        normal_mask = (kappa_2d >= kappa_vmin) & (kappa_2d <= kappa_vmax)
        outlier_low_mask = kappa_2d < kappa_vmin  # 小さい外れ値
        outlier_high_mask = kappa_2d > kappa_vmax  # 大きい外れ値
        normal_indices = np.where(normal_mask)[0]
        outlier_low_indices = np.where(outlier_low_mask)[0]
        outlier_high_indices = np.where(outlier_high_mask)[0]
        
        texts_preview = []
        for i in range(len(X_fisher)):
            if "texts" in st.session_state and i < len(st.session_state.texts):
                text = st.session_state.texts[i]
                texts_preview.append(text[:100] + "..." if len(text) > 100 else text)
            else:
                texts_preview.append("")
        
        # Build hover texts with curvature and metric matrix if available and ellipses are shown
        hover_texts = []
        for i, text in enumerate(texts_preview):
            hover_str = f"Curvature: {kappa_2d[i]:.4f}<br>Text: {text}"
            if show_ellipses and G_2d is not None and i < len(G_2d):
                G_i = G_2d[i]
                hover_str += f"<br><br>2D Metric Matrix:<br>"
                hover_str += f"[{G_i[0,0]:.4f}, {G_i[0,1]:.4f}]<br>"
                hover_str += f"[{G_i[1,0]:.4f}, {G_i[1,1]:.4f}]"
            hover_texts.append(hover_str)
        
        # Plot normal values with colormap
        if len(normal_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=X_fisher[normal_indices, 0],
                    y=X_fisher[normal_indices, 1],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=kappa_2d[normal_indices],
                        colorscale='RdYlBu_r',
                        cmin=kappa_vmin,
                        cmax=kappa_vmax,
                        colorbar=dict(title="Scalar Curvature", len=0.5, y=0.75),
                        line=dict(width=0.8, color='black')
                    ),
                    name="Curvature",
                    text=[hover_texts[i] for i in normal_indices],
                    hovertemplate='%{text}<extra></extra>',
                    legendgroup="group1",
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Plot small outliers in black
        if len(outlier_low_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=X_fisher[outlier_low_indices, 0],
                    y=X_fisher[outlier_low_indices, 1],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='black',
                        line=dict(width=0.8, color='black')
                    ),
                    name="Curvature (outliers - low)",
                    text=[hover_texts[i] for i in outlier_low_indices],
                    hovertemplate='%{text}<extra></extra>',
                    legendgroup="group1",
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Plot large outliers in white
        if len(outlier_high_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=X_fisher[outlier_high_indices, 0],
                    y=X_fisher[outlier_high_indices, 1],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='white',
                        line=dict(width=0.8, color='black')
                    ),
                    name="Curvature (outliers - high)",
                    text=[hover_texts[i] for i in outlier_high_indices],
                    hovertemplate='%{text}<extra></extra>',
                    legendgroup="group1",
                    showlegend=False
                ),
                row=1, col=1
            )
    elif show_labels:
        # Label-based coloring
        for label in unique_labels:
            mask = np.array(labels) == label
            texts_preview = []
            indices_list = []
            for i in np.where(mask)[0]:
                if "texts" in st.session_state and i < len(st.session_state.texts):
                    text = st.session_state.texts[i]
                    texts_preview.append(text[:100] + "..." if len(text) > 100 else text)
                else:
                    texts_preview.append("")
                indices_list.append(i)
            
            # Build hover texts with metric matrix if available and ellipses are shown
            hover_texts = []
            for idx, text in zip(indices_list, texts_preview):
                hover_str = f"{label}<br>Text: {text}"
                if show_ellipses and G_2d is not None and idx < len(G_2d):
                    G_i = G_2d[idx]
                    hover_str += f"<br><br>2D Metric Matrix:<br>"
                    hover_str += f"[{G_i[0,0]:.4f}, {G_i[0,1]:.4f}]<br>"
                    hover_str += f"[{G_i[1,0]:.4f}, {G_i[1,1]:.4f}]"
                hover_texts.append(hover_str)
            
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
    else:
        # No coloring - gray
        texts_preview = []
        for i in range(len(X_fisher)):
            if "texts" in st.session_state and i < len(st.session_state.texts):
                text = st.session_state.texts[i]
                texts_preview.append(text[:100] + "..." if len(text) > 100 else text)
            else:
                texts_preview.append("")
        
        # Build hover texts with metric matrix if available and ellipses are shown
        hover_texts = []
        for i, text in enumerate(texts_preview):
            hover_str = text if text else "Point"
            if show_ellipses and G_2d is not None and i < len(G_2d):
                G_i = G_2d[i]
                hover_str += f"<br><br>2D Metric Matrix:<br>"
                hover_str += f"[{G_i[0,0]:.4f}, {G_i[0,1]:.4f}]<br>"
                hover_str += f"[{G_i[1,0]:.4f}, {G_i[1,1]:.4f}]"
            hover_texts.append(hover_str)
        
        fig.add_trace(
            go.Scatter(
                x=X_fisher[:, 0],
                y=X_fisher[:, 1],
                mode='markers',
                marker=dict(size=8, color='gray', line=dict(width=0.5, color='black')),
                name="Points",
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>',
                legendgroup="group1",
                showlegend=False
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
    config = update_config(knn_k, dataset_name, num_datasets, samples_per_dataset)
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
            G_fisher, G_euclidean, labels, unique_labels, label_counts, texts, G_2d, kappa_2d = load_data(config)
            
            # Store texts in session state for plotly
            st.session_state.texts = texts
            
            # Compute MDS
            X_fisher = compute_mds(G_fisher, seed=config["seed"])
            X_euclidean = compute_mds(G_euclidean, seed=config["seed"])
            
            # Create comparison plot
            fig = create_plotly_figure(X_fisher, X_euclidean, G_fisher, G_euclidean, 
                                     labels, unique_labels, label_counts, G_2d, kappa_2d, show_edges, show_ellipses, show_labels, show_curvature)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create distance scatter plot
            D_fisher = _all_pairs_shortest_path_matrix(G_fisher)
            D_euclidean = _all_pairs_shortest_path_matrix(G_euclidean)
            
            scatter_fig = create_distance_scatter_plot(D_fisher, D_euclidean, labels, unique_labels)
            st.plotly_chart(scatter_fig, use_container_width=False)
            
            # Create Fisher metric plot with ellipses
            if G_2d is not None:
                st.subheader("🎯 Fisher Metric Visualization with Ellipses")
                fisher_fig = create_fisher_metric_plot(X_fisher, G_fisher, labels, unique_labels, label_counts, G_2d, kappa_2d, show_edges, show_ellipses, show_labels, show_curvature)
                st.plotly_chart(fisher_fig, use_container_width=True)
            
            # Create scalar curvature histogram (colored by labels)
            if kappa_2d is not None:
                st.subheader("📈 Scalar Curvature Histogram (by Label)")
                fig_hist = go.Figure()
                
                # Get colors for labels
                import plotly.colors as pc
                colors = pc.qualitative.Set3[:len(unique_labels)]
                label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
                
                # Add histogram for each label
                for label in unique_labels:
                    mask = np.array(labels) == label
                    kappa_label = kappa_2d[mask]
                    
                    fig_hist.add_trace(go.Histogram(
                        x=kappa_label,
                        nbinsx=50,
                        name=f"{label} ({label_counts.get(label, 0)})",
                        marker_color=label_to_color[label],
                        opacity=1.0,  # Full opacity for stacked histogram
                        marker_line=dict(
                            color='white',
                            width=0.5
                        ),
                        hovertemplate=f'Label: {label}<br>Range: %{{x}}<br>Count: %{{y}}<extra></extra>'
                    ))
                
                # Calculate statistics
                kappa_mean = np.mean(kappa_2d)
                kappa_variance = np.var(kappa_2d)
                
                # Paper-style layout: clean and professional
                fig_hist.update_layout(
                    title=dict(
                        text="Scalar Curvature Distribution",
                        font=dict(size=18, family="Arial, sans-serif"),
                        x=0.5,
                        xanchor='center'
                    ),
                    xaxis=dict(
                        title=dict(
                            text="Scalar Curvature (κ)",
                            font=dict(size=14, family="Arial, sans-serif")
                        ),
                        tickfont=dict(size=12, family="Arial, sans-serif"),
                        showgrid=True,
                        gridcolor='rgba(0,0,0,0.1)',
                        gridwidth=1,
                        showline=True,
                        linewidth=1,
                        linecolor='black',
                        mirror=True
                    ),
                    yaxis=dict(
                        title=dict(
                            text="Frequency",
                            font=dict(size=14, family="Arial, sans-serif")
                        ),
                        tickfont=dict(size=12, family="Arial, sans-serif"),
                        showgrid=True,
                        gridcolor='rgba(0,0,0,0.1)',
                        gridwidth=1,
                        showline=True,
                        linewidth=1,
                        linecolor='black',
                        mirror=True
                    ),
                    height=500,
                    barmode='stack',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=True,
                    legend=dict(
                        title=dict(
                            text="Labels",
                            font=dict(size=12, family="Arial, sans-serif")
                        ),
                        font=dict(size=11, family="Arial, sans-serif"),
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.01,  # Move slightly left to avoid overlap
                        bgcolor='white',
                        bordercolor='black',
                        borderwidth=1,
                        itemclick="toggleothers",
                        itemdoubleclick="toggle"
                    ),
                    annotations=[
                        dict(
                            text=f"Mean: {kappa_mean:.4f}, Variance: {kappa_variance:.4f}",
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=1.03,  # Lower position to avoid overlap
                            showarrow=False,
                            font=dict(size=11, family="Arial, sans-serif"),
                            align='center',
                            bgcolor='white',
                            bordercolor='black',
                            borderwidth=0.5,
                            borderpad=3
                        )
                    ],
                    margin=dict(l=60, r=120, t=80, b=60)  # Add margins to ensure everything fits
                )
                
                # Add mean line (subtle, without annotation to avoid overlap with legend)
                fig_hist.add_vline(
                    x=kappa_mean,
                    line_dash="dash",
                    line_color="black",
                    line_width=1.5,
                    opacity=0.5,
                    annotation_text="",  # No annotation to avoid overlap
                    annotation_position="top"
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Display statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{kappa_mean:.4f}")
                with col2:
                    st.metric("Variance", f"{kappa_variance:.4f}")
                with col3:
                    st.metric("Std Dev", f"{np.std(kappa_2d):.4f}")
                with col4:
                    st.metric("Range", f"[{np.min(kappa_2d):.4f}, {np.max(kappa_2d):.4f}]")
            
            # Create MDS eigenvalues bar chart
            st.subheader("📊 MDS Eigenvalues")
            eigenvals_fisher, explained_ratio_fisher = compute_mds_eigenvalues(G_fisher)
            eigenvals_euclidean, explained_ratio_euclidean = compute_mds_eigenvalues(G_euclidean)
            
            # Create bar chart for eigenvalues
            n_show = min(20, len(eigenvals_fisher), len(eigenvals_euclidean))
            
            fig_eigen = make_subplots(
                rows=1, cols=2,
                subplot_titles=("MDS Eigenvalues (Fisher Metric)", "MDS Eigenvalues (Euclidean Distance)"),
                horizontal_spacing=0.15
            )
            
            # Fisher eigenvalues
            fig_eigen.add_trace(
                go.Bar(
                    x=list(range(1, n_show + 1)),
                    y=eigenvals_fisher[:n_show],
                    name="Fisher",
                    marker_color='steelblue',
                    text=[f'{r:.3f}' for r in explained_ratio_fisher[:n_show]],
                    textposition='outside',
                    hovertemplate='Dimension %{x}<br>Eigenvalue: %{y:.4f}<br>Explained Ratio: %{text}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Euclidean eigenvalues
            fig_eigen.add_trace(
                go.Bar(
                    x=list(range(1, n_show + 1)),
                    y=eigenvals_euclidean[:n_show],
                    name="Euclidean",
                    marker_color='coral',
                    text=[f'{r:.3f}' for r in explained_ratio_euclidean[:n_show]],
                    textposition='outside',
                    hovertemplate='Dimension %{x}<br>Eigenvalue: %{y:.4f}<br>Explained Ratio: %{text}<extra></extra>'
                ),
                row=1, col=2
            )
            
            fig_eigen.update_xaxes(title_text="Dimension", row=1, col=1)
            fig_eigen.update_yaxes(title_text="Eigenvalue", row=1, col=1)
            fig_eigen.update_xaxes(title_text="Dimension", row=1, col=2)
            fig_eigen.update_yaxes(title_text="Eigenvalue", row=1, col=2)
            fig_eigen.update_layout(height=500, showlegend=False)
            
            st.plotly_chart(fig_eigen, use_container_width=True)
            
            # Display statistics
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Fisher Metric MDS:**")
                st.write(f"- Total variance: {eigenvals_fisher.sum():.4f}")
                st.write(f"- 2D explained ratio: {explained_ratio_fisher[:2].sum():.4f}")
                st.write(f"- Top 5 eigenvalues: {', '.join([f'{v:.4f}' for v in eigenvals_fisher[:5]])}")
            
            with col2:
                st.write("**Euclidean MDS:**")
                st.write(f"- Total variance: {eigenvals_euclidean.sum():.4f}")
                st.write(f"- 2D explained ratio: {explained_ratio_euclidean[:2].sum():.4f}")
                st.write(f"- Top 5 eigenvalues: {', '.join([f'{v:.4f}' for v in eigenvals_euclidean[:5]])}")
            
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
        G_fisher, G_euclidean, labels, unique_labels, label_counts, texts, G_2d, kappa_2d = load_data(default_config)
        
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
                                   labels, unique_labels, label_counts, G_2d, kappa_2d, show_edges, show_ellipses, show_labels, show_curvature)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create distance scatter plot
        D_fisher = _all_pairs_shortest_path_matrix(G_fisher)
        D_euclidean = _all_pairs_shortest_path_matrix(G_euclidean)
        
        scatter_fig = create_distance_scatter_plot(D_fisher, D_euclidean, labels, unique_labels)
        st.plotly_chart(scatter_fig, use_container_width=False)
        
        # Create Fisher metric plot with ellipses
        if G_2d is not None:
            st.subheader("🎯 Fisher Metric Visualization with Ellipses")
            fisher_fig = create_fisher_metric_plot(X_fisher, G_fisher, labels, unique_labels, label_counts, G_2d, kappa_2d, show_edges, show_ellipses, show_labels, show_curvature)
            st.plotly_chart(fisher_fig, use_container_width=True)
        
        # Create scalar curvature histogram (colored by labels)
        if kappa_2d is not None:
            st.subheader("📈 Scalar Curvature Histogram (by Label)")
            fig_hist = go.Figure()
            
            # Get colors for labels
            import plotly.colors as pc
            colors = pc.qualitative.Set3[:len(unique_labels)]
            label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
            
            # Add histogram for each label
            for label in unique_labels:
                mask = np.array(labels) == label
                kappa_label = kappa_2d[mask]
                
                fig_hist.add_trace(go.Histogram(
                    x=kappa_label,
                    nbinsx=50,
                    name=f"{label} ({label_counts.get(label, 0)})",
                    marker_color=label_to_color[label],
                    opacity=1.0,  # Full opacity for stacked histogram
                    marker_line=dict(
                        color='white',
                        width=0.5
                    ),
                    hovertemplate=f'Label: {label}<br>Range: %{{x}}<br>Count: %{{y}}<extra></extra>'
                ))
            
            # Calculate statistics
            kappa_mean = np.mean(kappa_2d)
            kappa_variance = np.var(kappa_2d)
            
            # Paper-style layout: clean and professional
            fig_hist.update_layout(
                title=dict(
                    text="Scalar Curvature Distribution",
                    font=dict(size=18, family="Arial, sans-serif"),
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    title=dict(
                        text="Scalar Curvature (κ)",
                        font=dict(size=14, family="Arial, sans-serif")
                    ),
                    tickfont=dict(size=12, family="Arial, sans-serif"),
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    gridwidth=1,
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    mirror=True
                ),
                yaxis=dict(
                    title=dict(
                        text="Frequency",
                        font=dict(size=14, family="Arial, sans-serif")
                    ),
                    tickfont=dict(size=12, family="Arial, sans-serif"),
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    gridwidth=1,
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    mirror=True
                ),
                height=500,
                barmode='stack',
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=True,
                legend=dict(
                    title=dict(
                        text="Labels",
                        font=dict(size=12, family="Arial, sans-serif")
                    ),
                    font=dict(size=11, family="Arial, sans-serif"),
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02,
                    bgcolor='white',
                    bordercolor='black',
                    borderwidth=1
                ),
                annotations=[
                    dict(
                        text=f"Mean: {kappa_mean:.4f}, Variance: {kappa_variance:.4f}",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=1.05,
                        showarrow=False,
                        font=dict(size=11, family="Arial, sans-serif"),
                        align='center'
                    )
                ]
            )
            
            # Add mean line (subtle, without annotation to avoid overlap with legend)
            fig_hist.add_vline(
                x=kappa_mean,
                line_dash="dash",
                line_color="black",
                line_width=1.5,
                opacity=0.5,
                annotation_text="",  # No annotation to avoid overlap
                annotation_position="top"
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{kappa_mean:.4f}")
            with col2:
                st.metric("Variance", f"{kappa_variance:.4f}")
            with col3:
                st.metric("Std Dev", f"{np.std(kappa_2d):.4f}")
            with col4:
                st.metric("Range", f"[{np.min(kappa_2d):.4f}, {np.max(kappa_2d):.4f}]")
        
except Exception as e:
    if "run_button" not in locals():
        st.info("👈 左側のパラメータを設定して「パイプライン実行」ボタンを押してください。")

# Handle HTML export
if export_html_button:
    try:
        with open("src/config.yaml", "r") as f:
            export_config = yaml.safe_load(f)
        
        base = Path(export_config["artifacts_dir"])
        fisher_graph_path = base / export_config["graphs_dir"] / "gpt2.gpickle"
        euclidean_graph_path = base / export_config["graphs_dir"] / "gpt2_euclidean.gpickle"
        meta_path = base / export_config["embeddings_dir"] / "meta.json"
        
        if fisher_graph_path.exists() and euclidean_graph_path.exists() and meta_path.exists():
            with st.spinner("HTMLファイルを生成中..."):
                output_path, files = export_plots_to_html(export_config, 
                                                         show_edges=show_edges, 
                                                         show_ellipses=show_ellipses, 
                                                         show_labels=show_labels, 
                                                         show_curvature=show_curvature)
                
                st.success(f"✅ HTMLエクスポートが完了しました！")
                st.info(f"📁 出力ディレクトリ: `{output_path}`")
                
                st.subheader("📄 生成されたファイル:")
                for name, file_path in files:
                    if file_path is not None:
                        st.write(f"- **{name}**: `{file_path.name}`")
                
                # Show download links
                st.subheader("📥 ダウンロード:")
                for name, file_path in files:
                    if file_path is not None:
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label=f"📄 {name}をダウンロード",
                                data=f.read(),
                                file_name=file_path.name,
                                mime="text/html",
                                key=f"download_{name}"
                            )
        else:
            st.error("❌ エクスポートするデータが見つかりません。まずパイプラインを実行してください。")
    except Exception as e:
        st.error(f"❌ エクスポート中にエラーが発生しました: {e}")
        st.exception(e)

