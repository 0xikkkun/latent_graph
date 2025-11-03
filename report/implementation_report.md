# LLM潜在空間可視化システム 実装レポート

## 概要

本レポートは、LLM（Large Language Model）の潜在空間における幾何学的解析システムの実装内容をまとめたものです。特に、Fisher計量を用いた距離測定とEuclidean距離の比較分析、および可視化機能の実装について詳細を記載します。

## 実装内容

### 1. Fisher計量の2次元射影と楕円可視化

#### 1.1 2次元Fisher計量の計算

**ファイル**: `src/project_metrics_2d.py`

MDS（Multi-Dimensional Scaling）により2次元に射影された各点において、元の高次元空間でのFisher計量を2次元空間に射影する処理を実装しました。

- MDSの固有値分解から最大2つの固有値を取得
- Fisher計量を対応する固有ベクトル空間に射影
- 各点における2次元Fisher計量行列を保存

```python
# MDSの固有値分解
evals, evecs = np.linalg.eigh(D)
idx = evals.argsort()[::-1]
evals, evecs = evals[idx], evecs[:, idx]

# Fisher計量を射影
V = evecs[:, :2]
G_2d = V.T @ G_i @ V
```

#### 1.2 楕円によるFisher計量の可視化

**ファイル**: `src/compare_fisher_euclidean.py`, `app_streamlit.py`

各点のFisher計量を楕円として可視化しました。

**楕円の描画方法**:
1. Fisher計量行列を固有値分解
2. 固有値の平方根を楕円の半軸長とする
3. 固有ベクトルから回転角を計算
4. MDS座標の範囲に対して適切なスケールを適用

```python
# 楕円の半軸長と回転角の計算
eigenvals, eigenvecs = np.linalg.eigh(G_i)
a = np.sqrt(eigenvals[0]) * scale  # 長軸
b = np.sqrt(eigenvals[1]) * scale  # 短軸
angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
```

**スケール因子**:
MDSの座標範囲の2.5%を目安に、典型的な楕円サイズに基づいて自動調整。

### 2. Streamlitによるインタラクティブ可視化

#### 2.1 可視化図の種類

**メイン比較図**:
- 左: Fisher計量によるMDS
- 右: Euclidean距離によるMDS
- 両方に楕円とエッジを表示

**Fisher計量図**（楕円表示用）:
- Fisher計量によるMDSに楕円とエッジを表示
- パラメータをタイトルに表示

**散布図**:
- Fisher距離 vs Euclidean距離の正規化散布図
- 1:1の参照線を表示
- アスペクト比1:1を維持

#### 2.2 表示オプション

サイドバーから以下を切り替え可能：
- **エッジの表示/非表示**: グラフの接続線
- **楕円の表示/非表示**: Fisher計量楕円

#### 2.3 図のサイズとアスペクト比

- すべての図を800x800ピクセルに統一
- アスペクト比1:1を維持（`scaleanchor`と`scaleratio`を使用）
- 散布図は`use_container_width=False`で表示

### 3. パラメータ管理とファイル名

#### 3.1 パラメータ表示

すべての図に実験パラメータを表示：
- `num_samples`: データサンプル数
- `num_datasets`: データセット数
- `knn_k`: k近傍グラフのk値

表示形式:
```
Parameters: num_samples=100, num_datasets=10, knn_k=10
```

#### 3.2 ファイル名へのパラメータ埋め込み

保存ファイル名にパラメータを含める：

```
gpt2_comparison_fisher_vs_euclidean_ns{num_samples}_nd{num_datasets}_k{knn_k}.png
gpt2_fisher_metric_ns{num_samples}_nd{num_datasets}_k{knn_k}.png
```

例: `gpt2_fisher_metric_ns100_nd10_k10.png`

### 4. Streamlit UI/UX改善

#### 4.1 インタラクティブな可視化

- Plotlyによるインタラクティブなグラフ
- マウスオーバーでテキスト内容を表示
- ズーム・パン操作に対応

#### 4.2 パラメータ調整

サイドバーで以下をスライダー調整可能：
- `num_samples`: 10-500（デフォルト100）
- `num_datasets`: 1-12（デフォルト10）
- `knn_k`: 3-20（デフォルト10）

#### 4.3 パイプライン実行

「パイプライン実行」ボタンで以下を順次実行：
1. データ抽出
2. Fisherグラフ構築
3. Euclideanグラフ構築
4. Fisher計量2D射影
5. 可視化生成

進捗バーで現在の処理を表示。

## 技術仕様

### 楕円のスケーリング手法

各点iにおけるFisher計量行列G_i（2×2）を以下の手順で楕円化：

1. **固有値分解**:
   ```python
   eigenvals, eigenvecs = np.linalg.eigh(G_i)
   ```

2. **半軸長の計算**:
   ```python
   a = sqrt(λ₁) × scale  # 長軸
   b = sqrt(λ₂) × scale  # 短軸
   ```

3. **回転角度**:
   ```python
   angle = arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
   ```

4. **スケール因子**:
   ```python
   mds_range = max(MDS座標) - min(MDS座標)
   typical_ellipse_size = median(sqrt(固有値))
   scale = (mds_range × 0.025) / typical_ellipse_size
   ```

### 距離行列の計算

**Fisher距離**（測地距離）:
- グラフの重み付き最短経路（Dijkstra）
- 重みはFisher計量による内部積

**Euclidean距離**:
- eta埋め込み空間でのL2距離

### MDS設定

- 2次元への埋め込み
- ランダムシード固定（再現性）
- 最大反復数1000
- 収束判定eps=1e-9

## 実装ファイル一覧

### メイン実装

- `src/compare_fisher_euclidean.py`: 比較可視化と楕円描画
- `src/project_metrics_2d.py`: Fisher計量の2次元射影
- `app_streamlit.py`: Streamlit UIとインタラクティブ可視化
- `src/config.yaml`: 実験パラメータ設定

### サポート実装

- `src/extract.py`: データ抽出とエンベディング計算
- `src/graph.py`: Fisher計量グラフ構築
- `src/graph_euclidean.py`: Euclidean距離グラフ構築
- `src/utils.py`: ユーティリティ関数

## 使用方法

### 1. Streamlitアプリ起動

```bash
streamlit run app_streamlit.py --server.port 8504
```

### 2. パラメータ設定

左サイドバーで以下を調整：
- データサンプル数
- データセット数
- k近傍グラフのk

### 3. パイプライン実行

「🚀 パイプライン実行」をクリック

### 4. 結果確認

自動的に以下が表示されます：
1. Fisher vs Euclidean比較図
2. 距離散布図
3. Fisher計量楕円図（表示オプションで制御）

## 生成されるファイル

### 画像ファイル

```
artifacts_gpt2_large_multi/plots/
├── gpt2_comparison_fisher_vs_euclidean_ns{num_samples}_nd{num_datasets}_k{knn_k}.png
└── gpt2_fisher_metric_ns{num_samples}_nd{num_datasets}_k{knn_k}.png
```

### データファイル

```
artifacts_gpt2_large_multi/
├── embeddings/gpt2_large_multi/
│   ├── eta.npy              # 埋め込みベクトル
│   └── meta.json            # メタデータ（ラベル、テキスト）
├── metrics/gpt2_large_multi/
│   └── G_theta.npy          # Fisher計量
├── metrics_2d/
│   └── G_theta_2d.npy       # 2次元射影Fisher計量
└── graphs/
    ├── gpt2.gpickle         # Fisherグラフ
    └── gpt2_euclidean.gpickle  # Euclideanグラフ
```

## 技術的な特徴

### 1. 幾何学的解釈

Fisher計量は、各点における情報幾何の局所構造を表現します。楕円により以下を可視化：
- **方向性**: 固有ベクトルが示す主要な方向
- **大きさ**: 固有値が示す局所的不確実性
- **形状**: 楕円の扁平度が示す等方的／異方的性

### 2. スケールの自動調整

MDSの座標範囲に応じて自動スケーリング：
- 小さすぎ：見えない
- 大きすぎ：重なり多すぎ
- 最適値：視認性と情報量のバランス

### 3. 色分けと凡例

データソース別に色分け：
- 各データセットに固有色
- 凡例にカウント表示
- 透明度で重なりを調整

## 今後の改善案

### パフォーマンス

- GPU利用の最適化
- 大規模データのバッチ処理
- キャッシュの導入

### 機能拡張

- 他の次元削減手法（t-SNE、UMAP）
- 3次元可視化
- アニメーションによる遷移可視化
- エクスポート機能（PDF、SVG）

### 分析機能

- クラスタ分析の統合
- 統計情報の表示
- 比較実験の一括実行

## まとめ

本実装により、LLM潜在空間の幾何学的構造を可視化し、Fisher計量とEuclidean距離の比較分析を可能にしました。特に楕円によるFisher計量の可視化は、各点での局所的な情報構造を直感的に理解するための有用な手段となります。

Streamlitによるインタラクティブなインターフェースは、研究者がパラメータを変更しながら実験を進めることを可能にし、実験の効率化に寄与します。

