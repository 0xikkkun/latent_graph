"""
2次元投影空間での接続係数とリーマン曲率テンソルの計算

高次元での計算量問題を解決するため、2次元投影された期待値座標（eta_2d）と
計量行列（G_2d）から曲率を計算します。
計算量: O(D⁴) → O(2⁴) = O(16) に大幅削減
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from utils import set_global_seed, ensure_dirs, save_npy, pinvh_stable
from extract import (
    _select_nonempty_texts,
    _compute_qkv_from_hidden,
    _concat_heads,
)


def _compute_third_cumulant_2d_gpu(z: torch.Tensor, w: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    GPU版: 2次元空間での3次キュムラントを計算
    
    Args:
        z: (T, 2) 2次元投影されたトークン埋め込みベクトル (GPU上)
        w: (T,) 重み (GPU上)
        eta: (2,) 2次元期待値座標 (GPU上)
    
    Returns:
        Γ_{ijk}: (2, 2, 2) 3次キュムラント (GPU上)
    """
    sum_w2 = (w ** 2).sum()
    alpha = 1.0 / max(1e-8, (1.0 - sum_w2).item())
    diff = z - eta[None, :]
    third_moment = torch.einsum('t,ti,tj,tk->ijk', w, diff, diff, diff)
    return alpha * third_moment


def _compute_fourth_cumulant_2d_gpu(z: torch.Tensor, w: torch.Tensor, eta: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """
    GPU版: 2次元空間での4次キュムラントの偏微分項を計算
    
    Args:
        z: (T, 2) 2次元投影されたトークン埋め込みベクトル (GPU上)
        w: (T,) 重み (GPU上)
        eta: (2,) 2次元期待値座標 (GPU上)
        g: (2, 2) 2次元計量行列 (GPU上)
    
    Returns:
        ∂_iΓ_{jkm}: (2, 2, 2, 2) 4次キュムラントから計算した偏微分 (GPU上)
    """
    sum_w2 = (w ** 2).sum()
    alpha = 1.0 / max(1e-8, (1.0 - sum_w2).item())
    diff = z - eta[None, :]
    
    fourth_moment = torch.einsum('t,ti,tj,tk,tl->ijkl', w, diff, diff, diff, diff)
    fourth_moment = alpha * fourth_moment
    
    g_terms = (
        torch.einsum('ij,kl->ijkl', g, g) +
        torch.einsum('ik,jl->ijkl', g, g) +
        torch.einsum('jk,il->ijkl', g, g)
    )
    
    partial_gamma = 0.5 * (fourth_moment - g_terms)
    return partial_gamma


def compute_connection_coefficient_2d_gpu(z: torch.Tensor, w: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """GPU版: 2次元空間での接続係数 Γ_{ijk} を計算"""
    third_cumulant = _compute_third_cumulant_2d_gpu(z, w, eta)
    return 0.5 * third_cumulant


def compute_connection_coefficient_mixed_2d_gpu(gamma_lower: torch.Tensor, g_inv: torch.Tensor) -> torch.Tensor:
    """GPU版: 2次元空間での混合形式の接続係数 Γ_{jk}^l を計算"""
    return torch.einsum('jkm,ml->jkl', gamma_lower, g_inv)


def compute_connection_derivative_2d_gpu(
    z: torch.Tensor,
    w: torch.Tensor,
    eta: torch.Tensor,
    g: torch.Tensor,
    g_inv: torch.Tensor,
    gamma_lower: torch.Tensor,
) -> torch.Tensor:
    """GPU版: 2次元空間での接続係数の偏微分 ∂_iΓ_{jk}^l を計算"""
    partial_gamma_lower_ijk = _compute_fourth_cumulant_2d_gpu(z, w, eta, g)
    partial_gamma_lower = partial_gamma_lower_ijk.permute(1, 2, 3, 0).contiguous()
    
    partial_g_inv = -0.5 * torch.einsum('lm,imn,ns->ils', g_inv, gamma_lower, g_inv)
    
    term1 = torch.einsum('ijkm,ml->ijkl', partial_gamma_lower, g_inv)
    term2 = torch.einsum('jkm,ilm->ijkl', gamma_lower, partial_g_inv)
    
    return term1 + term2


def compute_riemann_curvature_2d_gpu(
    partial_gamma_mixed: torch.Tensor,
    gamma_mixed: torch.Tensor,
) -> torch.Tensor:
    """GPU版: 2次元空間でのリーマン曲率テンソル R_{ijk}^l を計算"""
    term1 = partial_gamma_mixed
    term2 = -partial_gamma_mixed.transpose(0, 1)
    term3 = -torch.einsum('iml,jkm->ijkl', gamma_mixed, gamma_mixed)
    term4 = -torch.einsum('jml,ikm->ijkl', gamma_mixed, gamma_mixed)
    return term1 + term2 + term3 + term4


def compute_scalar_curvature_2d_gpu(R: torch.Tensor, g: torch.Tensor, g_inv: torch.Tensor) -> float:
    """GPU版: 2次元空間でのスカラー曲率 κ を計算"""
    D = R.shape[0]
    if D <= 1:
        return 0.0
    
    R_lower = torch.einsum('ijkl,lm->ijkm', R, g)
    kappa_sum = torch.einsum('ijkm,im,jk->', R_lower, g_inv, g_inv)
    kappa = kappa_sum.item() / (D * (D - 1))
    return float(kappa)


def run(config_path: str) -> None:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_global_seed(int(cfg["seed"]))

    base_dir = cfg["artifacts_dir"]
    subdirs = [
        cfg["embeddings_dir"],
        cfg["metrics_dir"],
        cfg["curvature_2d_dir"],
    ]
    paths = ensure_dirs(base_dir, *subdirs)

    # データ読み込み
    num_datasets = int(cfg.get("num_datasets", 4))  # Fallback for backward compatibility
    num_samples = int(cfg.get("num_samples", 100))  # Fallback if samples_per_dataset not provided
    samples_per_dataset = cfg.get("samples_per_dataset", None)
    texts, labels = _select_nonempty_texts(
        cfg["dataset"]["name"],
        cfg["dataset"]["config"],
        cfg["dataset"]["split"],
        num_samples,
        int(cfg["seed"]),
        num_datasets,
        samples_per_dataset,
    )

    # 2次元投影データを読み込み
    metrics_2d_dir = Path(base_dir) / "metrics_2d"
    eta_2d = np.load(metrics_2d_dir / "eta_2d.npy")  # (N, 2)
    G_2d = np.load(metrics_2d_dir / "G_theta_2d.npy")  # (N, 2, 2)
    P = np.load(metrics_2d_dir / "P.npy")  # (D, 2)

    N = eta_2d.shape[0]
    D = P.shape[0]  # 元の次元数
    max_length = int(cfg["max_length"])
    lambda_reg = float(cfg["lambda_reg"])

    # モデル読み込み
    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager"
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # データをGPUに転送（GPU表示の前に転送）
    eta_2d_gpu = torch.from_numpy(eta_2d).float().to(device)  # (N, 2)
    G_2d_gpu = torch.from_numpy(G_2d).float().to(device)  # (N, 2, 2)
    P_gpu = torch.from_numpy(P).float().to(device)  # (D, 2)

    # GPU使用状況を表示
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version (PyTorch): {torch.version.cuda}")
        # モデルがGPU上にあるか確認
        print(f"Model device: {next(model.parameters()).device}")
        # データがGPU上にあるか確認
        print(f"eta_2d device: {eta_2d_gpu.device}")
        print(f"G_2d device: {G_2d_gpu.device}")
        # 初期GPUメモリ使用量
        torch.cuda.reset_peak_memory_stats()
        print(f"Initial GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA is not available, using CPU!")
    print(f"Computing curvature in 2D projected space (D=2)")
    print(f"{'='*60}\n")

    # 接続係数、リーマン曲率テンソル、スカラー曲率を保存するリスト
    gamma_list: list = []
    riemann_list: list = []
    kappa_list: list = []
    
    # 統計情報を記録
    rank_deficient_samples = []
    high_condition_samples = []
    near_singular_samples = []
    large_inv_error_samples = []

    # モデルの設定
    gpt = model.transformer  # type: ignore[attr-defined]
    last_block = gpt.h[-1]
    c_attn = last_block.attn.c_attn
    n_head: int = gpt.config.n_head

    for i, text in enumerate(tqdm(texts, desc="Computing 2D curvature", unit="sample")):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
            )

        hidden_states: Tuple[torch.Tensor, ...] = outputs.hidden_states  # type: ignore
        h_in_last = hidden_states[-2]  # (B,T,D)

        seq_len = int(attention_mask.sum().item())
        t_star = seq_len - 1

        q, k, v = _compute_qkv_from_hidden(h_in_last, c_attn, n_head)
        v_concat = _concat_heads(v).squeeze(0).float()  # (T, D)

        if outputs.attentions is not None and len(outputs.attentions) > 0:
            attn_last = outputs.attentions[-1]  # type: ignore
            attn = attn_last.squeeze(0)  # (H, T, T)
        else:
            attn = torch.ones((n_head, max_length, max_length), device=device) / max_length

        w = attn[:, t_star, :].mean(dim=0)  # (T,)
        valid_mask = attention_mask.squeeze(0).bool()
        w = w * valid_mask
        w = w[:seq_len]
        w = w / (w.sum() + 1e-12)

        z = v_concat[:seq_len, :]  # (T_valid, D)
        # 2次元に投影
        z_2d = torch.einsum('td,dk->tk', z, P_gpu)  # (T_valid, 2)
        
        eta_2d_i = eta_2d_gpu[i]  # (2,)
        g_2d = G_2d_gpu[i]  # (2, 2)
        
        # pinvhはCPUで実行
        g_2d_cpu = g_2d.cpu().numpy()
        
        # ランク落ちと特異性のチェック
        rank = np.linalg.matrix_rank(g_2d_cpu)
        det = np.linalg.det(g_2d_cpu)
        cond_num = np.linalg.cond(g_2d_cpu)
        
        # ランク落ちチェック（2x2行列なのでrank=2であるべき）
        if rank < 2:
            print(f"Warning: Sample {i} has rank-deficient metric matrix (rank={rank}), skipping")
            rank_deficient_samples.append(i)
            # NaNを保存してスキップ
            gamma_list.append(np.full((2, 2, 2), np.nan, dtype=np.float32))
            riemann_list.append(np.full((2, 2, 2, 2), np.nan, dtype=np.float32))
            kappa_list.append(np.nan)
            continue
        
        # 条件数チェック（大きすぎると逆行列が不安定）
        if cond_num > 1e10:
            print(f"Warning: Sample {i} has high condition number ({cond_num:.2e}), may cause numerical instability")
            high_condition_samples.append((i, cond_num))
        
        # 行列式チェック（0に近いと特異）
        if abs(det) < 1e-10:
            print(f"Warning: Sample {i} has near-singular metric matrix (det={det:.2e}), may cause numerical instability")
            near_singular_samples.append((i, det))
        
        # 対称性チェック（計量行列は対称であるべき）
        if not np.allclose(g_2d_cpu, g_2d_cpu.T, atol=1e-6):
            print(f"Warning: Sample {i} has non-symmetric metric matrix")
        
        g_inv_2d_cpu = pinvh_stable(g_2d_cpu, jitter=lambda_reg)
        
        # 逆行列の検証（G @ G_inv ≈ I を確認）
        verification = g_2d_cpu @ g_inv_2d_cpu
        identity = np.eye(2)
        inv_error = np.linalg.norm(verification - identity, 'fro')
        if inv_error > 1e-3:
            print(f"Warning: Sample {i} has large inverse matrix error ({inv_error:.2e}), rank={rank}, cond={cond_num:.2e}, det={det:.2e}")
            large_inv_error_samples.append((i, inv_error, rank, cond_num, det))
        
        g_inv_2d = torch.from_numpy(g_inv_2d_cpu).float().to(device)  # (2, 2)

        # 2次元空間での接続係数 Γ_{ijk} を計算
        gamma_lower = compute_connection_coefficient_2d_gpu(z_2d, w[:seq_len], eta_2d_i)  # (2, 2, 2)

        # 混合形式の接続係数 Γ_{jk}^l を計算
        gamma_mixed = compute_connection_coefficient_mixed_2d_gpu(gamma_lower, g_inv_2d)  # (2, 2, 2)

        # 接続係数の偏微分 ∂_iΓ_{jk}^l を計算
        partial_gamma_mixed = compute_connection_derivative_2d_gpu(
            z_2d, w[:seq_len], eta_2d_i, g_2d, g_inv_2d, gamma_lower
        )  # (2, 2, 2, 2)

        # リーマン曲率テンソル R_{ijk}^l を計算
        R = compute_riemann_curvature_2d_gpu(partial_gamma_mixed, gamma_mixed)  # (2, 2, 2, 2)

        # スカラー曲率 κ を計算
        kappa = compute_scalar_curvature_2d_gpu(R, g_2d, g_inv_2d)

        gamma_list.append(gamma_lower.detach().cpu().numpy().astype(np.float32))
        riemann_list.append(R.detach().cpu().numpy().astype(np.float32))
        kappa_list.append(kappa)

    # 結果を保存
    gamma_arr = np.stack(gamma_list, axis=0)  # (N, 2, 2, 2)
    riemann_arr = np.stack(riemann_list, axis=0)  # (N, 2, 2, 2, 2)
    kappa_arr = np.array(kappa_list, dtype=np.float32)  # (N,)

    save_npy(
        Path(base_dir) / cfg["curvature_2d_dir"] / "gamma_ijk_2d.npy", gamma_arr
    )
    save_npy(
        Path(base_dir) / cfg["curvature_2d_dir"] / "riemann_curvature_2d.npy", riemann_arr
    )
    save_npy(
        Path(base_dir) / cfg["curvature_2d_dir"] / "scalar_curvature_2d.npy", kappa_arr
    )

    print("Saved:")
    print("  gamma_ijk_2d:", Path(base_dir) / cfg["curvature_2d_dir"] / "gamma_ijk_2d.npy")
    print(
        "  riemann_curvature_2d:",
        Path(base_dir) / cfg["curvature_2d_dir"] / "riemann_curvature_2d.npy",
    )
    print(
        "  scalar_curvature_2d:",
        Path(base_dir) / cfg["curvature_2d_dir"] / "scalar_curvature_2d.npy",
    )
    print(f"\nScalar curvature statistics (2D):")
    
    # NaNを除外した統計
    kappa_valid = kappa_arr[~np.isnan(kappa_arr)]
    if len(kappa_valid) > 0:
        print(f"  Mean: {kappa_valid.mean():.6f}")
        print(f"  Std: {kappa_valid.std():.6f}")
        print(f"  Min: {kappa_valid.min():.6f}")
        print(f"  Max: {kappa_valid.max():.6f}")
    else:
        print(f"  All values are NaN!")
    
    # 統計情報の要約
    print(f"\n{'='*60}")
    print(f"Metric Matrix Quality Check Summary:")
    print(f"{'='*60}")
    print(f"  Total samples: {N}")
    print(f"  Rank-deficient samples (skipped): {len(rank_deficient_samples)}")
    if rank_deficient_samples:
        print(f"    Samples: {rank_deficient_samples[:10]}{'...' if len(rank_deficient_samples) > 10 else ''}")
    print(f"  High condition number (>1e10): {len(high_condition_samples)}")
    if high_condition_samples:
        print(f"    Top 5 worst: {[(idx, f'{cond:.2e}') for idx, cond in sorted(high_condition_samples, key=lambda x: x[1], reverse=True)[:5]]}")
    print(f"  Near-singular matrices (|det|<1e-10): {len(near_singular_samples)}")
    if near_singular_samples:
        print(f"    Top 5 worst: {[(idx, f'{det:.2e}') for idx, det in sorted(near_singular_samples, key=lambda x: abs(x[1]))[:5]]}")
    print(f"  Large inverse error (>1e-3): {len(large_inv_error_samples)}")
    if large_inv_error_samples:
        print(f"    Top 5 worst: {[(idx, f'{err:.2e}') for idx, err, _, _, _ in sorted(large_inv_error_samples, key=lambda x: x[1], reverse=True)[:5]]}")
    
    # 最終的なGPUメモリ使用量を表示
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(0) / 1e9
        mem_reserved = torch.cuda.memory_reserved(0) / 1e9
        mem_peak = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"\n{'='*60}")
        print(f"Final GPU Memory Usage:")
        print(f"  Allocated: {mem_allocated:.2f} GB")
        print(f"  Reserved: {mem_reserved:.2f} GB")
        print(f"  Peak: {mem_peak:.2f} GB")
        print(f"{'='*60}\n")
    
    # NaNの統計
    nan_count = np.isnan(kappa_arr).sum()
    if nan_count > 0:
        print(f"  NaN values in curvature: {nan_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    args = parser.parse_args()
    run(args.config)

