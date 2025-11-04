"""
接続係数とリーマン曲率テンソルの計算

理論に基づき、標本推定により以下を計算：
- 接続係数 Γ_{ijk}
- リーマン曲率テンソル R_{ijk}^l
- スカラー曲率 κ
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from utils import set_global_seed, ensure_dirs, save_npy, pinvh_stable
from extract import (
    _select_nonempty_texts,
    _compute_qkv_from_hidden,
    _concat_heads,
)


def _compute_third_cumulant_gpu(z: torch.Tensor, w: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    GPU版: 3次キュムラントを計算: E[(z_i - η_i)(z_j - η_j)(z_k - η_k)]
    
    Args:
        z: (T, D) トークン埋め込みベクトル (GPU上)
        w: (T,) 重み (GPU上)
        eta: (D,) 期待値座標 (GPU上)
    
    Returns:
        Γ_{ijk}: (D, D, D) 3次キュムラント (GPU上)
    """
    # 有効サンプル補正
    sum_w2 = (w ** 2).sum()
    alpha = 1.0 / max(1e-8, (1.0 - sum_w2).item())
    
    # 差分: (T, D)
    diff = z - eta[None, :]
    
    # 重み付き3次モーメント: E[(z_i - η_i)(z_j - η_j)(z_k - η_k)]
    # einsum: 't,ti,tj,tk->ijk'
    third_moment = torch.einsum('t,ti,tj,tk->ijk', w, diff, diff, diff)
    
    third_moment = alpha * third_moment
    
    return third_moment


def _compute_fourth_cumulant_gpu(z: torch.Tensor, w: torch.Tensor, eta: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """
    GPU版: 4次キュムラントの偏微分項を計算: ∂_iΓ_{jkm}
    
    理論:
    ∂_i∂_j∂_k∂_lψ(θ) = 2∂_lΓ_{ijk}
                   = ∂_i∂_j∂_kψ(θ) + g_{ij}*g_{kl} + g_{ik}*g_{jl} + g_{jk}*g_{il}
    
    Args:
        z: (T, D) トークン埋め込みベクトル (GPU上)
        w: (T,) 重み (GPU上)
        eta: (D,) 期待値座標 (GPU上)
        g: (D, D) 計量行列 (GPU上)
    
    Returns:
        ∂_iΓ_{jkm}: (D, D, D, D) 4次キュムラントから計算した偏微分 (GPU上)
    """
    sum_w2 = (w ** 2).sum()
    alpha = 1.0 / max(1e-8, (1.0 - sum_w2).item())
    
    diff = z - eta[None, :]  # (T, D)
    
    # 4次モーメント: E[(z_i - η_i)(z_j - η_j)(z_k - η_k)(z_l - η_l)]
    # einsum: 't,ti,tj,tk,tl->ijkl'
    fourth_moment = torch.einsum('t,ti,tj,tk,tl->ijkl', w, diff, diff, diff, diff)
    fourth_moment = alpha * fourth_moment
    
    # 4次キュムラントから偏微分項を計算
    # ∂_lΓ_{ijk} = (1/2) * [∂_i∂_j∂_k∂_lψ - (g_{ij}g_{kl} + g_{ik}g_{jl} + g_{jk}g_{il})]
    # ここで ∂_i∂_j∂_k∂_lψ = fourth_moment
    # g_terms を効率的に計算
    g_terms = (
        torch.einsum('ij,kl->ijkl', g, g) +  # g_{ij} * g_{kl}
        torch.einsum('ik,jl->ijkl', g, g) +  # g_{ik} * g_{jl}
        torch.einsum('jk,il->ijkl', g, g)    # g_{jk} * g_{il}
    )
    
    partial_gamma = 0.5 * (fourth_moment - g_terms)
    
    return partial_gamma


def compute_connection_coefficient_gpu(z: torch.Tensor, w: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    GPU版: 接続係数 Γ_{ijk} を計算
    
    Γ_{ijk} = (1/2) * E[(z_i - η_i)(z_j - η_j)(z_k - η_k)]
    
    Args:
        z: (T, D) トークン埋め込みベクトル (GPU上)
        w: (T,) 重み (GPU上)
        eta: (D,) 期待値座標 (GPU上)
    
    Returns:
        Γ_{ijk}: (D, D, D) 接続係数 (GPU上)
    """
    third_cumulant = _compute_third_cumulant_gpu(z, w, eta)
    return 0.5 * third_cumulant


def compute_connection_coefficient_mixed_gpu(gamma_lower: torch.Tensor, g_inv: torch.Tensor) -> torch.Tensor:
    """
    GPU版: 混合形式の接続係数 Γ_{jk}^l を計算
    
    Γ_{jk}^l = Γ_{jkm} g^{ml}
    
    Args:
        gamma_lower: (D, D, D) 下添字の接続係数 Γ_{jkm} (GPU上)
        g_inv: (D, D) 計量の逆行列 g^{ml} (GPU上)
    
    Returns:
        Γ_{jk}^l: (D, D, D) 混合形式の接続係数 (GPU上)
    """
    # einsum: 'jkm,ml->jkl'
    return torch.einsum('jkm,ml->jkl', gamma_lower, g_inv)


def compute_connection_derivative_gpu(
    z: torch.Tensor,
    w: torch.Tensor,
    eta: torch.Tensor,
    g: torch.Tensor,
    g_inv: torch.Tensor,
    gamma_lower: torch.Tensor,
) -> torch.Tensor:
    """
    GPU版: 接続係数の偏微分 ∂_iΓ_{jk}^l を計算
    
    ∂_iΓ_{jk}^l = ∂_iΓ_{jkm}·g^{lm} + Γ_{jkm}·∂_i g^{lm}
    
    ここで:
    ∂_i g^{ls} = -g^{lm}·(1/2)Γ_{imn}·g^{ns}
    
    Args:
        z: (T, D) トークン埋め込みベクトル (GPU上)
        w: (T,) 重み (GPU上)
        eta: (D,) 期待値座標 (GPU上)
        g: (D, D) 計量行列 (GPU上)
        g_inv: (D, D) 計量の逆行列 (GPU上)
        gamma_lower: (D, D, D) 下添字の接続係数 (GPU上)
    
    Returns:
        ∂_iΓ_{jk}^l: (D, D, D, D) 接続係数の偏微分 (GPU上)
    """
    D = g.shape[0]
    
    # 4次キュムラントから ∂_iΓ_{jkm} を計算
    # 注意: _compute_fourth_cumulant_gpu は ∂_lΓ_{ijk} を返すが、
    # 実際には ∂_iΓ_{jkm} が必要なので、インデックスを適切に並び替える
    partial_gamma_lower_ijk = _compute_fourth_cumulant_gpu(z, w, eta, g)  # (D, D, D, D) = ∂_lΓ_{ijk}
    
    # インデックスを並び替えて ∂_iΓ_{jkm} を得る
    # ∂_lΓ_{ijk} -> ∂_iΓ_{jkm} の変換: ijk -> jkm, l -> i
    # つまり、partial_gamma_lower[i, j, k, l] = ∂_lΓ_{ijk}
    # 必要なのは ∂_iΓ_{jkm} = partial_gamma_lower[j, k, m, i]
    # より効率的に転置で実現
    partial_gamma_lower = partial_gamma_lower_ijk.permute(1, 2, 3, 0).contiguous()  # (D, D, D, D)
    
    # ∂_i g^{ls} を計算: ∂_i g^{ls} = -g^{lm}·(1/2)Γ_{imn}·g^{ns}
    # einsum: 'lm,imn,ns->ils'
    partial_g_inv = -0.5 * torch.einsum('lm,imn,ns->ils', g_inv, gamma_lower, g_inv)
    
    # ∂_iΓ_{jk}^l = ∂_iΓ_{jkm}·g^{lm} + Γ_{jkm}·∂_i g^{lm}
    # 第1項: einsum('ijkm,ml->ijkl', partial_gamma_lower, g_inv)
    # 第2項: einsum('jkm,ilm->ijkl', gamma_lower, partial_g_inv)
    term1 = torch.einsum('ijkm,ml->ijkl', partial_gamma_lower, g_inv)
    term2 = torch.einsum('jkm,ilm->ijkl', gamma_lower, partial_g_inv)
    
    partial_gamma_mixed = term1 + term2
    
    return partial_gamma_mixed


def compute_riemann_curvature_gpu(
    partial_gamma_mixed: torch.Tensor,
    gamma_mixed: torch.Tensor,
) -> torch.Tensor:
    """
    GPU版: リーマン曲率テンソル R_{ijk}^l を計算
    
    R_{ijk}^l = ∂_iΓ_{jk}^l - ∂_jΓ_{ik}^l - Γ_{im}^l Γ_{jk}^m - Γ_{jm}^l Γ_{ik}^m
    
    Args:
        partial_gamma_mixed: (D, D, D, D) 接続係数の偏微分 ∂_iΓ_{jk}^l (GPU上)
        gamma_mixed: (D, D, D) 混合形式の接続係数 Γ_{jk}^l (GPU上)
    
    Returns:
        R_{ijk}^l: (D, D, D, D) リーマン曲率テンソル (GPU上)
    """
    D = gamma_mixed.shape[0]
    
    # R_{ijk}^l = ∂_iΓ_{jk}^l - ∂_jΓ_{ik}^l - Γ_{im}^l Γ_{jk}^m - Γ_{jm}^l Γ_{ik}^m
    # 第1項: ∂_iΓ_{jk}^l
    term1 = partial_gamma_mixed  # (D, D, D, D)
    
    # 第2項: -∂_jΓ_{ik}^l (iとjを交換)
    term2 = -partial_gamma_mixed.transpose(0, 1)  # (D, D, D, D)
    
    # 第3項: -Γ_{im}^l Γ_{jk}^m
    # einsum: 'iml,jkm->ijkl'
    term3 = -torch.einsum('iml,jkm->ijkl', gamma_mixed, gamma_mixed)
    
    # 第4項: -Γ_{jm}^l Γ_{ik}^m
    # einsum: 'jml,ikm->ijkl'
    term4 = -torch.einsum('jml,ikm->ijkl', gamma_mixed, gamma_mixed)
    
    R = term1 + term2 + term3 + term4
    
    return R


def compute_scalar_curvature_gpu(R: torch.Tensor, g: torch.Tensor, g_inv: torch.Tensor) -> float:
    """
    GPU版: スカラー曲率 κ を計算
    
    κ = (1/(n(n-1))) * R_{ijkm} g^{im} g^{jk}
    
    ここで R_{ijkm} = R_{ijk}^l g_{lm}
    
    Args:
        R: (D, D, D, D) リーマン曲率テンソル R_{ijk}^l (GPU上)
        g: (D, D) 計量行列 g_{lm} (GPU上)
        g_inv: (D, D) 計量の逆行列 g^{lm} (GPU上)
    
    Returns:
        κ: スカラー曲率
    """
    D = R.shape[0]
    
    if D <= 1:
        return 0.0
    
    # R_{ijkm} = R_{ijk}^l g_{lm} を計算
    # einsum: 'ijkl,lm->ijkm'
    R_lower = torch.einsum('ijkl,lm->ijkm', R, g)  # (D, D, D, D)
    
    # スカラー曲率: κ = (1/(n(n-1))) * R_{ijkm} g^{im} g^{jk}
    # einsum: 'ijkm,im,jk->'
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
        cfg["curvature_dir"],
    ]
    paths = ensure_dirs(base_dir, *subdirs)

    # データ読み込み
    num_datasets = int(cfg.get("num_datasets", 4))
    texts, labels = _select_nonempty_texts(
        cfg["dataset"]["name"],
        cfg["dataset"]["config"],
        cfg["dataset"]["split"],
        int(cfg["num_samples"]),
        int(cfg["seed"]),
        num_datasets,
    )

    # モデル読み込み
    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # attn_implementationを'eager'に設定してoutput_attentionsを有効化
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager"  # sdpaではoutput_attentionsがサポートされていないためeagerを使用
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # GPU使用状況を表示
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"{'='*60}\n")

    max_length = int(cfg["max_length"])
    lambda_reg = float(cfg["lambda_reg"])

    # 既存のデータを読み込み
    eta_arr = np.load(Path(base_dir) / cfg["embeddings_dir"] / "eta.npy")  # (N, D)
    G_arr = np.load(Path(base_dir) / cfg["metrics_dir"] / "G_theta.npy")  # (N, D, D)

    N, D = eta_arr.shape

    # 接続係数、リーマン曲率テンソル、スカラー曲率を保存するリスト
    gamma_list: list = []
    riemann_list: list = []
    kappa_list: list = []

    # モデルの設定
    gpt = model.transformer  # type: ignore[attr-defined]
    last_block = gpt.h[-1]
    c_attn = last_block.attn.c_attn
    n_head: int = gpt.config.n_head

    # データをGPUに転送（計量行列はCPUでpinvhを計算する必要があるため、後で転送）
    eta_arr_gpu = torch.from_numpy(eta_arr).float().to(device)  # (N, D)
    G_arr_gpu = torch.from_numpy(G_arr).float().to(device)  # (N, D, D)

    for i, text in enumerate(tqdm(texts, desc="Computing curvature", unit="sample")):

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

        # 最後のレイヤーの入力
        hidden_states: Tuple[torch.Tensor, ...] = outputs.hidden_states  # type: ignore
        h_in_last = hidden_states[-2]  # (B,T,D)

        # seq_lenを先に計算
        seq_len = int(attention_mask.sum().item())
        t_star = seq_len - 1

        # q,k,v を計算
        q, k, v = _compute_qkv_from_hidden(h_in_last, c_attn, n_head)
        v_concat = _concat_heads(v).squeeze(0).float()  # (T, D) - GPU上に保持

        # アテンション重み
        if outputs.attentions is not None and len(outputs.attentions) > 0:
            attn_last = outputs.attentions[-1]  # type: ignore
            attn = attn_last.squeeze(0)  # (H, T, T)
        else:
            # attentionsが取得できない場合、均等重みを使用
            print(f"Warning: attentions not available for sample {i}, using uniform weights")
            attn = torch.ones((n_head, max_length, max_length), device=device) / max_length

        w = attn[:, t_star, :].mean(dim=0)  # (T,)
        valid_mask = attention_mask.squeeze(0).bool()
        w = w * valid_mask
        w = w[:seq_len]
        w = w / (w.sum() + 1e-12)  # (T_valid,) - GPU上に保持

        z = v_concat[:seq_len, :]  # (T_valid, D) - GPU上に保持
        eta = eta_arr_gpu[i]  # (D,) - GPU上に保持

        # 計量行列とその逆行列
        g = G_arr_gpu[i]  # (D, D) - GPU上に保持
        # pinvhはCPUで実行する必要があるため、一時的にCPUに転送
        g_cpu = g.cpu().numpy()
        g_inv_cpu = pinvh_stable(g_cpu, jitter=lambda_reg)
        g_inv = torch.from_numpy(g_inv_cpu).float().to(device)  # (D, D) - GPUに戻す

        # 接続係数 Γ_{ijk} を計算（GPU上）
        gamma_lower = compute_connection_coefficient_gpu(z, w, eta)  # (D, D, D)

        # 混合形式の接続係数 Γ_{jk}^l を計算（GPU上）
        gamma_mixed = compute_connection_coefficient_mixed_gpu(gamma_lower, g_inv)  # (D, D, D)

        # 接続係数の偏微分 ∂_iΓ_{jk}^l を計算（GPU上）
        partial_gamma_mixed = compute_connection_derivative_gpu(
            z, w, eta, g, g_inv, gamma_lower
        )  # (D, D, D, D)

        # リーマン曲率テンソル R_{ijk}^l を計算（GPU上）
        R = compute_riemann_curvature_gpu(partial_gamma_mixed, gamma_mixed)  # (D, D, D, D)

        # スカラー曲率 κ を計算（GPU上）
        kappa = compute_scalar_curvature_gpu(R, g, g_inv)

        # CPUに転送して保存用リストに追加（勾配計算を無効化）
        gamma_list.append(gamma_lower.detach().cpu().numpy().astype(np.float32))
        riemann_list.append(R.detach().cpu().numpy().astype(np.float32))
        kappa_list.append(kappa)

    # 結果を保存
    gamma_arr = np.stack(gamma_list, axis=0)  # (N, D, D, D)
    riemann_arr = np.stack(riemann_list, axis=0)  # (N, D, D, D, D)
    kappa_arr = np.array(kappa_list, dtype=np.float32)  # (N,)

    save_npy(
        Path(base_dir) / cfg["curvature_dir"] / "gamma_ijk.npy", gamma_arr
    )
    save_npy(
        Path(base_dir) / cfg["curvature_dir"] / "riemann_curvature.npy", riemann_arr
    )
    save_npy(
        Path(base_dir) / cfg["curvature_dir"] / "scalar_curvature.npy", kappa_arr
    )

    print("Saved:")
    print("  gamma_ijk:", Path(base_dir) / cfg["curvature_dir"] / "gamma_ijk.npy")
    print(
        "  riemann_curvature:",
        Path(base_dir) / cfg["curvature_dir"] / "riemann_curvature.npy",
    )
    print(
        "  scalar_curvature:",
        Path(base_dir) / cfg["curvature_dir"] / "scalar_curvature.npy",
    )
    print(f"\nScalar curvature statistics:")
    print(f"  Mean: {kappa_arr.mean():.6f}")
    print(f"  Std: {kappa_arr.std():.6f}")
    print(f"  Min: {kappa_arr.min():.6f}")
    print(f"  Max: {kappa_arr.max():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    args = parser.parse_args()
    run(args.config)

