import argparse
import math
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import set_global_seed, ensure_dirs, save_npy, save_json, save_csv_diag


def _select_nonempty_texts(dataset_name: str, config: str, split: str, num: int, seed: int, num_datasets: int = 4) -> Tuple[List[str], List[str]]:
    if dataset_name == "newsgroup":
        return _load_newsgroup_dataset(num, seed)
    elif dataset_name == "multi_source":
        return _load_multi_source_dataset(num, seed, num_datasets)
    elif dataset_name == "JeanKaddour/minipile" or dataset_name == "minipile":
        return _load_minipile_dataset(num, seed, num_datasets)
    else:
        # Original single dataset loading
        ds = load_dataset(dataset_name, config, split=split)
        texts: List[str] = []
        labels: List[str] = []
        
        for ex in ds:
            t = (ex.get("text") or "").strip()
            if t:
                texts.append(t)
                # Extract label from meta field
                meta = ex.get("meta", {})
                if isinstance(meta, dict):
                    label = (meta.get("pile_set_name") or 
                            meta.get("source") or 
                            meta.get("subset") or 
                            "unknown")
                else:
                    label = "unknown"
                labels.append(str(label))
            if len(texts) >= num:
                break
        
        if len(texts) < num:
            raise RuntimeError(f"Not enough non-empty samples in {dataset_name}/{config}:{split}")
        
        return texts, labels


def _load_multi_source_dataset(num: int, seed: int, num_datasets: int = 4) -> Tuple[List[str], List[str]]:
    """Load multiple datasets to simulate The Pile with different sources"""
    import random
    random.seed(seed)
    
    # Define multiple datasets with their labels (using stable Parquet-based datasets)
    # Smaller, more reliable datasets for better performance
    datasets_config = [
        ("wikimedia/wikipedia", "20231101.en", "train", "Wikipedia"),
        ("wikitext", "wikitext-2-raw-v1", "test", "WikiText"),
        ("wikitext", "wikitext-2-raw-v1", "validation", "WikiText-Val"),
        ("squad", "plain_text", "train", "SQuAD"),
        ("imdb", None, "train", "IMDB"),
        ("ag_news", None, "train", "AG-News"),
        ("yelp_review_full", None, "train", "Yelp"),
        ("amazon_polarity", None, "train", "Amazon"),
        ("tweet_eval", "sentiment", "train", "Tweet"),
        ("rotten_tomatoes", None, "train", "RottenTomatoes"),
        ("dbpedia_14", None, "train", "DBpedia"),
        ("tatoeba", None, "train", "Tatoeba"),
    ]
    
    # Limit the number of datasets to use
    num_datasets = min(num_datasets, len(datasets_config))
    datasets_config = datasets_config[:num_datasets]
    
    # Alternative simpler approach: create synthetic labels from wikitext
    # if True:  # Use this simpler approach for now
    #     print("Using simplified approach with synthetic labels...")
    #     return _create_synthetic_labels(num, seed)
    
    texts: List[str] = []
    labels: List[str] = []
    
    # Calculate samples per dataset
    samples_per_dataset = max(1, num // len(datasets_config))
    remaining = num
    
    for dataset_name, config, split, label in datasets_config:
        if remaining <= 0:
            break
            
        try:
            print(f"Loading {dataset_name} ({label})...")
            if config:
                ds = load_dataset(dataset_name, config, split=split, streaming=True)
            else:
                ds = load_dataset(dataset_name, split=split, streaming=True)
            
            # Sample texts from this dataset
            dataset_texts = []
            for ex in ds:
                t = (ex.get("text") or ex.get("content") or "").strip()
                if t and len(t) > 50:  # Filter very short texts
                    dataset_texts.append(t)
                if len(dataset_texts) >= samples_per_dataset:  # Get enough samples
                    break
            
            # Randomly sample from this dataset
            if dataset_texts:
                sample_size = min(samples_per_dataset, len(dataset_texts), remaining)
                sampled_texts = random.sample(dataset_texts, sample_size)
                texts.extend(sampled_texts)
                labels.extend([label] * sample_size)
                remaining -= sample_size
                print(f"  Loaded {sample_size} samples from {label}")
            
        except Exception as e:
            print(f"  Warning: Failed to load {dataset_name}: {e}")
            continue
    
    # If we still need more samples, fill with a fallback dataset
    if remaining > 0:
        try:
            print("Loading fallback dataset (wikitext)...")
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for ex in ds:
                t = (ex.get("text") or "").strip()
                if t:
                    texts.append(t)
                    labels.append("WikiText")
                    remaining -= 1
                    if remaining <= 0:
                        break
        except Exception as e:
            print(f"Warning: Fallback dataset failed: {e}")
    
    if len(texts) < num:
        print(f"Warning: Only found {len(texts)} samples, requested {num}")
    
    return texts[:num], labels[:num]


def _load_minipile_dataset(num: int, seed: int, num_datasets: int = 10) -> Tuple[List[str], List[str]]:
    """Load MiniPile dataset and extract samples from different sources"""
    import random
    random.seed(seed)
    
    # Load MiniPile dataset
    print("Loading JeanKaddour/minipile...")
    try:
        ds = load_dataset("JeanKaddour/minipile", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading MiniPile: {e}")
        print("Falling back to multi_source dataset...")
        return _load_multi_source_dataset(num, seed, num_datasets)
    
    texts: List[str] = []
    labels: List[str] = []
    
    # Collect all samples with their sources
    all_samples = []
    for ex in ds:
        text = (ex.get("text") or "").strip()
        if text and len(text) > 50:
            # Extract source from meta field
            meta = ex.get("meta", {})
            if isinstance(meta, dict):
                source = meta.get("pile_set_name") or meta.get("source") or "unknown"
            else:
                source = "unknown"
            all_samples.append((text, source))
    
    print(f"Found {len(all_samples)} samples from MiniPile")
    
    # Group by source
    from collections import defaultdict
    source_groups = defaultdict(list)
    for text, source in all_samples:
        source_groups[source].append(text)
    
    # Get unique sources
    unique_sources = list(source_groups.keys())
    print(f"Found {len(unique_sources)} unique sources: {unique_sources}")
    
    # Limit number of datasets
    num_datasets = min(num_datasets, len(unique_sources))
    selected_sources = unique_sources[:num_datasets]
    print(f"Using {num_datasets} sources: {selected_sources}")
    
    # Calculate samples per source
    samples_per_source = max(1, num // num_datasets)
    remaining = num
    
    # Sample from each source
    for source in selected_sources:
        if remaining <= 0:
            break
        
        source_texts = source_groups[source]
        sample_size = min(samples_per_source, len(source_texts), remaining)
        
        # Randomly sample from this source
        if source_texts:
            sampled_texts = random.sample(source_texts, sample_size)
            texts.extend(sampled_texts)
            labels.extend([source] * sample_size)
            remaining -= sample_size
            print(f"  Loaded {sample_size} samples from {source}")
    
    # Shuffle to mix sources
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    texts = list(texts[:num])
    labels = list(labels[:num])
    
    print(f"Total loaded: {len(texts)} samples")
    return texts[:num], labels[:num]


def _load_newsgroup_dataset(num: int, seed: int) -> Tuple[List[str], List[str]]:
    """Load 20 Newsgroups dataset with proper labels using sklearn"""
    import random
    from sklearn.datasets import fetch_20newsgroups
    
    random.seed(seed)
    
    # Load the dataset using sklearn
    newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    
    # 20 Newsgroups category names
    category_names = newsgroups.target_names
    
    texts = []
    labels = []
    
    for text, label_id in zip(newsgroups.data, newsgroups.target):
        text = text.strip()
        label_name = category_names[label_id]
        
        if text and len(text) > 50:  # Filter short texts
            texts.append(text)
            labels.append(label_name)
            
            if len(texts) >= num:
                break
    
    # Shuffle to get random distribution
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    texts = list(texts[:num])
    labels = list(labels[:num])
    
    print(f"Loaded {len(texts)} samples from 20 Newsgroups")
    print(f"Categories found: {set(labels)}")
    
    # Count labels
    from collections import Counter
    label_counts = Counter(labels)
    print(f"Category distribution: {dict(label_counts)}")
    
    return texts, labels


def _create_synthetic_labels(num: int, seed: int) -> Tuple[List[str], List[str]]:
    """Create synthetic labels based on text content patterns"""
    import random
    random.seed(seed)
    
    # Load wikitext as base
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = []
    labels = []
    
    # Define label patterns based on content
    label_patterns = {
        "Science": ["science", "research", "study", "experiment", "theory", "scientific", "biology", "chemistry", "physics"],
        "History": ["history", "historical", "war", "battle", "ancient", "century", "empire", "kingdom"],
        "Geography": ["country", "city", "mountain", "river", "ocean", "continent", "population", "capital"],
        "Culture": ["culture", "art", "music", "literature", "tradition", "festival", "religion"],
        "Technology": ["technology", "computer", "software", "internet", "digital", "electronic", "programming"],
        "General": []  # Default category
    }
    
    for ex in ds:
        t = (ex.get("text") or "").strip()
        if t and len(t) > 50:
            texts.append(t)
            
            # Determine label based on content
            t_lower = t.lower()
            assigned_label = "General"
            for label, keywords in label_patterns.items():
                if any(keyword in t_lower for keyword in keywords):
                    assigned_label = label
                    break
            
            labels.append(assigned_label)
            
            if len(texts) >= num:
                break
    
    # Ensure we have a good distribution
    if len(texts) >= num:
        # Shuffle to get random distribution
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)
        texts = list(texts[:num])
        labels = list(labels[:num])
    
    print(f"Created {len(texts)} samples with labels: {set(labels)}")
    return texts, labels


def _compute_qkv_from_hidden(hidden: torch.Tensor, c_attn: torch.nn.Linear, n_head: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # hidden: (B, T, D), c_attn projects to 3D, split into q,k,v
    B, T, D = hidden.shape
    qkv = c_attn(hidden)  # (B, T, 3D)
    q, k, v = qkv.split(D, dim=-1)
    # reshape to (B, n_head, T, d_head) then concat heads back for v later if needed
    d_head = D // n_head
    q = q.view(B, T, n_head, d_head).permute(0, 2, 1, 3).contiguous()  # (B, H, T, d_head)
    k = k.view(B, T, n_head, d_head).permute(0, 2, 1, 3).contiguous()
    v = v.view(B, T, n_head, d_head).permute(0, 2, 1, 3).contiguous()
    return q, k, v


def _concat_heads(v: torch.Tensor) -> torch.Tensor:
    # v: (B, H, T, d_head) -> (B, T, H*d_head)
    B, H, T, d_head = v.shape
    return v.permute(0, 2, 1, 3).contiguous().view(B, T, H * d_head)


def _weighted_covariance(z: np.ndarray, w: np.ndarray, eta: np.ndarray, lambda_reg: float) -> np.ndarray:
    # z: (T, D), w: (T,), eta: (D,)
    # effective sample correction
    sum_w2 = float((w ** 2).sum())
    alpha = 1.0 / max(1e-8, (1.0 - sum_w2))
    diff = z - eta[None, :]
    # (D,D)
    G = alpha * (diff.T * w) @ diff
    # regularization
    G = G + lambda_reg * np.eye(G.shape[0], dtype=G.dtype)
    return G


def run(config_path: str) -> None:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_global_seed(int(cfg["seed"]))

    base_dir = cfg["artifacts_dir"]
    subdirs = [
        cfg["embeddings_dir"],
        cfg["metrics_dir"],
        cfg["graphs_dir"],
        cfg["geodesic_dir"],
        cfg["plots_dir"],
    ]
    paths = ensure_dirs(base_dir, *subdirs)

    # data
    num_datasets = int(cfg.get("num_datasets", 4))  # Default to 4 if not specified
    texts, labels = _select_nonempty_texts(
        cfg["dataset"]["name"], cfg["dataset"]["config"], cfg["dataset"]["split"], int(cfg["num_samples"]), int(cfg["seed"]), num_datasets  # seed is unused but kept
    )

    # tokenizer and model
    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_length = int(cfg["max_length"])  # fixed length input

    etas: List[np.ndarray] = []
    G_list: List[np.ndarray] = []
    meta: List[Dict[str, str]] = []

    # identify last block c_attn and n_head
    gpt = model.transformer  # type: ignore[attr-defined]
    last_block = gpt.h[-1]
    c_attn = last_block.attn.c_attn  # Linear
    n_head: int = gpt.config.n_head

    for i, text in enumerate(texts):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{len(texts)}...")
        
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=True)

        # last hidden state BEFORE attention of last layer: use hidden_states from transformer blocks
        # transformers returns: hidden_states[0]=embeddings, ..., hidden_states[-1]=final hidden after last block
        # We approximate last block input as hidden_states[-2]
        hidden_states: Tuple[torch.Tensor, ...] = outputs.hidden_states  # type: ignore
        h_in_last = hidden_states[-2]  # (B,T,D)

        # q,k,v at last layer from input to c_attn
        q, k, v = _compute_qkv_from_hidden(h_in_last, c_attn, n_head)
        v_concat = _concat_heads(v).squeeze(0).float().detach().cpu().numpy()  # (T, D)

        # attention of last layer: tuple(len=num_layers). Take last, shape (B,H,T,T)
        attn_last = outputs.attentions[-1]  # type: ignore
        attn = attn_last.squeeze(0)  # (H, T, T)

        # pick last valid token position t*
        seq_len = int(attention_mask.sum().item())
        t_star = seq_len - 1

        # average heads on row t* over columns lambda; then mask to valid tokens only
        w = attn[:, t_star, :].mean(dim=0)  # (T,)
        valid_mask = attention_mask.squeeze(0).bool()
        w = w * valid_mask
        w = w[:seq_len]
        w = w / (w.sum() + 1e-12)
        w_np = w.float().cpu().numpy()  # (T_valid,)

        z_np = v_concat[:seq_len, :]  # (T_valid, D)
        eta = (w_np[:, None] * z_np).sum(axis=0)  # (D,)
        G_theta = _weighted_covariance(z_np, w_np, eta, float(cfg["lambda_reg"]))  # (D,D)

        etas.append(eta.astype(np.float32))
        G_list.append(G_theta.astype(np.float32))
        meta.append({"text": text, "label": labels[i]})

    eta_arr = np.stack(etas, axis=0)  # (N,D)
    G_arr = np.stack(G_list, axis=0)  # (N,D,D)

    # save artifacts
    save_npy(Path(base_dir) / cfg["embeddings_dir"] / "eta.npy", eta_arr)
    save_npy(Path(base_dir) / cfg["metrics_dir"] / "G_theta.npy", G_arr)
    save_csv_diag(Path(base_dir) / cfg["metrics_dir"] / "G_theta_diag.csv", G_arr)
    # Create label statistics
    unique_labels = list(set(labels))
    label_counts = {label: labels.count(label) for label in unique_labels}
    
    save_json(Path(base_dir) / cfg["embeddings_dir"] / "meta.json", {
        "num": len(texts), 
        "model": model_name,
        "texts": texts,  # Save texts for visualization
        "labels": labels,
        "unique_labels": unique_labels,
        "label_counts": label_counts
    })

    print("Saved:")
    print("  eta:", (Path(base_dir) / cfg["embeddings_dir"] / "eta.npy"))
    print("  G_theta:", (Path(base_dir) / cfg["metrics_dir"] / "G_theta.npy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    args = parser.parse_args()
    run(args.config)
