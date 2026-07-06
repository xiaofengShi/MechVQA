"""
VQA 数据集划分脚本（按 data_source 分层；同图多 QA 绑定；Chinese-CLIP 多模态聚类）

依赖：
  pip install torch transformers scikit-learn pillow tqdm numpy matplotlib modelscope(optional)

说明：
- 你可以把 model_name_or_path 设为 ModelScope 下载后的本地目录
- 或者设为 ModelScope repo_id，并确保安装了 modelscope（会自动 snapshot_download）
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple, Optional
import random

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .taxonomy import normalize_capability, normalize_difficulty, normalize_subcategory
except ImportError:
    from taxonomy import normalize_capability, normalize_difficulty, normalize_subcategory

# ---------------------------
# Matplotlib fonts (Chinese)
# ---------------------------

def setup_matplotlib_fonts():
    """设置 matplotlib 字体，支持中文显示（若找不到则退化为英文 label）"""
    import matplotlib.font_manager as fm

    chinese_fonts = [
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans CN",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}

    selected = None
    for fn in chinese_fonts:
        if fn in available:
            selected = fn
            break

    if selected:
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
        print(f"[FONT] Using: {selected}")
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        print("[FONT] WARNING: No Chinese font found; will fallback to English labels when possible.")

    plt.rcParams["axes.unicode_minus"] = False

setup_matplotlib_fonts()

DISPLAY_LABEL_ALIASES = {
    "中文": "Chinese",
    "英文": "English",
    "未分类": "Uncategorized",
}

def safe_label(x: str) -> str:
    value = DISPLAY_LABEL_ALIASES.get(x, x)
    value = normalize_capability(value)
    value = normalize_difficulty(value)
    value = normalize_subcategory(value)
    return value


# ---------------------------
# I/O
# ---------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def save_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------
# Field extractors
# ---------------------------

def get_metadata_value(record: Dict[str, Any], field: str, default: str = "unknown") -> str:
    md = record.get("metadata", {})
    v = md.get(field, default)
    if v is None:
        return default
    if isinstance(v, str) and not v.strip():
        return default
    if field == "capability":
        return normalize_capability(v) or default
    if field == "difficulty":
        return normalize_difficulty(v) or default
    if field == "subcategory":
        return normalize_subcategory(v) or default
    return str(v)


def get_data_source(record: Dict[str, Any]) -> str:
    return get_metadata_value(record, "data_source", default="unknown")


def get_image_path(record: Dict[str, Any]) -> str:
    imgs = record.get("images", [])
    if isinstance(imgs, list) and len(imgs) > 0:
        return str(imgs[0])
    return ""


def extract_question_text(record: Dict[str, Any]) -> str:
    """
    优先用 metadata.original_q（你给的样例里有），否则从 messages 里抽取。
    - 会去掉 <image> 前缀
    - 只取第一行“问题句子”（避免把后面的 instruction 一起编码进去）
    """
    md = record.get("metadata", {})
    if isinstance(md, dict) and md.get("original_q"):
        q = str(md["original_q"]).strip()
        q = q.replace("<image>", "").strip()
        return q

    msgs = record.get("messages", [])
    if not isinstance(msgs, list):
        return ""
    for m in msgs:
        if m.get("role") == "user":
            content = str(m.get("content", "")).strip()
            content = content.replace("<image>", "").strip()
            # 只取第一段非空行
            for line in content.splitlines():
                line = line.strip()
                if line:
                    return line
            return content
    return ""


def build_text_for_embedding(record: Dict[str, Any], add_meta: bool = True) -> str:
    q = extract_question_text(record)
    if not add_meta:
        return q
    cap = get_metadata_value(record, "capability", default="unknown")
    sub = get_metadata_value(record, "subcategory", default="unknown")
    # 用短前缀，避免把 prompt 撑爆，也让类别信息参与 embedding
    return f"{q}\n[capability]{cap}\n[subcategory]{sub}"


# ---------------------------
# Model resolve & embedding
# ---------------------------

def resolve_model_dir(model_name_or_path: str) -> str:
    """
    - 如果是本地目录：直接用
    - 否则尝试用 modelscope.snapshot_download(repo_id) 拉到本地，再返回本地目录
    """
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    try:
        from modelscope import snapshot_download
        local_dir = snapshot_download(model_name_or_path)
        if os.path.isdir(local_dir):
            return local_dir
        return model_name_or_path
    except Exception:
        # 没装 modelscope 或下载失败：让 transformers 自己处理（如果是 HF repo id 则可用）
        return model_name_or_path

@torch.inference_mode()
def encode_texts_chinese_clip(
    model,
    processor,
    texts,
    device: str,
    batch_size: int = 128,
    max_length: int = 64,
):
    all_emb = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encode texts"):
        batch = texts[i:i+batch_size]
        inputs = processor(
            text=batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model.text_model(**inputs, return_dict=True)

        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            # 推荐：CLS pooling（最接近 CLIP 的用法）
            pooled = out.last_hidden_state[:, 0]

            # 如果你更偏好 mean pooling（有时更稳）：
            # mask = inputs.get("attention_mask", None)
            # if mask is None:
            #     pooled = out.last_hidden_state.mean(dim=1)
            # else:
            #     mask = mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
            #     pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        feat = model.text_projection(pooled)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        all_emb.append(feat.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(all_emb, axis=0)

def _load_image_rgb(path: str) -> Optional[Image.Image]:
    if not path or not os.path.exists(path):
        return None
    try:
        img = Image.open(path)
        # GIF / PNG / JPG 统一转 RGB
        return img.convert("RGB")
    except Exception:
        return None


@torch.inference_mode()
def encode_images_chinese_clip(
    model,
    processor,
    image_paths: List[str],
    device: str,
    batch_size: int = 32
) -> np.ndarray:
    all_emb = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encode images"):
        batch_paths = image_paths[i:i+batch_size]
        imgs = []
        valid_mask = []
        for p in batch_paths:
            img = _load_image_rgb(p)
            if img is None:
                imgs.append(Image.new("RGB", (336, 336), (0, 0, 0)))  # 占位
                valid_mask.append(False)
            else:
                imgs.append(img)
                valid_mask.append(True)

        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feat = model.get_image_features(**inputs)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        feat = feat.detach().cpu().numpy()

        # 对无效图片：置零向量，避免引入噪声
        for j, ok in enumerate(valid_mask):
            if not ok:
                feat[j, :] = 0.0

        all_emb.append(feat)

    return np.concatenate(all_emb, axis=0)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def fuse_group_embeddings(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    mode: str = "concat",   # "concat" or "sum"
    w_img: float = 1.0,
    w_txt: float = 1.0
) -> np.ndarray:
    """
    img_emb/txt_emb 均假设已 L2 normalize（Chinese-CLIP 常规用法）
    """
    if mode == "sum":
        fused = w_img * img_emb + w_txt * txt_emb
        return l2_normalize(fused)

    if mode == "concat":
        fused = np.concatenate([w_img * img_emb, w_txt * txt_emb], axis=1)
        return l2_normalize(fused)

    raise ValueError(f"Unknown fuse mode: {mode}")


# ---------------------------
# Clustering & split
# ---------------------------

def cluster_labels(emb: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    n = len(emb)
    if n < 3:
        return np.zeros(n, dtype=int)

    k = min(n_clusters, n // 3)
    k = max(k, 2)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
    labels = km.fit_predict(emb)
    return labels


def stratified_split_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(indices)
    if n < 3:
        return indices, np.array([], dtype=int), np.array([], dtype=int)

    # 过小类合并到最大类，避免 StratifiedShuffleSplit 报错
    cnt = Counter(labels)
    adjusted = labels.copy()
    small = [lab for lab, c in cnt.items() if c < 2]
    if small and len(cnt) > 1:
        largest = max(cnt.items(), key=lambda x: x[1])[0]
        for lab in small:
            adjusted[adjusted == lab] = largest

    vt = val_ratio + test_ratio
    try:
        s1 = StratifiedShuffleSplit(n_splits=1, test_size=vt, random_state=random_state)
        train_mask, vt_mask = next(s1.split(np.arange(n), adjusted))
    except ValueError:
        rng = np.random.RandomState(random_state)
        perm = rng.permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        tr = indices[perm[:n_train]]
        va = indices[perm[n_train:n_train+n_val]]
        te = indices[perm[n_train+n_val:]]
        return tr, va, te

    tr_idx = indices[train_mask]
    vt_idx = indices[vt_mask]
    vt_lab = adjusted[vt_mask]

    if len(vt_idx) < 2:
        return tr_idx, vt_idx, np.array([], dtype=int)

    rel_val = val_ratio / (val_ratio + test_ratio)
    try:
        s2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - rel_val, random_state=random_state)
        val_mask, test_mask = next(s2.split(np.arange(len(vt_idx)), vt_lab))
        va_idx = vt_idx[val_mask]
        te_idx = vt_idx[test_mask]
    except ValueError:
        rng = np.random.RandomState(random_state + 1)
        perm = rng.permutation(len(vt_idx))
        n_val = int(len(vt_idx) * rel_val)
        va_idx = vt_idx[perm[:n_val]]
        te_idx = vt_idx[perm[n_val:]]

    return tr_idx, va_idx, te_idx


def sample_jsonl(
    input_path: str,
    output_path: str,
    n_samples: int,
    random_state: int = 42
) -> None:
    """
    从输入的 JSONL 文件中随机采样指定数量的记录，保存到新文件
    
    Args:
        input_path: 输入 JSONL 文件路径
        output_path: 输出 JSONL 文件路径
        n_samples: 采样数量
        random_state: 随机种子
    """
    print(f"Loading data from: {input_path}")
    records = load_jsonl(input_path)
    print(f"Total records: {len(records)}")
    
    if n_samples >= len(records):
        print(f"Sample size ({n_samples}) >= total records ({len(records)}), using all data")
        sampled = records
    else:
        random.seed(random_state)
        sampled = random.sample(records, n_samples)
        print(f"Randomly sampled {n_samples} records")
    
    save_jsonl(sampled, output_path)
    print(f"Saved to: {output_path}")


# ---------------------------
# Analysis & Visualization
# ---------------------------

def _topk_with_other(counter: Counter, top_k: int = 30) -> Tuple[List[str], List[int]]:
    """取 Top-K 类别，其余合并为 Other。返回 (labels, counts)。"""
    items = counter.most_common()
    if len(items) <= top_k:
        labels = [k for k, _ in items]
        counts = [v for _, v in items]
        return labels, counts

    top = items[:top_k]
    rest = items[top_k:]
    labels = [k for k, _ in top] + ["Other"]
    counts = [v for _, v in top] + [sum(v for _, v in rest)]
    return labels, counts


def analyze_split_distribution(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    fields: List[str],
    title: str = ""
) -> None:
    print("\n" + "=" * 80)
    print(f"[Distribution Analysis] {title}")
    print("=" * 80)

    for field in fields:
        train_dist = Counter(get_metadata_value(r, field) for r in train_records)
        val_dist = Counter(get_metadata_value(r, field) for r in val_records)
        test_dist = Counter(get_metadata_value(r, field) for r in test_records)

        all_values = sorted(set(train_dist) | set(val_dist) | set(test_dist))

        print(f"\nField: {field}")
        print(f"{'Value':>30} | {'Train':>8} | {'Val':>8} | {'Test':>8} | {'Train%':>8} | {'Val%':>8} | {'Test%':>8}")
        print("-" * 90)

        for v in all_values:
            tr = train_dist.get(v, 0)
            va = val_dist.get(v, 0)
            te = test_dist.get(v, 0)
            tot = tr + va + te
            if tot == 0:
                continue
            print(f"{v:>30} | {tr:>8} | {va:>8} | {te:>8} | {tr/tot*100:>7.1f}% | {va/tot*100:>7.1f}% | {te/tot*100:>7.1f}%")

    print("\n" + "=" * 80)


def visualize_split_distribution(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    fields: List[str],
    output_dir: str,
    title_prefix: str = "",
    embeddings: Optional[np.ndarray] = None,     # for t-SNE (usually group embeddings)
    split_ids: Optional[np.ndarray] = None,      # 0=train,1=val,2=test for each row in embeddings
    max_tsne_points: int = 5000,
    top_k: int = 30,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # 1) bar charts per field
    for field in fields:
        _plot_field_distribution(train_records, val_records, test_records, field, output_dir, title_prefix, top_k=top_k)

    # 2) dataset size pie chart
    _plot_split_pie_chart(train_records, val_records, test_records, output_dir, title_prefix)

    # 3) heatmap for first 2 fields (if provided)
    if len(fields) >= 2:
        _plot_stratify_heatmap(train_records, val_records, test_records, fields[0], fields[1], output_dir, title_prefix, top_k=top_k)

    # 4) t-SNE (if embeddings + split_ids)
    if embeddings is not None and split_ids is not None:
        _plot_tsne_visualization(
            embeddings=embeddings,
            split_ids=split_ids,
            output_dir=output_dir,
            title_prefix=title_prefix,
            max_points=max_tsne_points
        )

    # 5) summary report
    plot_summary_report(train_records, val_records, test_records, fields, output_dir, title_prefix)

    print(f"[VIS] Saved visualizations to: {output_dir}")


def _plot_field_distribution(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    field: str,
    output_dir: str,
    title_prefix: str,
    top_k: int = 30,
) -> None:
    train_dist = Counter(get_metadata_value(r, field) for r in train_records)
    val_dist = Counter(get_metadata_value(r, field) for r in val_records)
    test_dist = Counter(get_metadata_value(r, field) for r in test_records)

    # 用总分布来选 Top-K
    all_dist = train_dist + val_dist + test_dist
    labels, _ = _topk_with_other(all_dist, top_k=top_k)

    def _get_count(dist: Counter, k: str) -> int:
        if k == "Other":
            # 其它类别合并
            return sum(v for kk, v in dist.items() if kk not in labels[:-1])
        return dist.get(k, 0)

    train_counts = [_get_count(train_dist, k) for k in labels]
    val_counts = [_get_count(val_dist, k) for k in labels]
    test_counts = [_get_count(test_dist, k) for k in labels]

    display_labels = [safe_label(k) for k in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # abs
    ax1 = axes[0]
    bars1 = ax1.bar(x - width, train_counts, width, label="Train", alpha=0.85)
    bars2 = ax1.bar(x, val_counts, width, label="Val", alpha=0.85)
    bars3 = ax1.bar(x + width, test_counts, width, label="Test", alpha=0.85)

    ax1.set_xlabel(field)
    ax1.set_ylabel("Count")
    ax1.set_title(f"{title_prefix} Distribution by {field} (Count)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_labels, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax1.annotate(f"{int(h)}", (b.get_x() + b.get_width()/2, h), xytext=(0, 2),
                             textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # pct within each value
    ax2 = axes[1]
    train_pcts, val_pcts, test_pcts = [], [], []
    for i in range(len(labels)):
        tot = train_counts[i] + val_counts[i] + test_counts[i]
        if tot == 0:
            train_pcts.append(0); val_pcts.append(0); test_pcts.append(0)
        else:
            train_pcts.append(train_counts[i]/tot*100)
            val_pcts.append(val_counts[i]/tot*100)
            test_pcts.append(test_counts[i]/tot*100)

    ax2.bar(x - width, train_pcts, width, label="Train", alpha=0.85)
    ax2.bar(x, val_pcts, width, label="Val", alpha=0.85)
    ax2.bar(x + width, test_pcts, width, label="Test", alpha=0.85)

    ax2.set_xlabel(field)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title(f"{title_prefix} Distribution by {field} (Pct within value)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_labels, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    out = os.path.join(output_dir, f"distribution_{field}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIS] saved: {out}")


def _plot_split_pie_chart(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    output_dir: str,
    title_prefix: str,
) -> None:
    sizes = [len(train_records), len(val_records), len(test_records)]
    labels = [f"Train\n{sizes[0]}", f"Val\n{sizes[1]}", f"Test\n{sizes[2]}"]
    explode = (0.02, 0.02, 0.02)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, explode=explode,
        autopct="%1.1f%%", shadow=True, startangle=90
    )
    ax.set_title(f"{title_prefix} Split Ratio (Total={sum(sizes)})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(output_dir, "split_pie_chart.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIS] saved: {out}")


def _plot_stratify_heatmap(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    field1: str,
    field2: str,
    output_dir: str,
    title_prefix: str,
    top_k: int = 25,
) -> None:
    all_records = train_records + val_records + test_records
    all1 = Counter(get_metadata_value(r, field1) for r in all_records)
    all2 = Counter(get_metadata_value(r, field2) for r in all_records)

    values1, _ = _topk_with_other(all1, top_k=top_k)
    values2, _ = _topk_with_other(all2, top_k=top_k)

    def _map_value(v: str, keep: List[str]) -> str:
        if v in keep[:-1]:
            return v
        return "Other" if keep[-1] == "Other" else v

    def _matrix(recs: List[Dict[str, Any]]) -> np.ndarray:
        m = np.zeros((len(values1), len(values2)), dtype=np.int32)
        idx1 = {v:i for i,v in enumerate(values1)}
        idx2 = {v:i for i,v in enumerate(values2)}
        for r in recs:
            v1 = _map_value(get_metadata_value(r, field1), values1)
            v2 = _map_value(get_metadata_value(r, field2), values2)
            m[idx1[v1], idx2[v2]] += 1
        return m

    mats = [
        (_matrix(train_records), "Train"),
        (_matrix(val_records), "Val"),
        (_matrix(test_records), "Test"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (m, name) in zip(axes, mats):
        im = ax.imshow(m, aspect="auto")
        ax.set_title(f"{title_prefix} {name} ({m.sum()} samples)")
        ax.set_xlabel(field2)
        ax.set_ylabel(field1)
        ax.set_xticks(np.arange(len(values2)))
        ax.set_yticks(np.arange(len(values1)))
        ax.set_xticklabels([safe_label(v) for v in values2], rotation=45, ha="right")
        ax.set_yticklabels([safe_label(v) for v in values1])

        # annotate small heatmap values
        if len(values1) <= 20 and len(values2) <= 20:
            mx = m.max() if m.max() > 0 else 1
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    if m[i, j] > 0:
                        ax.text(j, i, int(m[i, j]), ha="center", va="center",
                                color="white" if m[i, j] > mx/2 else "black", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.85)

    plt.suptitle(f"{title_prefix} Heatmap: {field1} x {field2}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, f"heatmap_{field1}_{field2}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIS] saved: {out}")


def _plot_tsne_visualization(
    embeddings: np.ndarray,
    split_ids: np.ndarray,
    output_dir: str,
    title_prefix: str,
    max_points: int = 5000,
) -> None:
    from sklearn.manifold import TSNE
    import sklearn

    n = len(embeddings)
    if n == 0:
        return

    # downsample if too many
    if n > max_points:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_points, replace=False)
        emb = embeddings[idx]
        sid = split_ids[idx]
    else:
        emb = embeddings
        sid = split_ids

    if len(emb) < 3:
        print(f"[VIS] skip t-SNE: need at least 3 points, got {len(emb)}")
        return

    perplexity = min(30, max(1, len(emb) // 3))
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    except TypeError:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    else:
        tsne.set_params(perplexity=perplexity)

    print("[VIS] t-SNE running ...")
    emb2d = tsne.fit_transform(emb)

    # 0=train,1=val,2=test
    color_map = {0: "#2ecc71", 1: "#3498db", 2: "#e74c3c"}
    colors = [color_map.get(int(x), "#95a5a6") for x in sid]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(emb2d[:, 0], emb2d[:, 1], c=colors, s=18, alpha=0.65)
    ax.set_title(f"{title_prefix} t-SNE of Split", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE dim-1")
    ax.set_ylabel("t-SNE dim-2")
    ax.grid(alpha=0.3)

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=color_map[0], label="Train", alpha=0.65),
        Patch(facecolor=color_map[1], label="Val", alpha=0.65),
        Patch(facecolor=color_map[2], label="Test", alpha=0.65),
    ]
    ax.legend(handles=legend, loc="upper right")

    plt.tight_layout()
    out = os.path.join(output_dir, "tsne_split.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIS] saved: {out}")


def plot_summary_report(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    fields: List[str],
    output_dir: str,
    title_prefix: str,
) -> None:
    total = len(train_records) + len(val_records) + len(test_records)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # pie
    ax1 = fig.add_subplot(gs[0, 0])
    sizes = [len(train_records), len(val_records), len(test_records)]
    ax1.pie(sizes, labels=["Train", "Val", "Test"], autopct="%1.1f%%", startangle=90)
    ax1.set_title(f"{title_prefix} Split Summary (Total={total})", fontsize=13, fontweight="bold")

    # stacked by first field
    ax2 = fig.add_subplot(gs[0, 1])
    if fields:
        f = fields[0]
        tr = Counter(get_metadata_value(r, f) for r in train_records)
        va = Counter(get_metadata_value(r, f) for r in val_records)
        te = Counter(get_metadata_value(r, f) for r in test_records)
        allc = tr + va + te
        keys, _ = _topk_with_other(allc, top_k=20)

        def _cnt(dist: Counter, k: str) -> int:
            if k == "Other":
                return sum(v for kk, v in dist.items() if kk not in keys[:-1])
            return dist.get(k, 0)

        xs = np.arange(len(keys))
        trc = np.array([_cnt(tr, k) for k in keys])
        vac = np.array([_cnt(va, k) for k in keys])
        tec = np.array([_cnt(te, k) for k in keys])

        ax2.bar(xs, trc, label="Train", alpha=0.85)
        ax2.bar(xs, vac, bottom=trc, label="Val", alpha=0.85)
        ax2.bar(xs, tec, bottom=trc+vac, label="Test", alpha=0.85)
        ax2.set_xticks(xs)
        ax2.set_xticklabels([safe_label(k) for k in keys], rotation=45, ha="right")
        ax2.set_title(f"{title_prefix} Stacked by {f}", fontsize=13, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)
        ax2.legend()

    # table
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    table_data = [["Dataset", "Count", "Percentage"]]
    table_data += [
        ["Train", str(len(train_records)), f"{len(train_records)/total*100:.1f}%"],
        ["Val",   str(len(val_records)),   f"{len(val_records)/total*100:.1f}%"],
        ["Test",  str(len(test_records)),  f"{len(test_records)/total*100:.1f}%"],
        ["Total", str(total), "100.0%"],
    ]

    for f in fields:
        table_data.append(["", "", ""])
        table_data.append([f"=== {f} ===", "", ""])
        all_dist = Counter(get_metadata_value(r, f) for r in (train_records + val_records + test_records))
        for k, v in all_dist.most_common(30):
            table_data.append([f"  {safe_label(k)}", str(v), f"{v/total*100:.1f}%"])
        if len(all_dist) > 30:
            rest = sum(v for _, v in all_dist.most_common()[30:])
            table_data.append(["  Other", str(rest), f"{rest/total*100:.1f}%"])

    table = ax3.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.38, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.35)

    # header style
    for i in range(3):
        table[(0, i)].set_facecolor("#34495e")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    plt.suptitle(f"{title_prefix} Summary Report", fontsize=16, fontweight="bold", y=0.98)
    out = os.path.join(output_dir, "summary_report.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIS] saved: {out}")



def split_vqa_dataset(
    input_path: str,
    output_dir: str,
    model_name_or_path: str,
    device: str = "cuda",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    n_clusters_per_source: int = 30,
    fuse_mode: str = "concat",  # "concat" or "sum"
    w_img: float = 1.0,
    w_txt: float = 1.0,
    add_meta_to_text: bool = True,
    inner_stratify_fields: Optional[List[str]] = None,  # e.g. ["capability", "subcategory"]
    random_state: int = 42,
    cache_dir: Optional[str] = None,
    embedding_cache_prefix: Optional[str] = None,
    n_samples: Optional[int] = None  # 新增参数：如果不为 None，则先采样
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Load records: {input_path}")
    records = load_jsonl(input_path)
    print(f"Total records: {len(records)}")
    
    # 新增：如果指定了采样数量，则先采样
    if n_samples is not None and n_samples < len(records):
        print(f"\n[SAMPLING] Randomly sampling {n_samples} records for testing...")
        random.seed(random_state)
        records = random.sample(records, n_samples)
        print(f"Sampled records: {len(records)}")

    # 1) 先按 data_source 划分 record 索引
    by_source: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_source[get_data_source(r)].append(i)

    print(f"Found data_source groups: {len(by_source)}")
    for k, v in sorted(by_source.items(), key=lambda x: -len(x[1])):
        print(f"  {k}: {len(v)} records")

    # 2) 在每个 data_source 内，按 image 聚合成 group
    # group_id: 0..n_groups-1 (global)
    group_key_to_gid: Dict[Tuple[str, str], int] = {}
    gid_to_info: List[Dict[str, Any]] = []
    record_to_gid = np.full(len(records), -1, dtype=int)

    def _new_group(source: str, img_path: str) -> int:
        gid = len(gid_to_info)
        gid_to_info.append({
            "data_source": source,
            "image_path": img_path,
            "record_indices": [],
            "meta_counts": defaultdict(Counter),  # field -> Counter
        })
        return gid

    for source, rec_ids in by_source.items():
        for rid in rec_ids:
            img_path = get_image_path(records[rid])
            key = (source, img_path)
            if key not in group_key_to_gid:
                group_key_to_gid[key] = _new_group(source, img_path)
            gid = group_key_to_gid[key]
            record_to_gid[rid] = gid
            gid_to_info[gid]["record_indices"].append(rid)

            # 统计 group 内 capability/subcategory 分布（用于 inner_stratify_fields）
            md = records[rid].get("metadata", {})
            if isinstance(md, dict):
                for f in ["capability", "subcategory", "language", "difficulty"]:
                    gid_to_info[gid]["meta_counts"][f][get_metadata_value(records[rid], f)] += 1

    n_groups = len(gid_to_info)
    print(f"Image-groups (by data_source + image_path): {n_groups}")

    # 3) 加载 Chinese-CLIP（可用本地目录）
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    resolved = resolve_model_dir(model_name_or_path)
    print(f"Model dir/path: {resolved}")

    model = ChineseCLIPModel.from_pretrained(
        resolved,
        local_files_only=os.path.isdir(resolved),  # 目录存在就不联网
        cache_dir=cache_dir
    ).to(device).eval()

    processor = ChineseCLIPProcessor.from_pretrained(
        resolved,
        local_files_only=os.path.isdir(resolved),
        cache_dir=cache_dir
    )

    # 4) 计算每条 record 的 text embedding，然后 group 内 mean pooling
    texts = [build_text_for_embedding(r, add_meta=add_meta_to_text) for r in records]

    # 可选：缓存
    if embedding_cache_prefix:
        text_cache = f"{embedding_cache_prefix}_text_record_emb.npy"
        if os.path.exists(text_cache):
            print(f"Load cached text embeddings: {text_cache}")
            text_record_emb = np.load(text_cache)
        else:
            text_record_emb = encode_texts_chinese_clip(model, processor, texts, device=device)
            np.save(text_cache, text_record_emb)
            print(f"Saved text embeddings: {text_cache}")
    else:
        text_record_emb = encode_texts_chinese_clip(model, processor, texts, device=device)

    d = text_record_emb.shape[1]
    group_text_sum = np.zeros((n_groups, d), dtype=np.float32)
    group_text_cnt = np.zeros((n_groups,), dtype=np.int32)

    for rid in range(len(records)):
        gid = record_to_gid[rid]
        if gid < 0:
            continue
        group_text_sum[gid] += text_record_emb[rid].astype(np.float32)
        group_text_cnt[gid] += 1

    group_text_emb = group_text_sum / np.maximum(group_text_cnt[:, None], 1)
    group_text_emb = l2_normalize(group_text_emb)

    # 5) 计算每个 group 的 image embedding
    group_image_paths = [g["image_path"] for g in gid_to_info]
    if embedding_cache_prefix:
        img_cache = f"{embedding_cache_prefix}_image_group_emb.npy"
        if os.path.exists(img_cache):
            print(f"Load cached image embeddings: {img_cache}")
            group_img_emb = np.load(img_cache)
        else:
            group_img_emb = encode_images_chinese_clip(model, processor, group_image_paths, device=device)
            np.save(img_cache, group_img_emb)
            print(f"Saved image embeddings: {img_cache}")
    else:
        group_img_emb = encode_images_chinese_clip(model, processor, group_image_paths, device=device)

    # 6) 融合得到 group embedding
    group_fused_emb = fuse_group_embeddings(
        img_emb=group_img_emb,
        txt_emb=group_text_emb,
        mode=fuse_mode,
        w_img=w_img,
        w_txt=w_txt
    )

    # 7) 逐 data_source 划分（保证不同 data_source 不混合）
    train_rids, val_rids, test_rids = [], [], []

    for source, rec_ids in by_source.items():
        # 取该 source 下的 group 列表（去重）
        source_gids = sorted({record_to_gid[rid] for rid in rec_ids if record_to_gid[rid] >= 0})
        source_gids = np.array(source_gids, dtype=int)

        if len(source_gids) == 0:
            continue

        print(f"\nSplit data_source='{source}': groups={len(source_gids)}, records={len(rec_ids)}")

        # 可选：在 source 内再按字段做“高层分层”（group-level majority）
        if inner_stratify_fields:
            buckets = defaultdict(list)
            for gid in source_gids:
                key_parts = []
                for f in inner_stratify_fields:
                    c = gid_to_info[gid]["meta_counts"].get(f, Counter())
                    key_parts.append(c.most_common(1)[0][0] if len(c) else "unknown")
                buckets["|".join(key_parts)].append(gid)

            tr_gids, va_gids, te_gids = [], [], []
            for bkey, b_gids in buckets.items():
                b_gids = np.array(b_gids, dtype=int)
                emb = group_fused_emb[b_gids]
                labs = cluster_labels(emb, n_clusters=n_clusters_per_source, random_state=random_state)
                tr, va, te = stratified_split_indices(
                    indices=b_gids, labels=labs,
                    train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                    random_state=random_state
                )
                tr_gids.extend(tr.tolist())
                va_gids.extend(va.tolist())
                te_gids.extend(te.tolist())

            tr_gids = np.array(tr_gids, dtype=int)
            va_gids = np.array(va_gids, dtype=int)
            te_gids = np.array(te_gids, dtype=int)
        else:
            emb = group_fused_emb[source_gids]
            labs = cluster_labels(emb, n_clusters=n_clusters_per_source, random_state=random_state)
            tr_gids, va_gids, te_gids = stratified_split_indices(
                indices=source_gids, labels=labs,
                train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                random_state=random_state
            )

        # gid -> record indices
        tr_set = set(tr_gids.tolist())
        va_set = set(va_gids.tolist())
        te_set = set(te_gids.tolist())

        for gid in tr_set:
            train_rids.extend(gid_to_info[gid]["record_indices"])
        for gid in va_set:
            val_rids.extend(gid_to_info[gid]["record_indices"])
        for gid in te_set:
            test_rids.extend(gid_to_info[gid]["record_indices"])

        # sanity check：该 source 内图片不交叉
        if (tr_set & va_set) or (tr_set & te_set) or (va_set & te_set):
            raise RuntimeError(f"[BUG] group overlap detected in source={source}")

        print(f"  group split: train={len(tr_set)}, val={len(va_set)}, test={len(te_set)}")

    # 8) 输出 JSONL
    train_records = [records[i] for i in train_rids]
    val_records = [records[i] for i in val_rids]
    test_records = [records[i] for i in test_rids]

    stem = Path(input_path).stem
    out_train = os.path.join(output_dir, f"{stem}_train.jsonl")
    out_val = os.path.join(output_dir, f"{stem}_val.jsonl")
    out_test = os.path.join(output_dir, f"{stem}_test.jsonl")

    save_jsonl(train_records, out_train)
    save_jsonl(val_records, out_val)
    save_jsonl(test_records, out_test)

    # 9) 全局检查：图片是否重复（跨 split）
    def _images_set(recs: List[Dict[str, Any]]) -> set:
        s = set()
        for r in recs:
            p = get_image_path(r)
            if p:
                s.add((get_data_source(r), p))  # 加 data_source 更严格
        return s

    s_tr = _images_set(train_records)
    s_va = _images_set(val_records)
    s_te = _images_set(test_records)

    inter = (s_tr & s_va) | (s_tr & s_te) | (s_va & s_te)
    if inter:
        raise RuntimeError(f"[FAIL] duplicate images across splits: {len(inter)}")
    

        # ---------------------------
    # 10) Visualization (record-level + group-level)
    # ---------------------------

    fields_for_vis = ["data_source", "language", "capability", "subcategory", "difficulty"]

    # (A) Record-level
    analyze_split_distribution(train_records, val_records, test_records, fields_for_vis, title="Record-level (QA items)")
    vis_dir_records = os.path.join(output_dir, "visualizations_records")
    visualize_split_distribution(
        train_records, val_records, test_records,
        fields_for_vis,
        output_dir=vis_dir_records,
        title_prefix="Record-level",
        embeddings=None,
        split_ids=None,
        top_k=30
    )

    # (B) Group-level: each image-group is one sample
    # build "group meta records" to reuse the same plotting functions
    train_gids = sorted(set(record_to_gid[i] for i in train_rids if record_to_gid[i] >= 0))
    val_gids   = sorted(set(record_to_gid[i] for i in val_rids   if record_to_gid[i] >= 0))
    test_gids  = sorted(set(record_to_gid[i] for i in test_rids  if record_to_gid[i] >= 0))

    def _group_majority(gid: int, field: str) -> str:
        c = gid_to_info[gid]["meta_counts"].get(field, Counter())
        return c.most_common(1)[0][0] if len(c) else "unknown"

    def _group_meta_record(gid: int) -> Dict[str, Any]:
        # 伪造一个 record，仅用于可视化/统计（images + metadata）
        md = {
            "data_source": gid_to_info[gid]["data_source"],
            "language": _group_majority(gid, "language"),
            "capability": _group_majority(gid, "capability"),
            "subcategory": _group_majority(gid, "subcategory"),
            # difficulty 可能在你数据里为空，仍然可画
            "difficulty": _group_majority(gid, "difficulty"),
        }
        return {"images": [gid_to_info[gid]["image_path"]], "metadata": md}

    train_group_records = [_group_meta_record(g) for g in train_gids]
    val_group_records   = [_group_meta_record(g) for g in val_gids]
    test_group_records  = [_group_meta_record(g) for g in test_gids]

    analyze_split_distribution(train_group_records, val_group_records, test_group_records, fields_for_vis, title="Group-level (unique images)")

    # t-SNE on group embeddings (recommended to visualize split at image-level)
    # build split_ids aligned with gid order 0..n_groups-1
    split_ids = np.full((n_groups,), -1, dtype=np.int32)
    split_ids[train_gids] = 0
    split_ids[val_gids]   = 1
    split_ids[test_gids]  = 2

    vis_dir_groups = os.path.join(output_dir, "visualizations_groups")
    visualize_split_distribution(
        train_group_records, val_group_records, test_group_records,
        fields_for_vis,
        output_dir=vis_dir_groups,
        title_prefix="Group-level",
        embeddings=group_fused_emb,   # 你前面算好的融合 embedding
        split_ids=split_ids,
        max_tsne_points=5000,
        top_k=30
    )

    print(f"\n[VIS] record-level plots: {vis_dir_records}")
    print(f"[VIS] group-level plots:  {vis_dir_groups}")


    print("\n" + "=" * 80)
    print("Done.")
    print(f"Total records: {len(records)}")
    print(f"Train: {len(train_records)} -> {out_train}")
    print(f"Val:   {len(val_records)} -> {out_val}")
    print(f"Test:  {len(test_records)} -> {out_test}")
    print(f"Total groups: {n_groups} (image uniqueness enforced)")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split VQA message-format JSONL into train/val/test with image-group binding."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--n-clusters-per-source", type=int, default=10)
    parser.add_argument("--fuse-mode", choices=["concat", "sum"], default="concat")
    parser.add_argument("--w-img", type=float, default=1.0)
    parser.add_argument("--w-txt", type=float, default=1.0)
    parser.add_argument("--no-meta-text", action="store_true")
    parser.add_argument(
        "--inner-stratify-field",
        action="append",
        default=None,
        dest="inner_stratify_fields",
        help="Optional metadata field for inner stratification. Can be passed multiple times.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--embedding-cache-prefix", default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    embedding_cache_prefix = args.embedding_cache_prefix or os.path.join(args.output_dir, "emb_cache")
    split_vqa_dataset(
        input_path=args.input_jsonl,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        n_clusters_per_source=args.n_clusters_per_source,
        fuse_mode=args.fuse_mode,
        w_img=args.w_img,
        w_txt=args.w_txt,
        add_meta_to_text=not args.no_meta_text,
        inner_stratify_fields=args.inner_stratify_fields,
        random_state=args.random_state,
        cache_dir=args.cache_dir,
        embedding_cache_prefix=embedding_cache_prefix,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()
