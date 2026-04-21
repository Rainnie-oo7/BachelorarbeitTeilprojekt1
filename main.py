import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer


# ============================================================
# Pfade relativ zu main.py
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT = (
    BASE_DIR
    / "BiomedCLIP_data_pipeline"
    / "_results"
    / "data"
    / "pubmed_open_access_file_list.txt"
)

DEFAULT_OUTPUT_DIR = BASE_DIR / "sampling_outputs"


# ============================================================
# Prototype texts for BERT fallback
# ============================================================

DEFAULT_MODALITY_PROTOTYPES: Dict[str, List[str]] = {
    "ct": [
        "computed tomography ct axial coronal sagittal contrast scan",
        "ct image showing anatomy lesion organ cross section",
    ],
    "mri": [
        "magnetic resonance imaging mri t1 t2 flair diffusion gadolinium",
        "mri image of brain spine body soft tissue joint",
    ],
    "xray": [
        "x ray radiograph chest xray plain film projection ap pa lateral",
        "radiographic image showing bones lungs chest abdomen extremities",
    ],
    "ultrasound": [
        "ultrasound sonography ultrasonography doppler echography echocardiography",
        "ultrasound image showing soft tissue vessels fetus abdominal organs",
    ],
}

DEFAULT_MR_SUBCLASS_PROTOTYPES: Dict[str, List[str]] = {
    "mr_t1": [
        "mri t1 weighted image t1w pre contrast post contrast gadolinium",
        "t1 weighted magnetic resonance imaging anatomy",
    ],
    "mr_t2": [
        "mri t2 weighted image t2w fluid bright signal",
        "t2 weighted magnetic resonance imaging edema fluid",
    ],
    "mr_flair": [
        "mri flair fluid attenuated inversion recovery brain lesion",
        "flair magnetic resonance imaging white matter lesion",
    ],
    "mr_dwi": [
        "mri diffusion weighted imaging dwi adc restricted diffusion",
        "diffusion weighted magnetic resonance imaging acute lesion",
    ],
    "mr_other": [
        "mri magnetic resonance imaging unspecified sequence anatomy pathology",
        "mri scan sequence not clearly t1 t2 flair or dwi",
    ],
}


# ============================================================
# Rules
# ============================================================

RULES_MODALITY = {
    "ultrasound": [
        r"\bultrasound\b",
        r"\bultrasonography\b",
        r"\bsonography\b",
        r"\bsonographic\b",
        r"\bechography\b",
        r"\bechocardiograph\w*\b",
        r"\bdoppler\b",
        r"\bcolor doppler\b",
        r"\bduplex\b",
    ],
    "xray": [
        r"\bx[- ]?ray\b",
        r"\bradiograph\w*\b",
        r"\bplain film\b",
        r"\bprojection radiograph\w*\b",
        r"\bap view\b",
        r"\bpa view\b",
        r"\blateral view\b",
        r"\broentgen\b",
        r"\bröntgen\b",
        r"\bfluoroscopy\b",
        r"\bfluoroscopic\b",
        r"\bc[- ]?arm\b",
    ],
    "ct": [
        r"\bct\b",
        r"\bcomputed tomography\b",
        r"\bhrct\b",
        r"\bmdct\b",
        r"\bcontrast[- ]enhanced ct\b",
        r"\baxial ct\b",
        r"\bcoronal ct\b",
        r"\bsagittal ct\b",
    ],
    "mri": [
        r"\bmri\b",
        r"\bmr image\b",
        r"\bmr imaging\b",
        r"\bmagnetic resonance\b",
        r"\bt1\b",
        r"\bt2\b",
        r"\bflair\b",
        r"\bdwi\b",
        r"\badc\b",
        r"\bgadolinium\b",
        r"\binversion recovery\b",
    ],
}

RULES_MR_SUBCLASS = {
    "mr_t1": [
        r"\bt1\b",
        r"\bt1[- ]weighted\b",
        r"\bt1w\b",
        r"\bgadolinium\b",
        r"\bpostcontrast\b",
        r"\bprecontrast\b",
    ],
    "mr_t2": [
        r"\bt2\b",
        r"\bt2[- ]weighted\b",
        r"\bt2w\b",
        r"\bfast spin echo\b",
        r"\bfse\b",
    ],
    "mr_flair": [
        r"\bflair\b",
        r"\bfluid attenuated inversion recovery\b",
    ],
    "mr_dwi": [
        r"\bdwi\b",
        r"\bdiffusion[- ]weighted\b",
        r"\badc\b",
        r"\brestricted diffusion\b",
    ],
}


# ============================================================
# Args
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic nested subset selection with Rules -> BERT and distribution analysis."
    )

    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Pfad zur TXT/CSV/JSONL-Datei")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Ausgabeordner")
    parser.add_argument("--model-dir", required=True, help="Lokaler BERT-Modellordner")

    parser.add_argument("--txt-sep", default="\t", help=r"Separator der TXT-Datei, Standard: \t")
    parser.add_argument("--txt-has-header", action="store_true")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--caption-column", default="caption")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--bert-margin-threshold", type=float, default=0.03)

    parser.add_argument("--master-pool-size", type=int, default=500000)
    parser.add_argument("--subset-small-size", type=int, default=200000)
    parser.add_argument("--subset-large-size", type=int, default=400000)

    parser.add_argument("--class-small-size", type=int, default=20000)
    parser.add_argument("--class-large-size", type=int, default=40000)

    parser.add_argument("--mr-small-per-subclass", type=int, default=4000)
    parser.add_argument("--mr-large-per-subclass", type=int, default=8000)

    parser.add_argument("--bins-small", type=int, default=100)
    parser.add_argument("--bins-large", type=int, default=100)

    parser.add_argument("--salt-master", default="pmc15m_master_v1")
    parser.add_argument("--salt-class", default="pmc15m_class_v1")
    parser.add_argument("--salt-mr", default="pmc15m_mr_v1")

    return parser.parse_args()


# ============================================================
# Loading
# ============================================================

def parse_txt_line(line: str, sep: str) -> Tuple[str, str]:
    line = line.rstrip("\n")
    parts = line.split(sep, maxsplit=1)
    if len(parts) != 2:
        raise ValueError(
            f"Zeile konnte nicht in Bildname + Caption getrennt werden. "
            f"Separator={repr(sep)}. Zeile: {line[:200]}"
        )
    return parts[0].strip(), parts[1].strip()


def load_txt_pairs(path: Path, sep: str, has_header: bool, image_column: str, caption_column: str) -> pd.DataFrame:
    rows = []

    with path.open("r", encoding="utf8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i == 0 and has_header:
                continue

            image_path, caption = parse_txt_line(line, sep=sep)
            rows.append({
                image_column: image_path,
                caption_column: caption,
            })

    if not rows:
        raise ValueError("Keine Zeilen gefunden.")

    return pd.DataFrame(rows)


def load_table(path_str: str, sep: str, has_header: bool, image_column: str, caption_column: str) -> pd.DataFrame:
    path = Path(path_str)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return load_txt_pairs(path, sep=sep, has_header=has_header, image_column=image_column, caption_column=caption_column)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return load_microsoft_pipeline_json(path)

    raise ValueError(f"Nicht unterstütztes Format: {path}")


def load_microsoft_pipeline_json(path: Path) -> pd.DataFrame:
    rows: List[dict] = []

    with path.open("r", encoding="utf8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            article = json.loads(line)
            pmcid = str(article.get("pmc", "") or article.get("pmcid", ""))
            pmid = str(article.get("pmid", ""))
            location = str(article.get("location", ""))
            figures = article.get("figures", []) or []

            for fig in figures:
                caption = str(fig.get("fig_caption", "") or "").strip()
                if not caption:
                    continue

                rows.append({
                    "pmcid": pmcid,
                    "pmid": pmid,
                    "location": location,
                    "image_path": str(fig.get("graphic_ref", "") or ""),
                    "fig_id": str(fig.get("fig_id", "") or ""),
                    "fig_label": str(fig.get("fig_label", "") or ""),
                    "pair_id": str(fig.get("pair_id", "") or ""),
                    "caption": caption,
                })

    if not rows:
        raise ValueError("Keine Figure-Captions gefunden.")

    return pd.DataFrame(rows)


# ============================================================
# Helpers
# ============================================================

def clean_caption(text: str) -> str:
    text = str(text or "")
    text = " ".join(text.replace("\n", " ").split())
    return text.strip()


def make_sample_id(image_path: str, caption: str) -> str:
    return hashlib.sha1(f"{image_path}|||{caption}".encode("utf8")).hexdigest()


def stable_hash_u64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def hash_score(sample_id: str, salt: str) -> int:
    return stable_hash_u64(f"{salt}::{sample_id}")


def regex_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


# ============================================================
# BERT
# ============================================================

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    vectors = []
    model.eval()

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out = model(**enc)
        emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        vectors.append(emb.cpu().numpy())

    return np.vstack(vectors)


def build_prototype_embeddings(
    prototypes: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int,
    max_length: int,
) -> Dict[str, np.ndarray]:
    result = {}
    for label, texts in prototypes.items():
        embs = encode_texts(texts, tokenizer, model, batch_size, max_length)
        proto = embs.mean(axis=0)
        proto = proto / np.linalg.norm(proto)
        result[label] = proto
    return result


def score_against_prototypes(
    text_embeddings: np.ndarray,
    prototype_embeddings: Dict[str, np.ndarray],
) -> pd.DataFrame:
    score_dict = {}
    for label, proto in prototype_embeddings.items():
        score_dict[label] = text_embeddings @ proto

    scores_df = pd.DataFrame(score_dict)
    top1 = scores_df.idxmax(axis=1)

    sorted_scores = np.sort(scores_df.to_numpy(), axis=1)
    top_vals = sorted_scores[:, -1]
    second_vals = sorted_scores[:, -2] if scores_df.shape[1] > 1 else np.zeros(len(scores_df))
    margins = top_vals - second_vals

    out = pd.DataFrame({
        "predicted_label": top1,
        "top_score": top_vals,
        "score_margin": margins,
    })

    return pd.concat([out, scores_df], axis=1)


# ============================================================
# Rules
# ============================================================

def rule_based_modality(caption: str) -> Optional[str]:
    caption = caption.lower()
    matches = []

    for label, patterns in RULES_MODALITY.items():
        if regex_any(caption, patterns):
            matches.append(label)

    if "mri" in matches:
        return "mri"
    if "ct" in matches:
        return "ct"
    if "ultrasound" in matches:
        return "ultrasound"
    if "xray" in matches:
        return "xray"

    return None


def rule_based_mr_subclass(caption: str) -> Optional[str]:
    caption = caption.lower()

    for label, patterns in RULES_MR_SUBCLASS.items():
        if regex_any(caption, patterns):
            return label

    return None


# ============================================================
# Classification: Rules -> BERT
# ============================================================

def classify_modalities_and_mr_subclasses(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int,
    max_length: int,
    bert_margin_threshold: float,
    caption_column: str,
) -> pd.DataFrame:
    df = df.copy()

    df["rule_modality"] = df[caption_column].map(rule_based_modality)
    df["rule_mr_subclass"] = df[caption_column].map(rule_based_mr_subclass)

    need_bert_mod = df["rule_modality"].isna()
    need_bert_mr = (df["rule_modality"] == "mri") & df["rule_mr_subclass"].isna()

    df["bert_modality"] = None
    df["bert_modality_score"] = np.nan
    df["bert_modality_margin"] = np.nan

    df["bert_mr_subclass"] = None
    df["bert_mr_score"] = np.nan
    df["bert_mr_margin"] = np.nan

    modality_proto_embs = None
    mr_proto_embs = None

    if need_bert_mod.any():
        texts = df.loc[need_bert_mod, caption_column].tolist()
        text_embs = encode_texts(texts, tokenizer, model, batch_size, max_length)

        modality_proto_embs = build_prototype_embeddings(
            DEFAULT_MODALITY_PROTOTYPES, tokenizer, model, batch_size, max_length
        )

        scored = score_against_prototypes(text_embs, modality_proto_embs)
        idx = df.index[need_bert_mod]

        df.loc[idx, "bert_modality"] = scored["predicted_label"].values
        df.loc[idx, "bert_modality_score"] = scored["top_score"].values
        df.loc[idx, "bert_modality_margin"] = scored["score_margin"].values

    if need_bert_mr.any():
        texts = df.loc[need_bert_mr, caption_column].tolist()
        text_embs = encode_texts(texts, tokenizer, model, batch_size, max_length)

        mr_proto_embs = build_prototype_embeddings(
            DEFAULT_MR_SUBCLASS_PROTOTYPES, tokenizer, model, batch_size, max_length
        )

        scored = score_against_prototypes(text_embs, mr_proto_embs)
        idx = df.index[need_bert_mr]

        df.loc[idx, "bert_mr_subclass"] = scored["predicted_label"].values
        df.loc[idx, "bert_mr_score"] = scored["top_score"].values
        df.loc[idx, "bert_mr_margin"] = scored["score_margin"].values

    final_modality = []
    decision_source_modality = []

    for _, row in df.iterrows():
        if pd.notna(row["rule_modality"]):
            final_modality.append(row["rule_modality"])
            decision_source_modality.append("rule")
        elif pd.notna(row["bert_modality"]) and float(row["bert_modality_margin"]) >= bert_margin_threshold:
            final_modality.append(row["bert_modality"])
            decision_source_modality.append("bert_fallback")
        else:
            final_modality.append(None)
            decision_source_modality.append("unresolved")

    df["final_modality"] = final_modality
    df["decision_source_modality"] = decision_source_modality

    final_mr_subclass = []
    decision_source_mr = []

    for _, row in df.iterrows():
        if row["final_modality"] != "mri":
            final_mr_subclass.append(None)
            decision_source_mr.append(None)
            continue

        if pd.notna(row["rule_mr_subclass"]):
            final_mr_subclass.append(row["rule_mr_subclass"])
            decision_source_mr.append("rule")
        elif pd.notna(row["bert_mr_subclass"]) and float(row["bert_mr_margin"]) >= bert_margin_threshold:
            final_mr_subclass.append(row["bert_mr_subclass"])
            decision_source_mr.append("bert_fallback")
        else:
            final_mr_subclass.append("mr_other")
            decision_source_mr.append("mr_default_other")

    df["final_mr_subclass"] = final_mr_subclass
    df["decision_source_mr"] = decision_source_mr

    return df


# ============================================================
# Master subset
# ============================================================

def build_master_subset(
    df: pd.DataFrame,
    master_pool_size: int,
    subset_large_size: int,
    subset_small_size: int,
    salt_master: str,
) -> pd.DataFrame:
    df = df.copy()
    df["master_score"] = df["sample_id"].map(lambda s: hash_score(s, salt_master))
    df = df.sort_values("master_score", kind="mergesort").reset_index(drop=True)

    if len(df) < master_pool_size:
        print(f"[Warnung] Weniger als {master_pool_size} Zeilen vorhanden: {len(df)}")
        master_pool_size = len(df)

    df = df.iloc[:master_pool_size].copy().reset_index(drop=True)
    df["master_rank"] = np.arange(len(df), dtype=np.int64)

    df["is_in_large_subset"] = df["master_rank"] < min(subset_large_size, len(df))
    df["is_in_small_subset"] = df["master_rank"] < min(subset_small_size, len(df))
    return df


# ============================================================
# Distributed deterministic selection
# ============================================================

def assign_bins_by_rank(df: pd.DataFrame, rank_col: str, subset_size: int, n_bins: int) -> pd.Series:
    bin_size = math.ceil(subset_size / n_bins)
    bins = (df[rank_col] // bin_size).clip(upper=n_bins - 1)
    return bins.astype(int)


def distributed_take(
    df_candidates: pd.DataFrame,
    target_n: int,
    bin_col: str,
    score_col: str,
) -> pd.DataFrame:
    if len(df_candidates) == 0:
        return df_candidates.copy()

    groups = {}
    for b, sub in df_candidates.groupby(bin_col, sort=True):
        groups[int(b)] = sub.sort_values(score_col, kind="mergesort").copy().reset_index(drop=False)

    bin_ids = sorted(groups.keys())
    pointers = {b: 0 for b in bin_ids}
    selected_original_indices = []

    while len(selected_original_indices) < target_n:
        progress = False

        for b in bin_ids:
            ptr = pointers[b]
            group = groups[b]
            if ptr < len(group):
                selected_original_indices.append(group.loc[ptr, "index"])
                pointers[b] += 1
                progress = True
                if len(selected_original_indices) >= target_n:
                    break

        if not progress:
            break

    return df_candidates.loc[selected_original_indices].copy()


def select_class_members_distributed(
    df_subset: pd.DataFrame,
    modality: str,
    target_large: int,
    target_small: int,
    n_bins: int,
    salt_class: str,
    rank_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pool = df_subset[df_subset["final_modality"] == modality].copy()

    if len(pool) == 0:
        return pool.copy(), pool.copy()

    pool["class_score"] = pool["sample_id"].map(lambda s: hash_score(s, f"{salt_class}::{modality}"))
    pool["bin_id"] = assign_bins_by_rank(pool, rank_col, subset_size=len(df_subset), n_bins=n_bins)

    selected_large = distributed_take(
        pool, min(target_large, len(pool)), bin_col="bin_id", score_col="class_score"
    )

    selected_large = selected_large.sort_values("class_score", kind="mergesort").reset_index(drop=True)
    selected_large["selected_large_rank"] = np.arange(len(selected_large), dtype=np.int64)

    selected_small = selected_large.iloc[:min(target_small, len(selected_large))].copy().reset_index(drop=True)
    selected_small["selected_small_rank"] = np.arange(len(selected_small), dtype=np.int64)

    return selected_large, selected_small


def select_mr_subclass_members_distributed(
    df_subset: pd.DataFrame,
    subclass: str,
    target_large: int,
    target_small: int,
    n_bins: int,
    salt_mr: str,
    rank_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pool = df_subset[
        (df_subset["final_modality"] == "mri") &
        (df_subset["final_mr_subclass"] == subclass)
    ].copy()

    if len(pool) == 0:
        return pool.copy(), pool.copy()

    pool["mr_score"] = pool["sample_id"].map(lambda s: hash_score(s, f"{salt_mr}::{subclass}"))
    pool["bin_id"] = assign_bins_by_rank(pool, rank_col, subset_size=len(df_subset), n_bins=n_bins)

    selected_large = distributed_take(
        pool, min(target_large, len(pool)), bin_col="bin_id", score_col="mr_score"
    )

    selected_large = selected_large.sort_values("mr_score", kind="mergesort").reset_index(drop=True)
    selected_large["selected_large_rank"] = np.arange(len(selected_large), dtype=np.int64)

    selected_small = selected_large.iloc[:min(target_small, len(selected_large))].copy().reset_index(drop=True)
    selected_small["selected_small_rank"] = np.arange(len(selected_small), dtype=np.int64)

    return selected_large, selected_small


# ============================================================
# Distribution analysis
# ============================================================

def compute_distribution_metrics(df_selected: pd.DataFrame, n_bins: int, subset_size: int, rank_col: str) -> pd.DataFrame:
    if len(df_selected) == 0:
        return pd.DataFrame(columns=["bin_id", "count", "fraction"])

    tmp = df_selected.copy()
    tmp["bin_id"] = assign_bins_by_rank(tmp, rank_col=rank_col, subset_size=subset_size, n_bins=n_bins)

    counts = tmp["bin_id"].value_counts().sort_index()
    all_bins = pd.Index(range(n_bins), name="bin_id")
    counts = counts.reindex(all_bins, fill_value=0)

    out = pd.DataFrame({
        "bin_id": counts.index,
        "count": counts.values,
    })
    out["fraction"] = out["count"] / max(len(df_selected), 1)
    return out


def summarize_distribution(bin_df: pd.DataFrame) -> pd.DataFrame:
    counts = bin_df["count"].to_numpy(dtype=float)

    if len(counts) == 0:
        return pd.DataFrame([{
            "n_bins": 0,
            "total_selected": 0,
            "mean_per_bin": np.nan,
            "std_per_bin": np.nan,
            "min_bin": np.nan,
            "max_bin": np.nan,
            "coef_var": np.nan,
            "max_abs_dev_from_mean": np.nan,
            "normalized_l1_to_uniform": np.nan,
        }])

    mean_val = counts.mean()
    std_val = counts.std(ddof=0)
    min_val = counts.min()
    max_val = counts.max()
    max_abs_dev = np.max(np.abs(counts - mean_val))

    # normierte L1-Abweichung von idealer Gleichverteilung
    probs = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
    uniform = np.full_like(probs, 1.0 / len(probs), dtype=float)
    normalized_l1 = np.abs(probs - uniform).sum()

    return pd.DataFrame([{
        "n_bins": len(counts),
        "total_selected": int(counts.sum()),
        "mean_per_bin": mean_val,
        "std_per_bin": std_val,
        "min_bin": min_val,
        "max_bin": max_val,
        "coef_var": (std_val / mean_val) if mean_val > 0 else np.nan,
        "max_abs_dev_from_mean": max_abs_dev,
        "normalized_l1_to_uniform": normalized_l1,
    }])


def save_distribution_plot(bin_df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(12, 4))
    plt.bar(bin_df["bin_id"], bin_df["count"])
    plt.xlabel("Bin")
    plt.ylabel("Anzahl ausgewählter Samples")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_class_pie_chart(df_subset: pd.DataFrame, out_path: Path, title: str) -> None:
    counts = df_subset["final_modality"].fillna("unresolved").value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(
        path_str=str(input_path),
        sep=args.txt_sep,
        has_header=args.txt_has_header,
        image_column=args.image_column,
        caption_column=args.caption_column,
    )

    if args.caption_column not in df.columns:
        raise ValueError(f"Caption-Spalte '{args.caption_column}' nicht gefunden.")
    if args.image_column not in df.columns:
        raise ValueError(f"Bildspalte '{args.image_column}' nicht gefunden.")

    df = df.copy()
    df[args.caption_column] = df[args.caption_column].map(clean_caption)
    df = df[df[args.caption_column].astype(bool)].reset_index(drop=True)

    df["sample_id"] = [
        make_sample_id(img, cap)
        for img, cap in zip(df[args.image_column], df[args.caption_column])
    ]
    df = df.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("Nach Bereinigung sind keine Samples übrig geblieben.")

    df_master = build_master_subset(
        df=df,
        master_pool_size=args.master_pool_size,
        subset_large_size=args.subset_large_size,
        subset_small_size=args.subset_small_size,
        salt_master=args.salt_master,
    )

    df_large = df_master[df_master["is_in_large_subset"]].copy().reset_index(drop=True)
    df_small = df_master[df_master["is_in_small_subset"]].copy().reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True)
    model.to("cpu")
    model.eval()

    df_large = classify_modalities_and_mr_subclasses(
        df=df_large,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        bert_margin_threshold=args.bert_margin_threshold,
        caption_column=args.caption_column,
    )

    df_small = df_large[df_large["master_rank"] < args.subset_small_size].copy().reset_index(drop=True)

    df_large.to_csv(output_dir / "subset_large_400k_scored.csv", index=False)
    df_small.to_csv(output_dir / "subset_small_200k_scored.csv", index=False)

    save_class_pie_chart(df_large, output_dir / "class_pie_large.png", "Klassenverteilung im 400k-Subset")
    save_class_pie_chart(df_small, output_dir / "class_pie_small.png", "Klassenverteilung im 200k-Subset")

    modalities = ["ultrasound", "xray", "ct", "mri"]
    class_summary_rows = []

    for modality in modalities:
        sel_large, sel_small = select_class_members_distributed(
            df_subset=df_large,
            modality=modality,
            target_large=args.class_large_size,
            target_small=args.class_small_size,
            n_bins=args.bins_large,
            salt_class=args.salt_class,
            rank_col="master_rank",
        )

        sel_large.to_csv(output_dir / f"{modality}_large_selected.csv", index=False)
        sel_small.to_csv(output_dir / f"{modality}_small_selected.csv", index=False)

        large_bin_df = compute_distribution_metrics(sel_large, args.bins_large, len(df_large), "master_rank")
        small_bin_df = compute_distribution_metrics(sel_small, args.bins_large, len(df_large), "master_rank")

        large_bin_df.to_csv(output_dir / f"{modality}_large_bin_distribution.csv", index=False)
        small_bin_df.to_csv(output_dir / f"{modality}_small_bin_distribution.csv", index=False)

        summarize_distribution(large_bin_df).to_csv(output_dir / f"{modality}_large_distribution_summary.csv", index=False)
        summarize_distribution(small_bin_df).to_csv(output_dir / f"{modality}_small_distribution_summary.csv", index=False)

        save_distribution_plot(
            large_bin_df,
            output_dir / f"{modality}_large_distribution.png",
            f"{modality}: Verteilung der 40k-Auswahl über das 400k-Subset"
        )
        save_distribution_plot(
            small_bin_df,
            output_dir / f"{modality}_small_distribution.png",
            f"{modality}: Verteilung der 20k-Auswahl über das 400k-Subset"
        )

        class_summary_rows.append({
            "label": modality,
            "available_in_large_subset": int((df_large["final_modality"] == modality).sum()),
            "selected_large": len(sel_large),
            "selected_small": len(sel_small),
        })

    mr_subclasses = ["mr_t1", "mr_t2", "mr_flair", "mr_dwi", "mr_other"]
    mr_summary_rows = []

    for subclass in mr_subclasses:
        sel_large, sel_small = select_mr_subclass_members_distributed(
            df_subset=df_large,
            subclass=subclass,
            target_large=args.mr_large_per_subclass,
            target_small=args.mr_small_per_subclass,
            n_bins=args.bins_large,
            salt_mr=args.salt_mr,
            rank_col="master_rank",
        )

        sel_large.to_csv(output_dir / f"{subclass}_large_selected.csv", index=False)
        sel_small.to_csv(output_dir / f"{subclass}_small_selected.csv", index=False)

        large_bin_df = compute_distribution_metrics(sel_large, args.bins_large, len(df_large), "master_rank")
        small_bin_df = compute_distribution_metrics(sel_small, args.bins_large, len(df_large), "master_rank")

        large_bin_df.to_csv(output_dir / f"{subclass}_large_bin_distribution.csv", index=False)
        small_bin_df.to_csv(output_dir / f"{subclass}_small_bin_distribution.csv", index=False)

        summarize_distribution(large_bin_df).to_csv(output_dir / f"{subclass}_large_distribution_summary.csv", index=False)
        summarize_distribution(small_bin_df).to_csv(output_dir / f"{subclass}_small_distribution_summary.csv", index=False)

        save_distribution_plot(
            large_bin_df,
            output_dir / f"{subclass}_large_distribution.png",
            f"{subclass}: Verteilung der Auswahl über das 400k-Subset"
        )
        save_distribution_plot(
            small_bin_df,
            output_dir / f"{subclass}_small_distribution.png",
            f"{subclass}: Verteilung der Auswahl über das 400k-Subset"
        )

        mr_summary_rows.append({
            "label": subclass,
            "available_in_large_subset": int(
                ((df_large["final_modality"] == "mri") & (df_large["final_mr_subclass"] == subclass)).sum()
            ),
            "selected_large": len(sel_large),
            "selected_small": len(sel_small),
        })

    pd.DataFrame(class_summary_rows).to_csv(output_dir / "class_summary.csv", index=False)
    pd.DataFrame(mr_summary_rows).to_csv(output_dir / "mr_subclass_summary.csv", index=False)

    print("\n===== Fertig =====")
    print(f"Eingabe: {input_path}")
    print(f"Gesamt geladen: {len(df)}")
    print(f"Master-Pool: {len(df_master)}")
    print(f"Large subset: {len(df_large)}")
    print(f"Small subset: {len(df_small)}")
    print(f"Ausgabeordner: {output_dir}")


if __name__ == "__main__":
    main()