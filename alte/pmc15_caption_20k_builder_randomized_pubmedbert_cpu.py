#!/usr/bin/env python3
"""
pmc15_caption_20k_builder_randomized_pubmedbert_cpu.py

Zweck
-----
- Liest Captions aus einer PMC-15M-nahen Tabelle/Datei ein
- zieht reproduzierbar eine Zufallsstichprobe (z. B. 20k Captions)
- lädt PubMedBERT / BiomedBERT lokal auf CPU
- berechnet Caption-Embeddings lokal
- vergleicht sie mit Text-Prototypen für Modalitäten
- schreibt ein Ergebnis-CSV mit Scores

Unterstützte Eingaben:
- .csv
- .jsonl
- .parquet

Beispiel:
python pmc15_caption_20k_builder_randomized_pubmedbert_cpu.py \
  --input /data/pmc15_captions.parquet \
  --output /data/pmc15_caption_20k_pubmedbert.csv \
  --sample-size 20000 \
  --caption-column caption \
  --id-column pmcid \
  --image-column image_file \
  --model-dir /models/BiomedBERT-base-uncased-abstract-fulltext
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# Du kannst diese Prototypen eng an deine Bachelorarbeit anpassen.
DEFAULT_MODALITY_PROTOTYPES = {
    "xray": [
        "chest x ray radiograph",
        "radiograph x ray image",
        "plain film xray",
        "frontal lateral radiograph",
    ],
    "ct": [
        "computed tomography ct scan",
        "axial ct image",
        "contrast enhanced ct",
        "coronal ct reconstruction",
    ],
    "mri": [
        "magnetic resonance imaging mri",
        "t1 weighted mri",
        "t2 weighted mri",
        "gadolinium enhanced mri",
    ],
    "ultrasound": [
        "ultrasound sonography image",
        "doppler ultrasound",
        "sonographic image",
        "echographic image",
    ],
    "histology": [
        "histology microscopy tissue section",
        "hematoxylin eosin stain",
        "microscopic pathology image",
        "immunohistochemistry slide",
    ]
}


@dataclass
class InputColumns:
    caption: str
    item_id: Optional[str]
    image: Optional[str]


class PubMedBERTEmbedder:
    def __init__(self, model_dir: str, max_length: int = 64) -> None:
        self.device = torch.device("cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            use_fast=True,
        )
        self.model = AutoModel.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()

        torch.set_num_threads(max(1, min(os.cpu_count() or 1, 8)))

    @staticmethod
    def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.model(**encoded)
            pooled = self.mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Pfad zu CSV / JSONL / Parquet")
    parser.add_argument("--output", required=True, help="Pfad zur Ausgabe-CSV")
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--caption-column", default="caption")
    parser.add_argument("--id-column", default="pmcid")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--model-dir", required=True, help="Lokales Modellverzeichnis")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-caption-len", type=int, default=8)
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument(
        "--modalities-json",
        default=None,
        help="Optional: JSON-Datei mit eigenen Modalitäts-Prototypen",
    )
    return parser.parse_args()


def load_modalities(path: Optional[str]) -> dict:
    if path is None:
        return DEFAULT_MODALITY_PROTOTYPES
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("modalities-json muss ein JSON-Objekt sein: {modalitaet: [texte,...]}")
    return data


def load_table(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Nicht unterstütztes Format: {suffix}")


def validate_columns(df: pd.DataFrame, cols: InputColumns) -> None:
    missing = [c for c in [cols.caption, cols.item_id, cols.image] if c and c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten: {missing}. Verfügbar: {list(df.columns)}")


def clean_caption(text: object) -> str:
    if text is None:
        return ""
    text = str(text)
    text = " ".join(text.replace("\n", " ").replace("\t", " ").split())
    return text.strip()


def prepare_dataframe(df: pd.DataFrame, cols: InputColumns, min_caption_len: int, keep_duplicates: bool) -> pd.DataFrame:
    out = df.copy()
    out[cols.caption] = out[cols.caption].map(clean_caption)
    out = out[out[cols.caption].str.len() >= min_caption_len].copy()

    if not keep_duplicates:
        subset = [cols.caption]
        if cols.item_id and cols.item_id in out.columns:
            subset = [cols.item_id, cols.caption]
        out = out.drop_duplicates(subset=subset).copy()

    out = out.reset_index(drop=True)
    return out


def randomized_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df.sample(n=sample_size, random_state=seed, replace=False).reset_index(drop=True)


def build_prototype_table(modalities: dict) -> pd.DataFrame:
    rows = []
    for modality, texts in modalities.items():
        if isinstance(texts, str):
            texts = [texts]
        for t in texts:
            rows.append({"modality": modality, "prototype_text": clean_caption(t)})
    return pd.DataFrame(rows)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.matmul(a, b.T)


def classify_with_prototypes(
    captions_df: pd.DataFrame,
    cols: InputColumns,
    embedder: PubMedBERTEmbedder,
    modalities: dict,
    batch_size: int,
) -> pd.DataFrame:
    proto_df = build_prototype_table(modalities)
    proto_emb = embedder.encode(proto_df["prototype_text"].tolist(), batch_size=batch_size)
    cap_emb = embedder.encode(captions_df[cols.caption].tolist(), batch_size=batch_size)

    sims = cosine_matrix(cap_emb, proto_emb)

    # Pro Modalität den besten Prototypen-Score nehmen
    modality_names = proto_df["modality"].tolist()
    unique_modalities = list(dict.fromkeys(modality_names))

    modality_score_matrix = np.full((len(captions_df), len(unique_modalities)), -1.0, dtype=np.float32)
    for j, modality in enumerate(unique_modalities):
        indices = [i for i, m in enumerate(modality_names) if m == modality]
        modality_score_matrix[:, j] = sims[:, indices].max(axis=1)

    best_idx = modality_score_matrix.argmax(axis=1)
    best_modality = [unique_modalities[i] for i in best_idx]
    best_score = modality_score_matrix[np.arange(len(captions_df)), best_idx]

    sorted_scores = np.sort(modality_score_matrix, axis=1)
    second_best = sorted_scores[:, -2] if modality_score_matrix.shape[1] >= 2 else np.zeros(len(captions_df))
    margin = best_score - second_best

    result = captions_df.copy()
    result["predicted_modality"] = best_modality
    result["modality_score"] = best_score
    result["score_margin"] = margin

    for j, modality in enumerate(unique_modalities):
        result[f"score__{modality}"] = modality_score_matrix[:, j]

    return result


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cols = InputColumns(
        caption=args.caption_column,
        item_id=args.id_column if args.id_column else None,
        image=args.image_column if args.image_column else None,
    )

    df = load_table(args.input)
    validate_columns(df, cols)
    df = prepare_dataframe(
        df,
        cols=cols,
        min_caption_len=args.min_caption_len,
        keep_duplicates=args.keep_duplicates,
    )
    sample_df = randomized_sample(df, sample_size=args.sample_size, seed=args.seed)

    modalities = load_modalities(args.modalities_json)
    embedder = PubMedBERTEmbedder(model_dir=args.model_dir, max_length=args.max_length)

    result_df = classify_with_prototypes(
        captions_df=sample_df,
        cols=cols,
        embedder=embedder,
        modalities=modalities,
        batch_size=args.batch_size,
    )

    sort_cols = ["modality_score", "score_margin"]
    result_df = result_df.sort_values(sort_cols, ascending=[False, False]).reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)

    print(f"Fertig. {len(result_df)} Zeilen nach {out_path} geschrieben.")
    print("Verteilung predicted_modality:")
    print(result_df["predicted_modality"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
