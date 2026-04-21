import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODALITY_PROTOTYPES: Dict[str, List[str]] = {
    "ct": [
        "computed tomography ct axial coronal sagittal scan",
        "ct image showing contrast enhanced lesion",
    ],
    "mri": [
        "magnetic resonance imaging mri t1 t2 flair diffusion gadolinium",
        "mri scan showing brain or soft tissue anatomy",
    ],
    "xray": [
        "x ray radiograph chest xray plain film",
        "radiographic image showing bones lungs or chest",
    ],
    "ultrasound": [
        "ultrasound sonography doppler echography",
        "ultrasound image showing soft tissue or fetal anatomy",
    ],
    "histology": [
        "histology microscopy stained tissue section hematoxylin eosin immunohistochemistry",
        "microscopic tissue image showing cells and staining",
    ]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample PMC captions and score them with local PubMedBERT/BiomedBERT on CPU. Supports Microsoft pipeline JSONL output."
    )
    parser.add_argument("--input", required=True, help="Pfad zu JSON/JSONL/CSV/Parquet")
    parser.add_argument("--output", required=True, help="Ausgabe als CSV oder Parquet")
    parser.add_argument("--model-dir", required=True, help="Lokaler Modellordner")
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--caption-column", default="caption")
    parser.add_argument("--id-column", default="pmcid")
    parser.add_argument("--image-column", default="image_path")
    return parser.parse_args()


# ------------------------------
# Loading / flattening input data
# ------------------------------

def load_table(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    suffixes = path.suffixes

    if suffixes and suffixes[-1] == ".csv":
        return pd.read_csv(path)
    if suffixes and suffixes[-1] == ".parquet":
        return pd.read_parquet(path)
    if suffixes and suffixes[-1] in {".json", ".jsonl"}:
        return load_microsoft_pipeline_json(path)

    raise ValueError(f"Nicht unterstütztes Format: {path}")



def load_microsoft_pipeline_json(path: Path) -> pd.DataFrame:
    """
    Unterstützt die von generate_pmc15_pipeline_outputs() erzeugte Struktur:
    pro Zeile ein Artikelobjekt mit article['figures'].
    """
    rows: List[dict] = []

    with path.open("r", encoding="utf8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                article = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Fehler beim JSON-Parsen in Zeile {line_no}: {e}") from e

            pmcid = str(article.get("pmc", "") or article.get("pmcid", ""))
            pmid = str(article.get("pmid", ""))
            location = str(article.get("location", ""))
            figures = article.get("figures", []) or []

            for fig in figures:
                caption = str(fig.get("fig_caption", "") or "").strip()
                if not caption:
                    continue

                rows.append(
                    {
                        "pmcid": pmcid,
                        "pmid": pmid,
                        "location": location,
                        "image_path": str(fig.get("graphic_ref", "") or ""),
                        "fig_id": str(fig.get("fig_id", "") or ""),
                        "fig_label": str(fig.get("fig_label", "") or ""),
                        "pair_id": str(fig.get("pair_id", "") or ""),
                        "caption": caption,
                    }
                )

    if not rows:
        raise ValueError(
            "Keine Figure-Captions gefunden. Prüfe, ob die Eingabedatei wirklich aus generate_pmc15_pipeline_outputs() stammt."
        )

    return pd.DataFrame(rows)


# ------------------------------
# Text / embedding helpers
# ------------------------------

def clean_caption(text: str) -> str:
    text = str(text or "")
    text = " ".join(text.replace("\n", " ").split())
    return text.strip()



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
    vectors: List[np.ndarray] = []
    model.eval()

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
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
    result: Dict[str, np.ndarray] = {}
    for modality, texts in prototypes.items():
        embs = encode_texts(texts, tokenizer, model, batch_size=batch_size, max_length=max_length)
        proto = embs.mean(axis=0)
        proto = proto / np.linalg.norm(proto)
        result[modality] = proto
    return result



def score_modalities(caption_embeddings: np.ndarray, prototype_embeddings: Dict[str, np.ndarray]) -> pd.DataFrame:
    score_dict: Dict[str, np.ndarray] = {}
    for modality, proto in prototype_embeddings.items():
        score_dict[modality] = caption_embeddings @ proto

    scores_df = pd.DataFrame(score_dict)
    top1 = scores_df.idxmax(axis=1)
    sorted_scores = np.sort(scores_df.to_numpy(), axis=1)
    top_vals = sorted_scores[:, -1]
    second_vals = sorted_scores[:, -2] if scores_df.shape[1] > 1 else np.zeros(len(scores_df))

    scores_df.insert(0, "predicted_modality", top1)
    scores_df.insert(1, "modality_score", top_vals)
    scores_df.insert(2, "score_margin", top_vals - second_vals)
    return scores_df


# ------------------------------
# Main
# ------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = load_table(args.input)

    if args.caption_column not in df.columns:
        if "caption" in df.columns:
            args.caption_column = "caption"
        else:
            raise ValueError(
                f"Caption-Spalte '{args.caption_column}' nicht gefunden. Verfügbare Spalten: {list(df.columns)}"
            )

    if args.id_column not in df.columns:
        if "pmcid" in df.columns:
            args.id_column = "pmcid"

    if args.image_column not in df.columns:
        if "image_path" in df.columns:
            args.image_column = "image_path"

    df = df.copy()
    df[args.caption_column] = df[args.caption_column].map(clean_caption)
    df = df[df[args.caption_column].astype(bool)].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("Nach Bereinigung sind keine Captions übrig geblieben.")

    n = min(args.sample_size, len(df))
    df = df.sample(n=n, random_state=args.seed).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True)
    model.to("cpu")
    model.eval()

    captions = df[args.caption_column].tolist()
    caption_embeddings = encode_texts(
        captions,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    prototype_embeddings = build_prototype_embeddings(
        DEFAULT_MODALITY_PROTOTYPES,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    score_df = score_modalities(caption_embeddings, prototype_embeddings)
    out_df = pd.concat([df.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        out_df.to_parquet(output_path, index=False)
    else:
        out_df.to_csv(output_path, index=False)

    print(f"Eingabe: {args.input}")
    print(f"Geladene Zeilen: {len(df)}")
    print(f"Ausgabe: {output_path}")
    print("Spalten:", list(out_df.columns))


if __name__ == "__main__":
    main()
