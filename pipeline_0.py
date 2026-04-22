# -*- coding: utf-8 -*-
"""
Arrow-Shards einlesen und Captions mit Rules + BERT klassifizieren.

Erwartete Ordnerstruktur:
dataset_root/
    data-00000-of-00059.arrow
    ...
    data-00058-of-00059.arrow
    dataset_info.json
    state.json

Beispiel:
python classify_open_pmc_arrow.py \
    --dataset_root /pfad/zum/open_pmc_dataset \
    --output_csv /pfad/zu/output/open_pmc_classified.csv \
    --biomedbert_path /home/b/Dokumente/biomedbert

Optional:
    --limit 10000
    --batch_size 64
    --bert_threshold 0.30
"""

from __future__ import annotations

import os
import re
import json
import math
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import pandas as pd
from tqdm import tqdm

from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel


# ============================================================
# Konfiguration
# ============================================================

TEXT_COLUMN_CANDIDATES = [
    "caption",
    "captions",
    "txt",
    "text",
    "sentence",
    "report",
    "description",
    "input",
]

# Reihenfolge ist wichtig:
# - explizite Kombiklassen zuerst
# - danach klare Einzelmodalitäten
RULE_LABELS_ORDER = [
    "ct_pet",
    "mri",
    "ct",
    "pet",
    "ultrasound",
    "xray",
    "microscopy",
    "chart_or_diagram",
]

CLASS_TEXTS: Dict[str, str] = {
    "xray": (
        "X-ray radiography, radiograph, plain film, chest x-ray, abdominal x-ray, "
        "skeletal radiograph, projection radiography, AP view, PA view, lateral view, "
        "portable x-ray, fluoroscopic x-ray image, c-arm x-ray guidance, radiographic examination."
    ),
    "ct": (
        "Computed tomography, CT scan, axial CT, coronal CT, sagittal CT, contrast-enhanced CT, "
        "non-contrast CT, computed tomographic imaging, Hounsfield units, helical CT."
    ),
    "mri": (
        "Magnetic resonance imaging, MRI, MR image, T1-weighted, T2-weighted, FLAIR, DWI, "
        "diffusion-weighted imaging, gadolinium-enhanced MRI, sagittal MR, axial MR, coronal MR."
    ),
    "ultrasound": (
        "Ultrasound imaging, ultrasonography, sonography, echography, echocardiography, "
        "doppler ultrasound, duplex sonography, transvaginal ultrasound, abdominal ultrasound."
    ),
    "pet": (
        "Positron emission tomography, PET scan, FDG PET, PET imaging, metabolic imaging, "
        "PET tracer uptake, positron emission tomographic image."
    ),
    "ct_pet": (
        "Combined PET/CT imaging, fused PET CT, PET-CT, PET/CT scan, hybrid PET CT, "
        "co-registered CT and PET image, attenuation-corrected PET CT."
    ),
    "microscopy": (
        "Microscopy, microscopic image, histology, histopathology, immunohistochemistry, "
        "hematoxylin eosin staining, H&E stain, tissue section, pathology slide."
    ),
    "chart_or_diagram": (
        "Chart, graph, plot, bar chart, line graph, diagram, schematic illustration, "
        "workflow figure, histogram, box plot, survival curve, ROC curve."
    ),
    "unknown": (
        "Unknown or not identifiable imaging modality, insufficient information, unclear caption."
    ),
}


# ============================================================
# Text-Normalisierung
# ============================================================

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_rules(text: str) -> str:
    text = normalize_text(text).lower()
    # vereinfachte Trennung
    text = re.sub(r"[_/\\\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# Harte Regeln
# ============================================================

def contains_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


RULES: Dict[str, List[str]] = {
    "ct_pet": [
        r"\bpet\s*/\s*ct\b",
        r"\bpet\s*-\s*ct\b",
        r"\bpetct\b",
        r"\bfused\s+pet\s+ct\b",
        r"\bhybrid\s+pet\s+ct\b",
        r"\bco[- ]registered\s+pet\s+ct\b",
        r"\bcombined\s+pet\s+ct\b",
    ],
    "mri": [
        r"\bmri\b",
        r"\bmr\b",
        r"\bmagnetic resonance\b",
        r"\bt1\b",
        r"\bt2\b",
        r"\bflair\b",
        r"\bdwi\b",
        r"\badc\b",
        r"\bgadolinium\b",
        r"\bmrcp\b",
        r"\bfmri\b",
    ],
    "ct": [
        r"\bct\b",
        r"\bcomputed tomography\b",
        r"\bcomputed tomographic\b",
        r"\bhelical ct\b",
        r"\bmultidetector ct\b",
        r"\bhrct\b",
        r"\bcect\b",
        r"\baxial ct\b",
        r"\bcoronal ct\b",
        r"\bsagittal ct\b",
        r"\bhounsfield\b",
    ],
    "pet": [
        r"\bpet\b",
        r"\bpositron emission tomography\b",
        r"\bfdg pet\b",
        r"\bpet scan\b",
        r"\bpet image\b",
        r"\btracer uptake\b",
        r"\bsuv(max)?\b",
    ],
    "ultrasound": [
        r"\bultrasound\b",
        r"\bultrasonography\b",
        r"\bsonography\b",
        r"\bsonographic\b",
        r"\bechography\b",
        r"\bechocardiography\b",
        r"\bdoppler\b",
        r"\bduplex\b",
        r"\btransvaginal\b",
        r"\btransrectal\b",
        r"\btransabdominal\b",
        r"\bcolor doppler\b",
    ],
    "xray": [
        r"\bx[- ]?ray\b",
        r"\bradiograph\b",
        r"\bradiographic\b",
        r"\bplain film\b",
        r"\bchest film\b",
        r"\bportable chest\b",
        r"\bap view\b",
        r"\bpa view\b",
        r"\blateral view\b",
        r"\bfluoroscopy\b",
        r"\bfluoroscopic\b",
        r"\bc[- ]arm\b",
    ],
    "microscopy": [
        r"\bmicroscopy\b",
        r"\bmicroscopic\b",
        r"\bhistology\b",
        r"\bhistopathology\b",
        r"\bimmunohistochemistry\b",
        r"\bh\s*&\s*e\b",
        r"\bhematoxylin\b",
        r"\beosin\b",
        r"\btissue section\b",
        r"\bpathology slide\b",
        r"\bstaining\b",
    ],
    "chart_or_diagram": [
        r"\bchart\b",
        r"\bgraph\b",
        r"\bplot\b",
        r"\bdiagram\b",
        r"\bschematic\b",
        r"\bworkflow\b",
        r"\bhistogram\b",
        r"\bbox plot\b",
        r"\broc curve\b",
        r"\bsurvival curve\b",
        r"\bkaplan[- ]meier\b",
        r"\bbar graph\b",
        r"\bline graph\b",
        r"\bscatter plot\b",
    ],
}


def rule_based_classify(text: str) -> Tuple[Optional[str], str]:
    """
    Gibt (label, reason) zurück.
    Falls keine Regel greift: (None, "no_rule_match")
    """
    t = normalize_for_rules(text)

    if not t:
        return None, "empty_text"

    for label in RULE_LABELS_ORDER:
        if contains_any(t, RULES[label]):
            return label, f"rule:{label}"

    return None, "no_rule_match"


# ============================================================
# BERT-Fallback
# ============================================================

class BertCaptionClassifier:
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 128,
    ):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True
        ).to(self.device)
        self.model.eval()

        self.labels = list(CLASS_TEXTS.keys())
        self.label_texts = [CLASS_TEXTS[label] for label in self.labels]
        self.label_embs = self._encode_texts(self.label_texts, batch_size=16)

    @torch.no_grad()
    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @torch.no_grad()
    def _encode_texts(self, texts: List[str], batch_size: int = 16) -> torch.Tensor:
        all_embs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            out = self.model(**enc)
            pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embs.append(pooled.cpu())

        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def predict_batch(
        self,
        texts: List[str],
        threshold: float = 0.30,
        batch_size: int = 32,
    ) -> List[Dict]:
        text_embs = self._encode_texts(texts, batch_size=batch_size)  # [N, D]
        sims = text_embs @ self.label_embs.T  # [N, C]

        results = []
        for i in range(sims.shape[0]):
            row = sims[i]
            best_idx = int(torch.argmax(row).item())
            best_score = float(row[best_idx].item())
            label = self.labels[best_idx]

            if best_score < threshold:
                label = "unknown"

            sorted_idx = torch.argsort(row, descending=True).tolist()
            top2 = [(self.labels[j], float(row[j].item())) for j in sorted_idx[:2]]

            results.append({
                "label": label,
                "score": best_score,
                "top2": top2,
            })

        return results


# ============================================================
# Arrow-Dateien einlesen
# ============================================================

def find_arrow_files(dataset_root: Path) -> List[Path]:
    files = sorted(dataset_root.glob("data-*.arrow"))
    return files


def load_arrow_shards(arrow_files: List[Path]) -> Dataset:
    datasets_list = []
    for fp in tqdm(arrow_files, desc="Lade Arrow-Shards"):
        ds = Dataset.from_file(str(fp))
        datasets_list.append(ds)

    if not datasets_list:
        raise FileNotFoundError("Keine Arrow-Dateien gefunden.")

    if len(datasets_list) == 1:
        return datasets_list[0]

    return concatenate_datasets(datasets_list)


def choose_text_column(columns: List[str]) -> str:
    lower_map = {c.lower(): c for c in columns}

    for cand in TEXT_COLUMN_CANDIDATES:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    # Fallback: erste String-ähnliche Spalte, die typisch klingt
    for c in columns:
        cl = c.lower()
        if any(token in cl for token in ["text", "caption", "txt", "report", "desc"]):
            return c

    raise ValueError(
        f"Keine Text-/Caption-Spalte gefunden. Vorhandene Spalten: {columns}"
    )


# ============================================================
# Hauptlogik
# ============================================================

def classify_dataset(
    dataset_root: Path,
    output_csv: Path,
    biomedbert_path: Optional[str],
    limit: Optional[int],
    batch_size: int,
    bert_threshold: float,
):
    arrow_files = find_arrow_files(dataset_root)
    if not arrow_files:
        raise FileNotFoundError(f"Keine data-*.arrow Dateien in {dataset_root} gefunden.")

    print(f"Gefundene Arrow-Dateien: {len(arrow_files)}")
    ds = load_arrow_shards(arrow_files)
    print(f"Gesamtanzahl Zeilen: {len(ds)}")
    print(f"Spalten: {ds.column_names}")

    # Erstes Sample anschauen
    sample = ds[0]

    print("\nKeys im Sample:")
    for k in sample.keys():
        print(k, type(sample[k]))

    print("\njsonl Inhalt (roh):")
    print(sample["jsonl"][:350000])

    text_col = choose_text_column(ds.column_names)
    print(f"Verwendete Textspalte: {text_col}")

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"Limit aktiv: {len(ds)} Zeilen")

    # Für Ausgabe nützliche Zusatzspalten
    extra_cols = []
    for c in ["__key__", "__url__"]:
        if c in ds.column_names:
            extra_cols.append(c)

    records = []
    fallback_texts = []
    fallback_indices = []

    print("Wende Regeln an ...")
    for idx in tqdm(range(len(ds)), desc="Rule-Klassifikation"):
        row = ds[idx]
        text = normalize_text(row.get(text_col, ""))

        label, reason = rule_based_classify(text)

        rec = {
            "row_id": idx,
            "text_column": text_col,
            "caption": text,
            "pred_label": label if label is not None else "",
            "decision_source": "rule" if label is not None else "",
            "decision_reason": reason,
            "bert_score": None,
            "bert_top2": None,
        }

        for c in extra_cols:
            rec[c] = row.get(c, None)

        records.append(rec)

        if label is None:
            fallback_indices.append(idx)
            fallback_texts.append(text)

    print(f"Rule-Treffer: {len(records) - len(fallback_indices)}")
    print(f"BERT-Fallback nötig für: {len(fallback_indices)}")

    if fallback_indices:
        if not biomedbert_path:
            print("Warnung: Kein --biomedbert_path angegeben. Unklare Fälle werden als 'unknown' gesetzt.")
            for idx in fallback_indices:
                records[idx]["pred_label"] = "unknown"
                records[idx]["decision_source"] = "no_bert_available"
                records[idx]["decision_reason"] = "rule_failed_and_no_bert_model"
        else:
            print("Lade BERT-Modell ...")
            bert_clf = BertCaptionClassifier(model_path=biomedbert_path)

            print("BERT-Fallback läuft ...")
            for start in tqdm(range(0, len(fallback_texts), batch_size), desc="BERT-Batches"):
                sub_texts = fallback_texts[start:start + batch_size]
                sub_preds = bert_clf.predict_batch(
                    sub_texts,
                    threshold=bert_threshold,
                    batch_size=min(16, batch_size)
                )
                sub_ids = fallback_indices[start:start + batch_size]

                for real_idx, pred in zip(sub_ids, sub_preds):
                    records[real_idx]["pred_label"] = pred["label"]
                    records[real_idx]["decision_source"] = "bert_fallback"
                    records[real_idx]["decision_reason"] = "rule_failed"
                    records[real_idx]["bert_score"] = pred["score"]
                    records[real_idx]["bert_top2"] = json.dumps(pred["top2"], ensure_ascii=False)

    df = pd.DataFrame(records)

    # Leere Captions markieren
    empty_mask = df["caption"].fillna("").astype(str).str.strip().eq("")
    df.loc[empty_mask, "pred_label"] = "unknown"
    df.loc[empty_mask, "decision_source"] = "empty_text"
    df.loc[empty_mask, "decision_reason"] = "empty_text"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\nFertig. CSV gespeichert unter:\n{output_csv}")

    print("\nVerteilung:")
    print(df["pred_label"].value_counts(dropna=False))

    print("\nDecision Source:")
    print(df["decision_source"].value_counts(dropna=False))


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Ordner mit data-*.arrow, dataset_info.json, state.json")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Pfad zur Ausgabedatei (.csv)")
    parser.add_argument("--biomedbert_path", type=str, default=None,
                        help="Lokaler Pfad zu BiomedBERT/PubMedBERT")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optionales Zeilenlimit zum Testen")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batchgröße für Fallback-Verarbeitung")
    parser.add_argument("--bert_threshold", type=float, default=0.30,
                        help="Mindestscore für BERT-Zuordnung, sonst unknown")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    output_csv = Path(args.output_csv)

    classify_dataset(
        dataset_root=dataset_root,
        output_csv=output_csv,
        biomedbert_path=args.biomedbert_path,
        limit=args.limit,
        batch_size=args.batch_size,
        bert_threshold=args.bert_threshold,
    )


if __name__ == "__main__":
    main()