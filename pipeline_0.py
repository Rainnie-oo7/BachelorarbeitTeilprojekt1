# -*- coding: utf-8 -*-

"""
open-pmc Arrow/WebDataset Klassifikation mit Rules + BERT

Erwartete Struktur:
dataset_root/
    data-00000-of-00059.arrow
    ...
    data-00058-of-00059.arrow
    dataset_info.json
    state.json

Beispiel:
python pipeline_open_pmc.py \
    --dataset_root /home/b/open_pmc \
    --output_csv /home/b/open_pmc_classified.csv \
    --biomedbert_path /home/b/Dokumente/biomedbert \
    --limit 1000
"""

from __future__ import annotations

import os
import re
import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel


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
    text = re.sub(r"[_/\\\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# JSONL aus WebDataset parsen
# ============================================================

def parse_jsonl_field(x):
    if x is None:
        return {}

    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="ignore")

    if isinstance(x, str):
        x = x.strip()
        if not x:
            return {}
        try:
            return json.loads(x)
        except Exception:
            return {}

    return {}


def extract_text_from_row(row) -> Tuple[str, dict]:
    """
    Extrahiert den Text für die Modalitätsklassifikation.

    Priorität:
    - full_caption + sub_caption
    - nur full_caption
    - nur sub_caption
    - dann intext_refs_summary
    - dann intext_refs
    """
    meta = parse_jsonl_field(row.get("jsonl"))

    full_caption = str(meta.get("full_caption", "") or "").strip()
    sub_caption = str(meta.get("sub_caption", "") or "").strip()
    intext_refs_summary = str(meta.get("intext_refs_summary", "") or "").strip()
    intext_refs = str(meta.get("intext_refs", "") or "").strip()

    if full_caption and sub_caption:
        text = full_caption + " " + sub_caption
    elif full_caption:
        text = full_caption
    elif sub_caption:
        text = sub_caption
    elif intext_refs_summary:
        text = intext_refs_summary
    elif intext_refs:
        text = intext_refs
    else:
        text = ""

    return text, meta


# ============================================================
# Harte Regeln
# ============================================================

RULE_LABELS_ORDER = [
    "ct_pet",
    "mri",
    "ct",
    "pet",
    "ultrasound",
    "xray",
    "endoscopy",
    "pathology",
    "microscopy",
    "chart_or_diagram",
]

RULES: Dict[str, List[str]] = {
    "ct_pet": [
        r"\bpet\s*/\s*ct\b",
        r"\bpet\s*-\s*ct\b",
        r"\bpetct\b",
        r"\bfused\s+pet\s+ct\b",
        r"\bhybrid\s+pet\s+ct\b",
        r"\bcombined\s+pet\s+ct\b",
        r"\bco[- ]registered\s+pet\s+ct\b",
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
    "endoscopy": [
        r"\bendoscopy\b",
        r"\bcolonoscopy\b",
        r"\bgastroscopy\b",
        r"\besophagogastroduodenoscopy\b",
        r"\blaparoscopy\b",
        r"\bbronchoscopy\b",
        r"\bduodenoscopy\b",
        r"\benteroscopy\b",
        r"\bcolono fiberscope\b",
        r"\bcf\b",
    ],
    "pathology": [
        r"\bpathological findings\b",
        r"\bbiopsy\b",
        r"\bbiopsy findings\b",
        r"\bhistopathology\b",
        r"\bpathology\b",
        r"\btissue specimen\b",
        r"\bcell infiltration\b",
        r"\beosinophil counts\b",
    ],
    "microscopy": [
        r"\bmicroscopy\b",
        r"\bmicroscopic\b",
        r"\bhistology\b",
        r"\bimmunohistochemistry\b",
        r"\bh\s*&\s*e\b",
        r"\bhematoxylin\b",
        r"\beosin\b",
        r"\btissue section\b",
        r"\bpathology slide\b",
        r"\bstaining\b",
        r"\bcells\/hpf\b",
        r"\bhigh[- ]power field\b",
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
        r"\bclinical course\b",
        r"\btherapeutic management\b",
        r"\bbody weight changes\b",
    ],
}


def contains_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def rule_based_classify(text: str) -> Tuple[Optional[str], str]:
    t = normalize_for_rules(text)

    if not t:
        return None, "empty_text"

    for label in RULE_LABELS_ORDER:
        if contains_any(t, RULES[label]):
            return label, f"rule:{label}"

    return None, "no_rule_match"


# ============================================================
# Zero-Shot-artiger BERT-Fallback
# ============================================================

CLASS_TEXTS: Dict[str, str] = {
    "xray": (
        "X-ray radiography, radiograph, plain radiograph, chest x-ray, abdominal x-ray, "
        "skeletal radiograph, projection radiography, AP view, PA view, lateral view, "
        "portable x-ray, fluoroscopic x-ray image, c-arm x-ray guidance."
    ),
    "ct": (
        "Computed tomography, CT scan, axial CT, coronal CT, sagittal CT, contrast-enhanced CT, "
        "non-contrast CT, computed tomographic imaging, Hounsfield units, helical CT."
    ),
    "mri": (
        "Magnetic resonance imaging, MRI, MR image, T1-weighted, T2-weighted, FLAIR, DWI, "
        "diffusion-weighted imaging, gadolinium-enhanced MRI."
    ),
    "ultrasound": (
        "Ultrasound imaging, ultrasonography, sonography, echography, echocardiography, "
        "doppler ultrasound, duplex sonography."
    ),
    "pet": (
        "Positron emission tomography, PET scan, FDG PET, PET imaging, metabolic imaging, "
        "PET tracer uptake."
    ),
    "ct_pet": (
        "Combined PET CT imaging, fused PET CT, PET-CT, PET CT scan, hybrid PET CT."
    ),
    "endoscopy": (
        "Endoscopy, colonoscopy, gastroscopy, bronchoscopy, enteroscopy, laparoscopic view, "
        "endoscopic findings, mucosal findings."
    ),
    "pathology": (
        "Pathology, pathological findings, biopsy findings, tissue pathology, inflammatory cells, "
        "eosinophil counts, pathological examination."
    ),
    "microscopy": (
        "Microscopy, microscopic image, histology, histopathology, pathology slide, tissue section, "
        "H and E stain, immunohistochemistry, staining."
    ),
    "chart_or_diagram": (
        "Chart, graph, plot, line graph, bar chart, histogram, box plot, schematic, workflow figure, "
        "timeline, clinical course figure, therapeutic management chart."
    ),
    "unknown": (
        "Unknown or not identifiable modality, insufficient information, unclear caption."
    ),
}


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
        text_embs = self._encode_texts(texts, batch_size=batch_size)
        sims = text_embs @ self.label_embs.T

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
# Arrow-Dateien laden
# ============================================================

def find_arrow_files(dataset_root: Path) -> List[Path]:
    return sorted(dataset_root.glob("data-*.arrow"))


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


# ============================================================
# Debug / Übersicht
# ============================================================

def inspect_first_sample(ds: Dataset):
    sample = ds[0]

    print("\n===== DEBUG: Erstes Sample =====")
    print("Spalten:", ds.column_names)

    for k, v in sample.items():
        print(f"{k}: {type(v)}")

    raw_jsonl = sample.get("jsonl")
    if isinstance(raw_jsonl, bytes):
        preview = raw_jsonl[:800]
    else:
        preview = str(raw_jsonl)[:800]

    print("\njsonl Vorschau:")
    print(preview)

    meta = parse_jsonl_field(raw_jsonl)
    print("\nJSON Keys:")
    print(list(meta.keys()))

    print("\nPMC_ID:", meta.get("PMC_ID"))
    print("image:", meta.get("image"))
    print("modality:", meta.get("modality"))
    print("full_caption:", str(meta.get("full_caption", ""))[:300])
    print("sub_caption:", str(meta.get("sub_caption", ""))[:300])


def inspect_modality_distribution(ds: Dataset, limit: int = 5000):
    counter = Counter()

    n = min(len(ds), limit)
    for i in tqdm(range(n), desc="Prüfe modality-Verteilung"):
        meta = parse_jsonl_field(ds[i].get("jsonl"))
        counter[meta.get("modality", "MISSING")] += 1

    print("\n===== modality-Verteilung (Ausschnitt) =====")
    for key, value in counter.most_common():
        print(f"{key}: {value}")


# ============================================================
# Hauptklassifikation
# ============================================================

def classify_dataset(
    dataset_root: Path,
    output_csv: Path,
    biomedbert_path: Optional[str],
    limit: Optional[int],
    batch_size: int,
    bert_threshold: float,
    inspect_only: bool = False,
):
    arrow_files = find_arrow_files(dataset_root)
    if not arrow_files:
        raise FileNotFoundError(f"Keine data-*.arrow Dateien in {dataset_root} gefunden.")

    print(f"Gefundene Arrow-Dateien: {len(arrow_files)}")
    ds = load_arrow_shards(arrow_files)

    print(f"Gesamtanzahl Zeilen: {len(ds)}")
    print(f"Spalten: {ds.column_names}")
    print("Text wird aus row['jsonl'] extrahiert.")

    inspect_first_sample(ds)
    inspect_modality_distribution(ds, limit=2000)

    if inspect_only:
        print("\ninspect_only=True -> keine Klassifikation ausgeführt.")
        return

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"\nLimit aktiv: {len(ds)} Zeilen")

    records = []
    fallback_texts = []
    fallback_record_positions = []

    print("\nWende Regeln an ...")
    for idx in tqdm(range(len(ds)), desc="Rule-Klassifikation"):
        row = ds[idx]
        text, meta = extract_text_from_row(row)
        text = normalize_text(text)

        label, reason = rule_based_classify(text)

        rec = {
            "row_id": idx,
            "pmc_id": meta.get("PMC_ID", ""),
            "image": meta.get("image", ""),
            "modality_gt": meta.get("modality", ""),
            "caption": text,
            "sub_caption": meta.get("sub_caption", ""),
            "full_caption": meta.get("full_caption", ""),
            "intext_refs_summary": meta.get("intext_refs_summary", ""),
            "intext_refs": meta.get("intext_refs", ""),
            "pred_label": label if label is not None else "",
            "decision_source": "rule" if label is not None else "",
            "decision_reason": reason,
            "bert_score": None,
            "bert_top2": None,
        }

        records.append(rec)

        if label is None:
            fallback_record_positions.append(len(records) - 1)
            fallback_texts.append(text)

    print(f"\nRule-Treffer: {len(records) - len(fallback_texts)}")
    print(f"BERT-Fallback nötig für: {len(fallback_texts)}")

    if fallback_texts:
        if not biomedbert_path:
            print("Warnung: Kein --biomedbert_path angegeben. Unklare Fälle werden auf 'unknown' gesetzt.")
            for pos in fallback_record_positions:
                records[pos]["pred_label"] = "unknown"
                records[pos]["decision_source"] = "no_bert_available"
                records[pos]["decision_reason"] = "rule_failed_and_no_bert_model"
        else:
            print("\nLade BERT-Modell ...")
            bert_clf = BertCaptionClassifier(model_path=biomedbert_path)

            print("BERT-Fallback läuft ...")
            for start in tqdm(range(0, len(fallback_texts), batch_size), desc="BERT-Batches"):
                sub_texts = fallback_texts[start:start + batch_size]
                sub_positions = fallback_record_positions[start:start + batch_size]

                preds = bert_clf.predict_batch(
                    sub_texts,
                    threshold=bert_threshold,
                    batch_size=min(16, batch_size)
                )

                for pos, pred in zip(sub_positions, preds):
                    records[pos]["pred_label"] = pred["label"]
                    records[pos]["decision_source"] = "bert_fallback"
                    records[pos]["decision_reason"] = "rule_failed"
                    records[pos]["bert_score"] = pred["score"]
                    records[pos]["bert_top2"] = json.dumps(pred["top2"], ensure_ascii=False)

    df = pd.DataFrame(records)

    empty_mask = df["caption"].fillna("").astype(str).str.strip().eq("")
    df.loc[empty_mask, "pred_label"] = "unknown"
    df.loc[empty_mask, "decision_source"] = "empty_text"
    df.loc[empty_mask, "decision_reason"] = "empty_text"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"\nCSV gespeichert unter:\n{output_csv}")

    print("\n===== Verteilung pred_label =====")
    print(df["pred_label"].value_counts(dropna=False))

    print("\n===== Verteilung decision_source =====")
    print(df["decision_source"].value_counts(dropna=False))

    print("\n===== Verteilung modality_gt =====")
    print(df["modality_gt"].value_counts(dropna=False).head(20))


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Ordner mit data-*.arrow, dataset_info.json, state.json"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Pfad zur Ausgabe-CSV"
    )
    parser.add_argument(
        "--biomedbert_path",
        type=str,
        default=None,
        help="Lokaler Pfad zu BiomedBERT/PubMedBERT"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionales Limit für erste Tests"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batchgröße für den BERT-Fallback"
    )
    parser.add_argument(
        "--bert_threshold",
        type=float,
        default=0.30,
        help="Schwelle für BERT-Fallback, sonst unknown"
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Nur Struktur/Beispiele ausgeben, keine Klassifikation"
    )

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
        inspect_only=args.inspect_only,
    )


if __name__ == "__main__":
    main()