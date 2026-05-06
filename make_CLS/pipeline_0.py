# -*- coding: utf-8 -*-

"""
open-pmc Klassifikation mit Rules + BERT + LLM + CNN
Erwartete Struktur:
dataset_root/
    data-00000-of-00059.arrow
    ...
    data-00058-of-00059.arrow
    dataset_info.json
    state.json
Beispiel:
python make_CLS/pipeline_0.py \
  --dataset_root /home/user/PycharmProjects/ba1pmc/PMC-41GB \
  --output_csv /home/user/PycharmProjects/ba1pmc/make_CLS/test_per5.csv \
  --biomedbert_path /home/user/Dokumente/biomedbert \
  --llamamistral_path /home/user/Dokumente/LLaMa_Mistral \
  --cnn1_path /home/user/Dokumente/cnn1/convu_try_3t.pth \
  --cnn2_path /home/user/Dokumente/cnn2/convu_folderlabel_mrt_body_mrt_hirn.pth \
  --cnn3_path /home/user/Dokumente/cnn3/cnn_multiclass.pth
  > log.txt 2>&1
"""

from __future__ import annotations

import os.path as osp
from pathlib import Path
from PIL import Image
import re
import json
import argparse

import hashlib
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
import io
from itertools import zip_longest

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForCausalLM

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

# Finale Klassen u. Mapping
FINAL_CLASSES = [
    "xray",
    "xray_fluoroskopie_angiographie",
    "us",
    "mrt_hirn",
    "mrt_body",
    "ct",
    "ct_kombimodalitaet_spect+ct_pet+ct"]
CNN1_CLASS_NAMES = [
    "mrt_prostata_t1",
    "mrt_prostata_t2",
    "mrt_hirn_flair",
    "mrt_hirn_t1",
    "mrt_hirn_t2",
    "mrt_hirn_t1_c",
    "ct",
    "ct_kombimodalitaet_spect+ct_pet+ct",
    "xray",
    "xray_fluoroskopie_angiographie",
    "us"]
CNN1_MAPPING = {
    "mrt_prostata_t1": "mrt_body",
    "mrt_prostata_t2": "mrt_body",
    "mrt_hirn_flair": "mrt_hirn",
    "mrt_hirn_t1": "mrt_hirn",
    "mrt_hirn_t2": "mrt_hirn",
    "mrt_hirn_t1_c": "mrt_hirn",

    "ct": "ct",
    "ct_kombimodalitaet_spect+ct_pet+ct": "ct_kombimodalitaet_spect+ct_pet+ct",
    "xray": "xray",
    "xray_fluoroskopie_angiographie": "xray_fluoroskopie_angiographie",
    "us": "us"}
CNN2_CLASS_NAMES = [
    "xray",
    "xray_fluoroskopie_angiographie",
    "mrt_hirn",
    "mrt_body"]
CNN2_MAPPING = {
    "xray": "xray",
    "xray_fluoroskopie_angiographie": "xray_fluoroskopie_angiographie",
    "mrt_hirn": "mrt_hirn",
    "mrt_body": "mrt_body"}
CNN3_CLASS_NAMES = [
    "chirurgie",
    "mikroskopie",
    "endoskopie",
    "haut",
    "chart",
    "histologie"]
#Gating fuer CNN1 mit CNN2 Fusion. Disjunkt. CNN1 kann gut (besser als 2) obere. CNN2 kann besser das untere
# nicht disjunkt. Sie sind asymmetrisch spezialisiert
CNN1_VALID = [
    "ct",
    "ct_kombimodalitaet_spect+ct_pet+ct",
    "us"]
CNN2_VALID = [
    "xray",
    "xray_fluoroskopie_angiographie",
    "mrt_hirn",
    "mrt_body"]
# fuer Weight Log-Scaling
CNN1_CLASS_COUNTS_RAW = {
    "xray": 54419,
    "xray_fluoroskopie_angiographie": 3000,
    "us": 23804,
    "mrt_hirn": 4528,
    "mrt_body": 1666,
    "ct": 98158,
    "ct_kombimodalitaet_spect+ct_pet+ct": 418723
}
CNN2_CLASS_COUNTS_RAW = {
    "xray": 60852,
    "xray_fluoroskopie_angiographie": 6433,
    "mrt_hirn": 337909,
    "mrt_body": 32006
}
WEIGHTS = {
    "rules": 0.2,
    "bert": 0.2,
    "cnn": 0.4,
    "llm": 0.2,

    "cnn3": 0.3
}

# gen. LLM (vLLM READY)
class LocalLLM:
    def __init__(self, model_path: str):

        model_path = Path(osp.normpath(model_path))

        if not model_path.exists():
            raise FileNotFoundError(f"LLM Pfad existiert nicht: {model_path}")

        print("Lade Mistral von:", model_path)

        self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        # verhindert Warnung
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_path,
            torch_dtype=torch.float32,
            local_files_only=True)

        self.model.to(self.device)
        self.model.eval()

        self.model.config.pad_token_id = self.model.config.eos_token_id

    def batch_predict(self, texts, batch_size=8):
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            prompts = [
                f"""Classify into one of {FINAL_CLASSES}.
Choose one label from {FINAL_CLASSES}

Answer ONLY the label.

Text: {text}
"""

                for text in batch
            ]

            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64
                )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for d in decoded:
                label = d.strip().lower()

                for cls in FINAL_CLASSES:
                    if cls in label:
                        results.append((cls, 0.7))
                        break
                else:
                    results.append(("unknown", 0.0))

        return results

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ThirdCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
"""
# CNN Mapping -> FINAL_CLASSES
# Top 1 -> Score (kein 0.7 mrt, 0.2 us usw. Softmax, sondern direktes 'mrt')
# def cnn_to_scores(label, mapping):
#     scores = {c: 0.0 for c in FINAL_CLASSES}
# 
#     if label in mapping:
#         mapped = mapping[label]
#         scores[mapped] = 1.0
# 
#     return scores
"""
# CNN Mapping -> FINAL_CLASSES
# mit Top-3 (Variante, besserer Ueberblick in einem Viewer o.ae)
def map_scores_to_final(full_scores, mapping):
    final_scores = {c: 0.0 for c in FINAL_CLASSES}

    for cls, prob in full_scores.items():
        if cls in mapping:
            final_cls = mapping[cls]
            final_scores[final_cls] += prob

    return final_scores
# CNN fusion mit log-scaled weights of derer Klassen
def fuse_cnn_weighted(cnn1_scores, cnn2_scores, w1, w2, alpha=0.5):
    fused = {}

    for c in FINAL_CLASSES:
        s1 = cnn1_scores.get(c, 0.0)
        s2 = cnn2_scores.get(c, 0.0)

        # Soft weighting
        s1_weighted = s1 * (1 + alpha * w1.get(c, 0.0))
        s2_weighted = s2 * (1 + alpha * w2.get(c, 0.0))

        fused[c] = (s1_weighted + s2_weighted)
    return fused

def compute_cnn_confidence_and_margin(cnn_scores):
    # Berechnet:
    # - cnn_conf   → höchste Wahrscheinlichkeit
    # - cnn_margin → Abstand zum zweitbesten Label
    # Args:
    #     cnn_scores (dict): {label: score}
    # Returns:
    #     cnn_conf (float)
    #     cnn_margin (float)
    #     top1_label (str)
    #     top2_label (str)

    if not cnn_scores:
        return 0.0, 0.0, "none", "none"

    # Sortiere nach Score (absteigend)
    sorted_items = sorted(
        cnn_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Top-1
    top1_label, top1_score = sorted_items[0]

    # Top-2 (falls vorhanden)
    if len(sorted_items) > 1:
        top2_label, top2_score = sorted_items[1]
    else:
        top2_label, top2_score = "none", 0.0

    # Confidence
    cnn_conf = top1_score

    # Margin
    cnn_margin = top1_score - top2_score

    return cnn_conf, cnn_margin, top1_label, top2_label

# log scaling weights anstatt hard gating of classes (asymmetric classes)
def compute_model_weights(class_counts, final_classes):
    import math

    weights = {}

    # Log scaling
    for c in final_classes:
        count = class_counts.get(c, 0)
        weights[c] = math.log(count + 1)

    # Normalisierung
    max_w = max(weights.values()) if weights else 1.0

    if max_w == 0:
        return {c: 0.0 for c in final_classes}

    for c in weights:
        weights[c] /= max_w  # [0,1]

    return weights

cnn1_weights = compute_model_weights(CNN1_CLASS_COUNTS_RAW, FINAL_CLASSES)
cnn2_weights = compute_model_weights(CNN2_CLASS_COUNTS_RAW, FINAL_CLASSES)

# normalize all in WEIGHTS, da s1_weighted = s1 * (1 + alpha * w1.get(c, 0.0)) = UEBER 1 !!!
def normalize_scores(scores):
    total = sum(scores.values())
    if total == 0:
        return scores
    return {k: v / total for k, v in scores.items()}

def is_cnn_uncertain(cnn_conf, cnn_margin, conf_threshold=0.4, margin_threshold=0.03):
    relative_margin = cnn_margin / (cnn_conf + 1e-6)
    # Entscheidet, ob CNN unsicher ist.
    # Returns: bool: True = unsicher → LLM prüfen
    return (cnn_conf < conf_threshold or relative_margin < margin_threshold)

# Rules macht subset. pre_fuse. Fusioniert Rules/CNN/BERT. LLM nur bei unsicheren Faellen als Richter
def quick_fuse(rule_scores, bert_scores, cnn_scores):
    final = {}

    for label in FINAL_CLASSES:
        final[label] = (
            WEIGHTS["rules"] * rule_scores.get(label, 0)
            + WEIGHTS["bert"] * bert_scores.get(label, 0)
            + WEIGHTS["cnn"] * cnn_scores.get(label, 0)    # ersetzt cnn1 u cnn2
        )

    sorted_labels = sorted(final.items(), key=lambda x: x[1], reverse=True)

    top1, top2 = sorted_labels[0], sorted_labels[1]

    margin = top1[1] - top2[1]

    confident = (
            margin > 0.10 * top1[1] # relativ gesehen
            and top1[1] > 0.2       # schwach
    )

    return top1[0], final, confident

# Final-Classes Fusion
def fuse_scores(rule, bert, cnn, llm, cnn3_conf):

    final = {}
    contrib = {}

    for label in FINAL_CLASSES:
        base = (
            WEIGHTS["rules"] * rule.get(label, 0)
            + WEIGHTS["bert"] * bert.get(label, 0)
            + WEIGHTS["cnn"] * cnn.get(label, 0)
            + WEIGHTS["llm"] * (llm[1] if llm[0] == label else 0)
        )

        final[label] = base * (1 - WEIGHTS["cnn3"] * cnn3_conf)

    sorted_labels = sorted(final.items(), key=lambda x: x[1], reverse=True)

    top1, top2 = sorted_labels[0], sorted_labels[1]

    margin = top1[1] - top2[1]

    uncertain = (
        margin < 0.10 * top1[1] # relativ gesehen
        or top1[1] < 0.2        # schwach
    )

    return top1[0], final, contrib, None, uncertain
def run_final_fusion(records, llm_results):

    final_records = []
    filtered_cnn3 = 0

    for r, llm_out in zip(records, llm_results):
        rule_scores = r.get("rule_scores", {c: 0.0 for c in FINAL_CLASSES})
        bert_scores = r.get("bert_scores", {c: 0.0 for c in FINAL_CLASSES})
        cnn_scores  = r.get("cnn_scores",  {c: 0.0 for c in FINAL_CLASSES})
        cnn3_conf   = r.get("cnn3_conf", 0.0)

        if cnn3_conf > 0.91:     # cnn3_conf=Wie wahrscheinlich ist dieses Bild KEINE Radiologie?
            #CNN3Filt xray 0.9227646589279175
            print("CNN3Filt", r["final_label"], r["cnn3_conf"])
            filtered_cnn3 += 1
            continue

        # ---------- FINAL LABEL ----------
        if not r.get("llm_needed", False):
            final_label = r.get("final_label", r.get("quick_label", "unknown"))
            final_scores = rule_scores
            uncertain = False
        else:
            final_label, final_scores, _, _, uncertain = fuse_scores(
                rule_scores,
                bert_scores,
                cnn_scores,
                llm_out,
                cnn3_conf
            )

        # ---------- DEBUG INFO ----------
        debug_info = {
            "final_label": final_label,
            "uncertain": uncertain,
            "final_scores": final_scores
        }

        # ---------- FINAL OUTPUT ----------
        final_records.append({
            "pmc_id": r.get("pmc_id"),
            "row_id": r.get("row_id"),
            "row": r.get("row"),

            "final_label": final_label,

            "RULE": r.get("rule_label"),
            "BERT": r.get("bert_label"),
            "CNN1top3": json.dumps(r.get("cnn1_top3", [])),
            "CNN2top3": json.dumps(r.get("cnn2_top3", [])),
            "LLM": llm_out[0],

            "BERT_score": r.get("bert_score", 0.0),
            "CNN1_scores": json.dumps(r.get("cnn1_scores", {})),
            "CNN2_scores": json.dumps(r.get("cnn2_scores", {})),
            "LLM_conf": llm_out[1],

            "Final Scores": json.dumps(final_scores),
            "Begründung": json.dumps(debug_info),

            "uncertain": uncertain,
            "caption": r.get("caption"),
            "modality_gt": r.get("modality_gt", "unknown"),
        })

    print(f"CNN3 gefiltert: {filtered_cnn3}")

    return final_records
# ============================================================
# Hash-Sampling (100k Ziel)
# ============================================================
def stable_hash_key(key: str) -> int:
    return int(hashlib.md5(key.encode()).hexdigest(), 16)

def stable_hash_record(record):
    key = f"{record['pmc_id']}_{record['row_id']}"
    return stable_hash_key(key)

def stable_hash_index(idx: int) -> int:
    return stable_hash_key(str(idx))

def select_balanced(records, per_class):
    buckets = {c: [] for c in FINAL_CLASSES}

    for r in records:
        label = r["final_label"]
        if label in buckets:
            buckets[label].append(r)

    final = []
    for c in buckets:
        sorted_bucket = sorted(
            buckets[c],
            key=stable_hash_record
        )
        final.extend(sorted_bucket[:per_class])

    return final

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
# Bild-Extractor aus Arrow, dient auch im Testrun als Inspizieren/Distr/Debug/Übersicht
# ============================================================
def bytes_to_pil_image(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    if isinstance(x, bytes):
        return Image.open(io.BytesIO(x)).convert("RGB")

    if isinstance(x, dict):
        if "bytes" in x and x["bytes"] is not None:
            return Image.open(io.BytesIO(x["bytes"])).convert("RGB")
        if "path" in x and x["path"]:
            return Image.open(x["path"]).convert("RGB")

    raise ValueError(f"Kann jpg-Feld nicht als Bild lesen. Typ: {type(x)}")

def get_image_from_row(row):
    if row is None:
        return None
    for key in ["image", "jpg", "png"]:
        if key in row:
            try:
                return bytes_to_pil_image(row[key])
            except Exception:
                continue
    return None

# ============================================================
# Harte Regeln
# ============================================================

RULE_LABELS_ORDER = [
    "ct",
    "us",
    "mrt_hirn",
    "mrt_body",
    "xray_fluoroskopie_angiographie",
    "ct_kombimodalitaet_spect+ct_pet+ct",
    "xray",

    "microscopy",
    "pathology",
    "surgery_real",
    "endoscopy",
    "chart_or_diagram",
]

MRT_HIRN_T1_SHORT = [
    r"\bbrain\b.*\bt1\b",
    r"\bhead\b.*\bt1\b",
    r"\bcerebr\w*\b.*\bt1\b",
    r"\bcranial\b.*\bt1\b",
    r"\bt1[- ]weighted\b.*\bbrain\b",
    r"\bbrain\s+mri\b.*\bt1\b",
]
MRT_HIRN_T2_SHORT = [
    r"\bbrain\b.*\bt2\b",
    r"\bhead\b.*\bt2\b",
    r"\bcerebr\w*\b.*\bt2\b",
    r"\bcranial\b.*\bt2\b",
    r"\bt2[- ]weighted\b.*\bbrain\b",
    r"\bbrain\s+mri\b.*\bt2\b",
]
MRT_HIRN_FLAIR_SHORT = [
    r"\bbrain\b.*\bflair\b",
    r"\bcranial\b.*\bflair\b",
    r"\bcerebr\w*\b.*\bflair\b",
    r"\bhead\b.*\bflair\b",
    r"\bflair\b.*\bbrain\b",
    r"\bfluid[- ]attenuated inversion recovery\b",
]
MRT_HIRN_T1_C_SHORT = [
    r"\bbrain\b.*\bt1\s*\+\s*c\b",
    r"\bpost[- ]contrast\b.*\bbrain\b.*\bmri\b",
    r"\bgadolinium[- ]enhanced\b.*\bbrain\b.*\bmri\b",
    r"\bcontrast[- ]enhanced\b.*\bbrain\b.*\bmri\b",
    r"\bt1\s*\+\s*c\b.*\bbrain\b",
    r"\bt1\s+with\s+contrast\b.*\bbrain\b",
]
# Kurze RULES MRT BODY – Sublisten
MRT_PROSTATA_T1_SHORT = [
    r"\bprostat\w*\b.*\bt1\b",
    r"\bt1[- ]weighted\b.*\bprostat\w*\b",
    r"\bprostate\s+mri\b.*\bt1\b",
    r"\bpelvic\s+mri\b.*\bprostat\w*\b.*\bt1\b",
]
MRT_PROSTATA_T2_SHORT = [
    r"\bprostat\w*\b.*\bt2\b",
    r"\bt2[- ]weighted\b.*\bprostat\w*\b",
    r"\bprostate\s+mri\b.*\bt2\b",
    r"\bpelvic\s+mri\b.*\bprostat\w*\b.*\bt2\b",
    r"\bzonal anatomy\b.*\bprostat\w*\b.*\bt2\b",
]
MRT_BODY_GENERAL_SHORT = [
    r"\bpelvic\s+mri\b",
    r"\bpelvic\s+mr\b",
    r"\bprostate\s+mri\b",
    r"\bprostate\s+mr\b",
    r"\bmpmri\b",
    r"\bmultiparametric\s+mri\b.*\bprostat\w*\b",
    r"\babdominal\s+mri\b",
    r"\bbreast\s+mri\b",
    r"\bliver\s+mri\b",
    r"\bkidney\s+mri\b",
    r"\brenal\s+mri\b",
    r"\bspine\s+mri\b",
    r"\bknee\s+mri\b",
    r"\bcardiac\s+mri\b",
    r"\bheart\s+mri\b",
]
# Lange RULES MRT HIRN – Sublisten
MRT_HIRN_T1_LONG = [
    r"\bbrain\b.*\bt1\b",
    r"\bhead\b.*\bt1\b",
    r"\bcerebr\w*\b.*\bt1\b",
    r"\bcranial\b.*\bt1\b",
    r"\bt1[- ]weighted\b.*\bbrain\b",
    r"\bbrain\s+mri\b.*\bt1\b",
]
MRT_HIRN_T2_LONG = [
    r"\bbrain\b.*\bt2\b",
    r"\bhead\b.*\bt2\b",
    r"\bcerebr\w*\b.*\bt2\b",
    r"\bcranial\b.*\bt2\b",
    r"\bt2[- ]weighted\b.*\bbrain\b",
    r"\bbrain\s+mri\b.*\bt2\b",
]
MRT_HIRN_FLAIR_LONG = [
    r"\bbrain\b.*\bflair\b",
    r"\bcranial\b.*\bflair\b",
    r"\bcerebr\w*\b.*\bflair\b",
    r"\bhead\b.*\bflair\b",
    r"\bflair\b.*\bbrain\b",
    r"\bfluid[- ]attenuated inversion recovery\b",
]
MRT_HIRN_T1_C_LONG = [
    r"\bbrain\b.*\bt1\s*\+\s*c\b",
    r"\bbrain\b.*\bt1\s*post[- ]contrast\b",
    r"\bpost[- ]contrast\b.*\bbrain\b.*\bmri\b",
    r"\bgadolinium[- ]enhanced\b.*\bbrain\b.*\bmri\b",
    r"\bcontrast[- ]enhanced\b.*\bbrain\b.*\bmri\b",
    r"\bt1\s*\+\s*c\b.*\bbrain\b",
    r"\bt1\s+with\s+contrast\b.*\bbrain\b",
]

# RULES MRT BODY – Sublisten
MRT_PROSTATA_T1_LONG = [
    r"\bprostat\w*\b.*\bt1\b",
    r"\bt1[- ]weighted\b.*\bprostat\w*\b",
    r"\bprostate\s+mri\b.*\bt1\b",
    r"\bpelvic\s+mri\b.*\bprostat\w*\b.*\bt1\b",
]
MRT_PROSTATA_T2_LONG = [
    r"\bprostat\w*\b.*\bt2\b",
    r"\bt2[- ]weighted\b.*\bprostat\w*\b",
    r"\bprostate\s+mri\b.*\bt2\b",
    r"\bpelvic\s+mri\b.*\bprostat\w*\b.*\bt2\b",
    r"\bzonal anatomy\b.*\bprostat\w*\b.*\bt2\b",
]
MRT_BODY_GENERAL_LONG = [
    r"\bpelvic\s+mri\b",
    r"\bpelvic\s+mr\b",
    r"\bprostate\s+mri\b",
    r"\bprostate\s+mr\b",
    r"\bmpmri\b",
    r"\bmultiparametric\s+mri\b.*\bprostat\w*\b",
    r"\babdominal\s+mri\b",
    r"\bbreast\s+mri\b",
    r"\bliver\s+mri\b",
    r"\bkidney\s+mri\b",
    r"\brenal\s+mri\b",
    r"\bspine\s+mri\b",
    r"\bknee\s+mri\b",
    r"\bcardiac\s+mri\b",
    r"\bheart\s+mri\b",
    r"\bwhole[- ]body\s+mri\b",

    r"\bmri\b.*\bprostat\w*\b",
    r"\bmr\b.*\bprostat\w*\b",
    r"\bmri\b.*\bpelvi\w*\b",
    r"\bmr\b.*\bpelvi\w*\b",
    r"\bdiffusion[- ]weighted\b.*\bprostat\w*\b",
    r"\bdwi\b.*\bprostat\w*\b",
    r"\badc\b.*\bprostat\w*\b",
    r"\bdynamic contrast[- ]enhanced\b.*\bprostat\w*\b",
    r"\bdce\b.*\bprostat\w*\b",

    r"\babdominal\s+mr\b",
    r"\bbreast\s+mr\b",
    r"\bliver\s+mr\b",
    r"\bkidney\s+mr\b",
    r"\brenal\s+mr\b",
    r"\bspine\s+mr\b",
    r"\bknee\s+mr\b",
    r"\bcardiac\s+mr\b",
    r"\bheart\s+mr\b",
]

def interleave_patterns(*pattern_lists):
    mixed = []
    for group in zip_longest(*pattern_lists):
        for pattern in group:
            if pattern is not None:
                mixed.append(pattern)
    return mixed
# Ich wollte Hirn zusammenfassen und Prostata (auch auf viel Koerper trainiert, hiess aber durch CNN so)
# Interleave kurze Regeln (Kartenmisch-artig) quasi da Reihenfolge jeweiliger Klasse gleich wichtig ist m1 = [a, b] und m2 = [c, d] wird m_hirn = [a, c, b, d]
MRT_HIRN_RULES_SHORT = interleave_patterns(
    MRT_HIRN_T1_SHORT,
    MRT_HIRN_T2_SHORT,
    MRT_HIRN_FLAIR_SHORT,
    MRT_HIRN_T1_C_SHORT,
)
MRT_BODY_RULES_SHORT = interleave_patterns(
    MRT_PROSTATA_T1_SHORT,
    MRT_PROSTATA_T2_SHORT,
    MRT_BODY_GENERAL_SHORT,
)
# Interleave lange Regeln
MRT_HIRN_RULES_LONG = interleave_patterns(
    MRT_HIRN_T1_LONG,
    MRT_HIRN_T2_LONG,
    MRT_HIRN_FLAIR_LONG,
    MRT_HIRN_T1_C_LONG,
)
MRT_BODY_RULES_LONG = interleave_patterns(
    MRT_PROSTATA_T1_LONG,
    MRT_PROSTATA_T2_LONG,
    MRT_BODY_GENERAL_LONG,
)
# Weitere Regeln - kurze
CT_HYBRID_RULES_SHORT = [
    r"\bpet\s*/\s*ct\b",
    r"\bpet-ct\b",
    r"\bspect\s*/\s*ct\b",
    r"\bspect-ct\b",
    r"\bfused\s+(pet|spect)\s*[-/]?\s*ct\b",
    r"\bhybrid\s+(pet|spect)\s*[-/]?\s*ct\b",
    r"\bco[- ]registered\s+(pet|spect)\s+(and\s+)?ct\b",
]
CT_RULES_SHORT = [
    r"\bct\b",
    r"\bcomputed tomography\b",
    r"\baxial ct\b",
    r"\bcoronal ct\b",
    r"\bsagittal ct\b",
    r"\bcontrast[- ]enhanced ct\b",
    r"\bhelical ct\b",
    r"\bmdct\b",
    r"\bhrct\b",
]
XRAY_ANGIOGRAPHY_RULES_SHORT = [
    r"\bangioplast\w*\b",
    r"\bangiography\w*\b",
    r"\bpta\b",
    r"\bptca\b",
    r"\bpci\b",
    r"\bfluoroscopy\b",
    r"\bc[- ]arm\b",
    r"\bangiograph\w*\b",
    r"\bdsa\b",
]
XRAY_RULES_SHORT = [
    r"\bx[- ]?ray\b",
    r"\bradiograph\b",
    r"\bplain film\b",
    r"\bap view\b",
    r"\bpa view\b",
    r"\blateral view\b",
]
US_RULES_SHORT = [
    r"\bultrasound\b",
    r"\bsonograph\w*\b",
    r"\bechograph\w*\b",
    r"\bdoppler\b",
]
MICROSCOPY_RULES_SHORT = [
    r"\bmicroscopy\b",
    r"\bhistology\b",
    r"\bimmunohistochemistry\b",
    r"\bh\s*&\s*e\b",
]

PATHOLOGY_RULES_SHORT = [
    r"\bpathology\b",
    r"\bbiopsy\b",
    r"\bpathological findings\b",
]

SURGERY_RULES_SHORT = [
    r"\bintraoperative\b",
    r"\bsurgical findings\b",
    r"\boperation\b",
]

ENDOSCOPY_RULES_SHORT = [
    r"\bendoscopy\b",
    r"\bcolonoscopy\b",
]

CHART_RULES_SHORT = [
    r"\bchart\b",
    r"\bgraph\b",
    r"\bplot\b",
]

# Weitere Regeln - lange
CT_HYBRID_RULES_LONG = [
    r"\bpet\s*/\s*ct\b",
    r"\bpet\s*-\s*ct\b",
    r"\bpetct\b",
    r"\bfused\s+pet\s*[-/]?\s*ct\b",
    r"\bhybrid\s+pet\s*[-/]?\s*ct\b",
    r"\bcombined\s+pet\s*[-/]?\s*ct\b",
    r"\bco[- ]registered\s+pet\s+(and\s+)?ct\b",
    r"\bspect\s*/\s*ct\b",
    r"\bspect\s*-\s*ct\b",
    r"\bspectct\b",
    r"\bfused\s+spect\s*[-/]?\s*ct\b",
    r"\bhybrid\s+spect\s*[-/]?\s*ct\b",
    r"\bcombined\s+spect\s*[-/]?\s*ct\b",
    r"\bco[- ]registered\s+spect\s+(and\s+)?ct\b",
    r"\bsingle[- ]photon emission computed tomography\s+(and|with)\s+ct\b",
    r"\bpositron emission tomography\s+(and|with)\s+computed tomography\b",
]
CT_RULES_LONG = [
    r"\bct\b",
    r"\bct scan\b",
    r"\bcomputed tomography\b",
    r"\bcomputed tomograph\w*\b",
    r"\baxial ct\b",
    r"\bcoronal ct\b",
    r"\bsagittal ct\b",
    r"\bcontrast[- ]enhanced ct\b",
    r"\bnon[- ]contrast ct\b",
    r"\bhelical ct\b",
    r"\bmultidetector ct\b",
    r"\bmdct\b",
    r"\bhrct\b",
    r"\bcect\b",
    r"\bhounsfield\b",
]
XRAY_ANGIOGRAPHY_RULES_LONG = [
    r"\bangioplast\w*\b",
    r"\bangiography\w*\b",
    r"\bballoon angioplasty\b",
    r"\bpercutaneous transluminal angioplasty\b",
    r"\bpta\b",
    r"\bptca\b",
    r"\bcoronary angioplasty\b",
    r"\bstent placement\b",
    r"\bpercutaneous coronary intervention\b",
    r"\bpci\b",
    r"\bfluoroscopy\b",
    r"\bfluoroscopic\b",
    r"\bc[- ]arm\b",
    r"\bx[- ]ray guided\b",
    r"\bfluoroscopic guidance\b",
    r"\breal[- ]time x[- ]ray\b",
    r"\bangiograph\w*\b",
    r"\bdigital subtraction angiography\b",
    r"\bdsa\b",
    r"\bcatheter angiography\b",
]
XRAY_RULES_LONG = [
    r"\bx[- ]?ray\b",
    r"\bxray\b",
    r"\bradiograph\b",
    r"\bradiographic\b",
    r"\bprojection radiography\b",
    r"\bfilm\b",
    r"\bportable x[- ]ray\b",
    r"\broentgen\b",
    r"\br\s*[öo]ntgen\b",
    r"\bdorsoplantar projection\b",
]
US_RULES_LONG = [
    r"\bultrasound\b",
    r"\bsonograph\w*\b",
    r"\bultrasonograph\w*\b",
    r"\bechograph\w*\b",
    r"\bechocardiograph\w*\b",
    r"\bdoppler\b",
    r"\bduplex sonograph\w*\b",
    r"\bendoscopic ultrasound\b",
    r"\beus\b",
    r"\bb[- ]mode ultrasound\b",
    r"\bcolor doppler\b",
    r"\bpower doppler\b",
    r"\btransvaginal\b",
    r"\btransrectal\b",
    r"\btransabdominal\b",
]
MICROSCOPY_RULES_LONG = [
    r"\bmicroscopy\b",
    r"\bmicroscopic\b",
    r"\bhistology\b",
    r"\bhistologic\w*\b",
    r"\bimmunohistochemistry\b",
    r"\bihc\b",
    r"\bh\s*&\s*e\b",
    r"\bhematoxylin\b",
    r"\beosin\b",
    r"\btissue section\b",
    r"\bpathology slide\b",
    r"\bstaining\b",
    r"\bcells\/hpf\b",
    r"\bhigh[- ]power field\b",
]
PATHOLOGY_RULES_LONG = [
    r"\bpathological findings\b",
    r"\bpathology\b",
    r"\bpathologic\w*\b",
    r"\bbiopsy\b",
    r"\bbiopsy findings\b",
    r"\btissue specimen\b",
    r"\bcell infiltration\b",
    r"\beosinophil counts\b",
    r"\binflammatory cell infiltration\b",
]
SURGERY_RULES_LONG = [
    r"\bintraoperative\b",
    r"\boperative findings\b",
    r"\bsurgical findings\b",
    r"\bsurgery\b",
    r"\bsurgical view\b",
    r"\boperation\b",
    r"\bresected specimen\b",
    r"\bgross specimen\b",
    r"\bmacroscopic\b",
    r"\bclinical photograph\b",
    r"\bphotograph\b",
    r"\bphoto\b",
    r"\bwound\b",
    r"\blesion photograph\b",
]
ENDOSCOPY_RULES_LONG = [
    r"\bendoscopy\b",
    r"\bendoscopic\b",
    r"\bcolonoscopy\b",
    r"\bgastroscopy\b",
    r"\besophagogastroduodenoscopy\b",
    r"\blaparoscopy\b",
    r"\bbronchoscopy\b",
    r"\bduodenoscopy\b",
    r"\benteroscopy\b",
    r"\bcolono fiberscope\b",
    r"\bcf\b",
]
CHART_RULES_LONG = [
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
    r"\btimeline\b",
]
LABEL_PRIORITY = {
    "ct": 100,
    "us": 90,
    "mrt_hirn": 85,
    "mrt_body": 80,
    "xray_fluoroskopie_angiographie": 80,
    "ct_kombimodalitaet_spect+ct_pet+ct": 70,
    "xray": 65,
    "microscopy": 54,
    "pathology": 50,
    "surgery_real": 45,
    "endoscopy": 35,
    "chart_or_diagram": 34,
}
# RULES
RULES_SHORT = [
    ("xray_fluoroskopie_angiographie", XRAY_ANGIOGRAPHY_RULES_SHORT),
    ("us", US_RULES_SHORT),
    ("ct_kombimodalitaet_spect+ct_pet+ct", CT_HYBRID_RULES_SHORT),
    ("mrt_hirn", MRT_HIRN_RULES_SHORT),
    ("mrt_body", MRT_BODY_RULES_SHORT),
    ("ct", CT_RULES_SHORT),
    ("xray", XRAY_RULES_SHORT),
    ("microscopy", MICROSCOPY_RULES_SHORT),
    ("pathology", PATHOLOGY_RULES_SHORT),
    ("surgery_real", SURGERY_RULES_SHORT),
    ("endoscopy", ENDOSCOPY_RULES_SHORT),
    ("chart_or_diagram", CHART_RULES_SHORT),
]
RULES_LONG = [
    ("xray_fluoroskopie_angiographie", XRAY_ANGIOGRAPHY_RULES_LONG),
    ("us", US_RULES_LONG),
    ("ct_kombimodalitaet_spect+ct_pet+ct", CT_HYBRID_RULES_LONG),
    ("mrt_hirn", MRT_HIRN_RULES_LONG),
    ("mrt_body", MRT_BODY_RULES_LONG),
    ("ct", CT_RULES_LONG),
    ("xray", XRAY_RULES_LONG),
    ("microscopy", MICROSCOPY_RULES_LONG),
    ("pathology", PATHOLOGY_RULES_LONG),
    ("surgery_real", SURGERY_RULES_LONG),
    ("endoscopy", ENDOSCOPY_RULES_LONG),
    ("chart_or_diagram", CHART_RULES_LONG),
]

def contains_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text) for p in patterns)

def rule_based_classify_with_rules(text: str, rules, label_priority=None):

    t = normalize_for_rules(text)

    if not t:
        return "unknown", "empty_text", "", {}

    if label_priority is None:
        label_priority = {}

    # =========================================================
    # Score_boni
    # =========================================================

    SPECIAL_CLASS_BONUS = {
        "ct": 8,
        "xray_fluoroskopie_angiographie": 5,
        "us": 2,
    }

    # =========================================================
    # Parallel matching
    # =========================================================

    hits = {}

    for label, patterns in rules:

        for pattern in patterns:

            try:
                matched = re.search(pattern, t)

            except re.error:
                print(f"Regex Fehler: {pattern}")
                continue

            if matched:

                if label not in hits:
                    hits[label] = {
                        "score": 0,
                        "patterns": [],
                    }

                hits[label]["score"] += 1
                hits[label]["patterns"].append(pattern)

    if not hits:
        return "unknown", "no_rule_match", "", {}

    strong_context_labels = [
        "surgery_real",
        "pathology",
        "microscopy",
        "endoscopy",
        "chart_or_diagram",
    ]

    for label in strong_context_labels:

        if label in hits:
            hits[label]["score"] += 3

    # =========================================================
    # Bonus fuer spezifische Radiologieklassen
    # =========================================================

    for label, bonus in SPECIAL_CLASS_BONUS.items():

        if label in hits:
            hits[label]["score"] += bonus

    # =========================================================
    # Harte Priorisierung
    # =========================================================

    # PET/CT und SPECT/CT fast immer sehr spezifisch
    if "ct_kombimodalitaet_spect+ct_pet+ct" in hits:

        matched_patterns = hits[
            "ct_kombimodalitaet_spect+ct_pet+ct"
        ]["patterns"]

        return (
            "ct_kombimodalitaet_spect+ct_pet+ct",
            "forced_hybrid_ct",
            " | ".join(matched_patterns[:5]),
            hits,
        )

    # Angiographie oft sonst von xray verdrängt
    if "xray_fluoroskopie_angiographie" in hits:

        matched_patterns = hits[
            "xray_fluoroskopie_angiographie"
        ]["patterns"]

        return (
            "xray_fluoroskopie_angiographie",
            "forced_angiography",
            " | ".join(matched_patterns[:5]),
            hits,
        )

    # print("\n===== RULE DEBUG =====")
    #
    # for cls in sorted(
    #     hits.keys(),
    #     key=lambda x: hits[x]["score"],
    #     reverse=True
    # ):
    #
    #     print(
    #         cls,
    #         "score=",
    #         hits[cls]["score"],
    #         "patterns=",
    #         hits[cls]["patterns"][:5]
    #     )

    best_label = sorted(
        hits.keys(),
        key=lambda label: (
            hits[label]["score"],
            label_priority.get(label, 0)
        ),
        reverse=True
    )[0]

    matched_patterns = hits[best_label]["patterns"]

    return (
        best_label,
        f"parallel_rule:{best_label}; hits={hits[best_label]['score']}",
        " | ".join(matched_patterns[:5]),
        hits,
    )

# ============================================================
# CNN Wrapper
# ============================================================
def predict_with_cnn(model, image, transform, device, class_names):
    if image is None:
        return [], {}

    try:
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)[0]  # shape: [num_classes]

        probs = probs.cpu().numpy()

        # Top-3
        top_indices = probs.argsort()[-3:][::-1]

        top3 = []
        for idx in top_indices:
            label = class_names[idx]
            prob = float(probs[idx])
            top3.append((label, prob))

        full_scores = {
            class_names[i]: float(probs[i])
            for i in range(len(class_names))
        }

        return top3, full_scores

    except Exception:
        return [], {}

def run_cnn(model, image, transform, device):
    if image is None:
        return "unknown"

    try:
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            _, pred = torch.max(output, 1)
        return str(pred.item())
    except:
        return "unknown"


# ============================================================
# Fast pre-sampling (nur Rules)
# ============================================================
# ============================================================
# Checksum
# ============================================================
def debug_sampling(records, per_class, class_key="rule_label"):
    print("\n===== Early Sampling Checksum&Debug =====")

    counts = Counter()
    missing = 0

    for r in records:
        lbl = r.get(class_key)

        if lbl is None:
            missing += 1
            continue

        lbl = str(lbl).lower().strip()

        if lbl not in FINAL_CLASSES:
            continue

        counts[lbl] += 1

    print("\nGefundene Klassen:")
    for k, v in sorted(counts.items()):
        print(f"{k}: {v}")

    print(f"\nFehlende Labels: {missing}")
    print(f"Total gültige Labels: {sum(counts.values())}")

    print("\n===== Checksum =====")

    ok = True
    for cls in FINAL_CLASSES:
        c = counts.get(cls, 0)

        if c != per_class:
            print(f"{cls}: {c} (expected {per_class})")
            ok = False
        else:
            print(f"{cls}: {c}")

    if not ok:
        raise ValueError("Sampling ist nicht balanced!")

    print("\nSampling korrekt balanced!")

def early_balanced_sampling(ds, per_class, limit=None):

    buckets = defaultdict(list)

    # Hash-basierte Reihenfolge (=reproduktiv zufaellig)
    indices = list(range(len(ds)))
    indices.sort(key=stable_hash_index)

    max_n = min(len(indices), limit) if limit is not None else len(indices)

    for idx in tqdm(indices[:max_n], desc="Early sampling (RULES only)"):

        row = ds[idx]

        text, meta = extract_text_from_row(row)
        text = normalize_text(text)

        rule_label, reason, matched, hits = rule_based_classify_with_rules(
            text, RULES_LONG, LABEL_PRIORITY)

        # print(f"{rule_label} | {text[:80]}")

        if rule_label not in FINAL_CLASSES:
            continue

        if rule_label == "unknown":
            print("Regellabel ist unknown TEXT:", text[:200])

        if len(buckets[rule_label]) >= per_class:
            continue

        buckets[rule_label].append({
            "row_id": idx,
            "pmc_id": meta.get("PMC_ID", ""),
            "caption": text,
            "row": row,
            "rule_label": rule_label,
            "modality_gt": meta.get("modality", "unknown"),
            "rule_reason": reason,
            "matched_patterns": matched.split(" | ") if matched else []
        })

        # Early Stop
        if all(len(buckets[c]) >= per_class for c in FINAL_CLASSES):
            print(f"\nAlle Klassen voll (per_class) bei idx={idx} → STOP")
            break

    # Flatten
    records = []
    for c in FINAL_CLASSES:
        records.extend(buckets[c])

    print(f"\nGesammelt: {len(records)} Samples")

    return records

# ============================================================
# Similarity-BERT-Klassifikation (Cosine-Ahnlichkeit zu gegebener Description)
# ============================================================

CLASS_TEXTS: Dict[str, str] = {
    "xray": (
        "X-ray radiography, radiograph, plain radiograph, chest x-ray, abdominal x-ray, "
        "skeletal radiograph, projection radiography, conventional x-ray, AP view, PA view, "
        "lateral view, portable x-ray, panoramic radiograph, dorsoplantar projection, "
        "plain film, frontal radiograph, lateral radiograph."
    ),
    "xray_fluoroskopie_angiographie": (
        "Fluoroscopy, fluoroscopic x-ray imaging, real-time x-ray imaging, c-arm fluoroscopy, "
        "x-ray guided interventional procedure, fluoroscopic guidance, contrast fluoroscopy, "
        "angiography, angiographic image, digital subtraction angiography, DSA, catheter angiography, "
        "vascular intervention, angioplasty, balloon angioplasty, percutaneous transluminal angioplasty, "
        "PTA, PTCA, coronary angioplasty, stent placement, percutaneous coronary intervention, PCI."
    ),
    "us": (
        "Ultrasound imaging, sonography, ultrasonography, echography, echocardiography, "
        "B-mode ultrasound, doppler ultrasound, duplex sonography, endoscopic ultrasound, EUS, "
        "color doppler, power doppler, sonographic examination, ultrasonographic image."
    ),
    "mrt_hirn": (
        "Brain MRI, cranial MRI, cerebral MRI, neuro MRI, head MRI, magnetic resonance imaging of the brain, "
        "T1-weighted brain MRI, T2-weighted brain MRI, FLAIR brain MRI, fluid-attenuated inversion recovery, "
        "DWI brain MRI, ADC brain MRI, contrast-enhanced T1 brain MRI, gadolinium-enhanced brain MRI, "
        "post-contrast brain MRI, axial brain MRI, sagittal brain MRI, coronal brain MRI."
    ),
    "mrt_body": (
        "Body MRI, magnetic resonance imaging outside the brain, abdominal MRI, pelvic MRI, prostate MRI, "
        "breast MRI, liver MRI, renal MRI, kidney MRI, cardiac MRI, heart MRI, spine MRI, knee MRI, "
        "whole-body MRI, multiparametric prostate MRI, mpMRI, prostate T1-weighted MRI, prostate T2-weighted MRI, "
        "diffusion-weighted prostate MRI, ADC map of prostate MRI, dynamic contrast-enhanced prostate MRI, DCE MRI, "
        "pelvic MR imaging of the prostate gland."
    ),
    "ct": (
        "Computed tomography, CT scan, computed tomographic image, axial CT, coronal CT, sagittal CT, "
        "contrast-enhanced CT, non-contrast CT, helical CT, multidetector CT, MDCT, HRCT, "
        "brain CT, chest CT, abdominal CT, pelvic CT, Hounsfield units."
    ),
    "ct_kombimodalitaet_spect+ct_pet+ct": (
        "Hybrid CT imaging with PET or SPECT, PET/CT, PET-CT, fused PET-CT image, hybrid PET/CT imaging, "
        "co-registered PET and CT, combined positron emission tomography and computed tomography, "
        "SPECT/CT, SPECT-CT, fused SPECT-CT image, hybrid single-photon emission computed tomography and CT, "
        "nuclear medicine hybrid imaging, tracer uptake co-registered with CT anatomy."
    ),
    "microscopy": (
        "Microscopy, microscopic image, histology, histopathology, pathology slide, tissue section, "
        "H and E stain, hematoxylin and eosin staining, immunohistochemistry, IHC staining, "
        "cells per high-power field, microscopic tissue morphology."
    ),
    "pathology": (
        "Pathology, pathological findings, biopsy findings, tissue pathology, biopsy specimen, "
        "inflammatory cells, eosinophil counts, pathological examination, tissue specimen, "
        "cell infiltration, histopathological diagnosis."
    ),
    "surgery_real": (
        "Real clinical photograph, photo, a photo of, photograph, surgical photograph, intraoperative photograph, operative findings, "
        "surgical field, surgical view, surgical photo, photo of surgery, photo of surgical, gross, macroscopic specimen, resected specimen, gross specimen, "
        "wound photograph, lesion photograph, real-world medical image from operation or clinical examination."
    ),
    "endoscopy": (
        "Endoscopy, colonoscopy, gastroscopy, bronchoscopy, enteroscopy, duodenoscopy, laparoscopic view, "
        "endoscopic findings, mucosal findings, colono fiberscope, gastrointestinal endoscopy."
    ),
    "chart_or_diagram": (
        "Chart, graph, plot, line graph, bar chart, histogram, box plot, schematic, workflow figure, "
        "timeline, clinical course figure, therapeutic management chart, Kaplan-Meier curve, ROC curve."
    ),
    "unknown": (
        "Unknown or not identifiable modality, insufficient information, unclear caption."
    ),
}

class BertSimilarityClassifier:
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
# Dense-linear-Layer-Softmax über Embeddings (klassisches antrainieren) BERT-Klassifikation
# ============================================================
class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        return self.classifier(cls)
# braucht Trainingsdaten

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
# Loop Kapsel
# ============================================================
class ModelContext:
    def __init__(
        self,
        bert_clf,
        cnn1_model,
        cnn2_model,
        cnn3_model,
        cnn_transform,
        device
    ):
        self.bert = bert_clf
        self.cnn1 = cnn1_model
        self.cnn2 = cnn2_model
        self.cnn3 = cnn3_model
        self.transform = cnn_transform
        self.device = device

def top_label(scores):
    if not scores:
        return "none"
    # Dict
    if isinstance(scores, dict):
        return max(scores, key=scores.get)
    # List[Tuple]
    if isinstance(scores, list):
        first = scores[0]
        # [("xray", 0.9), ...]
        if isinstance(first, tuple):
            return max(scores, key=lambda x: x[1])[0]
        # [{"label":..., "score":...}]
        if isinstance(first, dict):
            return max(scores, key=lambda x: x.get("score", 0)).get("label", "none")

    return "none"
# ============================================================
# Verarbeitung (Ph. 3)
# ============================================================
def process_single_record(r, ctx: ModelContext, missing_classes=None, cnn_thresh=0.82, bert_thresh=0.88, cnn_margin_thresh=0.18):
    """
    Das ist mein Phase-3 Code für ein Sample
    (Rules + CNN + BERT + LLM Entscheidung)
    """

    text = r.get("caption")
    row = r.get("row")

    # =========================
    # RULES
    # =========================
    rule_scores = {c: 0.0 for c in FINAL_CLASSES}
    if r.get("rule_label") in FINAL_CLASSES:
        rule_scores[r["rule_label"]] = 1.0

    # =========================
    # BERT
    # =========================
    if ctx.bert:
        if text is None:
            text = ""

        if not isinstance(text, str):
            text = str(text)

        text = text.strip()

        if len(text) == 0:
            text = "unknown"

        bert_out = ctx.bert.predict_batch([text])[0]
        bert_scores = {c: 0.0 for c in FINAL_CLASSES}
        if bert_out["label"] in FINAL_CLASSES:
            bert_scores[bert_out["label"]] = bert_out["score"]
    else:
        bert_scores = {c: 0.0 for c in FINAL_CLASSES}
        bert_out = {"label": "none", "score": 0.0}
    # =========================Umgehung, weil manchmal selbst row None war
    if row is None:
        print("\nRow ist None")
        return None
    # =========================
    # CNN1 + CNN2
    # =========================
    image = get_image_from_row(row)

    cnn1_top3, cnn1_full = predict_with_cnn(
        ctx.cnn1, image, ctx.transform, ctx.device, CNN1_CLASS_NAMES)
    cnn1_scores = map_scores_to_final(cnn1_full, CNN1_MAPPING)

    cnn2_top3, cnn2_full = predict_with_cnn(
        ctx.cnn2, image, ctx.transform, ctx.device, CNN2_CLASS_NAMES)
    cnn2_scores = map_scores_to_final(cnn2_full, CNN2_MAPPING)

    cnn_scores = fuse_cnn_weighted(cnn1_scores, cnn2_scores, cnn1_weights, cnn2_weights, alpha=0.5)
    cnn_scores = normalize_scores(cnn_scores)
    # =========================
    # CNN3 Filtering + CNN1&CNN2-Uncertainty Filtering
    # =========================
    cnn_conf, cnn_margin, _, _ = compute_cnn_confidence_and_margin(cnn_scores)

    cnn_uncertain = is_cnn_uncertain(cnn_conf, cnn_margin)

    cnn3_top3, cnn3_full = predict_with_cnn(
        ctx.cnn3, image, ctx.transform, ctx.device, CNN3_CLASS_NAMES)
    cnn3_pred = top_label(cnn3_top3)
    CNN3_STRONG = 0.93
    CNN3_MEDIUM = 0.65
    rule_conf = max(rule_scores.values())
    bert_conf = max(bert_scores.values())
    cnn_conf = max(cnn_scores.values())
    if cnn3_full:
        cnn3_conf = max(cnn3_full.values())
    else:
        cnn3_conf = 0.0
    if cnn3_conf >= CNN3_STRONG:
        # sehr sicher Rauswurf
        r["is_filtered"] = True
        r["filter_reason"] = "cnn3_strong"

        print(f"CNN3Filt {cnn3_pred} {cnn3_conf}")

        return r

    elif cnn3_conf >= CNN3_MEDIUM:
        # unsicher. nur filtern wenn andere schwach
        if bert_conf <= 0.5 and cnn_conf <= 0.5:
            return {"filtered": True, "reason": "cnn3_medium"}

    # =========================
    # Filtering Entscheidung ueber CNN1/CNN2/RULES/BERT-einzelweises Scoring
    # =========================
    # Idee:
    # CNN3 darf nur filtern, wenn: Konsens
    # - gehört zu Filterklassen
    # - CNN3 ziemlich sicher
    # - andere Modelle nicht stark sind

    CNN3_CONF_MIN = 0.745  # (0.55) aber hochgetan: Ueber 0.745 sind es Trash-Bilder
    PRE_CONF_MAX = 0.7  # (0.35) (0.66) aber hochgedrillt: nach 0.7 sind sich die anderen modelle sicher
    # Klassen der Modelle muessen >0.7 haben. Muellmann moechte nur >=0.75.
    # Wenn Muellmann selbst unsicher ist, ist >0.7 gut genug, um als Klasse angenommen zu werden

    if (
            cnn3_conf > CNN3_CONF_MIN and
            bert_conf < PRE_CONF_MAX and
            cnn_conf < PRE_CONF_MAX
    ):
        return {"filtered": True, "reason": "noconsent"}
    # =========================
    # Quick Fusion (ohne LLM) laeuft schneller
    # Rules macht subset. pre_fuse Fusioniert Rules/CNN/BERT. LLM nur bei unsicheren Faellen als Richter
    # =========================
    quick_label, quick_scores, confident = quick_fuse(
        rule_scores,
        bert_scores,
        cnn_scores)

    # =========================
    # LLM Entscheidung KONFLIKT + CNN LOGIK
    # =========================

    rule_pred = top_label(rule_scores)
    bert_pred = top_label(bert_scores)
    cnn_pred = top_label(cnn_scores)

    # ============================================================
    # Nur fehlende Klassen weiter prüfen
    # ============================================================
    if missing_classes is not None:
        if (
                rule_pred not in missing_classes
                and cnn_pred not in missing_classes
                and bert_pred not in missing_classes
        ):
            return None
    conflict_level = len({rule_pred, bert_pred, cnn_pred})
    #cnn_conf oben schon definiert
    bert_conf = max(bert_scores.values()) if bert_scores else 0.0

    BERT_STRONG = 0.75
    CNN_EIGENTLICH_STRONG = 0.675

    # =========================
    # LLM Entscheidung
    # =========================
    use_llm = False #default

    # 🔴 Fall 1: CNN unsicher + kein Konsens
    if cnn_uncertain and conflict_level >= 2:
        use_llm = True

    # 🟢 Fall 2: kompletter Konflikt LLM
    if conflict_level == 3:
        if cnn_conf > 0.6 and cnn_margin > 0.17:
            use_llm = False  # CNN sicher.CNN entscheidet
        elif rule_conf >= 1.0 and conflict_level == 1:
            use_llm = False  # Rule_conf immer 1 wenn R. findet, aber mindestens conf lvl soll gelten
        elif bert_conf >= 0.75:
            use_llm = False  # BERT sicher. entscheidet
       # 🔴
        elif rule_pred != cnn_pred and bert_pred != cnn_pred:
            use_llm = True
        else:
            use_llm = True
        print(
            f"R:{rule_pred} "
            f"B:{bert_pred} "
            f"C:{cnn_pred} "
            f"| lvl={conflict_level} "
            f"| conf={cnn_conf:.2f} "
            f"| margin={cnn_margin:.2f} "
            f"| LLM={use_llm}"
        )

    # 🟢 Fall 3: alles einig → KEIN LLM
    elif conflict_level == 1:
        use_llm = False
        print(
            f"R:{rule_pred} "
            f"B:{bert_pred} "
            f"C:{cnn_pred} "
            f"| lvl={conflict_level} "
            f"| conf={cnn_conf:.2f} "
            f"| margin={cnn_margin:.2f} "
            f"| LLM={use_llm}"
        )
    # 🟢 Fall 4: 2 stimmen überein + CNN sicher
    elif conflict_level == 2 and cnn_conf > CNN_EIGENTLICH_STRONG and cnn_margin > 0.1:
        use_llm = False
        print(
            f"R:{rule_pred} "
            f"B:{bert_pred} "
            f"C:{cnn_pred} "
            f"| lvl={conflict_level} "
            f"| conf={cnn_conf:.2f} "
            f"| margin={cnn_margin:.2f} "
            f"| LLM={use_llm}"
        )
    # 🔴 Fall 5: Konflikt gegeben, CNN weiss es mittelmaessig
    elif conflict_level >= 2 and 0.49 <= cnn_conf < CNN_EIGENTLICH_STRONG:
        use_llm = True
    # 🔴 Fall 6: Konflikt gegeben, CNN weiss es mittelmaessig, keine Diskrepanz
    elif conflict_level >= 2 and cnn_margin <= 0.024:
        use_llm = True

    # 🟢 Fall 7: BERT stark + CNN stabil
    elif bert_conf >= BERT_STRONG:
        use_llm = False
        print(
            f"R:{rule_pred} "
            f"B:{bert_pred} "
            f"C:{cnn_pred} "
            f"| lvl={conflict_level} "
            f"| conf={cnn_conf:.2f} "
            f"| margin={cnn_margin:.2f} "
            f"| LLM={use_llm}"
        )
    
    else:
        use_llm = False

    if not use_llm:
        r.update({
            "final_label": quick_label,
            "llm_needed": False
        })
    else:
        r.update({
            "quick_label": quick_label,
            "llm_needed": True
        })

    # =========================
    # Debug + Speicherung
    # =========================
    result = {
        "pmc_id": r["pmc_id"],
        "row_id": r["row_id"],
        "caption": text,
        "row": row,

        "final_label": quick_label if not use_llm else None,
        "quick_label": quick_label,
        "llm_needed": use_llm,

        "rule_scores": rule_scores,
        "bert_scores": bert_scores,
        "cnn_scores": cnn_scores,

        "cnn_conf": cnn_conf,
        "cnn_margin": cnn_margin,

        "cnn1_top3": cnn1_top3,
        "cnn2_top3": cnn2_top3,

        "cnn3_conf": cnn3_conf,
    }

    return result

def process_batch(records0, ctx):

    processed = []
    filtered_counts = Counter()

    for r in tqdm(records0, desc="Processing"):

        out = process_single_record(r, ctx)
        # Wichtig ist hier nicht "if r is..." u. "if r.get...", da r fuer den iterativen fill beibehalten w.soll und nicht verfaelscht werden soll!
        if out is None:
            filtered_counts["unknown"] += 1
            continue

        if out.get("filtered"):       #cnn3certainty & cnn-uncertainty & Konsens R/B/C-Modelle
            filtered_counts[out.get("reason", "unknown")] += 1
            continue

        processed.append(out)

    print(f"Filtered: {filtered_counts}")
    return processed

# ============================================================
# ITERATIVES PRESAMPLING + FAST REFILL
# Fuhrt nach Loop den Loop nochmal mit besteh. Presamp auf und fuhrt
# nach Loop den Loop mit neu gesuchten Presamp auf und
# Fuellt die fehlenden Klassen per_class neu auf
# ============================================================
# Voraussetzung: existieren bereits:
#
# FINAL_CLASSES
# stable_hash_record()
# process_single_record()
# run_final_fusion()
# extract_text_from_row()

# ============================================================
# Iteratives presampling + fast refill Hilfsfunktion
# ============================================================
def count_labels(records):
    c = Counter()

    for r in records:
        label = r.get("final_label")
        if label in FINAL_CLASSES:
            c[label] += 1
    return c

# ============================================================
# Fast Local Balancing
# ============================================================
def fast_local_balancing(existing, pool, per_class):
    buckets = {c: [] for c in FINAL_CLASSES}

    # EXISTING

    for r in existing:

        label = r.get("final_label")

        if label in FINAL_CLASSES:
            buckets[label].append(r)

    used = set(
        (r["pmc_id"], r["row_id"])
        for r in existing
    )

    # Aus presamp. Deterministisch

    pool = sorted(pool, key=stable_hash_record)

    # FILL

    for r in pool:
        key = (r["pmc_id"], r["row_id"])

        if key in used:
            continue

        label = r.get("final_label")

        if label not in FINAL_CLASSES:
            continue
        if len(buckets[label]) >= per_class:
            continue

        buckets[label].append(r)
        used.add(key)
        print(f"+ refill {label}")

        done = all(len(buckets[c]) >= per_class for c in FINAL_CLASSES)

        if done:
            break

    # FINAL

    final = []

    for c in FINAL_CLASSES:

        bucket = sorted(
            buckets[c],
            key=stable_hash_record
        )

        final.extend(bucket[:per_class])

    return final

# ============================================================
# PRESAMPLE AUS GROSSEM DATASET
# ============================================================

def sample_new_chunk(ds, global_used, sample_size=100):

    candidates = []
# --------------------------------------------------------
# Hash-stabile Reihenfolge
# --------------------------------------------------------
    for idx in range(len(ds)):
        row = ds[idx]
        try:
            text, meta = extract_text_from_row(row)
        except Exception:
            continue

        key = (meta.get("PMC_ID", ""),idx)

        if key in global_used:
            continue

        record = {
            "pmc_id": meta.get("PMC_ID", ""),
            "row_id": idx}

        h = stable_hash_record(record)

        candidates.append((h, idx))
    candidates.sort(key=lambda x: x[0])

    selected = []

    for _, idx in candidates[:sample_size]:
        row = ds[idx]
        try:
            text, meta = extract_text_from_row(row)
        except Exception:
            continue

        r = {
            "row": row,
            "caption": text,
            "pmc_id": meta.get("PMC_ID", ""),
            "row_id": idx,
            "rule_label": "unknown"}

        selected.append(r)

    return selected

# ============================================================
# Volle Inferenz
# ============================================================
def run_full_inference(records, ctx):
    processed = []
    llm_results = []

    for r in records:
        out = process_single_record(r, ctx)

        if out is None:
            continue

        if out.get("is_filtered"):
            continue

        processed.append(out)
        # Dummy LLM Result
        llm_results.append(("unknown", 0.0))

    initial_records = run_final_fusion(processed, llm_results)

    return initial_records

# ============================================================
# Hauptpipeline
# ============================================================
def build_balanced_dataset(
        ds,
        ctx,
        existing_records,
        initial_presample,
        refill_presample,
        per_class,
        max_rounds
):
    # --------------------------------------------------------
    # Global
    # --------------------------------------------------------
    accepted_records = list(existing_records)

    remaining_pool = []

    # echte Dublettenkontrolle
    global_used = set()

    # ========================================================
    # Refill State
    # ========================================================
    refill_cursor = 0

    stagnation_counter = 0

    # ========================================================
    # Rounds
    # ========================================================
    for round_idx in range(max_rounds):
        # Lokale Fehlversuche innerhalb dieser Round
        failed_counter = 0

        print("\n" + "=" * 60)
        print(f"ROUND {round_idx}")
        print("=" * 60)

        # ====================================================
        # Status
        # ====================================================
        counts = count_labels(accepted_records)

        print("\nAktuelle Verteilung:")

        for c in FINAL_CLASSES:
            print(c, counts[c])

        done = all(
            counts[c] >= per_class
            for c in FINAL_CLASSES
        )

        if done:
            print("\n✔ Dataset vollständig balanced")
            break

        # ====================================================
        # Fehlende Klassen
        # ====================================================
        missing_classes = {
            cls for cls in FINAL_CLASSES
            if counts[cls] < per_class
        }

        print("\nFehlende Klassen:")
        print(missing_classes)

        if not missing_classes:
            print("Alle Klassen gefüllt.")
            break
        # ====================================================
        # Micro Decay. Alle 250 erfolglosen Samples Thresholds leicht lockern. Hauptrounds erhalten trotzdem urspruenglichen Decay-Wert!
        # 2it/sek =/= micro round duration ~2.1 min. DENN Erfolge+Fehlversuche dabei
        # ====================================================
        micro_round = failed_counter // 40

        effective_round = round_idx + micro_round

        print(f"\nMicro Round: {micro_round}")
        print(f"Effective Round: {effective_round}")

        # ====================================================
        # Adaptive Thresholds
        # ====================================================
        cnn_thresh = adaptive_threshold(
            start=0.82,
            min_value=0.55,
            refill_round=effective_round,
            decay=0.04
        )

        bert_thresh = adaptive_threshold(
            start=0.88,
            min_value=0.60,
            refill_round=effective_round,
            decay=0.035
        )

        cnn_margin_thresh = adaptive_threshold(
            start=0.18,
            min_value=0.08,
            refill_round=effective_round,
            decay=0.015
        )

        print(f"\nCNN Threshold: {cnn_thresh:.3f}")
        print(f"BERT Threshold: {bert_thresh:.3f}")
        print(f"Margin Threshold: {cnn_margin_thresh:.3f}")

        # ====================================================
        # Presample Size
        # ====================================================
        sample_size = (
            refill_presample
            # if round_idx == 0
            # else initial_presample
        )

        # ====================================================
        # Dataset Ende?
        # ====================================================
        if refill_cursor >= len(ds):

            print("\nDataset vollständig durchsucht.")
            break

        # ====================================================
        # Chunk bestimmen
        # ====================================================
        start_idx = refill_cursor

        end_idx = min(len(ds),
            refill_cursor + sample_size)

        print(f"\nChunk: {start_idx} -> {end_idx}")

        candidate_chunk = ds.select(
            range(start_idx, end_idx)
        )

        indices = list(range(len(candidate_chunk)))

        random.shuffle(indices)

        candidate_chunk = candidate_chunk.select(indices)

        refill_cursor = end_idx

        print("Chunkgröße:", len(candidate_chunk))
        # ====================================================
        # Raw Arrow -> Standard Records
        # ====================================================

        presample = []

        for local_idx, row in enumerate(candidate_chunk):
            key = row["__key__"]
            # --------------------------------------------
            # Arrow-Webds hat rohe Felder ['__key__', '__url__', 'jpg', 'jsonl']. Fuer global genutzt ja/nein Filter reicht blosses __key__
            # Bereits benutzt?
            # --------------------------------------------
            if key in global_used:
                continue

            global_used.add(key)
            # --------------------------------------------
            # Text extrahieren
            # --------------------------------------------
            text, meta = extract_text_from_row(row)

            text = normalize_text(text)
            # --------------------------------------------
            # Leere Texte ignorieren
            # --------------------------------------------
            if text is None:
                continue

            if not isinstance(text, str):
                text = str(text)

            text = text.strip()

            if len(text) == 0:
                continue
            # --------------------------------------------
            # Standard Record erzeugen
            # --------------------------------------------
            record = {
                "row_id": start_idx + local_idx,
                "pmc_id": meta.get("PMC_ID", ""),
                "caption": text,
                "row": row,
                "modality_gt": meta.get("modality", "unknown")}

            presample.append(record)

        print("Nach global_used:", len(presample))

        if len(presample) == 0:
            print("\nLeerer Presample Chunk")
            continue
        # ====================================================
        # Full Inference
        # ====================================================
        print("\nFull inference ...")

        processed = run_full_inference(
            presample,
            ctx)

        print("Processed:", len(processed))

        # ====================================================
        # Existing Keys
        # ====================================================
        accepted_keys = set(
            (r["pmc_id"], r["row_id"])
            for r in accepted_records
        )

        new_accepts = []

        new_remaining = []

        # ====================================================
        # Process Results
        # ====================================================
        for r in processed:

            key = (
                r["pmc_id"],
                r["row_id"]
            )

            # --------------------------------------------
            # Bereits akzeptiert?
            # --------------------------------------------
            if key in accepted_keys:
                continue

            label = r.get("final_label")

            # --------------------------------------------
            # Ungültiges Label
            # --------------------------------------------
            if label not in FINAL_CLASSES:
                new_remaining.append(r)
                continue

            # --------------------------------------------
            # Nur fehlende Klassen priorisieren
            # --------------------------------------------
            if label not in missing_classes:
                new_remaining.append(r)
                continue

            # --------------------------------------------
            # Counts aktualisieren
            # --------------------------------------------
            counts = count_labels(
                accepted_records + new_accepts
            )

            # --------------------------------------------
            # Klasse bereits voll?
            # --------------------------------------------
            if counts[label] >= per_class:
                new_remaining.append(r)
                continue

            # ====================================================
            # Adaptive Qualitätsprüfung
            # ====================================================
            cnn_conf = r.get("cnn_conf", 0.0)

            bert_conf = r.get("bert_conf", 0.0)

            cnn_margin = r.get("cnn_margin", 0.0)

            rule_pred = r.get("rule_pred")

            accept = False

            # --------------------------------------------
            # Rules dominant
            # --------------------------------------------
            if rule_pred == label:
                accept = True

            # --------------------------------------------
            # CNN akzeptieren
            # --------------------------------------------
            elif (
                cnn_conf >= cnn_thresh
                and cnn_margin >= cnn_margin_thresh
            ):
                accept = True

            # --------------------------------------------
            # BERT akzeptieren
            # --------------------------------------------
            elif (
                bert_conf >= bert_thresh
            ):
                accept = True

            # --------------------------------------------
            # Accept / Remaining
            # --------------------------------------------
            if accept:
                new_accepts.append(r)
            else:
                new_remaining.append(r)
                # ====================================================
                # Fehlversuch zählen
                # ====================================================
                failed_counter += 1

        # ====================================================
        # Merge
        # ====================================================
        accepted_records.extend(new_accepts)

        remaining_pool.extend(new_remaining)

        print("\nNeue Accepts:", len(new_accepts))
        print("Remaining pool:", len(remaining_pool))

        # ====================================================
        # Fast Local Balancing
        # ====================================================
        accepted_records = fast_local_balancing(
            existing=accepted_records,
            pool=remaining_pool,
            per_class=per_class)

        # ====================================================
        # Status nach Balancing
        # ====================================================
        print("\nNach local balancing:")

        counts = count_labels(accepted_records)

        refill_added = sum(counts.values()) - sum(counts.values())

        print(f"\nRefill Added: {refill_added}")

        for c in FINAL_CLASSES:
            print(c, counts[c])

        # ====================================================
        # Stagnation Detection
        # ====================================================
        if len(new_accepts) == 0:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        print(f"\nStagnation Counter: {stagnation_counter}")

        if stagnation_counter >= 100:
            print("\nRefill stagniert -> Abbruch")
            break

    # ========================================================
    # Final Dedup
    # ========================================================
    final = []

    seen = set()

    for r in accepted_records:

        key = (
            r["pmc_id"],
            r["row_id"]
        )

        if key in seen:
            continue

        seen.add(key)

        final.append(r)

    # ========================================================
    # Final Status
    # ========================================================
    print("\n" + "=" * 60)
    print("FINAL")
    print("=" * 60)

    counts = count_labels(final)

    for c in FINAL_CLASSES:
        print(c, counts[c])

    return final

# ============================================================
# Inspizieren/Distr/Debug/Übersicht
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

# Labels-Verteilung aus mitgelief. Metadaten
def inspect_modality_distribution(ds: Dataset, limit: int = 5000):
    counter = Counter()

    n = min(len(ds), limit)
    for i in tqdm(range(n), desc="Prüfe modality-Verteilung"):
        meta = parse_jsonl_field(ds[i].get("jsonl"))
        counter[meta.get("modality", "MISSING")] += 1

    print("\n===== modality-Verteilung (Ausschnitt) =====")
    for key, value in counter.most_common():
        print(f"{key}: {value}")
# Labels-Verteilung aus meinen Ergebnissen
def inspect_final_distribution(records, limit=None):
    counter = Counter()

    if limit is not None:
        records = records[:limit]

    for r in records:
        label = r.get("final_label", "MISSING")
        counter[label] += 1

    print("\n===== FINAL LABEL VERTEILUNG =====")
    total = sum(counter.values())

    for key, value in counter.most_common():
        perc = (value / total) * 100 if total > 0 else 0
        print(f"{key}: {value} ({perc:.2f}%)")

def compute_filter_stats(records):
    total = len(records)

    filtered = [r for r in records if r.get("is_filtered", False)]
    kept = [r for r in records if not r.get("is_filtered", False)]

    reasons = Counter(r.get("filter_reason", "unknown") for r in filtered)

    return {
        "total": total,
        "filtered_count": len(filtered),
        "kept_count": len(kept),
        "reasons": reasons
    }

def save_selected_images(records, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    for r in records:
        row = r["row"]

        img = get_image_from_row(row)
        if img is None:
            continue

        pmc_id = r.get("pmc_id", "unknown")
        row_id = r.get("row_id", "unknown")

        fname = f"{pmc_id}_{row_id}.jpg"
        path = output_dir / fname

        try:
            img.save(path)
            saved += 1
        except Exception:
            continue

    print(f"Gespeichert: {saved} Bilder in {output_dir}")
# ============================================================
# hilft das Laden von fehlenden Klassen im Refill durch leichtere Tresholds um unnoetiges langes Suchen zu vermeiden
# ============================================================
def adaptive_threshold(
        start: float,
        min_value: float,
        refill_round: int,
        decay: float
) -> float:

    value = start - refill_round * decay

    return max(min_value, value)


# ============================================================
# Hauptklassifikation
# ============================================================

def classify_dataset(
    dataset_root: Path,
    output_csv: Path,
    biomedbert_path: Optional[str],
    llamamistral_path: str,
    cnn1_path: str,
    cnn2_path: str,
    cnn3_path: str,
    limit: Optional[int],
    inspectlimit: int,
    per_class: int,
    initial_presample: int,
    refill_presample: int,
    max_rounds: int,
    inspect_only: bool = False,
):
    device = "cpu"
    # ============================================================
    # Phase 1: Datensatz initialisieren
    # ============================================================
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
        print(f"\nLimit aktiv: max {limit} Iterationen im Early Sampling")
        print(f"\nLimit aktiv: {len(ds)} Zeilen")

    print("\n\nSchnelle Vorauswahl Early Sampler...")
    # ============================================================
    # Phase 2: schnelle Vorauswahl Early Sampler
    # ============================================================
    records0 = early_balanced_sampling(ds, per_class, limit)
    # Gewaehrleist, dass wirklich alle Klassen Eintraganzahl=per_class haben.
    debug_sampling(records0, per_class)
    # BERT
    bert_clf = None
    if biomedbert_path:
        print("\nInitialize BERT Similarity...")
        bert_clf = BertSimilarityClassifier(model_path=biomedbert_path)

    # CNN
    print("\nInitialize Convolutional Neural Network...")
    cnn1_model = SimpleCNN(num_classes=11)
    cnn1_model.load_state_dict(torch.load(cnn1_path, map_location=device))
    cnn1_model.to(device).eval()

    cnn2_model = SimpleCNN(num_classes=4)
    cnn2_model.load_state_dict(torch.load(cnn2_path, map_location=device))
    cnn2_model.to(device).eval()

    cnn3_model = ThirdCNN(num_classes=6)
    cnn3_model.load_state_dict(torch.load(cnn3_path, map_location=device))
    cnn3_model.to(device).eval()

    cnn_transform = transform

    ctx = ModelContext(
        bert_clf,
        cnn1_model,
        cnn2_model,
        cnn3_model,
        cnn_transform,
        device
    )
    # ============================================================
    # Phase 3: Records sammeln (ohne LLM)
    # ============================================================

    print("\nExtrahiere Records + RULES + CNN + BERT ...")
    print("\nPhase 3: CNN + BERT auf Subset")
    records = process_batch(records0, ctx)
    print("records nach batch:", len(records))

    # ============================================================
    # Phase 4: LLM nur bei unsicheren Fällen
    # ============================================================

    print("\nInitialize LLaMa Mistral ...")
    llm = LocalLLM(model_path=llamamistral_path)
    #debug
    llm_needed_count = sum(r.get("llm_needed", False) for r in records)
    no_llm_count = sum(not r.get("llm_needed", False) for r in records)
    total_count = len(records)

    print("\n===== LLM USAGE =====")
    print(f"LLM nötig:      {llm_needed_count}")
    print(f"Ohne LLM:       {no_llm_count}")
    print(f"Gesamt:         {total_count}")
    print(f"LLM Anteil:     {llm_needed_count / total_count * 100:.2f}%")
    texts = [r["caption"] for r in records if r.get("llm_needed", False)]
    for r in records[:20]:
        print(r.keys())
        break
    llm_results_partial = llm.batch_predict(texts)

    # Mapping zurück
    llm_iter = iter(llm_results_partial)

    llm_results = []
    for r in records:
        if r.get("llm_needed", False):
            llm_results.append(next(llm_iter))
        else:
            llm_results.append((r.get("final_label", "unknown"), 0.5))  # fake confidence

    # ============================================================
    # Volle iterative Build Pipeline: Phase 5: Final fusion, Phase 6: Post-Balancing
    # ============================================================
    records = build_balanced_dataset(
        ds=ds,
        ctx=ctx,
        existing_records=run_final_fusion(records, llm_results),
        per_class=per_class,
        # erstes großes Presample
        initial_presample=initial_presample,
        #100
        # spätere Nachlade-Chunks
        refill_presample=refill_presample,
        #100
        # Sicherheitslimit
        max_rounds=max_rounds
        #20
    )
# ============================================================
# Phase 7: Bilder speichern (f. Viewer)
# ============================================================
    image_output_dir = output_csv.parent / "exported_images"

    print("\nSpeichere ausgewählte Bilder ...")
    save_selected_images(records, image_output_dir)
# ============================================================
# Phase 8: Endergebnisse Head ausgeben
# ============================================================
    df = pd.DataFrame(records)

    empty_mask = df["caption"].fillna("").astype(str).str.strip().eq("")
    df.loc[empty_mask, "final_label"] = "unknown"
    df.loc[empty_mask, "Begründung"] = "empty_text"
    df.loc[empty_mask, "Begründung"] = "empty_text"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"\nCSV gespeichert unter:\n{output_csv}")

    print("\n===== Verteilung final_label =====")
    print(df["final_label"].value_counts(dropna=False))

    print("\n===== Verteilung decision_source =====")
    print(df["Begründung"].value_counts(dropna=False))

    print("\n===== Verteilung modality_gt =====")
    print(df["modality_gt"].value_counts(dropna=False).head(20))

    print(f"\n===== Verteilung von mir klassifizierter Modalitaeten. Inspectlimit: {inspectlimit}=====")
    inspect_final_distribution(records, limit=inspectlimit)

    print("\n===== FILTERING =====")
    stats = compute_filter_stats(records)

    print(f"Original:        {stats['total']}")
    print(f"Weggefiltert:   {stats['filtered_count']}")
    print(f"Übrig:          {stats['kept_count']}")

    print("\n--- Gründe ---")
    for reason, count in stats["reasons"].items():
        print(f"{reason}: {count}")

    if len(records0) > 0:
        print(f"Filter-Rate: {stats['filtered_count'] / len(records0) * 100:.2f}%")
        print(f"Filter-Rate_kept: {stats['filtered_count'] / stats['kept_count'] * 100:.2f}%")

    print("\n===== DEBUG =====")
    print("LLM needed:", sum(r.get("llm_needed", False) for r in records))
    print("No LLM:", sum(not r.get("llm_needed", False) for r in records))
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
        "--llamamistral_path",
        type=str,
        required=True,
        help="Lokaler Pfad zum generativen LLM, LLaMA/Mistral Modell"
    )
    parser.add_argument(
        "--cnn1_path",
        type=str,
        required=True,
        help="Lokaler Pfad zum custom CNN1"
    )
    parser.add_argument(
        "--cnn2_path",
        type=str,
        required=True,
        help="Lokaler Pfad zum custom CNN2"
    )
    parser.add_argument(
        "--cnn3_path",
        type=str,
        required=True,
        help="Lokaler Pfad zum custom filtering CNN3"
    )
    parser.add_argument(
        "--per_class",
        type=int,
        default=5,
        help="Anzahl Samples pro finaler Klasse (hash-balanced)"
    )
    parser.add_argument(
        "--initial_presample",
        type=int,
        default=50000,
        help="Menge des ersten großes Presample vor Refill"
    )
    parser.add_argument(
        "--refill_presample",
        type=int,
        default=300,
        help="Anzahl spätere Nachlade-Chunks Postsample *im* Refill"
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=20,
        help="Anzahl Sicherheitslimit im Refill (=Runden Runs Durchlaeufe)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionales Limit für erste Tests"
    )
    parser.add_argument(
        "--inspectlimit",
        type=int,
        default=35,
        help="Limit für Inspektionsfunktionen (z.B. Verteilungen)"
    )
    parser.add_argument(
        "--bert_threshold",
        type=float,
        default=0.30,
        help="Schwelle Fallback für BERT-Gang, sonst unknown"
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
        llamamistral_path=args.llamamistral_path,
        cnn1_path=args.cnn1_path,
        cnn2_path=args.cnn2_path,
        cnn3_path=args.cnn3_path,
        inspectlimit=args.inspectlimit,
        limit=args.limit,
        per_class=args.per_class,
        initial_presample=args.initial_presample,
        refill_presample=args.refill_presample,
        max_rounds=args.max_rounds,
        inspect_only=args.inspect_only,
    )

if __name__ == "__main__":
    main()