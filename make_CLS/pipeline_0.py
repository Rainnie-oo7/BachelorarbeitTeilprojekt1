# -*- coding: utf-8 -*-

"""
open-pmc Arrow/WebDataset Klassifikation mit Rules + BERT + LLM + CNN

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
  --output_csv /home/user/PycharmProjects/ba1pmc/make_CLS/test_500.csv \
  --biomedbert_path /home/user/Dokumente/biomedbert \
  --llamamistral_path /home/user/Dokumente/LLaMa_Mistral \
  --cnn1_path /home/user/PycharmProjects/ba2roco/cnn/convu_try_3t.pth \
  --cnn2_path /home/user/PycharmProjects/ba2roco/cnn/convu_folderlabel_mrt_body_mrt_hirn.pth \
  --per_class 50 \
  --limit 50
"""

from __future__ import annotations

import os
import os.path as osp
from pathlib import Path
import numpy as np
from PIL import Image
import re
import json
import argparse

import hashlib
from collections import Counter
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
from vllm import LLM, SamplingParams

# Finale Klassen u. Mapping
FINAL_CLASSES = [
    "xray",
    "xray_fluoroskopie_angiographie",
    "us",
    "mrt_hirn",
    "mrt_body",
    "ct",
    "ct_kombimodalitaet_spect+ct_pet+ct"
]
CNN1_CLASS_NAMES = {
    "mrt_prostata_t1"
    "mrt_prostata_t2"
    "mrt_hirn_flair'"
    "mrt_hirn_t1"
    "mrt_hirn_t2"
    "mrt_hirn_t1_c"
    "ct",
    "ct_kombimodalitaet_spect+ct_pet+ct",
    "xray",
    "xray_fluoroskopie_angiographie",
    "us"
}
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
    "us": "us",
}
CNN2_CLASS_NAMES = {
    "xray",
    "xray_fluoroskopie_angiographie",
    "mrt_hirn",
    "mrt_body"
}
CNN2_MAPPING = {
    "xray": "xray",
    "xray_fluoroskopie_angiographie": "xray_fluoroskopie_angiographie",
    "mrt_hirn": "mrt_hirn",
    "mrt_body": "mrt_body",
}
WEIGHTS = {
    "rules": 0.2,
    "bert": 0.2,
    "cnn1": 0.2,
    "cnn2": 0.2,
    "llm": 0.2
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
Return JSON: {{"label": "...", "confidence": 0.0}}

{text}
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
                try:
                    js = json.loads(d[d.find("{"):])
                    results.append((js["label"], js["confidence"]))
                except:
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

# Final-Classes Fusion
def fuse_scores(rule, bert, cnn1, cnn2, llm):
    final = {}
    contrib = {}

    for label in FINAL_CLASSES:
        contrib[label] = {
            "rules": WEIGHTS["rules"] * rule.get(label, 0),
            "bert": WEIGHTS["bert"] * bert.get(label, 0),
            "cnn1": WEIGHTS["cnn1"] * cnn1.get(label, 0),
            "cnn2": WEIGHTS["cnn2"] * cnn2.get(label, 0),
            "llm": WEIGHTS["llm"] * (llm[1] if llm[0] == label else 0),
        }

        final[label] = sum(contrib[label].values())

    sorted_labels = sorted(final.items(), key=lambda x: x[1], reverse=True)

    top1, top2 = sorted_labels[0], sorted_labels[1]
    uncertain = (top1[1] - top2[1]) < 0.1

    explanation = sorted(contrib[top1[0]].items(), key=lambda x: x[1], reverse=True)

    return top1[0], final, contrib, explanation, uncertain

# ============================================================
# Hash-Sampling (100k Ziel)
# ============================================================
def stable_hash(text):
    return int(hashlib.md5(text.encode()).hexdigest(), 16)

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
            key=lambda x: stable_hash(x["caption"])
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
# Bild-Extractor aus Arrow
# ============================================================

def get_image_from_row(row):
    img = row.get("image", None)

    if img is None:
        return None

    try:
        # Bytes (meistens)
        if isinstance(img, bytes):
            return Image.open(io.BytesIO(img)).convert("RGB")

        # HF datasets Image-Objekt
        if hasattr(img, "convert"):
            return img.convert("RGB")

        if isinstance(img, np.ndarray):
            return Image.fromarray(img).convert("RGB")

    except Exception:
        return None

    return None

# ============================================================
# Harte Regeln
# ============================================================

RULE_LABELS_ORDER = [
    "ct_kombimodalitaet_spect+ct_pet+ct",
    "mrt_hirn",
    "mrt_body",
    "ct",
    "xray_fluoroskopie_angiographie",
    "xray",
    "us",

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
    r"\angiography\w*\b",
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
    r"\bplain film\b",
    r"\bchest film\b",
    r"\bportable chest\b",
    r"\bportable x[- ]ray\b",
    r"\broentgen\b",
    r"\br\s*[öo]ntgen\b"
    r"\bap view\b",
    r"\bpa view\b",
    r"\blateral view\b",
    r"\bfrontal radiograph\b",
    r"\blateral radiograph\b",
    r"\bpanoramic radiograph\b",
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
    "xray_fluoroskopie_angiographie": 100,
    "us": 90,
    "ct_kombimodalitaet_spect+ct_pet+ct": 85,
    "mrt_hirn": 80,
    "mrt_body": 70,
    "ct": 60,
    "xray": 55,
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

    hits = {}

    for label, patterns in rules:
        for pattern in patterns:
            if re.search(pattern, t):
                if label not in hits:
                    hits[label] = {
                        "score": 0,
                        "patterns": [],
                    }

                hits[label]["score"] += 1
                hits[label]["patterns"].append(pattern)

    if not hits:
        return "unknown", "no_rule_match", "", {}

    # Zusatzlogik: starke Bildtyp-Klassen gegen zufällige Modalitätswörter schützen
    # Beispiel: intraoperative findings + CT im Text.
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

    # Priorität als Tie-Breaker
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
# CNN1 Wrapper u. CNN2 Wrapper
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
# ============================================================
# CNN1 Wrapper u. CNN2 Wrapper
# ============================================================
def run_cnn1(model, image, transform, device):
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

def run_cnn2(model, image, transform, device):
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
# Zero-Shot-artige BERT-Klassifikation
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

# ============================================================
# Hauptklassifikation
# ============================================================

def classify_dataset(
    dataset_root: Path,
    output_csv: Path,
    biomedbert_path: Optional[str],
    llamamistral_path: str,
    c1path: str,
    c2path: str,
    limit: Optional[int],
    batch_size: int,
    bert_threshold: float,
    per_class: int,
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

    print("\nInitialize models & preparing batch ...")
    batch_texts = []
    batch_indices = []

    print("\nInitialize LLaMaMistral ...")
    # LLaMaMistral initialisieren
    llm = LocalLLM(model_path=llamamistral_path)

    print("\nInitialize BERT ...")
    # BERT initialisieren
    bert_clf = None
    if biomedbert_path:
        bert_clf = BertCaptionClassifier(model_path=biomedbert_path)

    for idx in tqdm(range(len(ds)), desc="Prepare LLM Batch"):
        row = ds[idx]
        text, meta = extract_text_from_row(row)
        text = normalize_text(text)

        batch_texts.append(text)
        batch_indices.append((idx, meta, text, row))

    # Complete vLLM LLaMa Mistral
    llm_results = llm.batch_predict(batch_texts)

    records = []

    device = "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # CNN1 laden
    cnn1_model = SimpleCNN(num_classes=11)
    cnn1_model.load_state_dict(torch.load(c1path, map_location=device))
    cnn1_model.to(device).eval()

    # CNN2 laden
    cnn2_model = SimpleCNN(num_classes=4)
    cnn2_model.load_state_dict(torch.load(c2path, map_location=device))
    cnn2_model.to(device).eval()

    cnn1_transform = transform
    cnn2_transform = transform
    print("\nWende Regeln an, Cosine den Text, Falte Bilder, Erstelle Scores")
    for (idx, meta, text, row), llm_out in zip(batch_indices, llm_results):

        # RULES
        rule_label, _, _, hits = rule_based_classify_with_rules(
            text, RULES_LONG, LABEL_PRIORITY
        )

        rule_scores = {c: 0.0 for c in FINAL_CLASSES}
        if rule_label in FINAL_CLASSES:
            rule_scores[rule_label] = 1.0

        # BERT
        if bert_clf:
            bert_out = bert_clf.predict_batch([text])[0]
            bert_scores = {c: 0.0 for c in FINAL_CLASSES}
            bert_scores[bert_out["label"]] = bert_out["score"]
        else:
            bert_scores = {c: 0.0 for c in FINAL_CLASSES}

        # CNN1 / CNN2
        image = get_image_from_row(row)
        # Top 1
        # cnn1_raw = run_cnn1(cnn1_model, image, cnn1_transform, device)
        # cnn2_raw = run_cnn2(cnn2_model, image, cnn2_transform, device)
        #
        # cnn1_scores = cnn_to_scores(cnn1_raw, CNN1_MAPPING)
        # cnn2_scores = cnn_to_scores(cnn2_raw, CNN2_MAPPING)

        # Top 3
        # CNN1
        cnn1_top3, cnn1_full = predict_with_cnn(
            cnn1_model, image, cnn1_transform, device, CNN1_CLASS_NAMES)

        cnn1_scores = map_scores_to_final(cnn1_full, CNN1_MAPPING)

        # CNN2
        cnn2_top3, cnn2_full = predict_with_cnn(
            cnn2_model, image, cnn2_transform, device, CNN2_CLASS_NAMES)

        cnn2_scores = map_scores_to_final(cnn2_full, CNN2_MAPPING)

        # Fusion
        final_label, final_scores, contrib, explanation, uncertain = fuse_scores(
            rule_scores,
            bert_scores,
            cnn1_scores,
            cnn2_scores,
            llm_out)

        records.append({
            "row_id": idx,

            "Final Label:": final_label,

            "RULE:": rule_label,
            "BERT:": bert_out["label"] if bert_clf else "none",
            "CNN1top3": json.dumps(cnn1_top3),
            "CNN2top3": json.dumps(cnn2_top3),
            "LaMa:": llm_out[0],

            "caption": text,

            "BERT_score": bert_out["score"] if bert_clf else 0.0,
            "CNN1_scores": json.dumps(cnn1_scores),
            "CNN2_scores": json.dumps(cnn2_scores),
            "LaMa_conf": llm_out[1],
            "Final Scores": json.dumps(final_scores),
            "Begründung": json.dumps(explanation),
            "uncertain": uncertain,
            "pmc_id": meta.get("PMC_ID", "")
        })

    records = select_balanced(records, per_class=per_class)
    print(f"\nBalanced Subset: {len(records)} Samples")

# ============================================================
# Endergebnisse Head ausgeben
# ============================================================
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
        "--per_class",
        type=int,
        default=5000,
        help="Anzahl Samples pro finaler Klasse (hash-balanced)"
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
        help="Batchgröße für den BERT-Gang"
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
        limit=args.limit,
        batch_size=args.batch_size,
        bert_threshold=args.bert_threshold,
        per_class=args.per_class,
        inspect_only=args.inspect_only,
    )


if __name__ == "__main__":
    main()