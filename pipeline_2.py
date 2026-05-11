# -*- coding: utf-8 -*-

"""
open-pmc Klassifikation mit Rules + CNN
Erwartete Struktur:
dataset_root/
    data-00000-of-00059.arrow
    ...
    data-00058-of-00059.arrow
    dataset_info.json
    state.json
Beispiel:
python make_CLS/pipeline_2.py \
  --dataset_root /home/user/PycharmProjects/ba1pmc/PMC-41GB \
  --pre_output_csv /home/user/PycharmProjects/ba1pmc/make_CLS/preoutput_per10.csv \
  --output_csv /home/user/PycharmProjects/ba1pmc/make_CLS/output_per10.csv \
  --cnn_path /home/user/Dokumente/cnnCLS/cnncls.pth \
  --cnn3_path /home/user/Dokumente/cnn3/cnn3filter.pth \
  --cnn_strongfilter 0.94 \
  --cnn_mediumfilter 0.6 \
  --cnn_thresh 0.60 \
  --micro_round_failures 500
  > logrun.txt 2>&1
"""

from __future__ import annotations

import os.path as osp
from pathlib import Path
from PIL import Image
import re
import json
import argparse
import numpy as np
import hashlib
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
import io
from itertools import zip_longest

import easyocr

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

# Klassen u. Mapping
FINAL_CLASSES = [
    "xray",
    "xray_fluoroskopie_angiographie",
    "us",
    "mrt_hirn",
    "mrt_body",
    "ct",
    "ct_kombimodalitaet_spect+ct_pet+ct"]
CNN_CLASS_NAMES = [        #strikt Reihenfolge!!!
    'ct',
    'ct_kombimodalitaet_spect+ct_pet+ct',
    'us',
    "mrt_body",
    "mrt_hirn",
    "xray",
    "xray_fluoroskopie_angiographie"
    ]

CNN3_CLASS_NAMES = [
    "histologie",
    "haut",
    "chart",
    "endoskopie",
    "mikroskopie",
    "chirurgie",
]

# ALLGEMEINE TIERE
ANIMAL_LIST = [
    # Maus / Ratte / Nagetiere
    r"\bmouse\b",
    r"\bmice\b",
    r"\bmurine\b",
    r"\bmus musculus\b",

    r"\brat\b",
    r"\brats\b",
    r"\brattus\b",
    r"\brattus norvegicus\b",

    r"\bwistar\b",
    r"\bwistar rat\b",
    r"\bsprague[- ]dawley\b",
    r"\bsprague dawley\b",
    r"\blong[- ]evans\b",
    r"\bfischer 344\b",
    r"\bcd[- ]1\b",

    r"\bguinea pig\b",
    r"\bhamster\b",
    r"\bgerbil\b",
    r"\bvole\b",
    # Kaninchen
    r"\brabbit\b",
    r"\brabbits\b",
    r"\boryctolagus\b",
    # Hunde / Katzen
    r"\bdog\b",
    r"\bdogs\b",
    r"\bcanine\b",
    r"\bcanines\b",
    r"\bbeagle\b",

    r"\bcat\b",
    r"\bcats\b",
    r"\bfeline\b",

    # --------------------------------------------------------
    # Schweine
    # --------------------------------------------------------
    r"\bpig\b",
    r"\bpigs\b",
    r"\bporcine\b",
    r"\bswine\b",
    r"\bmini[- ]pig\b",
    r"\bgottingen\b",
    # Affen / Primaten
    r"\bmonkey\b",
    r"\bmonkeys\b",
    r"\bprimate\b",
    r"\bprimate model\b",
    r"\bmacaque\b",
    r"\brhesus\b",
    r"\bchimpanzee\b",
    r"\bbaboon\b",
    r"\bnonhuman primate\b",
    r"\bnhp\b",
    # Schafe / Ziegen / Rinder
    r"\bsheep\b",
    r"\bovine\b",
    r"\bcow\b",
    r"\bcalf\b",
    r"\bgoat\b",
    r"\bovine model\b",
    # Pferde
    r"\bhorse\b",
    r"\bequine\b",
    # Gefluegel / Vögel
    r"\bchicken\b",
    r"\bhen\b",
    r"\bavian\b",
    r"\bbird\b",
    r"\bpoultry\b",
    # Fisch / Amphibien
    r"\bzebrafish\b",
    r"\bfish\b",
    r"\bxenopus\b",
    r"\bfrog\b",
    # Insekten
    r"\bdrosophila\b",
    r"\bfruit fly\b",
    r"\binsect\b",
    r"\bmosquito\b",
    # Sonstiges
    r"\banimal model\b",
    r"\bexperimental animal\b",
    r"\bpreclinical\b",
    r"\bin vivo mouse\b",
    r"\bin vivo rat\b",
    r"\bmurine model\b",
]

ANIMAL_REGEX = re.compile(
    "|".join(ANIMAL_LIST),
    flags=re.IGNORECASE)

def contains_animal_terms(text: str):
    # Gibt: (True, match) oder (False, None)
    if not isinstance(text, str):
        return False, None
    match = ANIMAL_REGEX.search(text)
    if match:
        return True, match.group(0)
    return False, None
# MULTIPANEL FILTER
MULTIPANEL_PATTERNS = [
    # (A) (B) (C)
    r"\([A-Z]\)",
    r"\([a-z]\)",
    # (A)- ...
    r"\([A-Z]\)\s*[-:]",
    r"\([a-z]\)\s*[-:]",
    # panel A
    r"\bpanel\s+[A-Z]\b",
    r"\bpanel\s+[a-z]\b",
    # Figure 2A
    r"\bfig(?:ure)?\.?\s*\d+[A-Z]\b",
    # A:
    r"^[A-Z]\s*:",
    r"^[a-z]\s*:",
    # A.
    r"^[A-Z]\.",
    r"^[a-z]\.",
    # multiple panels mention
    r"\bpanels\b",
    r"\bsubfigure\b",
    r"\bsubfigures\b",
]

MULTIPANEL_REGEX = re.compile(
    "|".join(MULTIPANEL_PATTERNS),
    flags=re.IGNORECASE
)

def is_multipanel_caption(text: str):
    if not isinstance(text, str):
        return False, None
    match = MULTIPANEL_REGEX.search(text)
    if match:
        return True, match.group(0)
    return False, None

def split_multipanel_caption(text):

    if not isinstance(text, str):
        return {}, ""

    pattern = r"""
    (?:\(([A-Z])\))                           # (A)
    |
    (?:\(([a-z])\))                           # (a)
    |
    (?:\((\d+)\))                             # (1)
    |
    (?:\bPanel\s+([A-Z])\b)                   # Panel A
    |
    (?:\bPanel\s+(\d+)\b)                     # Panel 1
    |
    (?:\bfig(?:ure)?\.?\s*\d+\s*([A-Z])\b)       # Fig 2A
    |
    (?:^|\n|\.\s|\;\s)([A-Z])[:\.]            # A:
    |
    (?:^|\n|\.\s|\;\s)([a-z])[:\.]            # a:
    |
    (?:^|\n|\.\s|\;\s)([A-Z])\s+(?=[A-Z])     # A CT image...
    |
    (?:^|\n|\.\s|\;\s)([a-z])\s+(?=[A-Z])     # a CT image...
    """

    matches = list(
        re.finditer(
            pattern,
            text,
            flags=re.VERBOSE | re.IGNORECASE
        )
    )

    if len(matches) == 0:
        return {}, text

    sections = {}

    # =====================================================
    # PREFIX vor erstem Panel
    # =====================================================

    prefix_text = text[:matches[0].start()].strip()

    # =====================================================
    # Panels
    # =====================================================

    for i, match in enumerate(matches):

        panel = None

        for g in match.groups():

            if g is not None:
                panel = str(g).upper()
                break

        # Kein echtes Panel gefunden
        if panel is None:
            continue

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        panel_text = text[start:end].strip()

        sections[panel] = panel_text

    return sections, prefix_text

ocrreader = easyocr.Reader(
    ['en'],
    gpu=False)

def run_ocr_pil(image):
    if image is None:
        return "", []
    try:

        results = ocrreader.readtext(
            np.array(image),
            detail=1,
            paragraph=False
        )

        texts = []

        for r in results:

            if len(r) >= 2:
                texts.append(str(r[1]))

        text = " ".join(texts)

        return text, results

    except Exception as e:

        print("\nOCR Fehler")
        print(e)

        return "", []

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

def is_cnn_uncertain(cnn_conf, cnn_margin, conf_threshold=0.4, margin_threshold=0.03):
    relative_margin = cnn_margin / (cnn_conf + 1e-6)
    # Entscheidet, ob CNN unsicher ist.
    # Returns: bool: True = unsicher → LLM prüfen
    return (cnn_conf < conf_threshold or relative_margin < margin_threshold)

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
# Lange RULES MRT HIRN – Sublisten
MRT_HIRN_T1_LONG = [
    r"\bbrain\b.*\bt1\b",
    r"\bhead\b.*\bt1\b",
    r"\bcerebr\w*\b.*\bt1\b",
    r"\bcranial\b.*\bt1\b",
    r"\bt1[- ]weighted\b.*\bbrain\b",
    r"\bbrain\s+mri\b.*\bt1\b",
    r"\bbrain\s+magnetic\s+resonance\b",
    r"\bmra\b",
    r"\binternal\s+carotid\s+artery\b",
    r"\bica\b",
]
MRT_HIRN_T2_LONG = [
    r"\bbrain\b.*\bt2\b",
    r"\bhead\b.*\bt2\b",
    r"\bcerebr\w*\b.*\bt2\b",
    r"\bcranial\b.*\bt2\b",
    r"\bt2[- ]weighted\b.*\bbrain\b",
    r"\bbrain\s+mri\b.*\bt2\b",
    r"\bbrain\s+magnetic\s+resonance\b",
    r"\bmra\b",
    r"\binternal\s+carotid\s+artery\b",
    r"\bica\b",
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
# Interleave lange Regeln (Kartenmisch-artig) quasi da Reihenfolge jeweiliger Klasse gleich wichtig ist m1 = [a, b] und m2 = [c, d] wird m_hirn = [a, c, b, d]
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
    r"\bultrasonic\b",
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
#Filter Regeln
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
    r"\bmicroscop\w*\b",
    r"\bconfocal\b",
    r"\bfluorescen\w*\b.*\bmicroscop\w*\b",
    r"\belectron\s+microscop\w*\b",
    r"\bsem\b",  # scanning electron microscopy
    r"\btem\b",  # transmission electron microscopy
    r"\bimmunofluorescen\w*\b",
    r"\bstained\s+section\b",
    r"\bcell\s+culture\b",
    r"\bhigh[- ]magnification\b",
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
    r"\bhistolog\w*\b",
    r"\bh&e\b",
    r"\bhematoxylin\b",
    r"\beosin\b",
    r"\bimmunohistochem\w*\b",
    r"\bparaffin[- ]embedded\b",
    r"\btissue\s+section\b",
    r"\bbiopsy\b",
    r"\bpatholog\w*\b",
    r"\bstroma\b",
]
SURGERY_RULES_LONG = [
    r"\bintraoperative\b",
    r"\boperative findings\b",
    r"\bsurgical findings\b",
    r"\bsurgery\b",
    r"\bsurgical\b",
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
    r"\bskin\b",
    r"\bcutaneous\b",
    r"\bdermatolog\w*\b",
    r"\bclinical\s+photograph\b",
    r"\bexternal\s+appearance\b",
    r"\blesion\b.*\bskin\b",
    r"\bface\b",
    r"\boral\s+cavity\b",
    r"\bphotograph\b",
    r"\bpatient\s+photo\b"]

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
    r"\bendoscop\w*\b",
    r"\bcolonoscopy\b",
    r"\bgastroscop\w*\b",
    r"\blaparoscop\w*\b",
    r"\bbronchoscop\w*\b",
    r"\bcystoscop\w*\b",
    r"\barthroscop\w*\b",
    r"\bduodenoscop\w*\b",
    r"\besophagoscop\w*\b",
    r"\bflexible\s+scope\b",
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
FILTER_CLASSES = {
    "microscopy": MICROSCOPY_RULES_LONG,
    "pathology": PATHOLOGY_RULES_LONG,
    "surgery_real": SURGERY_RULES_LONG,
    "endoscopy": ENDOSCOPY_RULES_LONG,
    "chart_or_diagram": CHART_RULES_LONG,
    }

def rule_based_classify_with_rules(text: str, rules, label_priority=None):

    t = normalize_for_rules(text)

    if not t:
        return "unknown", "empty_text", "", {}

    if label_priority is None:
        label_priority = {}

    # Parallel matching
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

    # Regeln Filter
    matched_filter_classes = [cls for cls in hits if cls in FILTER_CLASSES]

    # Falls irgendeine starke Filterklasse matched
    if matched_filter_classes:
        best_filter = sorted(
            matched_filter_classes,
            key=lambda x: (hits[x]["score"], label_priority.get(x, 0)),
            reverse=True)[0]

        matched_patterns = hits[best_filter]["patterns"]

        return (
            best_filter,
            f"forced_filter:{best_filter}",
            " | ".join(matched_patterns[:5]),
            hits)


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

# ============================================================
# Rules müssen eindeutig sein CNN muss zustimmen
# ============================================================
AGREEMENT_CNN_CONF = 0.80
AGREEMENT_CNN_MARGIN = 0.20


def get_top_prediction(score_dict):

    if not score_dict:
        return None, 0.0, 0.0

    sorted_items = sorted(
        score_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top1_label, top1_score = sorted_items[0]

    if len(sorted_items) > 1:
        top2_score = sorted_items[1][1]
    else:
        top2_score = 0.0

    margin = top1_score - top2_score

    return top1_label, top1_score, margin

# ============================================================
# Fast pre-sampling (nur Rules)
# ============================================================
# ============================================================
# Checksum
# ============================================================
def debug_sampling(records, per_class, class_key="rule_pred"):
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
        print("\nWARNUNG: Early sampling noch nicht vollständig balanced.")

    print("\nSampling korrekt balanced!")

def early_balanced_sampling(ds, per_class, limit=None, early_presample=None):

    buckets = defaultdict(list)

    indices = list(range(len(ds)))
    random.shuffle(indices)

    max_n = min(len(indices), limit) if limit is not None else len(indices)

    for n_processed, idx in enumerate(tqdm(indices[:max_n], desc="Early sampling (Rules, Panelsearch, Rule & Animals Filtering)")):

        row = ds[idx]

        text, meta = extract_text_from_row(row)
        text = normalize_text(text)

        is_animal, animal_match = contains_animal_terms(text)

        if is_animal:
            # rejected_samples.append({
            #     "reason": "animal_detected",
            #     "matched_term": animal_match,
            #     "caption": text
            # })
            continue

        # MULTIPANEL FILTER
        # =========================================================
        # MULTIPANEL + OCR
        # =========================================================

        is_multi, multi_match = is_multipanel_caption(text)

        if is_multi:

            image = get_image_from_row(row)

            if image is not None:

                ocr_text, ocr_meta = run_ocr_pil(image)

                panel_sections, prefix_text = split_multipanel_caption(text)

                VALID_PANELS = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"}

                found_panels = []

                for token in ocr_text.split():

                    token = re.sub(
                        r"[^A-Za-z0-9]",
                        "",
                        token
                    ).upper()

                    if token in VALID_PANELS:
                        found_panels.append(token)

                # duplicates entfernen
                found_panels = list(dict.fromkeys(found_panels))

                selected_panel_texts = []

                for p in found_panels:

                    if p in panel_sections:
                        combined_text = (
                                prefix_text + " " + panel_sections[p]
                        ).strip()

                        selected_panel_texts.append(combined_text)

                # OCR erfolgreich
                if len(selected_panel_texts) > 0:

                    text = " ".join(selected_panel_texts)

                # OCR erfolglos
                else:

                    # Fallback:
                    # ganzen Text behalten
                    pass

        rule_pred, reason, matched, hits = rule_based_classify_with_rules(
            text, RULES_LONG, LABEL_PRIORITY)
        # print("rule_pred:", rule_pred)
        # print("text:", text[:250])
        #Regelfilter
        if rule_pred not in FINAL_CLASSES:
            continue

        if rule_pred == "unknown":
            print("Regellabel ist unknown TEXT:", text[:200])

        if len(buckets[rule_pred]) >= per_class:
            continue

        buckets[rule_pred].append({
            "row_id": idx,
            "pmc_id": meta.get("PMC_ID", ""),
            "caption": text,
            "row": row,
            "rule_pred": rule_pred,
            "rule_reason": reason,
            "rule_hits": hits,
            "modality_gt": meta.get("modality", "unknown"),
            "matched_patterns": matched.split(" | ") if matched else []
        })

        # Early Stop
        if all(len(buckets[c]) >= per_class for c in FINAL_CLASSES):
            print(f"\nAlle Klassen voll (per_class) bei idx={idx} → STOP")
            break

        if early_presample is not None and n_processed >= early_presample:
            print(f"\nEarly presample limit erreicht: {early_presample}")
            break
    # Flatten
    records = []
    for c in FINAL_CLASSES:
        records.extend(buckets[c])

    print(f"\nGesammelt: {len(records)} Samples")

    return records

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
        cnn_model,
        cnn3_model,
        cnn_transform,
        device
    ):
        self.cnn = cnn_model
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

def enrich_debug_fields(
    r,
    text,
    original_text,
    row,

    rule_pred,
    rule_scores,

    cnn_pred,
    cnn_top3,

    cnn3_pred,
    cnn3_conf,

    final_label,
    final_conf,

    decision_source
):

    r["caption"] = text
    r["full_caption"] = original_text

    r["row"] = row

    r["rule_pred"] = rule_pred
    r["rule_scores"] = rule_scores

    r["cnn_pred"] = cnn_pred

    r["cnn_top3"] = cnn_top3

    r["cnn3_pred"] = cnn3_pred
    r["cnn3_conf"] = cnn3_conf

    r["final_label"] = final_label
    r["final_conf"] = final_conf
    r["final_margin"] = final_conf

    r["decision_source"] = decision_source

    r["ocr_used"] = r.get("ocr_used", False)
    r["ocr_meta"] = r.get("ocr_meta", [])
    r["ocr_text"] = r.get("ocr_text", "")
    r["selected_panels"] = r.get("selected_panels", [])

    r["rule_reason"] = r.get("rule_reason", "")
    r["rule_hits"] = r.get("rule_hits", {})
    r["matched_patterns"] = r.get("matched_patterns", [])
    r["modality_gt"] = r.get("modality_gt", "unknown")

    return r

# ============================================================
# Verarbeitung (Ph. 3)
# ============================================================
def process_single_record(r, ctx: ModelContext, cnn_strongfilter,
cnn_mediumfilter, cnn_thresh):
    # Das ist mein Phase-3 Code für ein Sample (Rules + CNN)
    text = r.get("caption")
    original_text = r.get("caption")
    row = r.get("row")
    rule_hits = r.get("rule_hits")

    # Wichtig OCR nur im Multipanel-Fall (i.e. Fig. 2A etc)
    is_multi, multi_match = is_multipanel_caption(text)
    if is_multi:
        image = get_image_from_row(row)
        if image is not None:
            ocr_text, ocr_meta = run_ocr_pil(image)
            r["ocr_meta"] = ocr_meta
            panel_sections, prefix_text = split_multipanel_caption(text)
            # print("\n===== PANEL DEBUG =====")
            # print("TEXT:")
            # print(text)
            #
            # print("\nPREFIX:")
            # print(prefix_text)
            #
            # print("\nPANELS FOUND:")
            # print(panel_sections.keys())
            #
            # for k, v in panel_sections.items():
            #     print(f"\nPANEL {k}:")
            #     print(v[:300])
            # OCR erkennt dominante Panels
            VALID_PANELS = {"A", "B", "C", "D", "E", "F"}
            found_panels = []
            for token in ocr_text.split():
                token = re.sub(r"[^A-Za-z0-9]", "", token).upper()
                if token in VALID_PANELS:
                    found_panels.append(token)

            found_panels = list(dict.fromkeys(found_panels))

            # Falls OCR z.B. A erkennt:
            # benutze nur Text von Panel A |----|
            selected_panel_texts = []
            for p in found_panels:
                if p in panel_sections:
                    combined_text = (prefix_text + " " + panel_sections[p]).strip()

                    selected_panel_texts.append(combined_text)

            # Falls OCR erfolgreich:
            # Caption ersetzen
            if len(selected_panel_texts) > 0:
                text = " ".join(selected_panel_texts)
                r["ocr_used"] = True
                r["ocr_text"] = ocr_text
                r["selected_panels"] = found_panels
            else:
                r["ocr_used"] = False
        else:
            r["ocr_used"] = False

    # =========================
    # RULES
    # =========================
    rule_scores = {c: 0.0 for c in FINAL_CLASSES}
    if r.get("rule_pred") in FINAL_CLASSES:
        rule_scores[r["rule_pred"]] = 1.0
        rule_pred = top_label(rule_scores)
        # Ultraschall wird immer genommen
        if r.get("rule_pred") == "us":

            final_label = "us"
            final_conf = 1.0
            cnn_conf_final = 1.0

            decision_source = "rule_us_override"

            cnn_pred = "us"
            cnn_top3 = [("us", 1.0)]

        else:
            # =========================
            # CNN
            # =========================
            image = get_image_from_row(row)

            cnn_top3, cnn_full = predict_with_cnn(
                ctx.cnn,
                image,
                ctx.transform,
                ctx.device,
                CNN_CLASS_NAMES
            )

            cnn_scores = cnn_full

            cnn_pred, cnn_conf_final, cnn_margin = get_top_prediction(cnn_scores)

            final_label = cnn_pred
            final_conf = cnn_conf_final

            decision_source = "cnn"

            cnn_pred = final_label

    # =========================
    # CNN3 Filtering + CNN-Uncertainty Filtering
    # =========================
    cnn3_top3, cnn3_full = predict_with_cnn(
        ctx.cnn3, image, ctx.transform, ctx.device, CNN3_CLASS_NAMES)
    cnn3_pred = top_label(cnn3_top3)

    if cnn3_full:
        cnn3_conf = max(cnn3_full.values())
    else:
        cnn3_conf = 0.0

    expert_conf = cnn_conf_final

    if cnn3_conf >= cnn_strongfilter:
        if expert_conf <= 0.65:
            r["is_filtered"] = True
            r["filter_reason"] = "cnn3_strong"

            r = enrich_debug_fields(
                r=r,
                text=text,
                original_text=original_text,
                row=row,

                rule_pred=rule_pred,
                rule_scores=rule_scores,

                cnn_pred=cnn_pred,
                cnn_top3=cnn_top3,

                cnn3_pred=cnn3_pred,
                cnn3_conf=cnn3_conf,

                final_label=final_label,
                final_conf=final_conf,

                decision_source=decision_source
            )

            return r

    elif cnn3_conf >= cnn_mediumfilter:
        if expert_conf <= 0.465:
            r["is_filtered"] = True
            r["filter_reason"] = "cnn3_medium"

            r = enrich_debug_fields(
                r=r,
                text=text,
                original_text=original_text,
                row=row,

                rule_pred=rule_pred,
                rule_scores=rule_scores,

                cnn_pred=cnn_pred,
                cnn_top3=cnn_top3,

                cnn3_pred=cnn3_pred,
                cnn3_conf=cnn3_conf,

                final_label=final_label,
                final_conf=final_conf,

                decision_source=decision_source
            )

            return r

    CNN3_CONF_MIN = 0.96
    PRE_CONF_MAX = 0.45

    if (
            cnn3_conf > CNN3_CONF_MIN
            and cnn_conf_final < PRE_CONF_MAX
    ):
        r["is_filtered"] = True
        r["filter_reason"] = "noconsent"

        r = enrich_debug_fields(
            r=r,
            text=text,
            original_text=original_text,
            row=row,

            rule_pred=rule_pred,
            rule_scores=rule_scores,

            cnn_pred=cnn_pred,
            cnn_top3=cnn_top3,

            cnn3_pred=cnn3_pred,
            cnn3_conf=cnn3_conf,

            final_label=final_label,
            final_conf=final_conf,

            decision_source=decision_source
        )

        return r

    # ============================================================
    # AGREEMENT FILTER
    # ============================================================

    rule_pred = r.get("rule_pred", "unknown")

    cnn_pred_label = final_label

    agreement_pass = (
            rule_pred == cnn_pred_label
            and cnn_conf_final >= 0.55
    )

    # Konflikt -> sofort rauswerfen
    if not agreement_pass:
        r["is_filtered"] = True
        r["filter_reason"] = "rule_cnn_disagreement"

        r = enrich_debug_fields(
            r=r,
            text=text,
            original_text=original_text,
            row=row,

            rule_pred=rule_pred,
            rule_scores=rule_scores,

            cnn_pred=cnn_pred,
            cnn_top3=cnn_top3,

            cnn3_pred=cnn3_pred,
            cnn3_conf=cnn3_conf,

            final_label=final_label,
            final_conf=final_conf,

            decision_source=decision_source
        )

        return r

    r["is_filtered"] = False

    r = enrich_debug_fields(
        r=r,
        text=text,
        original_text=original_text,
        row=row,

        rule_pred=rule_pred,
        rule_scores=rule_scores,

        cnn_pred=cnn_pred,
        cnn_top3=cnn_top3,

        cnn3_pred=cnn3_pred,
        cnn3_conf=cnn3_conf,

        final_label=final_label,
        final_conf=final_conf,

        decision_source=decision_source
    )

    return r



def process_batch(
    presample,
    ctx,
    cnn_strongfilter,
    cnn_mediumfilter,
    cnn_thresh,
    disagreement_dir=None):

    processed = []
    filtered_counts = Counter()
    disagreement_records = []

    # Parallele Verarbeitung
    with ThreadPoolExecutor(max_workers=10) as ex:

        futures = [
            ex.submit(
                process_single_record,
                r,
                ctx,
                cnn_strongfilter,
                cnn_mediumfilter,
                cnn_thresh
            )
            for r in presample
        ]

        for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="CNN Processing"):
            try:
                out = fut.result()

            except Exception as e:
                filtered_counts["thread_exception"] += 1

                print("\nThread Error")
                print(e)

                continue
            # Wichtig ist hier nicht "if r is..." u. "if r.get...", da r fuer den iterativen fill beibehalten w.soll und nicht verfaelscht werden soll!
            if out is None:
                filtered_counts["unknown"] += 1
                continue

            if out.get("is_filtered"):

                reason = out.get("filter_reason", "unknown")

                filtered_counts[reason] += 1

                # ====================================================
                # DISAGREEMENT SAVE
                # ====================================================

                if reason == "rule_cnn_disagreement":
                    disagreement_records.append(out)

                continue

            processed.append(out)

    print(f"\nFiltered: {filtered_counts}")
    print(f"records nach batch: {len(processed)}")
    # ====================================================
    # SAVE DISAGREEMENTS
    # ====================================================

    if disagreement_dir is not None and len(disagreement_records) > 0:

        disagreement_dir = Path(disagreement_dir)

        disagreement_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # --------------------------------------------
        # CSV
        # --------------------------------------------

        csv_records = []

        for r in disagreement_records:

            tmp = dict(r)

            tmp.pop("row", None)

            csv_records.append(tmp)

        df_dis = pd.DataFrame(csv_records)

        csv_path = disagreement_dir / "rule_cnn_disagreements.csv"

        df_dis.to_csv(
            csv_path,
            index=False,
            encoding="utf-8"
        )

        print(f"\nDisagreement CSV gespeichert:")
        print(csv_path)

        # --------------------------------------------
        # Bilder
        # --------------------------------------------

        img_dir = disagreement_dir / "images"

        save_selected_images(
            disagreement_records,
            img_dir
        )

        print(f"Disagreement Bilder gespeichert:")
        print(img_dir)
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
# zufallsrecord()
# process_single_record()
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
        for r in existing)

    random.shuffle(pool)

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
        random.shuffle(buckets[c])
        final.extend(buckets[c][:per_class])

    return final

# ============================================================
# PRESAMPLE AUS GROSSEM DATASET
# ============================================================
def sample_new_chunk(ds, global_used, sample_size=100):

    indices = list(range(len(ds)))
    random.shuffle(indices)

    selected = []

    for idx in indices:

        row = ds[idx]

        try:
            text, meta = extract_text_from_row(row)

        except Exception:
            continue

        key = (meta.get("PMC_ID", ""), idx)

        if key in global_used:
            continue

        r = {
            "row": row,
            "caption": text,
            "pmc_id": meta.get("PMC_ID", ""),
            "row_id": idx,
            "rule_pred": "unknown"
        }

        selected.append(r)

        if len(selected) >= sample_size:
            break

    return selected

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
        max_rounds,
        cnn_strongfilter,
        cnn_mediumfilter,
        cnn_thresh,
        micro_round_failures,
        output_csv):
    # --------------------------------------------------------
    # Global
    # --------------------------------------------------------
    global_failed_counter = 0
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
            for c in FINAL_CLASSES)

        if done:
            print("\nDataset vollständig balanced")
            break

        # ====================================================
        # Fehlende Klassen
        # ====================================================
        missing_classes = {
            cls for cls in FINAL_CLASSES
            if counts[cls] < per_class}

        print("\nFehlende Klassen:")
        print(missing_classes)

        if not missing_classes:
            print("Alle Klassen gefüllt.")
            break
        # ====================================================
        # Micro Decay. Alle micro_round_failures erfolglosen Samples Thresholds leicht lockern. Hauptrounds erhalten trotzdem urspruenglichen Decay-Wert!
        # 2it/sek =/= micro round duration ~2.1 min. DENN Erfolge+Fehlversuche dabei
        # ====================================================
        micro_round = global_failed_counter // micro_round_failures

        effective_round = round_idx + micro_round

        print(f"\nMicro Round: {micro_round}")
        print(f"Effective Round: {effective_round}")

        # ====================================================
        # Adaptive Thresholds
        # ====================================================
        cnn_thresh = adaptive_threshold(
            start=cnn_thresh,
            min_value=0.55,
            refill_round=effective_round,
            decay=0.04
        )

        print(f"\nCNN Threshold: {cnn_thresh:.3f}")

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

        end_idx = min(len(ds), refill_cursor + sample_size)

        print(f"\nChunk: {start_idx} -> {end_idx}")

        candidate_chunk = ds.select(
            range(start_idx, end_idx))

        indices = list(range(len(candidate_chunk)))

        random.shuffle(indices)

        candidate_chunk = candidate_chunk.select(indices)

        refill_cursor = end_idx

        print("Chunkgröße:", len(candidate_chunk))
        # ====================================================
        # Raw Arrow -> Standard Records
        # ====================================================
        presample = []

        for local_idx, row in enumerate(
                tqdm(
                    candidate_chunk,
                    desc="Prepare Presample",
                    total=len(candidate_chunk),
                    leave=False
                )):
            key = row["__key__"]
            # --------------------------------------------
            # Arrow-Webds hat rohe Felder ['__key__', '__url__', 'jpg', 'jsonl']. Fuer global genutzt ja/nein Filter reicht blosses __key__
            # Bereits benutzt?
            # --------------------------------------------
            if key in global_used:
                continue
            # --------------------------------------------
            # Text extrahieren
            # --------------------------------------------
            text, meta = extract_text_from_row(row)

            text = normalize_text(text)

            rule_pred, reason, matched, hits = rule_based_classify_with_rules(
                text,
                RULES_LONG,
                LABEL_PRIORITY
            )
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
                "rule_pred": rule_pred,
                "rule_reason": reason,
                "rule_hits": hits,
                "matched_patterns": matched.split(" | ") if matched else [],
                "modality_gt": meta.get("modality", "unknown")}

            global_used.add(key)

            presample.append(record)

        print("Nach global_used:", len(presample))
        processed = process_batch(
                        presample,
                        ctx,
                        cnn_strongfilter=cnn_strongfilter,
                        cnn_mediumfilter=cnn_mediumfilter,
                        cnn_thresh=cnn_thresh,
                        disagreement_dir=output_csv.parent / "disagreements")

        if len(presample) == 0:
            print("\nLeerer Presample Chunk")
            continue

        # ====================================================
        # Existing Keys
        # ====================================================
        accepted_keys = set(
            (r["pmc_id"], r["row_id"])
            for r in accepted_records)
        new_accepts = []
        new_remaining = []

        # ====================================================
        # Process Results
        # ====================================================
        for r in processed:
            key = (r["pmc_id"], r["row_id"])

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
            counts = count_labels(accepted_records + new_accepts)

            # --------------------------------------------
            # Klasse bereits voll?
            # --------------------------------------------
            if counts[label] >= per_class:
                new_remaining.append(r)
                continue

            # ====================================================
            # Adaptive Qualitätsprüfung
            # ====================================================
            final_conf = r.get("final_conf", 0.0)
            final_margin = r.get("final_margin", 0.0)

            accept_threshold = adaptive_threshold(
                start=0.58,
                min_value=0.40,
                refill_round=effective_round,
                decay=0.02
            )

            accept = final_conf >= accept_threshold
            # --------------------------------------------
            # Accept / Remaining
            # --------------------------------------------
            if accept:
                new_accepts.append(r)
            else:
                new_remaining.append(r)
                # Fehlversuch zählen
                global_failed_counter += 1

        # ====================================================
        # Merge
        # ====================================================
        accepted_records.extend(new_accepts)

        remaining_pool.extend(new_remaining)

        before_counts = count_labels(accepted_records)

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
        after_counts = count_labels(accepted_records)

        refill_added =  sum(after_counts.values()) - sum(before_counts.values())

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
        # ====================================================
        # ROUND CHECKPOINT SAVE
        # ====================================================
        print("\nSpeichere Round-Checkpoint ...")
        round_dir = output_csv.parent / "round_checkpoints"

        round_dir.mkdir(parents=True, exist_ok=True)
        # CSV
        checkpoint_csv = (
                round_dir
                / f"round_{effective_round:03d}.csv"
        )

        save_records = []

        for rr in accepted_records:
            tmp = dict(rr)

            # Arrow-Zeilen entfernen
            tmp.pop("row", None)

            save_records.append(tmp)

        df_round = pd.DataFrame(save_records)

        df_round.to_csv(
            checkpoint_csv,
            index=False,
            encoding="utf-8")
        print(f"Checkpoint CSV: {checkpoint_csv}")
        # Bilder
        checkpoint_img_dir = (round_dir / "images_after_rounds")

        # überschreibt einfach dieselbe Datei. Neue Bilder werden einfach ergänzt.
        save_selected_images(
            accepted_records,
            checkpoint_img_dir)

        print(f"Checkpoint Bilder: {checkpoint_img_dir}")
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
            r["row_id"])
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
        row = r.get("row")
        if row is None:
            print("WARNUNG: row fehlt")
            continue

        img = get_image_from_row(row)

        if img is None:
            continue

        pmc_id = r.get("pmc_id", "unknown")
        row_id = r.get("row_id", "unknown")

        label = r.get("final_label")

        if label is None:
            label = r.get("rule_pred", "unknown")

        class_dir = output_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{pmc_id}_{row_id}.jpg"

        path = class_dir / fname

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
    pre_output_csv: Path,
    output_csv: Path,
    cnn_path: str,
    cnn3_path: str,
    limit: Optional[int],
    inspectlimit: int,
    per_class: int,
    early_presample: int,
    initial_presample: int,
    refill_presample: int,
    max_rounds: int,
    cnn_strongfilter: float,
    cnn_mediumfilter: float,
    cnn_thresh: float,
    micro_round_failures: int,

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
    records0 = early_balanced_sampling(ds, per_class, limit, early_presample)
    # Gewaehrleist, dass wirklich alle Klassen Eintraganzahl=per_class haben.
    debug_sampling(records0, per_class)
    print("\nSpeichere Early-Rules Dataset ...")

    early_rules_dir = output_csv.parent / "early_rules_dataset"

    early_rules_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Bilder speichern
    # ------------------------------------------------------------

    save_selected_images(
        records0,
        early_rules_dir / "images"
    )

    # ------------------------------------------------------------
    # CSV speichern
    # ------------------------------------------------------------

    csv_records = []

    for r in records0:
        tmp = dict(r)

        tmp.pop("row", None)

        csv_records.append(tmp)

    df_early = pd.DataFrame(csv_records)

    early_csv = early_rules_dir / "early_rules.csv"

    df_early.to_csv(
        early_csv,
        index=False,
        encoding="utf-8"
    )

    # CNN
    print("\nInitialize Convolutional Neural Network...")
    cnn_model = SimpleCNN(num_classes=7)
    cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn_model.to(device).eval()

    cnn3_model = ThirdCNN(num_classes=6)
    cnn3_model.load_state_dict(torch.load(cnn3_path, map_location=device))
    cnn3_model.to(device).eval()

    cnn_transform = transform

    ctx = ModelContext(
        cnn_model,
        cnn3_model,
        cnn_transform,
        device)

    # ============================================================
    # Phase 3: Records sammeln
    # ============================================================
    print("\nExtrahiere Records + RULES + CNN ...")
    print("\nPhase 3: CNN auf Subset")
    records = process_batch(records0, ctx, cnn_strongfilter, cnn_mediumfilter, cnn_thresh, disagreement_dir=output_csv.parent / "disagreements")
    print("records nach batch:", len(records))

    # ============================================================
    # Phase 4: Final fusion

    existing_records = records
    # ============================================================
    # Volle iterative Build Pipeline:
    # Phase 5: Post-Balancing
    # ============================================================

    records = build_balanced_dataset(ds,
        ctx,
        existing_records,
        initial_presample,
        refill_presample,
        per_class,
        max_rounds,
        cnn_strongfilter,
        cnn_mediumfilter,
        cnn_thresh,
        micro_round_failures,
        output_csv)

    # ============================================================
    # Phase: Endergebnisse Head ausgeben
    # ============================================================
    final_output_dir = output_csv.parent / "final_presample_images"

    print("\nSpeichere Early-Presample Bilder ...")

    save_selected_images(
        records0,
        final_output_dir
    )

    # existing_records haben 'row', entfernen>nicht explodiert
    for r in records:
        r.pop("row", None)
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
        "--pre_output_csv",
        type=str,
        required=True,
        help="Pfad zur Presample-Ausgabe-CSV"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Pfad zur Final-Ausgabe-CSV"
    )
    parser.add_argument(
        "--cnn_path",
        type=str,
        required=True,
        help="Lokaler Pfad zum custom CNN"
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
        default=10,
        # default=5000,
        help="Anzahl Samples pro finaler Klasse (hash-balanced)"
    )
    parser.add_argument(
        "--early_presample",
        type=int,
        default=200,
        help="Nur diese Anzahl Samples in ersten Run (=Phase 2) klassifizieren"
    )
    parser.add_argument(
        "--initial_presample",
        type=int,
        # default=50000,
        # default=6000,
        default=500,
        help="Menge des ersten großes Presample vor Refill"
    )
    parser.add_argument(
        "--refill_presample",
        type=int,
        # default=200,
        default=300,
        # default=25000,
        help="Anzahl spätere Nachlade-Chunks Postsample *im* Refill"
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=68,
        help="Anzahl Sicherheitslimit im Refill (=Runden Runs Durchlaeufe)"
    )
    parser.add_argument(
        "--cnn_thresh",
        type=float,
        default=0.82,
        help="CNN confidence threshold für sichere Entscheidungen."
    )
    #höherer Threshold = strenger / konservativer niedrigerer Threshold = toleranter / mehr akzeptierte Fälle
    parser.add_argument(
        "--cnn_mediumfilter",
        type=float,
        default=0.5,
        help="CNN confidence threshold für sichere Entscheidungen."
    )
    parser.add_argument(
        "--cnn_strongfilter",
        type=float,
        default=0.75,
        help="CNN confidence threshold für sichere Entscheidungen."
    )
    parser.add_argument(
        "--micro_round_failures",
        type=int,
        # default=30,
        default=400,
        help="Anzahl erfolgloser Samples bis Micro-Round erhöht wird"
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
        "--inspect_only",
        action="store_true",
        help="Nur Struktur/Beispiele ausgeben, keine Klassifikation"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    pre_output_csv = Path(args.pre_output_csv)
    output_csv = Path(args.output_csv)

    classify_dataset(
        dataset_root=dataset_root,
        pre_output_csv=pre_output_csv,
        output_csv=output_csv,
        cnn_path=args.cnn_path,
        cnn3_path=args.cnn3_path,
        inspectlimit=args.inspectlimit,
        limit=args.limit,
        per_class=args.per_class,
        early_presample=args.early_presample,
        initial_presample=args.initial_presample,
        refill_presample=args.refill_presample,
        max_rounds=args.max_rounds,
        cnn_thresh=args.cnn_thresh,
        cnn_strongfilter=args.cnn_strongfilter,
        cnn_mediumfilter=args.cnn_mediumfilter,
        micro_round_failures=args.micro_round_failures,

        inspect_only=args.inspect_only,
    )

if __name__ == "__main__":
    main()