# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets
from PIL import Image, ImageTk
import io
import tkinter as tk
from tkinter import ttk, messagebox
import torch
import torch.nn as nn
import torchvision.transforms as transforms

CNN_MODEL_PATH = "/home/b/PycharmProjects/ba2roco/cnn/convu_try_3t.pth"

CLASS_NAMES_CNN = [
    'xray',
    'xray_fluoroskopie_angiographie',
    'us',
    'mrt_hirn_flair',
    'mrt_hirn_t1',
    'mrt_hirn_t2',
    'mrt_hirn_t1_c',
    'mrt_prostata_t1',
    'mrt_prostata_t2',
    'ct',
    'ct_kombimodalitaet_spect+ct_pet+ct'
]
#mrt_hirn zusamenfassen
#mrt_body zusammenfassen
#microscopy
#pathology
#surgery_real
#=10 Klassen
#chart_or_diagram
#endoscopy


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

def load_cnn_model(model_path, class_names):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleCNN(num_classes=len(class_names))

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    return model, transform, device

@torch.no_grad()
def predict_with_cnn(pil_img, model, transform, device, class_names):
    img = pil_img.convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    pred_label = class_names[pred_idx]
    pred_prob = float(probs[pred_idx].item())

    topk = torch.topk(probs, k=min(3, len(class_names)))

    top3 = []
    for idx, prob in zip(topk.indices.tolist(), topk.values.tolist()):
        top3.append((class_names[idx], float(prob)))

    return pred_label, pred_prob, top3

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


def find_arrow_files(dataset_root: Path):
    return sorted(dataset_root.glob("data-*.arrow"))


def load_arrow_shards(dataset_root: Path):
    arrow_files = find_arrow_files(dataset_root)

    if not arrow_files:
        raise FileNotFoundError(f"Keine data-*.arrow Dateien gefunden in: {dataset_root}")

    datasets_list = []
    for fp in arrow_files:
        print("Lade:", fp.name)
        datasets_list.append(Dataset.from_file(str(fp)))

    if len(datasets_list) == 1:
        return datasets_list[0]

    return concatenate_datasets(datasets_list)


def bytes_to_pil_image(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    if isinstance(x, bytes):
        return Image.open(io.BytesIO(x)).convert("RGB")

    # Hugging Face Image Feature kann manchmal dict sein
    if isinstance(x, dict):
        if "bytes" in x and x["bytes"] is not None:
            return Image.open(io.BytesIO(x["bytes"])).convert("RGB")
        if "path" in x and x["path"]:
            return Image.open(x["path"]).convert("RGB")

    raise ValueError(f"Kann jpg-Feld nicht als Bild lesen. Typ: {type(x)}")


class OpenPMCViewer:
    def __init__(self, root, ds, df, cnn_model=None, cnn_transform=None, cnn_device=None):
        self.root = root
        self.ds = ds
        self.df = df.reset_index(drop=True)
        self.index = 0
        self.photo = None

        self.cnn_model = cnn_model
        self.cnn_transform = cnn_transform
        self.cnn_device = cnn_device

        self.root.title("Open-PMC Bild-CSV Viewer")

        self.main = ttk.Frame(root, padding=10)
        self.main.pack(fill=tk.BOTH, expand=True)

        # Steuerung
        control = ttk.Frame(self.main)
        control.pack(fill=tk.X)

        self.prev_btn = ttk.Button(control, text="← Vorheriges", command=self.prev_image)
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = ttk.Button(control, text="Nächstes →", command=self.next_image)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        ttk.Label(control, text="Index / Nummer:").pack(side=tk.LEFT, padx=(20, 5))

        self.entry = ttk.Entry(control, width=12)
        self.entry.pack(side=tk.LEFT)

        self.go_btn = ttk.Button(control, text="Anzeigen", command=self.go_to_entry)
        self.go_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(control, text="")
        self.status_label.pack(side=tk.LEFT, padx=20)

        # Bild
        self.image_label = ttk.Label(self.main)
        self.image_label.pack(pady=10)

        # Infos
        info_frame = ttk.Frame(self.main)
        info_frame.pack(fill=tk.BOTH, expand=True)

        self.info_text = tk.Text(info_frame, height=18, wrap=tk.WORD)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(info_frame, command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)

        # Tastatur
        root.bind("<Left>", lambda event: self.prev_image())
        root.bind("<Right>", lambda event: self.next_image())
        root.bind("<Return>", lambda event: self.go_to_entry())

        self.show_current()

    def show_current(self):
        if self.index < 0:
            self.index = 0
        if self.index >= len(self.df):
            self.index = len(self.df) - 1

        csv_row = self.df.iloc[self.index]

        row_id = int(csv_row["row_id"])
        ds_row = self.ds[row_id]

        pil_img = bytes_to_pil_image(ds_row["jpg"])

        rules_bert_label = csv_row.get("pred_label", "")
        cnn_text = "CNN prediction: nicht geladen"

        if self.cnn_model is not None:
            try:
                cnn_label, cnn_prob, cnn_top3 = predict_with_cnn(
                    pil_img,
                    self.cnn_model,
                    self.cnn_transform,
                    self.cnn_device,
                    CLASS_NAMES_CNN
                )

                cnn_text = f"CNN prediction: {cnn_label} ({cnn_prob:.4f})\n"
                cnn_text += "CNN top3:\n"
                for label_name, prob in cnn_top3:
                    cnn_text += f"  - {label_name}: {prob:.4f}\n"

            except Exception as e:
                cnn_text = f"CNN-Fehler: {e}"
        # Bild skalieren
        max_w, max_h = 900, 650
        pil_img.thumbnail((max_w, max_h))

        self.photo = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=self.photo)

        meta = parse_jsonl_field(ds_row.get("jsonl"))

        image_name = csv_row.get("image", meta.get("image", ""))
        pmc_id = csv_row.get("pmc_id", meta.get("PMC_ID", ""))
        rules_short_label = csv_row.get("pred_label_short", "")
        rules_short_reason = csv_row.get("decision_reason_short", "")
        rules_short_pattern = csv_row.get("matched_pattern_short", "")

        rules_long_label = csv_row.get("pred_label_long", csv_row.get("pred_label", ""))
        rules_long_reason = csv_row.get("decision_reason_long", "")
        rules_long_pattern = csv_row.get("matched_pattern_long", "")
        modality_gt = csv_row.get("modality_gt", meta.get("modality", ""))

        caption = csv_row.get("caption", "")
        full_caption = csv_row.get("full_caption", meta.get("full_caption", ""))
        sub_caption = csv_row.get("sub_caption", meta.get("sub_caption", ""))
        intext_refs = csv_row.get("intext_refs", meta.get("intext_refs", ""))

        self.status_label.config(
            text=f"{self.index + 1}/{len(self.df)} | row_id={row_id} | {image_name}"
        )

        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(self.index + 1))

        self.info_text.delete("1.0", tk.END)

        text = f"""CSV-Index: {self.index + 1}
row_id: {row_id}
PMC_ID: {pmc_id}
Image: {image_name}

modality_gt: {modality_gt}

kurze Regeln: ||| lange Regeln:
Label: {rules_short_label} ||| {rules_long_label}
Reason: {rules_short_reason} ||| {rules_long_reason}
Pattern: {rules_short_pattern} ||| {rules_long_pattern}

{cnn_text}

Caption:
{caption}

Full Caption:
{full_caption}

Sub Caption:
{sub_caption}

Intext References:
{intext_refs}
"""
        self.info_text.insert(tk.END, text)

    def next_image(self):
        if self.index < len(self.df) - 1:
            self.index += 1
            self.show_current()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_current()

    def go_to_entry(self):
        value = self.entry.get().strip()

        if not value:
            return

        # Fall 1: einfache Zahl = CSV-Index 1 bis 500
        if value.isdigit():
            number = int(value)
            if 1 <= number <= len(self.df):
                self.index = number - 1
                self.show_current()
                return

            # Fall 2: Zahl aus Dateiname suchen, z. B. 000123
            matches = self.df[
                self.df["image"].astype(str).str.contains(value, regex=False, na=False)
            ]

            if len(matches) > 0:
                self.index = int(matches.index[0])
                self.show_current()
                return

        # Fall 3: kompletter oder teilweiser Dateiname
        matches = self.df[
            self.df["image"].astype(str).str.contains(value, regex=False, na=False)
        ]

        if len(matches) > 0:
            self.index = int(matches.index[0])
            self.show_current()
            return

        messagebox.showwarning("Nicht gefunden", f"Kein Eintrag gefunden für: {value}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", required=True, type=str)
    parser.add_argument("--csv_path", required=True, type=str)
    parser.add_argument("--limit", default=500, type=int)

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    csv_path = Path(args.csv_path)

    print("Lade CSV:", csv_path)
    df = pd.read_csv(csv_path)

    if "row_id" not in df.columns:
        raise ValueError("CSV braucht eine Spalte 'row_id'.")

    if "image" not in df.columns:
        print("Warnung: CSV hat keine Spalte 'image'. Suche per Dateiname ist dann eingeschränkt.")

    df = df.head(args.limit).copy()
    print(f"Verwende erste {len(df)} CSV-Einträge.")

    print("Lade Arrow-Dataset...")
    ds = load_arrow_shards(dataset_root)

    CLASS_NAMES_CNN = [
        'xray',
        'xray_fluoroskopie_angiographie',
        'us',
        'mrt_hirn_flair',
        'mrt_hirn_t1',
        'mrt_hirn_t2',
        'mrt_hirn_t1_c',
        'mrt_prostata_t1',
        'mrt_prostata_t2',
        'ct',
        'ct_kombimodalitaet_spect+ct_pet+ct'
    ]

    CNN_MODEL_PATH = "/home/b/PycharmProjects/ba2roco/cnn/convu_try_3t.pth"

    print("Lade CNN-Modell...")
    cnn_model, cnn_transform, cnn_device = load_cnn_model(
        CNN_MODEL_PATH,
        CLASS_NAMES_CNN
    )
    print(f"CNN geladen auf: {cnn_device}")
    print("Starte Viewer...")
    root = tk.Tk()
    root.geometry("1100x950")

    app = OpenPMCViewer(
        root,
        ds,
        df,
        cnn_model=cnn_model,
        cnn_transform=cnn_transform,
        cnn_device=cnn_device
    )
    root.mainloop()


if __name__ == "__main__":
    main()