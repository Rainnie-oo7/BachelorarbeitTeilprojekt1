# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
from pathlib import Path
import json
import threading

# =========================
# CONFIG
# =========================
CSV_PATH = "test_500.csv"
IMAGE_DIR = "exported_images"
THUMB_SIZE = (200, 200)

# =========================
# Async Image Loader
# =========================
class AsyncImageLoader:
    def __init__(self):
        self.cache = {}

    def load(self, path, callback):
        if path in self.cache:
            callback(self.cache[path])
            return

        def worker():
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail(THUMB_SIZE)
                self.cache[path] = img
                callback(img)
            except:
                callback(None)

        threading.Thread(target=worker, daemon=True).start()


# =========================
# Main App
# =========================
class DebugViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("PMC Debug Viewer")

        self.df = pd.read_csv(CSV_PATH)
        self.image_dir = Path(IMAGE_DIR)

        self.loader = AsyncImageLoader()

        self.filtered_df = self.df.copy()
        self.current_index = None

        self.build_ui()
        self.populate_list()

    # =========================
    # UI
    # =========================
    def build_ui(self):

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # LEFT PANEL
        left = tk.Frame(main_frame)
        left.pack(side="left", fill="y")

        self.listbox = tk.Listbox(left, width=50)
        self.listbox.pack(side="left", fill="y")

        scrollbar = tk.Scrollbar(left)
        scrollbar.pack(side="right", fill="y")

        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)

        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        # FILTERS
        filter_frame = tk.Frame(left)
        filter_frame.pack(fill="x")

        tk.Label(filter_frame, text="Filter label").pack()
        self.filter_label = tk.Entry(filter_frame)
        self.filter_label.pack(fill="x")

        tk.Button(filter_frame, text="Apply Filter", command=self.apply_filter).pack(fill="x")

        # RIGHT PANEL
        right = tk.Frame(main_frame)
        right.pack(side="right", fill="both", expand=True)

        self.image_label = tk.Label(right)
        self.image_label.pack()

        self.text = tk.Text(right, height=20)
        self.text.pack(fill="both", expand=True)

    # =========================
    # List Population
    # =========================
    def populate_list(self):
        self.listbox.delete(0, tk.END)

        for i, row in self.filtered_df.iterrows():
            label = row["final_label"]
            text = row["caption"][:50]
            self.listbox.insert(tk.END, f"{i} | {label} | {text}")

    # =========================
    # Filter
    # =========================
    def apply_filter(self):
        val = self.filter_label.get().lower()

        if not val:
            self.filtered_df = self.df.copy()
        else:
            self.filtered_df = self.df[
                self.df["final_label"].astype(str).str.lower().str.contains(val)
            ]

        self.populate_list()

    # =========================
    # Selection
    # =========================
    def on_select(self, event):
        if not self.listbox.curselection():
            return

        idx = self.listbox.curselection()[0]
        row = self.filtered_df.iloc[idx]

        self.show_detail(row)

    # =========================
    # Detail View
    # =========================
    def show_detail(self, row):

        pmc_id = row["pmc_id"]
        row_id = row["row_id"]

        img_path = self.image_dir / f"{pmc_id}_{row_id}.jpg"

        def update_image(img):
            if img is None:
                return
            self.tk_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.tk_img)

        self.loader.load(img_path, lambda img: self.root.after(0, update_image, img))

        # TEXT INFO
        self.text.delete("1.0", tk.END)

        info = []

        info.append(f"FINAL: {row['final_label']}")
        info.append(f"GT: {row.get('modality_gt', 'unknown')}")
        info.append(f"UNCERTAIN: {row.get('uncertain', False)}")

        info.append("\n--- MODELS ---")
        info.append(f"RULE: {row['RULE']}")
        info.append(f"BERT: {row['BERT']}")
        info.append(f"LLM: {row['LLM']}")

        info.append("\nCNN1:")
        info.append(row["CNN1top3"])

        info.append("\nCNN2:")
        info.append(row["CNN2top3"])

        info.append("\n--- SCORES ---")
        info.append(row["Final Scores"])

        info.append("\n--- CAPTION ---")
        info.append(row["caption"])

        # WHY PANEL
        info.append("\n--- WHY ---")
        try:
            dbg = json.loads(row["Begründung"])
            info.append(json.dumps(dbg, indent=2))
        except:
            info.append("No debug info")

        self.text.insert(tk.END, "\n".join(info))


# =========================
# RUN
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = DebugViewer(root)
    root.mainloop()