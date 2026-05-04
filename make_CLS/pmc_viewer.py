# -*- coding: utf-8 -*-

import json
import threading
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import pandas as pd


# ============================================================
# ASYNC IMAGE LOADER
# ============================================================

class ImageLoader(threading.Thread):
    def __init__(self, path, callback):
        super().__init__()
        self.path = path
        self.callback = callback

    def run(self):
        try:
            img = Image.open(self.path).convert("RGB")
            img = img.resize((512, 512))
        except:
            img = None

        self.callback(img)


# ============================================================
# MAIN VIEWER
# ============================================================

class AsyncViewer:

    def __init__(self, root, csv_path, image_dir, chunk_size=200):

        self.root = root
        self.root.title("🚀 PMC Async Viewer")

        self.csv_path = csv_path
        self.image_dir = Path(image_dir)

        self.chunk_size = chunk_size
        self.chunk_index = 0

        self.df = None
        self.filtered_df = None

        self.index = 0

        # ============================================================
        # UI
        # ============================================================

        top_frame = tk.Frame(root)
        top_frame.pack(fill=tk.X)

        # Filter
        tk.Label(top_frame, text="Filter Label:").pack(side=tk.LEFT)
        self.filter_entry = tk.Entry(top_frame, width=20)
        self.filter_entry.pack(side=tk.LEFT)

        tk.Button(top_frame, text="Apply Filter", command=self.apply_filter).pack(side=tk.LEFT)

        # Search
        tk.Label(top_frame, text="Search:").pack(side=tk.LEFT)
        self.search_entry = tk.Entry(top_frame, width=30)
        self.search_entry.pack(side=tk.LEFT)

        tk.Button(top_frame, text="Search", command=self.apply_search).pack(side=tk.LEFT)

        # Navigation
        nav_frame = tk.Frame(root)
        nav_frame.pack()

        tk.Button(nav_frame, text="<< Prev", command=self.prev).pack(side=tk.LEFT)
        tk.Button(nav_frame, text="Next >>", command=self.next).pack(side=tk.LEFT)

        # Image
        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Scrollable Text
        text_frame = tk.Frame(root)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.text = tk.Text(text_frame, wrap=tk.WORD)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame, command=self.text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.config(yscrollcommand=scrollbar.set)

        # Load first chunk
        self.load_next_chunk()

        self.show_sample()

    # ============================================================
    # CSV LOADING (LAZY)
    # ============================================================

    def load_next_chunk(self):

        print(f"Loading chunk {self.chunk_index}")

        try:
            df_chunk = pd.read_csv(
                self.csv_path,
                skiprows=self.chunk_index * self.chunk_size,
                nrows=self.chunk_size
            )
        except:
            return

        if self.df is None:
            self.df = df_chunk
        else:
            self.df = pd.concat([self.df, df_chunk], ignore_index=True)

        self.filtered_df = self.df

        self.chunk_index += 1

    # ============================================================
    # FILTER / SEARCH
    # ============================================================

    def apply_filter(self):
        label = self.filter_entry.get().strip()

        if not label:
            self.filtered_df = self.df
        else:
            self.filtered_df = self.df[self.df["final_label"].str.contains(label, na=False)]

        self.index = 0
        self.show_sample()

    def apply_search(self):
        term = self.search_entry.get().strip().lower()

        if not term:
            self.filtered_df = self.df
        else:
            self.filtered_df = self.df[
                self.df["caption"].str.lower().str.contains(term, na=False)
            ]

        self.index = 0
        self.show_sample()

    # ============================================================
    # IMAGE LOADING (ASYNC)
    # ============================================================

    def load_image_async(self, pmc_id, row_id):

        path = self.image_dir / f"{pmc_id}_{row_id}.jpg"

        def callback(img):
            if img is None:
                return

            img_tk = ImageTk.PhotoImage(img)

            def update():
                self.image_label.configure(image=img_tk)
                self.image_label.image = img_tk

            self.root.after(0, update)

        loader = ImageLoader(path, callback)
        loader.start()

    # ============================================================
    # DISPLAY
    # ============================================================

    def show_sample(self):

        if self.filtered_df is None or len(self.filtered_df) == 0:
            return

        if self.index >= len(self.filtered_df):
            self.load_next_chunk()

        row = self.filtered_df.iloc[self.index]

        pmc_id = row.get("pmc_id", "")
        row_id = row.get("row_id", "")

        # Async image load
        self.load_image_async(pmc_id, row_id)

        caption = row.get("caption", "")
        final_label = row.get("final_label", "")
        uncertain = row.get("uncertain", "")

        # DEBUG JSON
        debug_raw = row.get("debug", "{}")

        try:
            if isinstance(debug_raw, str) and len(debug_raw) < 50000:
                debug = json.loads(debug_raw)
            else:
                debug = {}
        except:
            debug = {}

        # ============================================================
        # TEXT OUTPUT
        # ============================================================

        output = []
        output.append(f"INDEX: {self.index}")
        output.append(f"PMC_ID: {pmc_id}")
        output.append(f"ROW_ID: {row_id}")

        output.append("\n--- CAPTION ---")
        output.append(str(caption)[:800])

        output.append("\n--- FINAL ---")
        output.append(f"Label: {final_label}")
        output.append(f"Uncertain: {uncertain}")

        output.append("\n--- DEBUG ---")
        output.append(json.dumps(debug, indent=2)[:2000])

        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, "\n".join(output))

    # ============================================================
    # NAVIGATION
    # ============================================================

    def next(self):
        self.index += 1
        if self.index >= len(self.filtered_df) - 5:
            self.load_next_chunk()
        self.show_sample()

    def prev(self):
        self.index = max(0, self.index - 1)
        self.show_sample()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    csv_path = "/home/b/PycharmProjects/ba1pmc/make_CLS/test_500.csv"
    img_path = "/home/b/PycharmProjects/ba1pmc/make_CLS/exported_images"

    root = tk.Tk()
    viewer = AsyncViewer(root, csv_path, img_path)
    root.mainloop()

"""
python make_CLS/pmc_viewer.py   --csv make_CLS/test_500.csv   --image_dir ./exported_images \
"""