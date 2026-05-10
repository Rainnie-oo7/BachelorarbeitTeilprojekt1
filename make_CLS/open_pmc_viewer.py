# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk
from pathlib import Path

import pandas as pd
import json
import threading
from queue import Queue

def build_image_index(image_dir):

    image_index = {}

    image_dir = Path(image_dir)

    for fp in image_dir.rglob("*"):

        if not fp.is_file():
            continue

        if fp.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        image_index[fp.name] = fp

    print(f"Indexed images: {len(image_index)}")

    return image_index

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "/home/b/PycharmProjects/ba1pmc/make_CLS/round_checkpoints/round_004.csv"
IMAGE_DIR = "/home/b/PycharmProjects/ba1pmc/make_CLS/round_checkpoints/images_after_rounds"
IMAGE_INDEX = build_image_index(IMAGE_DIR)

THUMB_SIZE = (900, 900)

BG = "#1e1e1e"
FG = "#dddddd"

FONT = ("Consolas", 10)


# ============================================================
# Async Image Loader
# ============================================================

class AsyncImageLoader:

    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()

    def preload(self, paths):

        for p in paths:
            if p in self.cache:
                continue

            threading.Thread(
                target=self._worker,
                args=(p,),
                daemon=True
            ).start()

    def _worker(self, path):

        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail(THUMB_SIZE)

            with self.lock:
                self.cache[path] = img

        except Exception:
            with self.lock:
                self.cache[path] = None

    def get(self, path):

        if path not in self.cache:
            self._worker(path)

        return self.cache.get(path)


# ============================================================
# MAIN VIEWER
# ============================================================
def build_image_path(image_dir, row):

    pmc_id = row.get("pmc_id", "unknown")
    row_id = row.get("row_id", "unknown")

    label = row.get("final_label")

    if pd.isna(label) or not label:
        label = row.get("rule_pred", "unknown")

    label = str(label)

    # Klassenordner
    path = (
        Path(image_dir)
        / label
        / f"{pmc_id}_{row_id}.jpg"
    )

    # Fallback alter Stil ohne Klassenordner
    if not path.exists():

        path = (
            Path(image_dir)
            / f"{pmc_id}_{row_id}.jpg"
        )

    return path

class PMCViewer:

    def __init__(self, root):

        self.root = root
        self.root.title("PMC DEBUG VIEWER")
        self.root.geometry("1800x1000")
        self.root.configure(bg=BG)

        print("Lade CSV ...")
        self.df = pd.read_csv(CSV_PATH, low_memory=False)

        self.filtered_df = self.df.copy()

        self.image_dir = Path(IMAGE_DIR)

        self.loader = AsyncImageLoader()

        self.current_idx = 0

        self.build_ui()

        self.populate_list()

        self.root.bind("<Down>", lambda e: self.next_item())
        self.root.bind("<Up>", lambda e: self.prev_item())

        self.root.bind("<Right>", lambda e: self.next_item())
        self.root.bind("<Left>", lambda e: self.prev_item())

        self.root.bind("f", lambda e: self.focus_search())

    # ============================================================
    # UI
    # ============================================================

    def build_ui(self):

        # =========================
        # MAIN PANED
        # =========================

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main, bg=BG)
        center = tk.Frame(main, bg=BG)
        right = tk.Frame(main, bg=BG)

        main.add(left, weight=1)
        main.add(center, weight=2)
        main.add(right, weight=3)

        # ============================================================
        # LEFT
        # ============================================================

        search_frame = tk.Frame(left, bg=BG)
        search_frame.pack(fill="x")

        tk.Label(
            search_frame,
            text="Filter",
            bg=BG,
            fg=FG
        ).pack()

        self.search_entry = tk.Entry(search_frame)
        self.search_entry.pack(fill="x")

        tk.Button(
            search_frame,
            text="Apply",
            command=self.apply_filter
        ).pack(fill="x")

        tk.Button(
            search_frame,
            text="Only uncertain",
            command=self.filter_uncertain
        ).pack(fill="x")

        self.listbox = tk.Listbox(
            left,
            bg="#252526",
            fg=FG,
            font=FONT
        )

        self.listbox.pack(fill="both", expand=True)

        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        # ============================================================
        # CENTER
        # ============================================================

        self.image_label = tk.Label(center, bg=BG)
        self.image_label.pack(fill="both", expand=True)

        # ============================================================
        # RIGHT
        # ============================================================

        self.info = ScrolledText(
            right,
            bg="#111111",
            fg=FG,
            insertbackground=FG,
            font=FONT
        )

        self.info.pack(fill="both", expand=True)

        # ============================================================
        # STATUSBAR
        # ============================================================

        self.status = tk.Label(
            self.root,
            text="READY",
            anchor="w",
            bg="#333333",
            fg="white"
        )

        self.status.pack(fill="x")

    # ============================================================
    # FILTER
    # ============================================================

    def apply_filter(self):

        val = self.search_entry.get().lower().strip()

        if not val:
            self.filtered_df = self.df.copy()

        else:

            mask = (
                self.df.astype(str)
                .apply(
                    lambda col: col.str.lower().str.contains(val, na=False)
                )
                .any(axis=1)
            )

            self.filtered_df = self.df[mask]

        self.populate_list()

    def filter_uncertain(self):

        if "uncertain" not in self.df.columns:
            return

        self.filtered_df = self.df[self.df["uncertain"] == True]

        self.populate_list()

    # ============================================================
    # LIST
    # ============================================================

    def populate_list(self):

        self.listbox.delete(0, tk.END)

        for idx, row in self.filtered_df.iterrows():

            txt = (
                f"{idx} | "
                f"{row.get('final_label','?')} | "
                f"{str(row.get('caption',''))[:60]}"
            )

            self.listbox.insert(tk.END, txt)

        self.status.config(
            text=f"{len(self.filtered_df)} Samples"
        )

    # ============================================================
    # NAVIGATION
    # ============================================================

    def next_item(self):

        if self.current_idx < len(self.filtered_df) - 1:

            self.current_idx += 1

            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(self.current_idx)

            self.show_current()

    def prev_item(self):

        if self.current_idx > 0:

            self.current_idx -= 1

            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(self.current_idx)

            self.show_current()

    # ============================================================
    # SELECTION
    # ============================================================

    def on_select(self, event):

        if not self.listbox.curselection():
            return

        self.current_idx = self.listbox.curselection()[0]

        self.show_current()

    # ============================================================
    # SHOW
    # ============================================================

    def show_current(self):

        row = self.filtered_df.iloc[self.current_idx]

        self.show_image(row)

        self.show_text(row)

        self.preload_neighbors()

    # ============================================================
    # IMAGE
    # ============================================================

    def show_image(self, row):

        path = build_image_path(
            self.image_dir,
            row
        )

        img = self.loader.get(path)

        if img is None:

            self.image_label.config(
                image="",
                text="NO IMAGE",
                fg="red"
            )

            return

        self.tk_img = ImageTk.PhotoImage(img)

        self.image_label.config(
            image=self.tk_img
        )

    # ============================================================
    # TEXT
    # ============================================================

    def show_text(self, row):

        self.info.delete("1.0", tk.END)

        lines = []

        lines.append("=" * 80)

        for col in row.index:

            val = row[col]

            try:

                if isinstance(val, str):

                    vv = val.strip()

                    if vv.startswith("{") or vv.startswith("["):

                        try:
                            parsed = json.loads(vv)

                            val = json.dumps(parsed, indent=2)

                        except:
                            pass

                lines.append(f"\n[{col}]")
                lines.append(str(val))

            except Exception as e:

                lines.append(f"{col}: ERROR {e}")

        self.info.insert(tk.END, "\n".join(lines))

    # ============================================================
    # PRELOAD
    # ============================================================

    def preload_neighbors(self):

        paths = []

        for delta in [-2, -1, 1, 2]:

            idx = self.current_idx + delta

            if idx < 0 or idx >= len(self.filtered_df):
                continue

            row = self.filtered_df.iloc[idx]

            path = build_image_path(
                self.image_dir,
                row
            )

            paths.append(path)

        self.loader.preload(paths)

    # ============================================================
    # SEARCH FOCUS
    # ============================================================

    def focus_search(self):

        self.search_entry.focus_set()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = PMCViewer(root)

    root.mainloop()