#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

# Prefer the original Microsoft package layout when available.
try:
    from pmc15_pipeline import data
    from pmc15_pipeline.utils import fs_utils
except Exception:
    import data  # type: ignore
    import fs_utils  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore


MODALITIES = ["ct", "mri", "xray", "ultrasound", "microscopy"]
FILELIST_HEADER = ["path", "title", "pmcid", "pmid", "code"]

PROTOTYPES: Dict[str, List[str]] = {
    "ct": [
        "computed tomography ct scan cross sectional imaging axial coronal sagittal contrast enhancement hounsfield density attenuation",
        "ct imaging demonstrates axial coronal or sagittal slices with contrast enhancement and tissue density findings",
    ],
    "mri": [
        "magnetic resonance imaging mri t1 t2 flair diffusion weighted dwi gadolinium signal intensity weighted sequence",
        "mri shows abnormal signal intensity on t1 t2 flair or diffusion weighted images with contrast enhancement",
    ],
    "xray": [
        "x ray radiograph radiography plain film chest x ray ap pa lateral upright portable supine opacity lucency",
        "radiographic examination reveals findings on chest radiograph or plain x ray film in ap or lateral view",
    ],
    "ultrasound": [
        "ultrasound sonography sonographic doppler transducer acoustic anechoic hyperechoic hypoechoic echogenic frequency echo",
        "ultrasound image demonstrates doppler flow and sonographic findings with anechoic or hyperechoic structures",
    ],
    "microscopy": [
        "microscopy histopathology histology microscopic micrograph biopsy tissue staining cellular nuclear cytoplasm morphology",
        "histological examination reveals stained tissue sections with cellular and nuclear morphology on microscopic image",
    ],
}

HEURISTIC_TERMS: Dict[str, Dict[str, int]] = {
    "ct": {
        "computed tomography": 12,
        "ct scan": 11,
        "ct": 8,
        "axial": 4,
        "coronal": 4,
        "sagittal": 4,
        "contrast": 2,
        "enhancement": 3,
        "density": 2,
        "hounsfield": 10,
        "attenuation": 4,
        "tomography": 4,
    },
    "mri": {
        "magnetic resonance imaging": 12,
        "mri": 10,
        "t1": 4,
        "t2": 4,
        "flair": 8,
        "dwi": 8,
        "adc": 8,
        "gadolinium": 8,
        "signal": 2,
        "intensity": 2,
        "weighted": 2,
        "diffusion weighted": 8,
    },
    "xray": {
        "x ray": 10,
        "x-ray": 10,
        "radiograph": 8,
        "radiographic": 8,
        "plain film": 8,
        "ap": 2,
        "pa": 2,
        "lateral": 3,
        "portable": 2,
        "upright": 2,
        "supine": 2,
        "opacity": 3,
        "lucency": 3,
    },
    "ultrasound": {
        "ultrasound": 10,
        "sonography": 9,
        "sonographic": 9,
        "sonogram": 8,
        "doppler": 7,
        "transducer": 4,
        "acoustic": 2,
        "anechoic": 7,
        "hyperechoic": 7,
        "hypoechoic": 7,
        "echogenic": 4,
        "echo": 2,
    },
    "microscopy": {
        "microscopy": 10,
        "microscopic": 10,
        "micrograph": 8,
        "histology": 10,
        "histological": 10,
        "histopathology": 10,
        "cellular": 4,
        "nuclear": 4,
        "cytoplasm": 4,
        "tissue": 2,
        "staining": 4,
        "morphology": 3,
        "biopsy": 3,
        "hematoxylin": 6,
        "eosin": 6,
        "immunohistochemistry": 7,
    },
}


@dataclass(slots=True)
class PubMedEntry:
    line_idx: int
    path: str
    title: str
    pmcid: str
    pmid: str
    code: str


@dataclass(slots=True)
class PairRecord:
    pair_id: str
    pmid: str
    pmc: str
    fig_id: str
    fig_label: str
    fig_caption: str
    graphic_ref: str
    location: str
    modality: str
    confidence: float
    article_line_idx: Optional[int]
    article_path: Optional[str]
    article_title: Optional[str]
    article_code: Optional[str]
    article_order: Optional[int]


class CaptionModalityClassifier:
    """Hybrid classifier that always maps a caption to one of the 5 target modalities."""

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.semantic_backend = "heuristic-only"
        self.label_names = MODALITIES[:]

        self.prototype_texts: List[str] = []
        self.prototype_owner: List[str] = []
        for label in self.label_names:
            for text in PROTOTYPES[label]:
                self.prototype_texts.append(text)
                self.prototype_owner.append(label)

        self.st_model = None
        self.prototype_embeddings = None
        self.tfidf = None
        self.prototype_matrix = None

        if SentenceTransformer is not None and np is not None:
            try:
                self.st_model = SentenceTransformer(embedding_model_name)
                self.prototype_embeddings = self.st_model.encode(
                    self.prototype_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                self.semantic_backend = f"sentence-transformers:{embedding_model_name}"
            except Exception:
                self.st_model = None
                self.prototype_embeddings = None

        if self.st_model is None and TfidfVectorizer is not None and cosine_similarity is not None:
            self.tfidf = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
            self.prototype_matrix = self.tfidf.fit_transform(self.prototype_texts)
            self.semantic_backend = "tfidf-cosine"

    @staticmethod
    def normalize(text: str) -> str:
        text = (text or "").lower()
        text = text.replace("/", " ")
        text = re.sub(r"[_\-]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return f" {text} "

    def heuristic_scores(self, text: str) -> Dict[str, float]:
        scores = {label: 0.0 for label in self.label_names}
        for label, terms in HEURISTIC_TERMS.items():
            for term, weight in terms.items():
                if re.search(rf"\b{re.escape(term)}\b", text):
                    scores[label] += float(weight)
        return scores

    def semantic_scores(self, text: str) -> Dict[str, float]:
        scores = {label: 0.0 for label in self.label_names}

        if self.st_model is not None and self.prototype_embeddings is not None and np is not None:
            emb = self.st_model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
            sims = np.dot(self.prototype_embeddings, emb)
            for sim, owner in zip(sims, self.prototype_owner):
                scores[owner] = max(scores[owner], float(sim))
            return scores

        if self.tfidf is not None and self.prototype_matrix is not None and cosine_similarity is not None:
            text_vec = self.tfidf.transform([text])
            sims = cosine_similarity(text_vec, self.prototype_matrix)[0]
            for sim, owner in zip(sims, self.prototype_owner):
                scores[owner] = max(scores[owner], float(sim))
            return scores

        return scores

    def classify(self, caption: str) -> Tuple[str, float, Dict[str, float]]:
        text = self.normalize(caption)
        h_scores = self.heuristic_scores(text)
        s_scores = self.semantic_scores(text)

        combined: Dict[str, float] = {}
        for label in self.label_names:
            combined[label] = h_scores[label] + (10.0 * s_scores[label])

        best_label, best_score = max(combined.items(), key=lambda x: x[1])
        ordered = sorted(combined.values(), reverse=True)
        second = ordered[1] if len(ordered) > 1 else 0.0
        conf = 0.5 if best_score <= 0 else max(0.0, min(1.0, (best_score - second + 1.0) / (best_score + 1.0)))
        return best_label, conf, combined


def get_repo_root_fallback() -> Path:
    try:
        return fs_utils.get_repo_root_path()
    except Exception:
        return Path.cwd()


def parse_args() -> argparse.Namespace:
    repo_root = get_repo_root_fallback()
    default_data_dir = repo_root / "_results" / "data"

    parser = argparse.ArgumentParser(
        description=(
            "Randomized PMC caption pipeline: download article archives using the Microsoft "
            "functions, parse captions with the Microsoft parser, classify figure captions into "
            "ct/mri/xray/ultrasound/microscopy, and build an exact 20k image-text subset."
        )
    )
    parser.add_argument("--file-list-path", type=Path, default=default_data_dir / "pubmed_open_access_file_list.txt")
    parser.add_argument("--output-root", type=Path, default=default_data_dir)
    parser.add_argument("--target-pairs", type=int, default=20_000)
    parser.add_argument("--batch-articles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-articles", type=int, default=None, help="Optional hard stop for debugging.")
    parser.add_argument("--clean-start", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-batch-artifacts", action="store_true")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_pubmed_file_list(file_list_path: Path) -> List[PubMedEntry]:
    entries: List[PubMedEntry] = []
    with file_list_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty file list: {file_list_path}")
        for line_idx, row in enumerate(reader, start=1):
            if len(row) < 5:
                continue
            path, title, pmcid, pmid, code = row[:5]
            entries.append(PubMedEntry(line_idx=line_idx, path=path, title=title, pmcid=pmcid, pmid=pmid, code=code))
    return entries


def write_file_list(entries: Sequence[PubMedEntry], output_path: Path) -> None:
    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(FILELIST_HEADER)
        for e in entries:
            writer.writerow([e.path, e.title, e.pmcid, e.pmid, e.code])


def iter_batches(items: Sequence[int], batch_size: int) -> Iterable[List[int]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start:start + batch_size])


def reset_outputs(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_seen_pair_ids(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pair_id = obj.get("pair_id")
            if pair_id:
                seen.add(pair_id)
    return seen


def read_processed_article_ids(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return set(manifest.get("processed_pmcs", []))


def save_manifest(path: Path, manifest: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def classify_parsed_articles(
    parsed_jsonl: Path,
    classifier: CaptionModalityClassifier,
    meta_by_pmc: Dict[str, PubMedEntry],
    article_order_lookup: Dict[str, int],
) -> List[PairRecord]:
    records: List[PairRecord] = []
    with parsed_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            article = json.loads(line)
            pmc_raw = str(article.get("pmc", "")).strip()
            pmc = pmc_raw if pmc_raw.startswith("PMC") else f"PMC{pmc_raw}" if pmc_raw else ""
            meta = meta_by_pmc.get(pmc)
            article_order = article_order_lookup.get(pmc)
            for fig in article.get("figures", []):
                caption = str(fig.get("fig_caption", "") or "").strip()
                graphic_ref = str(fig.get("graphic_ref", "") or "").strip()
                pair_id = str(fig.get("pair_id", "") or "").strip()
                if not caption or not graphic_ref or not pair_id:
                    continue
                modality, confidence, _ = classifier.classify(caption)
                records.append(
                    PairRecord(
                        pair_id=pair_id,
                        pmid=str(article.get("pmid", "") or ""),
                        pmc=pmc,
                        fig_id=str(fig.get("fig_id", "") or ""),
                        fig_label=str(fig.get("fig_label", "") or ""),
                        fig_caption=caption,
                        graphic_ref=graphic_ref,
                        location=str(article.get("location", "") or ""),
                        modality=modality,
                        confidence=float(confidence),
                        article_line_idx=meta.line_idx if meta else None,
                        article_path=meta.path if meta else None,
                        article_title=meta.title if meta else None,
                        article_code=meta.code if meta else None,
                        article_order=article_order,
                    )
                )
    return records


def write_stats(path: Path, rows: Sequence[dict], target_pairs: int, processed_articles: int, classifier_backend: str) -> None:
    counts = Counter(row["modality"] for row in rows)
    total = len(rows)
    stats = {
        "target_pairs": target_pairs,
        "actual_pairs": total,
        "processed_articles": processed_articles,
        "classifier_backend": classifier_backend,
        "modalities": {
            label: {
                "count": counts.get(label, 0),
                "percent_of_pairs": round((100.0 * counts.get(label, 0) / total), 6) if total else 0.0,
            }
            for label in MODALITIES
        },
    }
    ensure_parent(path)
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()

    output_root = args.output_root
    compressed_dir = output_root / "pubmed_open_access_files_compressed"
    decompressed_dir = output_root / "pubmed_open_access_files"
    parsed_jsonl = output_root / "pubmed_parsed_data.jsonl"
    modality_jsonl = output_root / "pubmed_parsed_data_modalities.jsonl"
    final_jsonl = output_root / f"pubmed_parsed_data_modalities_{args.target_pairs}.jsonl"
    stats_json = output_root / "pubmed_parsed_data_modality_stats.json"
    manifest_json = output_root / f"pubmed_selection_manifest_{args.target_pairs}.json"
    batch_root = output_root / "_batch_work"

    if args.clean_start and args.resume:
        raise ValueError("Use either --clean-start or --resume, not both.")

    if args.clean_start:
        reset_outputs([compressed_dir, decompressed_dir, parsed_jsonl, modality_jsonl, final_jsonl, stats_json, manifest_json, batch_root])

    entries = read_pubmed_file_list(args.file_list_path)
    if not entries:
        raise ValueError(f"No entries found in file list: {args.file_list_path}")

    rng = random.Random(args.seed)
    shuffled_indices = list(range(len(entries)))
    rng.shuffle(shuffled_indices)
    if args.max_articles is not None:
        shuffled_indices = shuffled_indices[: args.max_articles]

    classifier = CaptionModalityClassifier(embedding_model_name=args.embedding_model)
    meta_by_pmc = {entry.pmcid: entry for entry in entries}

    processed_pmcs = read_processed_article_ids(manifest_json) if args.resume else set()
    seen_pair_ids = load_seen_pair_ids(modality_jsonl) if args.resume else set()
    collected_rows = load_jsonl(modality_jsonl) if args.resume else []
    article_order_lookup: Dict[str, int] = {row["pmc"]: row.get("article_order") for row in collected_rows if row.get("pmc")}

    print(f"Loaded {len(entries):,} article entries from {args.file_list_path}")
    print(f"Sampling mode: randomized across the full list with seed={args.seed}")
    print(f"Classifier backend: {classifier.semantic_backend}")
    print(f"Already collected pairs: {len(collected_rows):,}")

    batch_counter = 0
    article_counter = len(processed_pmcs)

    for batch_indices in iter_batches(shuffled_indices, args.batch_articles):
        batch_entries = [entries[i] for i in batch_indices if entries[i].pmcid not in processed_pmcs]
        if not batch_entries:
            continue

        batch_counter += 1
        batch_dir = batch_root / f"batch_{batch_counter:05d}"
        batch_compressed = batch_dir / "pubmed_open_access_files_compressed"
        batch_decompressed = batch_dir / "pubmed_open_access_files"
        batch_file_list = batch_dir / "pubmed_open_access_file_list.txt"
        batch_parsed_jsonl = batch_dir / "pubmed_parsed_data.jsonl"

        write_file_list(batch_entries, batch_file_list)
        print(f"\n=== Batch {batch_counter} | articles: {len(batch_entries)} | collected pairs: {len(collected_rows):,}/{args.target_pairs:,} ===")

        data.download_pubmed_files_from_list(
            file_list_path=batch_file_list,
            output_folder_path=batch_compressed,
            subset_size=None,
        )
        data.decompress_pubmed_files(
            input_folder_path=batch_compressed,
            output_folder_path=batch_decompressed,
        )
        data.generate_pmc15_pipeline_outputs(
            decompressed_folder=batch_decompressed,
            output_file_path=batch_parsed_jsonl,
        )

        parsed_articles = load_jsonl(batch_parsed_jsonl)
        append_jsonl(parsed_jsonl, parsed_articles)

        for entry in batch_entries:
            article_order_lookup[entry.pmcid] = article_order_lookup.get(entry.pmcid, article_counter)
            article_counter += 1

        pair_records = classify_parsed_articles(
            parsed_jsonl=batch_parsed_jsonl,
            classifier=classifier,
            meta_by_pmc=meta_by_pmc,
            article_order_lookup=article_order_lookup,
        )

        new_rows: List[dict] = []
        for rec in pair_records:
            if rec.pair_id in seen_pair_ids:
                continue
            seen_pair_ids.add(rec.pair_id)
            row = asdict(rec)
            new_rows.append(row)
            collected_rows.append(row)

        append_jsonl(modality_jsonl, new_rows)

        for entry in batch_entries:
            processed_pmcs.add(entry.pmcid)

        manifest = {
            "seed": args.seed,
            "target_pairs": args.target_pairs,
            "batch_articles": args.batch_articles,
            "classifier_backend": classifier.semantic_backend,
            "embedding_model": args.embedding_model,
            "file_list_path": str(args.file_list_path),
            "output_root": str(output_root),
            "processed_articles_count": len(processed_pmcs),
            "processed_pmcs": sorted(processed_pmcs),
            "collected_pairs": len(collected_rows),
            "batches_completed": batch_counter,
        }
        save_manifest(manifest_json, manifest)
        write_stats(stats_json, collected_rows, args.target_pairs, len(processed_pmcs), classifier.semantic_backend)

        if not args.keep_batch_artifacts:
            shutil.rmtree(batch_dir, ignore_errors=True)

        if len(collected_rows) >= args.target_pairs:
            break

    if len(collected_rows) < args.target_pairs:
        raise RuntimeError(
            f"Only collected {len(collected_rows):,} pairs after processing {len(processed_pmcs):,} articles. "
            f"Increase --max-articles or rerun without a hard stop."
        )

    final_rng = random.Random(args.seed + 1)
    final_rows = final_rng.sample(collected_rows, args.target_pairs)
    final_rows.sort(key=lambda r: (str(r.get("pmc", "")), str(r.get("pair_id", ""))))

    if final_jsonl.exists():
        final_jsonl.unlink()
    append_jsonl(final_jsonl, final_rows)
    write_stats(stats_json, final_rows, args.target_pairs, len(processed_pmcs), classifier.semantic_backend)

    final_manifest = {
        "seed": args.seed,
        "target_pairs": args.target_pairs,
        "batch_articles": args.batch_articles,
        "classifier_backend": classifier.semantic_backend,
        "embedding_model": args.embedding_model,
        "sampling_policy": "randomized article order across the full file list; final exact subset sampled reproducibly from all collected valid pairs",
        "file_list_path": str(args.file_list_path),
        "output_root": str(output_root),
        "processed_articles_count": len(processed_pmcs),
        "processed_pmcs": sorted(processed_pmcs),
        "collected_pairs_before_final_sampling": len(collected_rows),
        "final_pairs": len(final_rows),
        "final_output_jsonl": str(final_jsonl),
    }
    save_manifest(manifest_json, final_manifest)

    print("\nFinished successfully.")
    print(f"Collected valid pairs before exact sampling: {len(collected_rows):,}")
    print(f"Final exact subset written to: {final_jsonl}")
    print(f"Stats written to: {stats_json}")
    print(f"Manifest written to: {manifest_json}")


if __name__ == "__main__":
    main()
