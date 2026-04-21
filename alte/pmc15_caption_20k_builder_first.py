#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from tqdm import tqdm

# Prefer the Microsoft package layout if available.
try:
    from pmc15_pipeline import data
    from pmc15_pipeline.utils import fs_utils
except Exception:
    import data  # type: ignore
    import fs_utils  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - optional dependency
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore


MODALITIES = ["ct", "mri", "xray", "ultrasound", "microscopy"]

PROTOTYPES: Dict[str, List[str]] = {
    "ct": [
        "computed tomography ct scan cross sectional axial coronal sagittal contrast enhancement hounsfield density",
        "ct imaging reveals contrast enhancement on axial or coronal reconstructed images",
    ],
    "mri": [
        "magnetic resonance imaging mri t1 t2 flair dwi gadolinium signal intensity weighted sequence",
        "mri shows abnormal signal intensity on t1 t2 or flair weighted images",
    ],
    "xray": [
        "x ray radiograph radiography chest x ray ap pa lateral portable upright supine opacity lucency",
        "radiographic examination reveals findings on chest radiograph or plain film",
    ],
    "ultrasound": [
        "ultrasound sonography sonographic doppler transducer acoustic anechoic hyperechoic hypoechoic echo",
        "ultrasound image demonstrates doppler flow and sonographic findings",
    ],
    "microscopy": [
        "microscopy histopathology histology micrograph biopsy tissue staining cellular nuclear cytoplasm morphology",
        "histological examination reveals cellular morphology on stained tissue section or microscopic image",
    ],
}

HEURISTIC_TERMS: Dict[str, Dict[str, int]] = {
    "ct": {
        "computed tomography": 12,
        "ct scan": 10,
        "ct image": 9,
        "ct imaging": 10,
        "axial": 3,
        "coronal": 3,
        "sagittal": 3,
        "contrast": 2,
        "enhancement": 3,
        "density": 2,
        "hounsfield": 8,
        "attenuation": 3,
        "tomography": 4,
        "reconstruction": 2,
    },
    "mri": {
        "magnetic resonance imaging": 12,
        "mri": 10,
        "t1": 3,
        "t2": 3,
        "flair": 6,
        "dwi": 6,
        "adc": 6,
        "gadolinium": 6,
        "signal": 2,
        "intensity": 2,
        "weighted": 2,
        "diffusion weighted": 6,
        "diffusion": 2,
    },
    "xray": {
        "x ray": 10,
        "x-ray": 10,
        "radiograph": 8,
        "radiographic": 7,
        "plain film": 8,
        "chest radiograph": 8,
        "chest x ray": 8,
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
        "doppler": 6,
        "transducer": 4,
        "frequency": 2,
        "acoustic": 2,
        "echo": 2,
        "anechoic": 6,
        "hyperechoic": 6,
        "hypoechoic": 6,
        "echogenic": 4,
    },
    "microscopy": {
        "microscopy": 10,
        "microscopic": 10,
        "micrograph": 8,
        "histology": 10,
        "histological": 10,
        "histopathology": 10,
        "cellular": 3,
        "nuclear": 3,
        "cytoplasm": 4,
        "tissue": 2,
        "staining": 4,
        "morphology": 3,
        "biopsy": 3,
        "hematoxylin": 5,
        "eosin": 5,
        "immunohistochemistry": 6,
    },
}

PREFIX_TERMS: Dict[str, List[str]] = {
    "ct": ["this ct scan shows", "ct imaging reveals"],
    "mri": ["this mri shows", "magnetic resonance imaging reveals"],
    "xray": ["this chest x ray shows", "radiographic examination reveals"],
    "ultrasound": ["this ultrasound shows", "sonographic examination reveals"],
    "microscopy": ["this microscopic image shows", "histological examination reveals"],
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
    article_order: int


class CaptionModalityClassifier:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_semantic_confidence: float = 0.30,
        min_margin: float = 0.05,
        heuristic_margin: float = 1.5,
    ):
        self.embedding_model_name = embedding_model_name
        self.min_semantic_confidence = min_semantic_confidence
        self.min_margin = min_margin
        self.heuristic_margin = heuristic_margin
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
        for label, prefixes in PREFIX_TERMS.items():
            for prefix in prefixes:
                if prefix in text:
                    scores[label] += 8.0
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
            q = self.tfidf.transform([text])
            sims = cosine_similarity(q, self.prototype_matrix)[0]
            for sim, owner in zip(sims, self.prototype_owner):
                scores[owner] = max(scores[owner], float(sim))
            return scores

        return scores

    def classify(self, caption: str, fig_label: str = "") -> Tuple[Optional[str], float, Dict[str, float], Dict[str, float]]:
        text = self.normalize(f"{fig_label} {caption}")
        h_scores = self.heuristic_scores(text)
        s_scores = self.semantic_scores(text)

        ranked_h = sorted(h_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_h_label, top_h_score = ranked_h[0]
        second_h_score = ranked_h[1][1] if len(ranked_h) > 1 else 0.0

        # Very strong heuristic hit wins directly.
        if top_h_score >= 9.0 and (top_h_score - second_h_score) >= self.heuristic_margin:
            conf = min(0.999, 0.65 + min(top_h_score, 20.0) / 40.0)
            return top_h_label, conf, h_scores, s_scores

        ranked_s = sorted(s_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_s_label, top_s_score = ranked_s[0]
        second_s_score = ranked_s[1][1] if len(ranked_s) > 1 else 0.0

        # Combine both views, but only accept if there is some evidence.
        combined = {
            label: h_scores[label] + 4.0 * s_scores[label]
            for label in self.label_names
        }
        ranked_c = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
        top_c_label, top_c_score = ranked_c[0]
        second_c_score = ranked_c[1][1] if len(ranked_c) > 1 else 0.0

        if top_s_score < self.min_semantic_confidence:
            return None, 0.0, h_scores, s_scores
        if (top_s_score - second_s_score) < self.min_margin and (top_c_score - second_c_score) < 0.75:
            return None, 0.0, h_scores, s_scores

        conf = max(top_s_score, min(0.999, top_c_score / 20.0))
        return top_c_label, float(conf), h_scores, s_scores


class Builder:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = self._resolve_repo_root(args.repo_root)
        self.results_dir = self.repo_root / "_results" / "data"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.file_list_path = Path(args.file_list_path) if args.file_list_path else (self.results_dir / "pubmed_open_access_file_list.txt")
        self.compressed_dir = self.results_dir / "pubmed_open_access_files_compressed"
        self.decompressed_dir = self.results_dir / "pubmed_open_access_files"
        self.parsed_jsonl_path = self.results_dir / "pubmed_parsed_data.jsonl"
        self.sampled_list_path = self.results_dir / "pubmed_open_access_file_list.sampled_for_20k_pairs.txt"
        self.modalities_jsonl_path = self.results_dir / "pubmed_parsed_data_modalities.jsonl"
        self.filtered_20k_jsonl_path = self.results_dir / "pubmed_parsed_data_modalities_20k.jsonl"
        self.stats_json_path = self.results_dir / "pubmed_parsed_data_modality_stats.json"
        self.manifest_path = self.results_dir / "pubmed_selection_manifest_20k.json"

        self.classifier = CaptionModalityClassifier(
            embedding_model_name=args.embedding_model,
            min_semantic_confidence=args.min_semantic_confidence,
            min_margin=args.min_semantic_margin,
            heuristic_margin=args.heuristic_margin,
        )

        self.target_pairs = args.target_pairs
        self.batch_articles = args.batch_articles
        self.max_articles = args.max_articles
        self.seed = args.seed
        self.randomize = args.randomize
        self.start_offset = args.start_offset
        self.resume = args.resume
        self.clean_start = args.clean_start

        self.entry_by_pmcid: Dict[str, PubMedEntry] = {}
        self.article_order_by_pmcid: Dict[str, int] = {}

    def _resolve_repo_root(self, explicit: Optional[str]) -> Path:
        if explicit:
            return Path(explicit).resolve()
        try:
            return fs_utils.get_repo_root_path()
        except Exception:
            return Path.cwd().resolve()

    def reset_dirs(self) -> None:
        for path in [self.compressed_dir, self.decompressed_dir]:
            if path.exists():
                shutil.rmtree(path)
        for path in [
            self.parsed_jsonl_path,
            self.sampled_list_path,
            self.modalities_jsonl_path,
            self.filtered_20k_jsonl_path,
            self.stats_json_path,
            self.manifest_path,
        ]:
            if path.exists():
                path.unlink()

    def load_entries(self) -> List[PubMedEntry]:
        entries: List[PubMedEntry] = []
        with self.file_list_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for line_idx, row in enumerate(reader, start=1):
                if len(row) != 5:
                    continue
                entry = PubMedEntry(
                    line_idx=line_idx,
                    path=row[0],
                    title=row[1],
                    pmcid=row[2],
                    pmid=row[3],
                    code=row[4],
                )
                entries.append(entry)
                self.entry_by_pmcid[entry.pmcid] = entry
        return entries

    def candidate_order(self, entries: List[PubMedEntry]) -> List[PubMedEntry]:
        selected = entries[self.start_offset :]
        if self.max_articles is not None:
            selected = selected[: self.max_articles]
        if self.randomize:
            rng = random.Random(self.seed)
            selected = selected[:]
            rng.shuffle(selected)
        return selected

    def write_sampled_list(self, batch: Sequence[PubMedEntry]) -> None:
        with self.sampled_list_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["path", "title", "pmcid", "pmid", "code"])
            for e in batch:
                writer.writerow([e.path, e.title, e.pmcid, e.pmid, e.code])

    def parse_existing_known_pairs(self) -> Dict[str, PairRecord]:
        if not self.modalities_jsonl_path.exists():
            return {}
        known: Dict[str, PairRecord] = {}
        with self.modalities_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                known[obj["pair_id"]] = PairRecord(**obj)
        return known

    def run(self) -> None:
        if self.clean_start and not self.resume:
            self.reset_dirs()

        entries = self.load_entries()
        if not entries:
            raise RuntimeError(f"No entries found in {self.file_list_path}")

        order = self.candidate_order(entries)
        if not order:
            raise RuntimeError("Candidate order is empty. Check start_offset/max_articles.")

        for article_order, entry in enumerate(order):
            self.article_order_by_pmcid[entry.pmcid] = article_order

        known_pairs = self.parse_existing_known_pairs() if self.resume else {}
        downloaded_pmcids = self._detect_downloaded_pmcids()

        if known_pairs:
            print(f"Resuming with {len(known_pairs)} known labeled pairs already stored.")
        print(f"Using semantic backend: {self.classifier.semantic_backend}")

        cursor = 0
        while len(known_pairs) < self.target_pairs and cursor < len(order):
            batch = order[cursor : cursor + self.batch_articles]
            cursor += len(batch)
            if not batch:
                break

            need_download = [e for e in batch if e.pmcid not in downloaded_pmcids]
            print(
                f"\nProcessing batch of {len(batch)} articles "
                f"({len(need_download)} new downloads). Cursor: {cursor}/{len(order)}"
            )

            if need_download:
                self.write_sampled_list(need_download)
                data.download_pubmed_files_from_list(
                    file_list_path=self.sampled_list_path,
                    output_folder_path=self.compressed_dir,
                    subset_size=None,
                )
                downloaded_pmcids.update(e.pmcid for e in need_download)

            # Keep the Microsoft folder names. This may re-extract existing archives, but stays close to their pipeline.
            data.decompress_pubmed_files(
                input_folder_path=self.compressed_dir,
                output_folder_path=self.decompressed_dir,
            )
            data.generate_pmc15_pipeline_outputs(
                decompressed_folder=self.decompressed_dir,
                output_file_path=self.parsed_jsonl_path,
            )

            known_pairs = self._rebuild_labeled_pairs()
            stats = self._compute_stats(known_pairs)
            self._write_stats(stats)

            print(
                f"Known pairs so far: {len(known_pairs)} / {self.target_pairs}. "
                f"Articles considered: {cursor}. Modalities: {stats['counts']}"
            )

        final_pairs = self._select_final_pairs(known_pairs)
        if len(final_pairs) < self.target_pairs:
            raise RuntimeError(
                f"Only {len(final_pairs)} known pairs available, below target {self.target_pairs}. "
                f"Increase max_articles or relax thresholds."
            )

        self._write_final_pairs(final_pairs)
        self._write_manifest(final_pairs, cursor, len(order), downloaded_pmcids)

        final_counts = Counter(p.modality for p in final_pairs)
        print("\nFinished.")
        print(f"Final pairs: {len(final_pairs)}")
        print(f"Final modality counts: {dict(final_counts)}")
        print(f"Saved final pairs to: {self.filtered_20k_jsonl_path}")
        print(f"Saved manifest to: {self.manifest_path}")

    def _detect_downloaded_pmcids(self) -> set[str]:
        pmcids: set[str] = set()
        if self.compressed_dir.exists():
            for path in self.compressed_dir.glob("PMC*.tar.gz"):
                pmcids.add(path.name.replace(".tar.gz", ""))
        return pmcids

    def _rebuild_labeled_pairs(self) -> Dict[str, PairRecord]:
        if not self.parsed_jsonl_path.exists():
            return {}

        labeled: Dict[str, PairRecord] = {}
        with self.parsed_jsonl_path.open("r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Classifying parsed captions", unit="article"):
                article = json.loads(line)
                pmc_value = str(article.get("pmc", ""))
                pmcid = pmc_value if pmc_value.startswith("PMC") else f"PMC{pmc_value}" if pmc_value else ""
                entry = self.entry_by_pmcid.get(pmcid)
                article_order = self.article_order_by_pmcid.get(pmcid, 10**12)

                for figure in article.get("figures", []):
                    pair_id = str(figure.get("pair_id", ""))
                    caption = str(figure.get("fig_caption", "")).strip()
                    fig_label = str(figure.get("fig_label", "")).strip()
                    graphic_ref = str(figure.get("graphic_ref", "")).strip()
                    if not pair_id or not caption or not graphic_ref:
                        continue
                    if not Path(graphic_ref).exists():
                        continue

                    modality, confidence, _, _ = self.classifier.classify(caption=caption, fig_label=fig_label)
                    if modality is None or modality not in MODALITIES:
                        continue

                    record = PairRecord(
                        pair_id=pair_id,
                        pmid=str(article.get("pmid", "")),
                        pmc=pmcid,
                        fig_id=str(figure.get("fig_id", "")),
                        fig_label=fig_label,
                        fig_caption=caption,
                        graphic_ref=graphic_ref,
                        location=str(article.get("location", "")),
                        modality=modality,
                        confidence=float(confidence),
                        article_line_idx=entry.line_idx if entry else None,
                        article_path=entry.path if entry else None,
                        article_title=entry.title if entry else None,
                        article_code=entry.code if entry else None,
                        article_order=article_order,
                    )
                    prev = labeled.get(pair_id)
                    if prev is None or record.confidence > prev.confidence:
                        labeled[pair_id] = record

        # Persist all currently known labeled pairs.
        with self.modalities_jsonl_path.open("w", encoding="utf-8") as out:
            for rec in sorted(labeled.values(), key=lambda r: (r.article_order, r.pair_id)):
                out.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
        return labeled

    def _select_final_pairs(self, known_pairs: Dict[str, PairRecord]) -> List[PairRecord]:
        pairs = sorted(known_pairs.values(), key=lambda r: (r.article_order, r.pair_id))
        selected = pairs[: self.target_pairs]
        return selected

    def _write_final_pairs(self, pairs: Sequence[PairRecord]) -> None:
        with self.filtered_20k_jsonl_path.open("w", encoding="utf-8") as f:
            for rec in pairs:
                f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")

    def _compute_stats(self, known_pairs: Dict[str, PairRecord]) -> Dict[str, object]:
        counts = Counter(rec.modality for rec in known_pairs.values())
        total = sum(counts.values())
        percentages = {
            m: (100.0 * counts[m] / total if total else 0.0)
            for m in MODALITIES
        }
        return {
            "total_known_pairs": total,
            "counts": dict(counts),
            "percentages": percentages,
            "semantic_backend": self.classifier.semantic_backend,
            "target_pairs": self.target_pairs,
        }

    def _write_stats(self, stats: Dict[str, object]) -> None:
        with self.stats_json_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    def _write_manifest(
        self,
        final_pairs: Sequence[PairRecord],
        cursor: int,
        total_candidates: int,
        downloaded_pmcids: Iterable[str],
    ) -> None:
        unique_pmcids = sorted({p.pmc for p in final_pairs if p.pmc})
        final_counts = Counter(p.modality for p in final_pairs)
        manifest = {
            "target_pairs": self.target_pairs,
            "final_pair_count": len(final_pairs),
            "final_unique_pmcids": len(unique_pmcids),
            "final_modality_counts": dict(final_counts),
            "semantic_backend": self.classifier.semantic_backend,
            "embedding_model": self.args.embedding_model,
            "min_semantic_confidence": self.args.min_semantic_confidence,
            "min_semantic_margin": self.args.min_semantic_margin,
            "heuristic_margin": self.args.heuristic_margin,
            "seed": self.seed,
            "randomize_article_order": self.randomize,
            "start_offset": self.start_offset,
            "batch_articles": self.batch_articles,
            "articles_examined": cursor,
            "total_candidate_articles": total_candidates,
            "downloaded_archive_count": len(set(downloaded_pmcids)),
            "file_list_path": str(self.file_list_path),
            "compressed_dir": str(self.compressed_dir),
            "decompressed_dir": str(self.decompressed_dir),
            "parsed_jsonl_path": str(self.parsed_jsonl_path),
            "all_labeled_jsonl_path": str(self.modalities_jsonl_path),
            "final_pairs_jsonl_path": str(self.filtered_20k_jsonl_path),
            "stats_json_path": str(self.stats_json_path),
            "final_pmcids": unique_pmcids,
        }
        with self.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download PMC OA article archives, parse figure captions via the Microsoft pipeline, "
            "classify modalities from captions, and build an exact 20k image-text subset without unknown labels."
        )
    )
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--file-list-path", type=str, default=None)
    parser.add_argument("--target-pairs", type=int, default=20_000)
    parser.add_argument("--batch-articles", type=int, default=500)
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--randomize", action="store_true", help="Randomize article order reproducibly using --seed.")
    parser.add_argument("--start-offset", type=int, default=0, help="Skip the first N rows from the OA list before sampling.")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--min-semantic-confidence", type=float, default=0.30)
    parser.add_argument("--min-semantic-margin", type=float, default=0.05)
    parser.add_argument("--heuristic-margin", type=float, default=1.5)
    parser.add_argument("--resume", action="store_true", help="Resume from existing downloaded files and labeled JSONL.")
    parser.add_argument("--clean-start", action="store_true", help="Delete previous _results/data pipeline artifacts before starting.")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    builder = Builder(args)
    builder.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
