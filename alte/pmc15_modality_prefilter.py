#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import requests
from tqdm import tqdm

# Reuse the existing helper from your repo.
# Works when this script is placed inside the same repository or when PYTHONPATH points to it.
try:
    from pmc15_pipeline.utils import fs_utils
except Exception:
    # fallback for running next to uploaded fs_utils.py
    import fs_utils  # type: ignore


# If your package is importable, reuse the same base URL constant the original pipeline uses.
# Otherwise fall back to the public PMC OA FTP URL shape.
try:
    from pmc15_pipeline.constants import PUBMED_OPEN_ACCESS_BASE_URL
except Exception:
    PUBMED_OPEN_ACCESS_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/"


MODALITIES = ["ct", "mri", "xray", "ultrasound", "microscopy", "unknown"]

PREFIX_PATTERNS = {
    "ct": [
        r"\bthis ct scan shows\b",
        r"\bct imaging reveals\b",
    ],
    "mri": [
        r"\bthis mri shows\b",
        r"\bmagnetic resonance imaging reveals\b",
    ],
    "xray": [
        r"\bthis chest x[ -]?ray shows\b",
        r"\bradiographic examination reveals\b",
    ],
    "ultrasound": [
        r"\bthis ultrasound shows\b",
        r"\bsonographic examination reveals\b",
    ],
    "microscopy": [
        r"\bthis microscopic image shows\b",
        r"\bhistological examination reveals\b",
    ],
}

KEYWORDS = {
    "ct": {
        "axial": 2,
        "coronal": 2,
        "sagittal": 2,
        "contrast": 1,
        "enhancement": 2,
        "density": 1,
        "hounsfield": 4,
        "computed tomography": 5,
        "ct scan": 4,
        "ct imaging": 4,
    },
    "mri": {
        "t1": 3,
        "t2": 3,
        "flair": 4,
        "dwi": 4,
        "gadolinium": 4,
        "signal": 1,
        "intensity": 1,
        "weighted": 2,
        "magnetic resonance imaging": 5,
        "mri": 4,
    },
    "xray": {
        " ap ": 2,
        " pa ": 2,
        "lateral": 2,
        "portable": 2,
        "upright": 2,
        "supine": 2,
        "opacity": 2,
        "lucency": 2,
        "radiograph": 4,
        "radiographic": 4,
        "x-ray": 5,
        "x ray": 5,
        "chest x-ray": 5,
    },
    "ultrasound": {
        "echo": 2,
        "doppler": 4,
        "transducer": 3,
        "frequency": 1,
        "acoustic": 2,
        "anechoic": 4,
        "hyperechoic": 4,
        "ultrasound": 5,
        "sonography": 5,
        "sonographic": 5,
    },
    "microscopy": {
        "cellular": 2,
        "nuclear": 2,
        "cytoplasm": 3,
        "tissue": 1,
        "staining": 3,
        "morphology": 2,
        "biopsy": 2,
        "histology": 5,
        "histological": 5,
        "histopathology": 5,
        "microscopy": 5,
        "microscopic": 5,
    },
}

NEGATIVE_HINTS = {
    "microscopy": ["ct", "mri", "x-ray", "x ray", "ultrasound", "sonograph"],
    "ct": ["microscopy", "histology", "histological", "histopathology"],
    "mri": ["microscopy", "histology", "histological", "histopathology"],
    "xray": ["microscopy", "histology", "histological", "histopathology"],
    "ultrasound": ["microscopy", "histology", "histological", "histopathology"],
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
class ClassificationResult:
    modality: str
    score: int
    scores: Dict[str, int]
    matched_terms: Dict[str, List[str]]
    text_hash: str


@dataclass(slots=True)
class SampledEntry:
    entry: PubMedEntry
    classification: ClassificationResult


def normalize_text(*parts: str) -> str:
    text = " | ".join(part.strip() for part in parts if part)
    text = text.lower()
    text = text.replace("/", " ")
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", f" {text} ").strip()
    return f" {text} "


def classify_modality(entry: PubMedEntry) -> ClassificationResult:
    text = normalize_text(entry.title, entry.path, entry.code)
    scores = {m: 0 for m in MODALITIES if m != "unknown"}
    matched_terms: Dict[str, List[str]] = defaultdict(list)

    for modality, patterns in PREFIX_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                scores[modality] += 6
                matched_terms[modality].append(f"prefix:{pattern}")

    for modality, weighted_keywords in KEYWORDS.items():
        for keyword, weight in weighted_keywords.items():
            if keyword.isalnum():
                pattern = rf"\b{re.escape(keyword)}\b"
                if re.search(pattern, text):
                    scores[modality] += weight
                    matched_terms[modality].append(keyword)
            else:
                if keyword in text:
                    scores[modality] += weight
                    matched_terms[modality].append(keyword)

    for modality, negatives in NEGATIVE_HINTS.items():
        for negative in negatives:
            if re.search(rf"\b{re.escape(negative)}\b", text):
                scores[modality] -= 1

    best_modality = max(scores, key=scores.get)
    best_score = scores[best_modality]

    if best_score <= 0:
        final_modality = "unknown"
        final_score = 0
    else:
        tied = [m for m, s in scores.items() if s == best_score]
        if len(tied) > 1:
            final_modality = "unknown"
            final_score = best_score
        else:
            final_modality = best_modality
            final_score = best_score

    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return ClassificationResult(
        modality=final_modality,
        score=final_score,
        scores=scores,
        matched_terms=dict(matched_terms),
        text_hash=text_hash,
    )


def iter_pubmed_entries(file_list_path: Path) -> Iterator[PubMedEntry]:
    with file_list_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return

        for line_idx, row in enumerate(reader, start=1):
            if len(row) != 5:
                # skip malformed lines, but preserve deterministic indexing of valid entries
                continue
            path, title, pmcid, pmid, code = row
            yield PubMedEntry(
                line_idx=line_idx,
                path=path,
                title=title,
                pmcid=pmcid,
                pmid=pmid,
                code=code,
            )


def classify_all_entries(
    file_list_path: Path,
    report_jsonl_path: Path,
    stats_json_path: Path,
    sampling_every: int,
) -> tuple[List[int], Dict[str, int], int]:
    report_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    stats_json_path.parent.mkdir(parents=True, exist_ok=True)

    modality_counts: Counter[str] = Counter()
    sampled_line_indices: List[int] = []
    total_entries = 0

    with report_jsonl_path.open("w", encoding="utf-8") as out:
        for entry in tqdm(iter_pubmed_entries(file_list_path), desc="Classifying all entries"):
            total_entries += 1
            cls = classify_modality(entry)
            modality_counts[cls.modality] += 1

            if entry.line_idx % sampling_every == 0:
                sampled_line_indices.append(entry.line_idx)

            out.write(
                json.dumps(
                    {
                        "line_idx": entry.line_idx,
                        "path": entry.path,
                        "title": entry.title,
                        "pmcid": entry.pmcid,
                        "pmid": entry.pmid,
                        "code": entry.code,
                        "modality": cls.modality,
                        "score": cls.score,
                        "scores": cls.scores,
                        "matched_terms": cls.matched_terms,
                        "text_hash": cls.text_hash,
                        "selected_every_n": entry.line_idx % sampling_every == 0,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    stats = {
        "file_list_path": str(file_list_path),
        "total_entries": total_entries,
        "sampling_every": sampling_every,
        "selected_every_n_count": len(sampled_line_indices),
        "modalities": {},
    }

    for modality in MODALITIES:
        count = modality_counts.get(modality, 0)
        percent = (count / total_entries * 100.0) if total_entries else 0.0
        stats["modalities"][modality] = {
            "count": count,
            "percent_of_all_entries": round(percent, 6),
        }

    with stats_json_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return sampled_line_indices, dict(modality_counts), total_entries


def print_stats(modality_counts: Dict[str, int], total_entries: int, sampled_count: int, sampling_every: int) -> None:
    print("\n===== Modalitätsstatistik über ALLE Einträge =====")
    print(f"Gesamtzahl klassifizierter Einträge: {total_entries}")
    print(f"Geplante systematische Auswahl: jeder {sampling_every}. Eintrag")
    print(f"Anzahl ausgewählter Kandidaten bei jeder-{sampling_every}-Regel: {sampled_count}")
    print()
    for modality in MODALITIES:
        count = modality_counts.get(modality, 0)
        percent = (count / total_entries * 100.0) if total_entries else 0.0
        print(f"{modality:12s}  count={count:10d}  percent_of_all={percent:9.4f}%")
    print("===============================================\n")


def download_selected_entries(
    file_list_path: Path,
    selected_line_indices: set[int],
    output_folder_path: Path,
    timeout: int = 60,
    skip_existing: bool = True,
) -> None:
    output_folder_path.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    downloaded = 0
    skipped_existing = 0
    failed: List[dict] = []

    entries = (entry for entry in iter_pubmed_entries(file_list_path) if entry.line_idx in selected_line_indices)
    for entry in tqdm(entries, total=len(selected_line_indices), desc="Downloading selected tar shards"):
        file_name = f"{entry.pmcid}.tar.gz"
        file_path = output_folder_path / file_name
        article_url = PUBMED_OPEN_ACCESS_BASE_URL + entry.path

        if skip_existing and file_path.exists():
            skipped_existing += 1
            continue

        try:
            response = session.get(article_url, timeout=timeout)
            response.raise_for_status()
            with file_path.open("wb") as f:
                f.write(response.content)
            downloaded += 1
        except requests.RequestException as exc:
            failed.append(
                {
                    "line_idx": entry.line_idx,
                    "pmcid": entry.pmcid,
                    "url": article_url,
                    "error": str(exc),
                }
            )

    print(f"Neu heruntergeladen: {downloaded}")
    print(f"Bereits vorhanden übersprungen: {skipped_existing}")
    print(f"Fehlgeschlagen: {len(failed)}")

    if failed:
        failed_path = output_folder_path / "failed_downloads.json"
        with failed_path.open("w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2, ensure_ascii=False)
        print(f"Fehlschläge gespeichert unter: {failed_path}")


def resolve_default_paths(file_list_path: Optional[str], output_folder_path: Optional[str]) -> tuple[Path, Path, Path, Path]:
    repo_root = fs_utils.get_repo_root_path()
    data_root = repo_root / "_results" / "data"

    list_path = Path(file_list_path) if file_list_path else data_root / "pubmed_open_access_file_list.txt"
    download_path = Path(output_folder_path) if output_folder_path else data_root / "pubmed_open_access_files_compressed_every7"
    report_jsonl_path = data_root / "pubmed_open_access_file_list_modality_report.jsonl"
    stats_json_path = data_root / "pubmed_open_access_file_list_modality_stats.json"
    return list_path, download_path, report_jsonl_path, stats_json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "1) Klassifiziert ALLE Einträge der pubmed_open_access_file_list.txt heuristisch nach Modalität. "
            "2) Druckt count und Prozentanteil bezogen auf ALLE Einträge. "
            "3) Lädt nur dann jeden n-ten Eintrag herunter, wenn --execute-download gesetzt ist."
        )
    )
    parser.add_argument("--file-list-path", type=str, default=None)
    parser.add_argument("--output-folder-path", type=str, default=None)
    parser.add_argument("--sampling-every", type=int, default=7)
    parser.add_argument(
        "--execute-download",
        action="store_true",
        help="Ohne diesen Schalter wird nur klassifiziert und berichtet, aber nichts heruntergeladen.",
    )
    args = parser.parse_args()

    file_list_path, output_folder_path, report_jsonl_path, stats_json_path = resolve_default_paths(
        args.file_list_path,
        args.output_folder_path,
    )

    if not file_list_path.exists():
        raise FileNotFoundError(f"Dateiliste nicht gefunden: {file_list_path}")

    selected_line_indices, modality_counts, total_entries = classify_all_entries(
        file_list_path=file_list_path,
        report_jsonl_path=report_jsonl_path,
        stats_json_path=stats_json_path,
        sampling_every=args.sampling_every,
    )

    print_stats(
        modality_counts=modality_counts,
        total_entries=total_entries,
        sampled_count=len(selected_line_indices),
        sampling_every=args.sampling_every,
    )

    print(f"JSONL-Report gespeichert unter: {report_jsonl_path}")
    print(f"Statistik gespeichert unter:    {stats_json_path}")

    if not args.execute_download:
        print(
            "\nKein Download ausgeführt. Das ist die Sicherheitsstufe vor deinem Absegnen. "
            "Wenn die Anteile für dich passen, starte denselben Befehl zusätzlich mit --execute-download."
        )
        return

    download_selected_entries(
        file_list_path=file_list_path,
        selected_line_indices=set(selected_line_indices),
        output_folder_path=output_folder_path,
    )


if __name__ == "__main__":
    main()
