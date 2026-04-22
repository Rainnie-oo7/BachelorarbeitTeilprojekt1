import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from lxml import etree
from tqdm import tqdm

import pubmed_parser
from BiomedCLIP_data_pipeline.pmc15_pipeline.constants import PUBMED_OPEN_ACCESS_BASE_URL, PUBMED_OPEN_ACCESS_FILE_LIST_URL
PIPELINE_PACKAGE_NAME = "pmc15_pipeline"



# ------------------------------------------------------------
# Pfade
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_OA_LIST_PATH = PROJECT_ROOT / "_results" / "data" / "pubmed_open_access_file_list.txt"
DEFAULT_COMPRESSED_DIR = PROJECT_ROOT / "_results" / "data" / "pubmed_open_access_files_compressed"
DEFAULT_EXTRACTED_DIR = PROJECT_ROOT / "_results" / "data" / "pubmed_open_access_files"
DEFAULT_MASTER_POOL_PARQUET = PROJECT_ROOT / "_results" / "data" / "master_pool_100k.parquet"
DEFAULT_MASTER_POOL_CSV = PROJECT_ROOT / "_results" / "data" / "master_pool_100k.csv"
DEFAULT_PROGRESS_JSON = PROJECT_ROOT / "_results" / "data" / "master_pool_progress.json"


# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic PMC master pool of figure-caption pairs."
    )

    parser.add_argument("--oa-list-path", default=str(DEFAULT_OA_LIST_PATH))
    parser.add_argument("--compressed-dir", default=str(DEFAULT_COMPRESSED_DIR))
    parser.add_argument("--extracted-dir", default=str(DEFAULT_EXTRACTED_DIR))
    parser.add_argument("--output-parquet", default=str(DEFAULT_MASTER_POOL_PARQUET))
    parser.add_argument("--output-csv", default=str(DEFAULT_MASTER_POOL_CSV))
    parser.add_argument("--progress-json", default=str(DEFAULT_PROGRESS_JSON))

    parser.add_argument("--target-pairs", type=int, default=100_000)
    parser.add_argument("--buffer-factor", type=float, default=1.10)
    parser.add_argument("--max-articles", type=int, default=None)

    parser.add_argument("--download-timeout", type=int, default=120)
    parser.add_argument("--article-salt", default="pmc_article_order_v1")
    parser.add_argument("--pair-salt", default="pmc_pair_order_v1")

    parser.add_argument("--force-redownload-list", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")

    return parser.parse_args()


# ------------------------------------------------------------
# Hash / IDs
# ------------------------------------------------------------
def stable_hash_u64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def hash_score(value: str, salt: str) -> int:
    return stable_hash_u64(f"{salt}::{value}")


def make_sample_id(pair_id: str, image_path: str, caption: str) -> str:
    base = f"{pair_id}|||{image_path}|||{caption}"
    return hashlib.sha1(base.encode("utf8")).hexdigest()


# ------------------------------------------------------------
# OA list
# ------------------------------------------------------------
def download_oa_list_if_needed(oa_list_path: Path, force_redownload: bool = False) -> None:
    oa_list_path.parent.mkdir(parents=True, exist_ok=True)

    if oa_list_path.exists() and not force_redownload:
        print(f"OA list already exists: {oa_list_path}")
        return

    print(f"Downloading OA list from {PUBMED_OPEN_ACCESS_FILE_LIST_URL}")
    resp = requests.get(PUBMED_OPEN_ACCESS_FILE_LIST_URL, timeout=120)
    resp.raise_for_status()

    with oa_list_path.open("wb") as f:
        f.write(resp.content)

    print(f"Saved OA list to: {oa_list_path}")


def load_oa_list(oa_list_path: Path) -> pd.DataFrame:
    rows: List[dict] = []

    with oa_list_path.open("r", encoding="utf8") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"OA list is empty: {oa_list_path}")

    # Header überspringen
    lines = lines[1:]

    for line_no, line in enumerate(lines, start=2):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 5:
            continue

        path, title, pmcid, pmid, code = parts
        rows.append(
            {
                "path": path,
                "title": title,
                "pmcid": pmcid,
                "pmid": pmid,
                "code": code,
                "article_key": f"{pmcid}|||{pmid}|||{path}",
            }
        )

    if not rows:
        raise ValueError("No valid rows parsed from OA list.")

    return pd.DataFrame(rows)


def build_deterministic_article_order(df: pd.DataFrame, salt: str) -> pd.DataFrame:
    df = df.copy()
    df["article_score"] = df["article_key"].map(lambda x: hash_score(x, salt))
    df = df.sort_values(["article_score", "pmcid"], kind="mergesort").reset_index(drop=True)
    df["article_rank"] = range(len(df))
    return df


# ------------------------------------------------------------
# Download / extract one article
# ------------------------------------------------------------
def download_article_if_needed(
    article_row: pd.Series,
    compressed_dir: Path,
    timeout: int,
) -> Path:
    compressed_dir.mkdir(parents=True, exist_ok=True)

    pmcid = str(article_row["pmcid"])
    rel_path = str(article_row["path"])
    url = PUBMED_OPEN_ACCESS_BASE_URL + rel_path
    out_path = compressed_dir / f"{pmcid}.tar.gz"

    if out_path.exists():
        return out_path

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    with out_path.open("wb") as f:
        f.write(resp.content)

    return out_path


def extract_article_to_folder(tar_path: Path, target_dir: Path) -> Path:
    article_extract_dir = target_dir / tar_path.stem.replace(".tar", "")
    article_extract_dir.mkdir(parents=True, exist_ok=True)

    sentinel = article_extract_dir / ".done"
    if sentinel.exists():
        return article_extract_dir

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(article_extract_dir)

    sentinel.write_text("ok", encoding="utf8")
    return article_extract_dir


# ------------------------------------------------------------
# Parse .nxml -> figure rows
# ------------------------------------------------------------
def parse_single_pubmed_nxml(nxml_path: Path) -> List[dict]:
    if not nxml_path.exists():
        return []

    try:
        output = pubmed_parser.parse_pubmed_caption(str(nxml_path.absolute()))
    except AttributeError:
        return []
    except etree.XMLSyntaxError:
        return []
    except Exception:
        return []

    if not output:
        return []

    figures = []
    pmid = str(output[0].get("pmid", ""))
    pmc = str(output[0].get("pmc", ""))
    location = Path(nxml_path).parent

    for figure_dict in output:
        inline_references = figure_dict.get("fig_refs", {}) or []
        if len(inline_references) > 0:
            # analog zur Originalpipeline nicht weiterverwenden
            pass

        fig_caption = str(figure_dict.get("fig_caption", "") or "").strip()
        fig_id = str(figure_dict.get("fig_id", "") or "")
        fig_label = str(figure_dict.get("fig_label", "") or "")

        if "graphic_ref" in figure_dict:
            graphic_ref = str(location / (figure_dict["graphic_ref"] + ".jpg"))
        else:
            graphic_ref = ""

        pair_id = f"{pmid}_{fig_id}"

        if not fig_caption:
            continue

        figures.append(
            {
                "pmid": pmid,
                "pmcid": pmc,
                "location": str(location),
                "image_path": graphic_ref,
                "fig_id": fig_id,
                "fig_label": fig_label,
                "pair_id": pair_id,
                "caption": fig_caption,
            }
        )

    return figures


def parse_extracted_article(article_extract_dir: Path) -> List[dict]:
    rows: List[dict] = []

    nxml_files = list(article_extract_dir.rglob("*.nxml"))
    for nxml_file in nxml_files:
        rows.extend(parse_single_pubmed_nxml(nxml_file))

    return rows


# ------------------------------------------------------------
# Build master pool
# ------------------------------------------------------------
def clean_caption(text: str) -> str:
    text = str(text or "")
    text = " ".join(text.replace("\n", " ").split())
    return text.strip()


def finalize_master_pool(df_pairs: pd.DataFrame, target_pairs: int, pair_salt: str) -> pd.DataFrame:
    df_pairs = df_pairs.copy()

    df_pairs["caption"] = df_pairs["caption"].map(clean_caption)
    df_pairs = df_pairs[df_pairs["caption"].astype(bool)].reset_index(drop=True)

    df_pairs["sample_id"] = [
        make_sample_id(pair_id, image_path, caption)
        for pair_id, image_path, caption in zip(
            df_pairs["pair_id"], df_pairs["image_path"], df_pairs["caption"]
        )
    ]

    df_pairs = df_pairs.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
    df_pairs["pair_score"] = df_pairs["sample_id"].map(lambda s: hash_score(s, pair_salt))
    df_pairs = df_pairs.sort_values(["pair_score", "sample_id"], kind="mergesort").reset_index(drop=True)

    if len(df_pairs) > target_pairs:
        df_pairs = df_pairs.iloc[:target_pairs].copy().reset_index(drop=True)

    df_pairs["master_rank"] = range(len(df_pairs))
    return df_pairs


def save_progress(progress_path: Path, payload: dict) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("w", encoding="utf8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    args = parse_args()

    oa_list_path = Path(args.oa_list_path).expanduser().resolve()
    compressed_dir = Path(args.compressed_dir).expanduser().resolve()
    extracted_dir = Path(args.extracted_dir).expanduser().resolve()
    output_parquet = Path(args.output_parquet).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    progress_json = Path(args.progress_json).expanduser().resolve()

    compressed_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    buffer_target = int(args.target_pairs * args.buffer_factor)

    print(f"Using pipeline package: {PIPELINE_PACKAGE_NAME}")
    print(f"Target pairs: {args.target_pairs}")
    print(f"Buffered collection target: {buffer_target}")

    download_oa_list_if_needed(oa_list_path, force_redownload=args.force_redownload_list)

    oa_df = load_oa_list(oa_list_path)
    oa_df = build_deterministic_article_order(oa_df, salt=args.article_salt)

    if args.max_articles is not None:
        oa_df = oa_df.iloc[:args.max_articles].copy().reset_index(drop=True)

    collected_rows: List[dict] = []
    articles_processed = 0
    parse_failures = 0

    for _, article_row in tqdm(oa_df.iterrows(), total=len(oa_df), desc="Articles"):
        pmcid = str(article_row["pmcid"])

        try:
            tar_path = download_article_if_needed(
                article_row=article_row,
                compressed_dir=compressed_dir,
                timeout=args.download_timeout,
            )
            extracted_article_dir = extract_article_to_folder(tar_path, extracted_dir)
            rows = parse_extracted_article(extracted_article_dir)

            for row in rows:
                row["article_rank"] = int(article_row["article_rank"])
                row["article_score"] = int(article_row["article_score"])
                row["source_archive"] = str(tar_path)
                collected_rows.append(row)

            articles_processed += 1

        except Exception as e:
            parse_failures += 1
            print(f"[Warnung] Artikel {pmcid} übersprungen: {e}")
            continue

        unique_pairs_so_far = len({
            make_sample_id(
                r.get("pair_id", ""),
                r.get("image_path", ""),
                clean_caption(r.get("caption", "")),
            )
            for r in collected_rows
            if clean_caption(r.get("caption", ""))
        })

        if unique_pairs_so_far >= buffer_target:
            print(f"Buffered target reached: {unique_pairs_so_far} unique pairs")
            break

    if not collected_rows:
        raise ValueError("No figure-caption pairs collected.")

    df_pairs = pd.DataFrame(collected_rows)
    df_master = finalize_master_pool(
        df_pairs=df_pairs,
        target_pairs=args.target_pairs,
        pair_salt=args.pair_salt,
    )

    df_master.to_parquet(output_parquet, index=False)
    df_master.to_csv(output_csv, index=False)

    progress_payload = {
        "pipeline_package": PIPELINE_PACKAGE_NAME,
        "oa_list_path": str(oa_list_path),
        "compressed_dir": str(compressed_dir),
        "extracted_dir": str(extracted_dir),
        "output_parquet": str(output_parquet),
        "output_csv": str(output_csv),
        "target_pairs": args.target_pairs,
        "buffer_factor": args.buffer_factor,
        "buffer_target": buffer_target,
        "articles_available": int(len(oa_df)),
        "articles_processed": int(articles_processed),
        "parse_failures": int(parse_failures),
        "pairs_collected_before_finalize": int(len(df_pairs)),
        "pairs_in_final_master_pool": int(len(df_master)),
    }
    save_progress(progress_json, progress_payload)

    print("\n===== Fertig =====")
    print(f"OA list: {oa_list_path}")
    print(f"Articles processed: {articles_processed}")
    print(f"Parse failures: {parse_failures}")
    print(f"Collected rows before final trim: {len(df_pairs)}")
    print(f"Final master pool: {len(df_master)}")
    print(f"Saved parquet: {output_parquet}")
    print(f"Saved csv: {output_csv}")
    print(f"Saved progress: {progress_json}")

    if not args.keep_intermediate:
        print("Zwischendateien bleiben erhalten. Es wurde nichts automatisch gelöscht.")


if __name__ == "__main__":
    main()