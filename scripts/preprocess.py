"""
preprocess.py
-------------
Reads all .txt blog files from data/raw/blogs/,
cleans them, and outputs:
  - data/processed/blog_corpus.jsonl   (one record per blog)
  - data/processed/blog_chunks.jsonl   (chunked records for training)
  - data/processed/train.jsonl         (85% split by document)
  - data/processed/val.jsonl           (15% split by document)
  - data/meta/blog_index.csv           (metadata tracker)
  - data/meta/preprocess_runs/*.json   (pipeline run metadata)
  - data/meta/logs/preprocess_*.log    (pipeline logs)

Usage:
  python scripts/preprocess.py
"""

import argparse
import json
import csv
import logging
import re
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
import unicodedata

# Pipeline version for lineage and auditability.
PIPELINE_VERSION = "2.0.0"

# Resolve paths from this file location so running from any cwd works.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ── Configuration ──────────────────────────────────────────────────────────────
RAW_DIR       = PROJECT_ROOT / "data/raw/blogs"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
META_DIR      = PROJECT_ROOT / "data/meta"
RUNS_DIR      = PROCESSED_DIR / "runs"
LOG_DIR       = META_DIR / "logs"
RUN_META_DIR  = META_DIR / "preprocess_runs"

MAX_CHUNK_TOKENS    = 320
CHUNK_OVERLAP_TOKENS = 48
MIN_CHUNK_TOKENS    = 80
MIN_ALPHA_RATIO     = 0.55
VAL_PERCENT         = 15
DEFAULT_TOKENIZER_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# ── Helpers ────────────────────────────────────────────────────────────────────

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def setup_logging(run_id: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"preprocess_{run_id}.log"

    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("Run log: %s", log_path)
    return logger


class TextTokenizer:
    def __init__(self, model_name: str):
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for token-aware chunking. "
                "Install with: pip install transformers sentencepiece"
            ) from exc

        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare blog dataset for local LLM training.")
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    return parser.parse_args()


def stable_int_hash(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16)


def split_from_blog_id(blog_id: str) -> str:
    bucket = stable_int_hash(blog_id) % 100
    return "val" if bucket < VAL_PERCENT else "train"

def parse_filename(filename: str) -> dict:
    """Extract blog_id and title from filename like blog_001_some_title.txt"""
    stem = Path(filename).stem
    # Remove trailing .ipynb if present (e.g. blog_001_title.ipynb)
    stem = stem.replace(".ipynb", "")
    parts = stem.split("_", 2)
    blog_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else stem
    title = parts[2].replace("_", " ").strip() if len(parts) >= 3 else stem
    return {"blog_id": blog_id, "title": title}


def clean_text(raw: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph structure."""
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line while preserving structure
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines).strip()
    return text


def count_tokens(text: str, tokenizer: TextTokenizer) -> int:
    return len(tokenizer.encode(text))


def is_low_quality_chunk(text: str, tokenizer: TextTokenizer) -> bool:
    stripped = text.strip()
    if not stripped:
        return True

    token_count = count_tokens(stripped, tokenizer)
    if token_count < MIN_CHUNK_TOKENS:
        return True

    alpha_chars = sum(ch.isalpha() for ch in stripped)
    alpha_ratio = alpha_chars / max(1, len(stripped))
    if alpha_ratio < MIN_ALPHA_RATIO:
        return True

    # Reject highly repetitive chunks.
    words = re.findall(r"\w+", stripped.lower())
    if not words:
        return True
    unique_ratio = len(set(words)) / len(words)
    if len(words) >= 20 and unique_ratio < 0.2:
        return True

    return False


def chunk_text(text: str, tokenizer: TextTokenizer, chunk_size: int = MAX_CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """
    Split text into overlapping token-level chunks.
    Paragraph boundaries are preferred when possible.
    """
    sep_ids = tokenizer.encode("\n\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current_ids: list[int] = []

    for para in paragraphs:
        para_ids = tokenizer.encode(para)

        # Handle paragraphs longer than chunk size using sliding token windows.
        if len(para_ids) > chunk_size:
            if current_ids:
                chunks.append(tokenizer.decode(current_ids).strip())
                current_ids = []

            start = 0
            while start < len(para_ids):
                end = min(start + chunk_size, len(para_ids))
                window_ids = para_ids[start:end]
                chunks.append(tokenizer.decode(window_ids).strip())
                if end == len(para_ids):
                    break
                start = max(0, end - overlap)
            continue

        proposed = len(current_ids) + (len(sep_ids) if current_ids else 0) + len(para_ids)
        if proposed > chunk_size and current_ids:
            chunks.append(tokenizer.decode(current_ids).strip())
            overlap_ids = current_ids[-overlap:] if overlap > 0 else []
            current_ids = overlap_ids + (sep_ids if overlap_ids else []) + para_ids
        else:
            if current_ids:
                current_ids.extend(sep_ids)
            current_ids.extend(para_ids)

    if current_ids:
        chunks.append(tokenizer.decode(current_ids).strip())

    return chunks


def doc_hash(text: str) -> str:
    """Short hash for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run():
    args = parse_args()
    run_started_at = utc_now_iso()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger = setup_logging(run_id)

    logger.info("Pipeline version: %s", PIPELINE_VERSION)
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Tokenizer model: %s", args.tokenizer_model)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RUN_META_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = TextTokenizer(args.tokenizer_model)

    txt_files = sorted(RAW_DIR.glob("*.txt"))
    if not txt_files:
        logger.warning("No .txt files found in %s. Exiting.", RAW_DIR)
        return

    logger.info("Found %d blog files.", len(txt_files))

    corpus_records  = []
    chunk_records   = []
    index_rows      = []
    seen_hashes     = set()
    dropped_low_quality = 0

    for filepath in txt_files:
        raw_text = filepath.read_text(encoding="utf-8", errors="replace")
        meta     = parse_filename(filepath.name)
        blog_id  = meta["blog_id"]
        title    = meta["title"]

        cleaned  = clean_text(raw_text)
        h        = doc_hash(cleaned)

        # Deduplication
        if h in seen_hashes:
            logger.info("  [SKIP duplicate] %s", filepath.name)
            continue
        seen_hashes.add(h)

        now = utc_now_iso()

        # ── Corpus record (one per blog) ───────────────────────────────────────
        corpus_rec = {
            "id":           blog_id,
            "title":        title,
            "source_file":  filepath.name,
            "hash":         h,
            "pipeline_version": PIPELINE_VERSION,
            "created_at":   now,
            "char_count":   len(cleaned),
            "token_count":  count_tokens(cleaned, tokenizer),
            "text":         cleaned,
        }
        corpus_records.append(corpus_rec)

        # ── Chunk records ──────────────────────────────────────────────────────
        chunks = chunk_text(cleaned, tokenizer=tokenizer)
        for i, chunk in enumerate(chunks):
            if is_low_quality_chunk(chunk, tokenizer):
                dropped_low_quality += 1
                continue

            split = split_from_blog_id(blog_id)
            token_count = count_tokens(chunk, tokenizer)
            chunk_records.append({
                "id":          f"{blog_id}_chunk_{i+1:03d}",
                "blog_id":     blog_id,
                "chunk_index": i + 1,
                "total_chunks": len(chunks),
                "split":       split,
                "token_count": token_count,
                "pipeline_version": PIPELINE_VERSION,
                "text":        chunk,
            })

        # ── Index row ──────────────────────────────────────────────────────────
        index_rows.append({
            "blog_id":     blog_id,
            "title":       title,
            "source_file": filepath.name,
            "source_path": str(filepath),
            "created_at":  now,
            "updated_at":  now,
            "tags":        "",
            "status":      "processed",
        })

        logger.info("  [OK] %s  -> %d chunk(s)", filepath.name, len(chunks))

    # ── Deterministic split (by document hash) ────────────────────────────────
    val_ids = {r["id"] for r in corpus_records if split_from_blog_id(r["id"]) == "val"}
    train_ids = {r["id"] for r in corpus_records if split_from_blog_id(r["id"]) == "train"}

    # Guardrail for very small datasets.
    if len(corpus_records) >= 2 and not val_ids:
        forced = min(corpus_records, key=lambda r: stable_int_hash(r["id"]))["id"]
        val_ids.add(forced)
        train_ids.discard(forced)
    if len(corpus_records) >= 2 and not train_ids:
        forced = max(corpus_records, key=lambda r: stable_int_hash(r["id"]))["id"]
        train_ids.add(forced)
        val_ids.discard(forced)

    train_chunks = [c for c in chunk_records if c["blog_id"] in train_ids]
    val_chunks   = [c for c in chunk_records if c["blog_id"] in val_ids]

    # ── Write outputs to both canonical paths and versioned run directory ─────
    run_out_dir = RUNS_DIR / run_id
    run_out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(PROCESSED_DIR / "blog_corpus.jsonl", corpus_records)
    write_jsonl(PROCESSED_DIR / "blog_chunks.jsonl", chunk_records)
    write_jsonl(PROCESSED_DIR / "train.jsonl", train_chunks)
    write_jsonl(PROCESSED_DIR / "val.jsonl", val_chunks)

    write_jsonl(run_out_dir / "blog_corpus.jsonl", corpus_records)
    write_jsonl(run_out_dir / "blog_chunks.jsonl", chunk_records)
    write_jsonl(run_out_dir / "train.jsonl", train_chunks)
    write_jsonl(run_out_dir / "val.jsonl", val_chunks)

    # ── Write blog_index.csv ───────────────────────────────────────────────────
    csv_path = META_DIR / "blog_index.csv"
    fieldnames = ["blog_id","title","source_file","source_path",
                  "created_at","updated_at","tags","status"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    with open(run_out_dir / "blog_index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    run_meta = {
        "run_id": run_id,
        "pipeline_version": PIPELINE_VERSION,
        "started_at": run_started_at,
        "finished_at": utc_now_iso(),
        "project_root": str(PROJECT_ROOT),
        "raw_dir": str(RAW_DIR),
        "processed_dir": str(PROCESSED_DIR),
        "run_output_dir": str(run_out_dir),
        "tokenizer_model": args.tokenizer_model,
        "config": {
            "max_chunk_tokens": MAX_CHUNK_TOKENS,
            "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
            "min_chunk_tokens": MIN_CHUNK_TOKENS,
            "min_alpha_ratio": MIN_ALPHA_RATIO,
            "val_percent": VAL_PERCENT,
        },
        "counts": {
            "documents_processed": len(corpus_records),
            "total_chunks": len(chunk_records),
            "dropped_low_quality_chunks": dropped_low_quality,
            "train_chunks": len(train_chunks),
            "val_chunks": len(val_chunks),
            "train_docs": len(train_ids),
            "val_docs": len(val_ids),
        },
    }

    with open(RUN_META_DIR / f"{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info(f"""
Done!
─────────────────────────────────────────
  Documents processed : {len(corpus_records)}
  Total chunks        : {len(chunk_records)}
  Dropped low-quality : {dropped_low_quality}
  Train chunks        : {len(train_chunks)}  ({len(train_ids)} docs)
  Val   chunks        : {len(val_chunks)}  ({len(val_ids)} docs)

Output files:
  {PROCESSED_DIR}/blog_corpus.jsonl
  {PROCESSED_DIR}/blog_chunks.jsonl
  {PROCESSED_DIR}/train.jsonl
  {PROCESSED_DIR}/val.jsonl
  {META_DIR}/blog_index.csv
  {RUN_META_DIR}/{run_id}.json
  {run_out_dir}
─────────────────────────────────────────
""".strip())


if __name__ == "__main__":
    run()
