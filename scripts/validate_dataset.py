"""
validate_dataset.py
-------------------
Validate processed dataset artifacts before training.

Checks:
- File presence and JSONL parseability
- Required schema fields
- Split consistency and leakage (blog_id overlap across train/val)
- Token bounds and short-chunk distribution
- Exact duplicate chunk text detection
- Basic quality signals (alphabetic ratio)

Usage:
  python scripts/validate_dataset.py
  python scripts/validate_dataset.py --strict
"""

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PROCESSED_DIR = PROJECT_ROOT / "data/processed"
META_DIR = PROJECT_ROOT / "data/meta"
VALIDATION_DIR = META_DIR / "validation_runs"

DEFAULT_MIN_CHUNK_TOKENS = 80
DEFAULT_MAX_CHUNK_TOKENS = 320
DEFAULT_NEAR_MIN_WINDOW = 20


REQUIRED_CHUNK_FIELDS = {
    "id",
    "blog_id",
    "chunk_index",
    "total_chunks",
    "split",
    "token_count",
    "pipeline_version",
    "text",
}


class ValidationState:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate processed dataset artifacts.")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--report-dir", default=str(VALIDATION_DIR))
    parser.add_argument("--min-chunk-tokens", type=int, default=DEFAULT_MIN_CHUNK_TOKENS)
    parser.add_argument("--max-chunk-tokens", type=int, default=DEFAULT_MAX_CHUNK_TOKENS)
    parser.add_argument("--near-min-window", type=int, default=DEFAULT_NEAR_MIN_WINDOW)
    parser.add_argument(
        "--max-near-min-percent",
        type=float,
        default=20.0,
        help="Warn if percentage of chunks in [min, min+window] exceeds this value.",
    )
    parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.55,
        help="Warn if chunks fall below this ratio of alphabetic characters.",
    )
    parser.add_argument(
        "--max-low-alpha-percent",
        type=float,
        default=2.0,
        help="Warn if percentage of chunks below min-alpha-ratio exceeds this value.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail validation when warnings exist.",
    )
    return parser.parse_args()


def read_jsonl(path: Path, state: ValidationState) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    state.error(f"{path.name}: invalid JSON on line {i}: {exc}")
                    continue
                if not isinstance(row, dict):
                    state.error(f"{path.name}: line {i} is not a JSON object")
                    continue
                rows.append(row)
    except OSError as exc:
        state.error(f"cannot read {path}: {exc}")
    return rows


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(ch.isalpha() for ch in text)
    return alpha / max(1, len(text))


def short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def validate_schema(rows: list[dict[str, Any]], state: ValidationState, file_label: str) -> None:
    for idx, row in enumerate(rows, start=1):
        missing = REQUIRED_CHUNK_FIELDS - set(row)
        if missing:
            state.error(f"{file_label}: record {idx} missing fields: {sorted(missing)}")
            continue
        if row["split"] not in {"train", "val"}:
            state.error(f"{file_label}: record {idx} invalid split='{row['split']}'")
        if not isinstance(row["token_count"], int):
            state.error(f"{file_label}: record {idx} token_count must be int")
        if not isinstance(row["text"], str):
            state.error(f"{file_label}: record {idx} text must be str")


def validate_dataset(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    state = ValidationState()

    processed_dir = Path(args.processed_dir)
    train_path = processed_dir / "train.jsonl"
    val_path = processed_dir / "val.jsonl"
    chunks_path = processed_dir / "blog_chunks.jsonl"

    for path in [train_path, val_path, chunks_path]:
        if not path.exists():
            state.error(f"required file not found: {path}")

    if state.errors:
        return 1, {
            "status": "failed",
            "errors": state.errors,
            "warnings": state.warnings,
            "counts": {},
            "metrics": {},
        }

    train_rows = read_jsonl(train_path, state)
    val_rows = read_jsonl(val_path, state)
    chunk_rows = read_jsonl(chunks_path, state)

    validate_schema(train_rows, state, "train.jsonl")
    validate_schema(val_rows, state, "val.jsonl")
    validate_schema(chunk_rows, state, "blog_chunks.jsonl")

    train_ids = {r.get("id") for r in train_rows}
    val_ids = {r.get("id") for r in val_rows}

    overlap_chunk_ids = train_ids.intersection(val_ids)
    if overlap_chunk_ids:
        state.error(f"train/val chunk id leakage detected: {len(overlap_chunk_ids)} overlapping ids")

    train_blog_ids = {r.get("blog_id") for r in train_rows}
    val_blog_ids = {r.get("blog_id") for r in val_rows}
    overlap_blog_ids = train_blog_ids.intersection(val_blog_ids)
    if overlap_blog_ids:
        state.error(f"train/val blog_id leakage detected: {len(overlap_blog_ids)} overlapping blog ids")

    split_mismatch = [r["id"] for r in train_rows if r.get("split") != "train"]
    split_mismatch += [r["id"] for r in val_rows if r.get("split") != "val"]
    if split_mismatch:
        state.error(f"split field mismatch in train/val files: {len(split_mismatch)} records")

    token_counts = [r.get("token_count", -1) for r in chunk_rows if isinstance(r.get("token_count"), int)]
    if token_counts:
        low_tokens = [t for t in token_counts if t < args.min_chunk_tokens]
        high_tokens = [t for t in token_counts if t > args.max_chunk_tokens]
        if low_tokens:
            state.error(
                f"{len(low_tokens)} chunks below min token threshold ({args.min_chunk_tokens})"
            )
        if high_tokens:
            state.error(
                f"{len(high_tokens)} chunks above max token threshold ({args.max_chunk_tokens})"
            )

        near_min_upper = args.min_chunk_tokens + max(0, args.near_min_window)
        near_min_count = sum(args.min_chunk_tokens <= t <= near_min_upper for t in token_counts)
        near_min_percent = (near_min_count / len(token_counts)) * 100.0
        if near_min_percent > args.max_near_min_percent:
            state.warn(
                "high concentration of near-min chunks: "
                f"{near_min_percent:.2f}% in [{args.min_chunk_tokens}, {near_min_upper}]"
            )
    else:
        state.error("blog_chunks.jsonl has no valid token_count values")
        near_min_count = 0
        near_min_percent = 0.0

    low_alpha_ids = []
    for row in chunk_rows:
        txt = row.get("text")
        if not isinstance(txt, str):
            continue
        ratio = alpha_ratio(txt)
        if ratio < args.min_alpha_ratio:
            low_alpha_ids.append(row.get("id", "<unknown>"))

    low_alpha_percent = (len(low_alpha_ids) / max(1, len(chunk_rows))) * 100.0
    if low_alpha_percent > args.max_low_alpha_percent:
        state.warn(
            f"low-alpha chunks exceed threshold: {low_alpha_percent:.2f}% "
            f"(limit {args.max_low_alpha_percent:.2f}%)"
        )

    seen_hashes: dict[str, str] = {}
    duplicate_chunk_count = 0
    for row in chunk_rows:
        txt = row.get("text")
        cid = row.get("id", "<unknown>")
        if not isinstance(txt, str):
            continue
        h = short_hash(txt)
        prev = seen_hashes.get(h)
        if prev is not None and prev != cid:
            duplicate_chunk_count += 1
        else:
            seen_hashes[h] = cid

    if duplicate_chunk_count > 0:
        state.warn(f"exact duplicate chunk texts detected: {duplicate_chunk_count}")

    counts = {
        "total_chunks": len(chunk_rows),
        "train_chunks": len(train_rows),
        "val_chunks": len(val_rows),
        "train_docs": len(train_blog_ids),
        "val_docs": len(val_blog_ids),
    }

    metrics: dict[str, Any] = {
        "token_stats": {
            "min": min(token_counts) if token_counts else None,
            "max": max(token_counts) if token_counts else None,
            "mean": round(statistics.mean(token_counts), 2) if token_counts else None,
            "median": round(statistics.median(token_counts), 2) if token_counts else None,
        },
        "near_min": {
            "window_min": args.min_chunk_tokens,
            "window_max": args.min_chunk_tokens + max(0, args.near_min_window),
            "count": near_min_count,
            "percent": round(near_min_percent, 2),
            "max_allowed_percent": args.max_near_min_percent,
        },
        "low_alpha": {
            "count": len(low_alpha_ids),
            "percent": round(low_alpha_percent, 2),
            "max_allowed_percent": args.max_low_alpha_percent,
            "min_alpha_ratio": args.min_alpha_ratio,
        },
        "duplicates": {
            "exact_text_duplicates": duplicate_chunk_count,
        },
        "leakage": {
            "overlap_chunk_ids": len(overlap_chunk_ids),
            "overlap_blog_ids": len(overlap_blog_ids),
        },
    }

    if state.errors:
        status = "failed"
        exit_code = 1
    elif args.strict and state.warnings:
        status = "failed_strict"
        exit_code = 2
    elif state.warnings:
        status = "passed_with_warnings"
        exit_code = 0
    else:
        status = "passed"
        exit_code = 0

    report = {
        "status": status,
        "validated_at": utc_now_iso(),
        "processed_dir": str(processed_dir),
        "config": {
            "min_chunk_tokens": args.min_chunk_tokens,
            "max_chunk_tokens": args.max_chunk_tokens,
            "near_min_window": args.near_min_window,
            "max_near_min_percent": args.max_near_min_percent,
            "min_alpha_ratio": args.min_alpha_ratio,
            "max_low_alpha_percent": args.max_low_alpha_percent,
            "strict": args.strict,
        },
        "counts": counts,
        "metrics": metrics,
        "errors": state.errors,
        "warnings": state.warnings,
    }
    return exit_code, report


def print_summary(report: dict[str, Any]) -> None:
    status = report["status"]
    counts = report.get("counts", {})
    token_stats = report.get("metrics", {}).get("token_stats", {})

    print("Dataset Validation")
    print("=" * 48)
    print(f"status         : {status}")
    print(f"total_chunks   : {counts.get('total_chunks')}")
    print(f"train_chunks   : {counts.get('train_chunks')}")
    print(f"val_chunks     : {counts.get('val_chunks')}")
    print(f"train_docs     : {counts.get('train_docs')}")
    print(f"val_docs       : {counts.get('val_docs')}")
    print(
        "token stats    : "
        f"min={token_stats.get('min')} "
        f"max={token_stats.get('max')} "
        f"mean={token_stats.get('mean')} "
        f"median={token_stats.get('median')}"
    )

    errors = report.get("errors", [])
    warnings = report.get("warnings", [])

    if errors:
        print("\nErrors:")
        for msg in errors:
            print(f"  - {msg}")

    if warnings:
        print("\nWarnings:")
        for msg in warnings:
            print(f"  - {msg}")


def main() -> int:
    args = parse_args()
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    exit_code, report = validate_dataset(args)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    report_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"validation_{report_id}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_summary(report)
    print(f"\nreport_path    : {report_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
