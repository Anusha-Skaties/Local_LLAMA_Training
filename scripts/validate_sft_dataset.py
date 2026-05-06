"""
validate_sft_dataset.py
----------------------
Validate conversation-style SFT dataset artifacts before training.

Checks:
- Required SFT files exist and parse as JSONL
- Conversation schema and message role order
- Split consistency and train/val leakage by blog_id and record id
- Basic response quality signals (empty responses, very short responses)
- Duplicate assistant responses (exact-text hash)

Usage:
  python scripts/validate_sft_dataset.py
  python scripts/validate_sft_dataset.py --strict
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

SFT_DIR = PROJECT_ROOT / "data/processed/sft"
REPORT_DIR = PROJECT_ROOT / "data/meta/validation_runs"

REQUIRED_TOP_KEYS = {
    "id",
    "blog_id",
    "split",
    "format",
    "messages",
    "metadata",
}

EXPECTED_ROLES = ["system", "user", "assistant"]


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
    parser = argparse.ArgumentParser(description="Validate conversation SFT dataset.")
    parser.add_argument("--sft-dir", default=str(SFT_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--min-assistant-chars", type=int, default=400)
    parser.add_argument("--max-short-assistant-percent", type=float, default=5.0)
    parser.add_argument("--max-duplicate-assistant-percent", type=float, default=70.0,
                        help="Warn only if duplicate assistant responses exceed this %% of total records. "
                             "Default 70%% allows for 3-template-per-blog datasets.")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path, state: ValidationState) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    state.error(f"{path.name}: invalid JSON on line {i}: {exc}")
                    continue
                if not isinstance(obj, dict):
                    state.error(f"{path.name}: line {i} is not a JSON object")
                    continue
                rows.append(obj)
    except OSError as exc:
        state.error(f"cannot read {path}: {exc}")
    return rows


def hash12(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def validate_row_schema(row: dict[str, Any], index: int, state: ValidationState, label: str) -> None:
    missing = REQUIRED_TOP_KEYS - set(row)
    if missing:
        state.error(f"{label}: record {index} missing keys: {sorted(missing)}")
        return

    split = row.get("split")
    if split not in {"train", "val"}:
        state.error(f"{label}: record {index} invalid split={split}")

    fmt = row.get("format")
    if fmt != "chatml_messages":
        state.warn(f"{label}: record {index} format='{fmt}', expected 'chatml_messages'")

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        state.error(f"{label}: record {index} messages must be a list with >= 3 items")
        return

    roles = []
    for m_idx, msg in enumerate(messages, start=1):
        if not isinstance(msg, dict):
            state.error(f"{label}: record {index} message {m_idx} is not an object")
            continue
        role = msg.get("role")
        content = msg.get("content")
        roles.append(role)
        if not isinstance(content, str) or not content.strip():
            state.error(f"{label}: record {index} message {m_idx} has empty content")

    # Enforce first three roles for deterministic template.
    if roles[:3] != EXPECTED_ROLES:
        state.error(
            f"{label}: record {index} first roles are {roles[:3]}, expected {EXPECTED_ROLES}"
        )


def validate_dataset(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    state = ValidationState()

    sft_dir = Path(args.sft_dir)
    train_path = sft_dir / "train_conversations.jsonl"
    val_path = sft_dir / "val_conversations.jsonl"
    all_path = sft_dir / "all_conversations.jsonl"

    for p in [train_path, val_path, all_path]:
        if not p.exists():
            state.error(f"required file not found: {p}")

    if state.errors:
        report = {
            "status": "failed",
            "validated_at": utc_now_iso(),
            "errors": state.errors,
            "warnings": state.warnings,
        }
        return 1, report

    train_rows = read_jsonl(train_path, state)
    val_rows = read_jsonl(val_path, state)
    all_rows = read_jsonl(all_path, state)

    for i, row in enumerate(train_rows, start=1):
        validate_row_schema(row, i, state, "train_conversations.jsonl")
        if row.get("split") != "train":
            state.error(f"train_conversations.jsonl: record {i} has split='{row.get('split')}'")

    for i, row in enumerate(val_rows, start=1):
        validate_row_schema(row, i, state, "val_conversations.jsonl")
        if row.get("split") != "val":
            state.error(f"val_conversations.jsonl: record {i} has split='{row.get('split')}'")

    for i, row in enumerate(all_rows, start=1):
        validate_row_schema(row, i, state, "all_conversations.jsonl")

    train_ids = {str(r.get("id")) for r in train_rows}
    val_ids = {str(r.get("id")) for r in val_rows}
    overlap_ids = train_ids.intersection(val_ids)
    if overlap_ids:
        state.error(f"record id leakage across train/val: {len(overlap_ids)}")

    train_blog_ids = {str(r.get("blog_id")) for r in train_rows}
    val_blog_ids = {str(r.get("blog_id")) for r in val_rows}
    overlap_blog_ids = train_blog_ids.intersection(val_blog_ids)
    if overlap_blog_ids:
        state.error(f"blog_id leakage across train/val: {len(overlap_blog_ids)}")

    all_ids = {str(r.get("id")) for r in all_rows}
    expected_all = train_ids.union(val_ids)
    if all_ids != expected_all:
        state.warn(
            "all_conversations.jsonl ids do not exactly match union(train,val). "
            f"all={len(all_ids)} union={len(expected_all)}"
        )

    assistant_lengths: list[int] = []
    short_assistant_count = 0
    duplicate_assistant_count = 0
    seen_hashes: set[str] = set()

    for row in all_rows:
        messages = row.get("messages", [])
        if not isinstance(messages, list) or len(messages) < 3:
            continue
        assistant_msg = messages[2].get("content") if isinstance(messages[2], dict) else None
        if not isinstance(assistant_msg, str):
            continue

        length = len(assistant_msg.strip())
        assistant_lengths.append(length)
        if length < args.min_assistant_chars:
            short_assistant_count += 1

        h = hash12(assistant_msg.strip())
        if h in seen_hashes:
            duplicate_assistant_count += 1
        else:
            seen_hashes.add(h)

    total = max(1, len(all_rows))
    short_percent = (short_assistant_count / total) * 100.0
    if short_percent > args.max_short_assistant_percent:
        state.warn(
            "high share of short assistant responses: "
            f"{short_percent:.2f}% < {args.min_assistant_chars} chars"
        )

    duplicate_percent = (duplicate_assistant_count / len(all_rows) * 100) if all_rows else 0.0
    if duplicate_percent > args.max_duplicate_assistant_percent:
        state.warn(
            f"duplicate assistant responses detected: {duplicate_assistant_count} "
            f"({duplicate_percent:.1f}% > threshold {args.max_duplicate_assistant_percent}%)"
        )

    counts = {
        "all_records": len(all_rows),
        "train_records": len(train_rows),
        "val_records": len(val_rows),
        "train_blog_ids": len(train_blog_ids),
        "val_blog_ids": len(val_blog_ids),
    }

    metrics = {
        "assistant_length_chars": {
            "min": min(assistant_lengths) if assistant_lengths else None,
            "max": max(assistant_lengths) if assistant_lengths else None,
            "mean": round(statistics.mean(assistant_lengths), 2) if assistant_lengths else None,
            "median": round(statistics.median(assistant_lengths), 2) if assistant_lengths else None,
            "short_count": short_assistant_count,
            "short_percent": round(short_percent, 2),
            "min_assistant_chars": args.min_assistant_chars,
            "max_short_assistant_percent": args.max_short_assistant_percent,
        },
        "leakage": {
            "overlap_record_ids": len(overlap_ids),
            "overlap_blog_ids": len(overlap_blog_ids),
        },
        "duplicates": {
            "assistant_duplicates": duplicate_assistant_count,
        },
    }

    if state.errors:
        status = "failed"
        code = 1
    elif args.strict and state.warnings:
        status = "failed_strict"
        code = 2
    elif state.warnings:
        status = "passed_with_warnings"
        code = 0
    else:
        status = "passed"
        code = 0

    report = {
        "status": status,
        "validated_at": utc_now_iso(),
        "sft_dir": str(sft_dir),
        "config": {
            "min_assistant_chars": args.min_assistant_chars,
            "max_short_assistant_percent": args.max_short_assistant_percent,
            "strict": args.strict,
        },
        "counts": counts,
        "metrics": metrics,
        "errors": state.errors,
        "warnings": state.warnings,
    }
    return code, report


def print_summary(report: dict[str, Any]) -> None:
    counts = report.get("counts", {})
    metrics = report.get("metrics", {}).get("assistant_length_chars", {})

    print("SFT Dataset Validation")
    print("=" * 48)
    print(f"status             : {report.get('status')}")
    print(f"all records        : {counts.get('all_records')}")
    print(f"train records      : {counts.get('train_records')}")
    print(f"val records        : {counts.get('val_records')}")
    print(f"train blog_ids     : {counts.get('train_blog_ids')}")
    print(f"val blog_ids       : {counts.get('val_blog_ids')}")
    print(
        "assistant chars    : "
        f"min={metrics.get('min')} max={metrics.get('max')} "
        f"mean={metrics.get('mean')} median={metrics.get('median')}"
    )

    errors = report.get("errors", [])
    warnings = report.get("warnings", [])

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")


def main() -> int:
    args = parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    code, report = validate_dataset(args)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"sft_validation_{run_id}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_summary(report)
    print(f"\nreport_path        : {report_path}")

    return code


if __name__ == "__main__":
    raise SystemExit(main())
