"""
build_sft_dataset.py
--------------------
Create conversation-style SFT JSONL files from blog_corpus.jsonl.

Input:
  - data/processed/blog_corpus.jsonl (one full blog per record)

Output:
  - data/processed/sft/train_conversations.jsonl
  - data/processed/sft/val_conversations.jsonl
  - data/processed/sft/all_conversations.jsonl
  - data/meta/sft_runs/<run_id>.json

Usage:
  python scripts/build_sft_dataset.py
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PROCESSED_DIR = PROJECT_ROOT / "data/processed"
META_DIR = PROJECT_ROOT / "data/meta"
SFT_DIR = PROCESSED_DIR / "sft"
SFT_RUNS_DIR = META_DIR / "sft_runs"

INPUT_CORPUS_PATH = PROCESSED_DIR / "blog_corpus.jsonl"
VAL_PERCENT = 15

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert technical blog writer. Follow the requested style exactly, "
    "be clear and practical, and use clean structure with meaningful headings."
)

STYLE_RULES = [
    "Use a practical, concise, and founder-friendly tone.",
    "Start with a sharp hook and explain why the topic matters.",
    "Use clear section headings and short paragraphs.",
    "Use real-world examples and actionable guidance.",
    "End with a crisp takeaway.",
]

PROMPT_TEMPLATES = [
    (
        "topic_style_brief",
        "Write a technical blog on '{title}'. "
        "Audience: engineering leaders and builders. "
        "Style rules: {style_rules}. "
        "Length target: 700-1000 words.",
    ),
    (
        "audience_outcome",
        "Create a blog post about '{title}' for startup CTOs. "
        "Make it practical and execution-focused. "
        "Style rules: {style_rules}. "
        "Include clear section headings and concrete examples.",
    ),
    (
        "structured_playbook",
        "Draft a production-grade explainer on '{title}'. "
        "Use this structure: hook, core concept, common mistakes, real-world example, checklist, conclusion. "
        "Style rules: {style_rules}.",
    ),
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stable_int_hash(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16)


def split_from_blog_id(blog_id: str, val_percent: int) -> str:
    bucket = stable_int_hash(blog_id) % 100
    return "val" if bucket < val_percent else "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conversation SFT dataset from blog corpus.")
    parser.add_argument("--input-corpus", default=str(INPUT_CORPUS_PATH))
    parser.add_argument("--output-dir", default=str(SFT_DIR))
    parser.add_argument("--val-percent", type=int, default=VAL_PERCENT)
    parser.add_argument("--variants-per-blog", type=int, default=3)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_user_prompt(title: str, template_text: str) -> str:
    style_rules = " ".join(STYLE_RULES)
    return template_text.format(title=title, style_rules=style_rules)


def build_records(
    corpus_rows: list[dict[str, Any]],
    val_percent: int,
    system_prompt: str,
    variants_per_blog: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    template_count = min(len(PROMPT_TEMPLATES), max(1, variants_per_blog))
    chosen_templates = PROMPT_TEMPLATES[:template_count]

    for row in corpus_rows:
        blog_id = str(row.get("id", ""))
        title = str(row.get("title", "")).strip() or blog_id
        assistant_text = str(row.get("text", "")).strip()
        source_file = str(row.get("source_file", ""))
        source_hash = str(row.get("hash", ""))
        pipeline_version = str(row.get("pipeline_version", ""))

        if not blog_id or not assistant_text:
            continue

        split = split_from_blog_id(blog_id, val_percent)

        for idx, (template_name, template_text) in enumerate(chosen_templates, start=1):
            user_prompt = build_user_prompt(title, template_text)
            rec_id = f"{blog_id}_sft_{idx:02d}"

            record = {
                "id": rec_id,
                "blog_id": blog_id,
                "title": title,
                "split": split,
                "format": "chatml_messages",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_text},
                ],
                "metadata": {
                    "prompt_template": template_name,
                    "source_file": source_file,
                    "source_hash": source_hash,
                    "source_pipeline_version": pipeline_version,
                    "created_at": utc_now_iso(),
                },
            }
            records.append(record)

    return records


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_corpus)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise SystemExit(f"Input corpus not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    SFT_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    corpus_rows = read_jsonl(input_path)
    sft_rows = build_records(
        corpus_rows=corpus_rows,
        val_percent=args.val_percent,
        system_prompt=args.system_prompt,
        variants_per_blog=args.variants_per_blog,
    )

    train_rows = [r for r in sft_rows if r["split"] == "train"]
    val_rows = [r for r in sft_rows if r["split"] == "val"]

    write_jsonl(output_dir / "all_conversations.jsonl", sft_rows)
    write_jsonl(output_dir / "train_conversations.jsonl", train_rows)
    write_jsonl(output_dir / "val_conversations.jsonl", val_rows)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_meta = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "input_corpus": str(input_path),
        "output_dir": str(output_dir),
        "val_percent": args.val_percent,
        "variants_per_blog": args.variants_per_blog,
        "system_prompt": args.system_prompt,
        "prompt_templates": [name for name, _ in PROMPT_TEMPLATES[: max(1, args.variants_per_blog)]],
        "counts": {
            "corpus_documents": len(corpus_rows),
            "all_records": len(sft_rows),
            "train_records": len(train_rows),
            "val_records": len(val_rows),
        },
    }

    with (SFT_RUNS_DIR / f"{run_id}.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print("SFT dataset build complete")
    print("-" * 48)
    print(f"input docs      : {len(corpus_rows)}")
    print(f"all records     : {len(sft_rows)}")
    print(f"train records   : {len(train_rows)}")
    print(f"val records     : {len(val_rows)}")
    print(f"output dir      : {output_dir}")
    print(f"run metadata    : {SFT_RUNS_DIR / f'{run_id}.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
