"""
evaluate_model.py
-----------------
Post-training evaluation for a QLoRA SFT adapter.

Runs inference on val_conversations.jsonl user prompts, compares generated
assistant responses against reference responses using:
  - ROUGE-L (structural / lexical overlap)
  - Style cosine similarity (sentence-transformers, semantic proximity to your blog corpus)

Outputs:
  - data/meta/eval_runs/<run_id>.json   (structured report)
  - Logged to MLflow if --mlflow-experiment is set

Usage:
  python scripts/evaluate_model.py \\
    --adapter-dir outputs/qlora_sft/<run_name>/adapter \\
    --base-model meta-llama/Llama-3.2-3B-Instruct \\
    --val-file data/processed/sft/val_conversations.jsonl \\
    --corpus-file data/processed/blog_corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_VAL_FILE = PROJECT_ROOT / "data/processed/sft/val_conversations.jsonl"
DEFAULT_CORPUS_FILE = PROJECT_ROOT / "data/processed/blog_corpus.jsonl"
DEFAULT_EVAL_DIR = PROJECT_ROOT / "data/meta/eval_runs"

MAX_NEW_TOKENS = 1024
GENERATION_TEMPERATURE = 0.2
GENERATION_TOP_P = 0.9


# ── Optional MLflow ─────────────────────────────────────────────────────────────
try:
    import mlflow  # type: ignore
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def setup_logging(run_id: str) -> logging.Logger:
    logger = logging.getLogger("evaluate_model")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QLoRA SFT adapter.")
    parser.add_argument("--adapter-dir", required=True, help="Path to saved adapter directory.")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--val-file", default=str(DEFAULT_VAL_FILE))
    parser.add_argument("--corpus-file", default=str(DEFAULT_CORPUS_FILE))
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit evaluation to N samples.")
    parser.add_argument("--mlflow-experiment", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_user_and_reference(record: dict[str, Any]) -> tuple[list[dict], str]:
    """Return (prompt_messages_without_assistant, reference_assistant_text)."""
    messages = record["messages"]
    reference = ""
    prompt_messages = []
    for msg in messages:
        if msg["role"] == "assistant":
            reference = msg["content"]
        else:
            prompt_messages.append(msg)
    return prompt_messages, reference


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt_messages: list[dict],
    max_new_tokens: int,
    logger: logging.Logger,
) -> str:
    text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=GENERATION_TEMPERATURE,
            top_p=GENERATION_TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def compute_rouge_l(generated: str, reference: str) -> float:
    try:
        from rouge_score import rouge_scorer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("rouge-score is required. Install with: pip install rouge-score") from exc

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(reference, generated)
    return score["rougeL"].fmeasure


def compute_style_similarity(generated: str, corpus_texts: list[str]) -> float:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers and scikit-learn are required. "
            "Install with: pip install sentence-transformers scikit-learn"
        ) from exc

    model = SentenceTransformer("all-MiniLM-L6-v2")
    gen_emb = model.encode([generated])
    # Sample up to 50 corpus texts to keep latency reasonable
    sample = corpus_texts[:50]
    ref_emb = model.encode(sample)
    return float(cosine_similarity(gen_emb, ref_emb).mean())


@dataclass
class EvalReport:
    run_id: str
    evaluated_at: str
    adapter_dir: str
    base_model: str
    val_file: str
    num_samples: int
    rouge_l: dict[str, float]
    style_similarity: dict[str, float]
    per_sample: list[dict[str, Any]]


def evaluate(args: argparse.Namespace) -> int:
    run_id = make_run_id()
    logger = setup_logging(run_id)
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ── MLflow setup ────────────────────────────────────────────────────────────
    if args.mlflow_experiment and _MLFLOW_AVAILABLE:
        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=f"eval_{run_id}")
        logger.info("MLflow run started: experiment=%s", args.mlflow_experiment)

    # ── Load model + adapter ─────────────────────────────────────────────────────
    logger.info("Loading base model: %s", args.base_model)
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from peft import PeftModel  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization: reduces memory from ~6 GB to ~2 GB.
    # Safe for evaluation on an 8 GB MacBook (Apple M2 unified memory).
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
        import bitsandbytes  # noqa - just check it's installed
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quant_config,
            device_map="auto",
        )
        logger.info("Loaded in 4-bit quantization (~2 GB RAM)")
    except (ImportError, Exception):
        # bitsandbytes not available (e.g. Apple Silicon without MPS support) — fall back
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        logger.info("Loaded in float32 (bitsandbytes not available)")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()
    logger.info("Adapter loaded from: %s", args.adapter_dir)

    # ── Load data ────────────────────────────────────────────────────────────────
    val_records = read_jsonl(Path(args.val_file))
    corpus_records = read_jsonl(Path(args.corpus_file))
    corpus_texts = [r.get("text", r.get("content", "")) for r in corpus_records if r.get("text") or r.get("content")]

    if args.max_samples:
        val_records = val_records[: args.max_samples]

    logger.info("Evaluating %d samples...", len(val_records))

    # ── Run inference + scoring ──────────────────────────────────────────────────
    per_sample: list[dict[str, Any]] = []
    rouge_l_scores: list[float] = []
    style_scores: list[float] = []

    for i, record in enumerate(val_records, start=1):
        blog_id = record.get("blog_id", f"unknown_{i}")
        prompt_messages, reference = extract_user_and_reference(record)

        logger.info("[%d/%d] Generating for blog_id=%s", i, len(val_records), blog_id)
        generated = generate_response(model, tokenizer, prompt_messages, args.max_new_tokens, logger)

        rouge = compute_rouge_l(generated, reference)
        style = compute_style_similarity(generated, corpus_texts)

        rouge_l_scores.append(rouge)
        style_scores.append(style)

        per_sample.append({
            "blog_id": blog_id,
            "record_id": record.get("id", ""),
            "rouge_l": round(rouge, 4),
            "style_similarity": round(style, 4),
            "generated_chars": len(generated),
            "reference_chars": len(reference),
        })
        logger.info("  ROUGE-L=%.4f  style_sim=%.4f", rouge, style)

    # ── Aggregate ────────────────────────────────────────────────────────────────
    rouge_summary = {
        "mean": round(statistics.mean(rouge_l_scores), 4),
        "median": round(statistics.median(rouge_l_scores), 4),
        "min": round(min(rouge_l_scores), 4),
        "max": round(max(rouge_l_scores), 4),
    }
    style_summary = {
        "mean": round(statistics.mean(style_scores), 4),
        "median": round(statistics.median(style_scores), 4),
        "min": round(min(style_scores), 4),
        "max": round(max(style_scores), 4),
    }

    logger.info("ROUGE-L summary: %s", rouge_summary)
    logger.info("Style similarity summary: %s", style_summary)

    # ── Save report ──────────────────────────────────────────────────────────────
    report = EvalReport(
        run_id=run_id,
        evaluated_at=utc_now_iso(),
        adapter_dir=str(args.adapter_dir),
        base_model=args.base_model,
        val_file=str(args.val_file),
        num_samples=len(val_records),
        rouge_l=rouge_summary,
        style_similarity=style_summary,
        per_sample=per_sample,
    )
    report_path = eval_dir / f"eval_{run_id}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)
    logger.info("Eval report saved: %s", report_path)

    # ── Log to MLflow ────────────────────────────────────────────────────────────
    if _MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.log_metrics({
            "eval/rouge_l_mean": rouge_summary["mean"],
            "eval/rouge_l_median": rouge_summary["median"],
            "eval/style_similarity_mean": style_summary["mean"],
            "eval/style_similarity_median": style_summary["median"],
        })
        mlflow.log_artifact(str(report_path))
        mlflow.end_run()

    return 0


def main() -> int:
    args = parse_args()
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
