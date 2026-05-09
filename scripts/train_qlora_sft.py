"""
train_qlora_sft.py
------------------
Production-oriented QLoRA SFT training entrypoint for conversation JSONL data.

Expected input schema (one JSON object per line):
{
  "id": "...",
  "blog_id": "...",
  "split": "train|val",
  "format": "chatml_messages",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {...}
}

Usage example:
python scripts/train_qlora_sft.py \
  --model-name meta-llama/Llama-3.2-3B-Instruct \
  --train-file data/processed/sft/train_conversations.jsonl \
  --val-file data/processed/sft/val_conversations.jsonl \
  --output-root outputs/qlora_sft \
  --max-seq-length 2048 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

# ── Optional MLflow ────────────────────────────────────────────────────────────
try:
    import mlflow  # type: ignore
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_TRAIN_FILE = PROJECT_ROOT / "data/processed/sft/train_conversations.jsonl"
DEFAULT_VAL_FILE = PROJECT_ROOT / "data/processed/sft/val_conversations.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/qlora_sft"
DEFAULT_RUN_META_DIR = PROJECT_ROOT / "data/meta/training_runs"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ── MLflow helpers ────────────────────────────────────────────────────────────

def _start_mlflow_run(
    experiment: str | None,
    tracking_uri: str | None,
    run_name: str,
    logger: logging.Logger,
) -> None:
    if not experiment or not _MLFLOW_AVAILABLE:
        return
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    mlflow.start_run(run_name=run_name)
    logger.info("MLflow run started: experiment=%s run_name=%s", experiment, run_name)


def _mlflow_log_params(args: "Args") -> None:
    if not _MLFLOW_AVAILABLE or not mlflow.active_run():
        return
    mlflow.log_params({
        "model_name": args.model_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "seed": args.seed,
        "use_4bit": args.use_4bit,
        "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "optim": args.optim,
    })


def _mlflow_log_metrics(
    train_metrics: dict,
    eval_metrics: dict,
    trainable_params: int,
    total_params: int,
) -> None:
    if not _MLFLOW_AVAILABLE or not mlflow.active_run():
        return
    loggable: dict[str, float] = {}
    for k, v in {**train_metrics, **eval_metrics}.items():
        if isinstance(v, (int, float)):
            loggable[k] = float(v)
    loggable["trainable_params"] = float(trainable_params)
    loggable["total_params"] = float(total_params)
    loggable["trainable_param_ratio"] = trainable_params / max(1, total_params)
    mlflow.log_metrics(loggable)


def _mlflow_log_artifact(path: Path) -> None:
    if not _MLFLOW_AVAILABLE or not mlflow.active_run():
        return
    mlflow.log_artifact(str(path))


def _end_mlflow_run() -> None:
    if not _MLFLOW_AVAILABLE or not mlflow.active_run():
        return
    mlflow.end_run()


def parse_dtype(name: str) -> torch.dtype:
    value = name.strip().lower()
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def parse_target_modules(raw: str) -> list[str]:
    modules = [p.strip() for p in raw.split(",") if p.strip()]
    if not modules:
        raise ValueError("--lora-target-modules cannot be empty")
    return modules


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_qlora_sft")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


@dataclass
class RunInfo:
    run_id: str
    run_name: str
    started_at: str
    finished_at: str
    model_name: str
    tokenizer_name: str
    output_dir: str
    train_file: str
    val_file: str
    train_rows: int
    val_rows: int
    trainable_params: int
    total_params: int
    eval_metrics: dict[str, Any]
    train_metrics: dict[str, Any]
    args: dict[str, Any]


class Args(argparse.Namespace):
    model_name: str
    tokenizer_name: str | None
    train_file: str
    val_file: str
    output_root: str
    run_name: str | None
    run_meta_dir: str
    seed: int

    max_seq_length: int
    packing: bool

    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    max_steps: int

    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float

    lr_scheduler_type: str
    optim: str

    eval_strategy: str
    eval_steps: int
    save_steps: int
    save_total_limit: int
    logging_steps: int

    gradient_checkpointing: bool
    group_by_length: bool
    dataloader_num_workers: int

    report_to: str

    use_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: str
    bnb_4bit_use_double_quant: bool

    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: str
    lora_bias: str
    lora_task_type: str

    trust_remote_code: bool
    mlflow_experiment: str | None
    mlflow_tracking_uri: str | None
    push_to_hub: bool
    hf_hub_repo_id: str | None


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Train QLoRA SFT model on conversation JSONL data.")

    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--val-file", default=str(DEFAULT_VAL_FILE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-meta-dir", default=str(DEFAULT_RUN_META_DIR))
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--packing", action="store_true")

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)

    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)

    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--optim", default="paged_adamw_8bit")

    parser.add_argument("--eval-strategy", default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=5)

    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--group-by-length", action="store_true", default=True)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)

    parser.add_argument("--report-to", default="none", help="none,tensorboard,wandb or comma-separated list")

    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb-4bit-compute-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true", default=True)

    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--lora-bias", default="none", choices=["none", "all", "lora_only"])
    parser.add_argument("--lora-task-type", default="CAUSAL_LM")

    parser.add_argument("--trust-remote-code", action="store_true", default=False)

    parser.add_argument("--push-to-hub", action="store_true", default=False,
                        help="Push trained adapter to HuggingFace Hub after training.")
    parser.add_argument("--hf-hub-repo-id", default=None,
                        help="HuggingFace Hub repo ID (e.g. username/llama-blog-sft). Required when --push-to-hub is set.")

    parser.add_argument(
        "--mlflow-experiment",
        default=None,
        help="MLflow experiment name. Enables MLflow tracking when set. On Azure ML, tracking URI is injected automatically.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="MLflow tracking URI. Falls back to MLFLOW_TRACKING_URI env var if not set.",
    )

    return parser.parse_args(namespace=Args())


def validate_paths(args: Args) -> tuple[Path, Path, Path, Path, Path]:
    train_path = Path(args.train_file)
    val_path = Path(args.val_file)
    output_root = Path(args.output_root)
    run_meta_dir = Path(args.run_meta_dir)

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val file not found: {val_path}")

    run_id = make_run_id()
    run_name = args.run_name or f"qlora_sft_{run_id}"
    output_dir = output_root / run_name
    log_path = output_dir / "train.log"

    output_dir.mkdir(parents=True, exist_ok=True)
    run_meta_dir.mkdir(parents=True, exist_ok=True)

    return train_path, val_path, output_dir, run_meta_dir, log_path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def to_chat_text(messages: list[dict[str, Any]], tokenizer: AutoTokenizer) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def prepare_dataset(train_file: Path, val_file: Path, tokenizer: AutoTokenizer, logger: logging.Logger) -> DatasetDict:
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(val_file),
        },
    )

    def map_row(row: dict[str, Any]) -> dict[str, Any]:
        messages = row.get("messages", [])
        text = to_chat_text(messages, tokenizer)
        return {"text": text}

    logger.info("Formatting conversations with tokenizer chat template...")
    dataset = dataset.map(map_row, desc="Apply chat template")

    return dataset


def count_trainable_params(model: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    return trainable, total


def parse_report_to(value: str) -> list[str] | str:
    raw = value.strip().lower()
    if raw == "none":
        return "none"
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts if parts else "none"


def train(args: Args) -> int:
    train_path, val_path, output_dir, run_meta_dir, log_path = validate_paths(args)
    logger = setup_logging(log_path)

    run_id = make_run_id()
    run_name = output_dir.name
    started_at = utc_now_iso()
    _start_mlflow_run(args.mlflow_experiment, args.mlflow_tracking_uri, run_name, logger)

    logger.info("Run name: %s", run_name)
    logger.info("Output dir: %s", output_dir)
    logger.info("Train file: %s", train_path)
    logger.info("Val file: %s", val_path)

    seed_everything(args.seed)
    _mlflow_log_params(args)

    tokenizer_name = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=args.trust_remote_code)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = args.use_4bit
    if use_4bit and not torch.cuda.is_available():
        logger.warning("CUDA not available; disabling 4-bit quantization for this run.")
        use_4bit = False

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=parse_dtype(args.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    # Recommended for many decoder-only training runs.
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type=args.lora_task_type,
        target_modules=parse_target_modules(args.lora_target_modules),
    )

    dataset = prepare_dataset(train_path, val_path, tokenizer, logger)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        group_by_length=args.group_by_length,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=parse_report_to(args.report_to),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        processing_class=tokenizer,
        dataset_text_field="text",
    )

    trainable_params, total_params = count_trainable_params(trainer.model)
    logger.info("Trainable params: %d", trainable_params)
    logger.info("Total params: %d", total_params)

    train_result = trainer.train()
    train_metrics = dict(train_result.metrics)

    eval_metrics: dict[str, Any] = {}
    if args.eval_strategy != "no":
        eval_metrics = dict(trainer.evaluate())
    _mlflow_log_metrics(train_metrics, eval_metrics, trainable_params, total_params)

    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))

    if args.push_to_hub:
        if not args.hf_hub_repo_id:
            logger.warning("--push-to-hub set but --hf-hub-repo-id not provided; skipping push.")
        else:
            logger.info("Pushing adapter to HuggingFace Hub: %s", args.hf_hub_repo_id)
            trainer.model.push_to_hub(args.hf_hub_repo_id)
            tokenizer.push_to_hub(args.hf_hub_repo_id)
            logger.info("Push complete: https://huggingface.co/%s", args.hf_hub_repo_id)

    # Save trainer state and args snapshot for reproducibility.
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))
    with (output_dir / "training_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    finished_at = utc_now_iso()

    run_info = RunInfo(
        run_id=run_id,
        run_name=run_name,
        started_at=started_at,
        finished_at=finished_at,
        model_name=args.model_name,
        tokenizer_name=tokenizer_name,
        output_dir=str(output_dir),
        train_file=str(train_path),
        val_file=str(val_path),
        train_rows=len(dataset["train"]),
        val_rows=len(dataset["validation"]),
        trainable_params=trainable_params,
        total_params=total_params,
        eval_metrics=eval_metrics,
        train_metrics=train_metrics,
        args=dict(vars(args)),
    )

    run_meta_path = run_meta_dir / f"{run_name}.json"
    with run_meta_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(run_info), f, ensure_ascii=False, indent=2)

    logger.info("Training complete.")
    logger.info("Adapter saved to: %s", output_dir / "adapter")
    logger.info("Run metadata: %s", run_meta_path)
    _mlflow_log_artifact(run_meta_path)
    _end_mlflow_run()

    return 0


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
