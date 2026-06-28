"""
generate_blog.py
----------------
Production-oriented inference entrypoint for blog generation.

Supports either:
1) Fully merged model directory (--model-dir), or
2) QLoRA adapter directory + base model (--adapter-dir + --base-model)

Examples:
python scripts/generate_blog.py \
  --adapter-dir outputs/qlora_sft/my_run/adapter \
  --base-model meta-llama/Llama-3.2-3B-Instruct \
  --prompt "Write a blog about evaluating LLM applications in production" \
  --output-file outputs/generated/blog.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs/named-outputs/model_output"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "outputs/generated/blog.txt"


def setup_logging(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("generate_blog")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers = []
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate blogs with a trained model or QLoRA adapter.")

    source = parser.add_argument_group("model source")
    source.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Path to a fully merged model directory.")
    source.add_argument("--adapter-dir", default=None, help="Path to a PEFT/QLoRA adapter directory.")
    source.add_argument("--base-model", default=None, help="Base model name/path (required when --adapter-dir is used).")

    prompt_group = parser.add_argument_group("prompt")
    prompt_group.add_argument("--prompt", default=None, help="Prompt text for blog generation.")
    prompt_group.add_argument("--prompt-file", default=None, help="Path to a UTF-8 text file containing prompt text.")
    prompt_group.add_argument("--system-prompt", default="You are an expert technical blog writer.")

    gen = parser.add_argument_group("generation")
    gen.add_argument("--max-new-tokens", type=int, default=1024)
    gen.add_argument("--temperature", type=float, default=0.7)
    gen.add_argument("--top-p", type=float, default=0.9)
    gen.add_argument("--top-k", type=int, default=50)
    gen.add_argument("--repetition-penalty", type=float, default=1.1)
    gen.add_argument("--num-return-sequences", type=int, default=1)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--device-map", default="auto", help="Transformers device_map value. Use 'cpu' for CPU-only.")
    runtime.add_argument("--trust-remote-code", action="store_true")
    runtime.add_argument("--verbose", action="store_true")

    output = parser.add_argument_group("output")
    output.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE), help="Where to save generated blog text.")
    output.add_argument("--metadata-file", default=None, help="Optional path to save generation metadata as JSON.")

    args = parser.parse_args()
    validate_args(args, parser)
    return args


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.adapter_dir and not args.base_model:
        parser.error("--base-model is required when --adapter-dir is provided.")

    if args.prompt and args.prompt_file:
        parser.error("Use either --prompt or --prompt-file, not both.")

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be > 0.")

    if args.num_return_sequences <= 0:
        parser.error("--num-return-sequences must be > 0.")

    if args.temperature < 0:
        parser.error("--temperature must be >= 0.")

    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1].")

    if args.top_k < 0:
        parser.error("--top-k must be >= 0.")

    if args.repetition_penalty <= 0:
        parser.error("--repetition-penalty must be > 0.")


def parse_dtype(dtype_name: str) -> torch.dtype | None:
    if dtype_name == "auto":
        return None
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt.strip()

    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file does not exist: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8").strip()

    return "Write a detailed technical blog post about building production-ready LLM systems."


def load_model_and_tokenizer(args: argparse.Namespace, logger: logging.Logger) -> tuple[Any, Any]:
    torch_dtype = parse_dtype(args.dtype)

    if args.adapter_dir:
        adapter_dir = Path(args.adapter_dir)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

        logger.info("Loading tokenizer from adapter directory...")
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=args.trust_remote_code)

        logger.info("Loading base model: %s", args.base_model)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch_dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )

        logger.info("Attaching adapter: %s", adapter_dir)
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    else:
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

        logger.info("Loading tokenizer from model directory...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=args.trust_remote_code)

        logger.info("Loading model from: %s", model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch_dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def build_model_input(tokenizer: Any, system_prompt: str, prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"


def generate_blog(args: argparse.Namespace, model: Any, tokenizer: Any, prompt: str) -> list[str]:
    model_input = build_model_input(tokenizer, args.system_prompt, prompt)
    inputs = tokenizer(model_input, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    do_sample = args.temperature > 0

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            top_k=args.top_k if do_sample else None,
            repetition_penalty=args.repetition_penalty,
            num_return_sequences=args.num_return_sequences,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = input_ids.shape[1]
    generations: list[str] = []
    for row in output_ids:
        generated_ids = row[prompt_len:]
        generations.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return generations


def write_outputs(args: argparse.Namespace, generations: list[str], metadata: dict[str, Any]) -> None:
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(generations), encoding="utf-8")

    if args.metadata_file:
        metadata_path = Path(args.metadata_file)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.verbose)
    set_global_seed(args.seed)

    try:
        prompt = resolve_prompt(args)
        model, tokenizer = load_model_and_tokenizer(args, logger)
        logger.info("Generating blog content...")
        generations = generate_blog(args, model, tokenizer, prompt)

        metadata = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": {
                "model_dir": args.model_dir,
                "adapter_dir": args.adapter_dir,
                "base_model": args.base_model,
            },
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                "num_return_sequences": args.num_return_sequences,
                "seed": args.seed,
            },
            "prompt_chars": len(prompt),
            "outputs": [
                {"index": i + 1, "chars": len(text)}
                for i, text in enumerate(generations)
            ],
        }

        write_outputs(args, generations, metadata)
        logger.info("Generated %d blog output(s).", len(generations))
        logger.info("Saved blog text to: %s", args.output_file)
        if args.metadata_file:
            logger.info("Saved metadata to: %s", args.metadata_file)
        return 0
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())