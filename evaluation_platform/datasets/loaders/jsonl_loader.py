"""
datasets/loaders/jsonl_loader.py
---------------------------------
JSONL dataset loader.

Handles two schemas:
  1. **SFT conversations format** (your existing data):
         {"id": "...", "messages": [...], "metadata": {...}}
     Extracts: input = user message, reference = assistant message

  2. **Flat evaluation format**:
         {"id": "...", "input": "...", "output": "...", "reference": "...",
          "contexts": [...], "metadata": {...}}

Auto-detects schema from the first record's keys.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from evaluation_platform.core.exceptions import DatasetFormatError
from evaluation_platform.core.protocols import BaseDatasetLoader
from evaluation_platform.core.registry import DatasetLoaderRegistry
from evaluation_platform.core.schemas import DatasetConfig, EvalSample, Message, TaskType
from evaluation_platform.logging_.structured import get_logger

log = get_logger(__name__)


def _extract_sft_sample(record: dict[str, Any], task_type: TaskType) -> EvalSample:
    """
    Convert an SFT-conversations record into an EvalSample.

    Schema:
        {
          "id": "blog_006_sft_01",
          "messages": [
              {"role": "system", "content": "..."},
              {"role": "user",   "content": "..."},   ← input
              {"role": "assistant", "content": "..."}  ← reference
          ],
          "metadata": {...}
        }
    """
    messages_raw: list[dict[str, Any]] = record.get("messages", [])
    messages = [Message(role=m["role"], content=m["content"]) for m in messages_raw]

    # Build prompt messages (everything except assistant turn)
    input_text = ""
    reference_text = ""
    for msg in messages_raw:
        if msg["role"] == "user":
            input_text = msg["content"]
        elif msg["role"] == "assistant":
            reference_text = msg["content"]

    if not input_text:
        raise DatasetFormatError(
            f"Record '{record.get('id', '?')}' has no user message."
        )

    metadata = {k: v for k, v in record.items() if k not in ("id", "messages")}

    return EvalSample(
        id=str(record.get("id", "")),
        task_type=task_type,
        input=input_text,
        output=None,           # filled later during inference
        reference=reference_text or None,
        messages=messages,
        metadata=metadata,
    )


def _extract_flat_sample(record: dict[str, Any], task_type: TaskType) -> EvalSample:
    """Convert a flat-format record into an EvalSample."""
    return EvalSample(
        id=str(record.get("id", "")),
        task_type=task_type,
        input=record["input"],
        output=record.get("output"),
        reference=record.get("reference"),
        contexts=record.get("contexts", []),
        metadata=record.get("metadata", {}),
    )


def _detect_schema(record: dict[str, Any]) -> str:
    """Return 'sft' or 'flat' based on the first record's keys."""
    if "messages" in record:
        return "sft"
    if "input" in record:
        return "flat"
    raise DatasetFormatError(
        f"Cannot detect schema from record keys: {list(record.keys())}. "
        "Expected either 'messages' (SFT format) or 'input' (flat format)."
    )


class JsonlDatasetLoader(BaseDatasetLoader):
    """
    Loads evaluation samples from JSONL files.

    Supports both the project's native SFT conversation format and a generic
    flat evaluation format.  Schema is auto-detected from the first record.
    """

    def load(self, config: DatasetConfig) -> list[EvalSample]:
        path = Path(config.path)
        if not path.exists():
            from evaluation_platform.core.exceptions import DatasetLoadError
            raise DatasetLoadError(f"JSONL file not found: {path}")

        task_type = TaskType(config.extra.get("task_type", TaskType.GENERATION))

        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    log.warning(
                        "Skipping malformed JSONL line",
                        file=str(path),
                        line=line_no,
                        error=str(exc),
                    )

        if not records:
            return []

        schema = _detect_schema(records[0])
        log.info("JSONL schema detected", schema=schema, file=str(path))

        # Apply filters if configured
        if config.filters:
            records = [
                r for r in records
                if all(r.get(k) == v for k, v in config.filters.items())
            ]

        # Shuffle before slicing so max_samples is representative
        if config.shuffle:
            rng = random.Random(config.shuffle_seed)
            rng.shuffle(records)

        if config.max_samples is not None:
            records = records[: config.max_samples]

        extractor = _extract_sft_sample if schema == "sft" else _extract_flat_sample
        samples: list[EvalSample] = []
        for record in records:
            try:
                samples.append(extractor(record, task_type))
            except DatasetFormatError as exc:
                log.warning("Skipping invalid record", error=str(exc))

        return samples

    def supports(self, format_name: str) -> bool:
        return format_name.lower() in ("jsonl", "json_lines")


# Register at import time
DatasetLoaderRegistry.register(JsonlDatasetLoader(), "jsonl", "json_lines")
