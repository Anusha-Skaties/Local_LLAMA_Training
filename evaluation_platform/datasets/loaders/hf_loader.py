"""
datasets/loaders/hf_loader.py
------------------------------
HuggingFace datasets loader.

Supports any HF dataset via datasets.load_dataset().  Column mapping is
configurable so any dataset schema can be adapted to EvalSample.

Example config:
    DatasetConfig(
        name="squad_v2",
        format="hf",
        hf_dataset_name="squad_v2",
        split="validation",
        extra={
            "column_map": {
                "input":     "question",
                "reference": "answers.text[0]",
                "contexts":  "context"
            }
        }
    )
"""
from __future__ import annotations

import random
from typing import Any

from evaluation_platform.core.exceptions import DatasetLoadError
from evaluation_platform.core.protocols import BaseDatasetLoader
from evaluation_platform.core.registry import DatasetLoaderRegistry
from evaluation_platform.core.schemas import DatasetConfig, EvalSample, TaskType
from evaluation_platform.logging_.structured import get_logger

log = get_logger(__name__)


def _resolve_column(record: dict[str, Any], column_path: str) -> Any:
    """
    Resolve a dotted column path from a record dict.

    Examples:
        "question"           → record["question"]
        "answers.text[0]"    → record["answers"]["text"][0]
    """
    # Handle list index notation: answers.text[0]
    import re
    parts = re.split(r"\.|\[(\d+)\]", column_path)
    current: Any = record
    for part in parts:
        if part is None or part == "":
            continue
        if part.isdigit():
            try:
                current = current[int(part)]
            except (IndexError, TypeError):
                return None
        else:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
    return current


class HuggingFaceDatasetLoader(BaseDatasetLoader):
    """
    Loads evaluation samples from any HuggingFace Hub dataset.

    Column mapping is fully configurable via config.extra["column_map"].
    Default column names: input, output, reference, contexts.
    """

    def load(self, config: DatasetConfig) -> list[EvalSample]:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise DatasetLoadError(
                "datasets package not installed. Run: pip install datasets"
            ) from exc

        dataset_name = config.hf_dataset_name or config.path
        if not dataset_name:
            raise DatasetLoadError(
                "HF loader requires either hf_dataset_name or path to be set."
            )

        log.info(
            "Loading HuggingFace dataset",
            name=dataset_name,
            split=config.split,
        )

        try:
            dataset = load_dataset(dataset_name, split=config.split, trust_remote_code=True)
        except Exception as exc:
            raise DatasetLoadError(
                f"Failed to load HF dataset '{dataset_name}': {exc}"
            ) from exc

        column_map: dict[str, str] = config.extra.get(
            "column_map",
            {
                "input": "input",
                "output": "output",
                "reference": "reference",
                "contexts": "contexts",
            },
        )
        task_type = TaskType(config.extra.get("task_type", TaskType.GENERATION))

        records: list[dict[str, Any]] = [dict(row) for row in dataset]

        if config.shuffle:
            rng = random.Random(config.shuffle_seed)
            rng.shuffle(records)

        if config.max_samples is not None:
            records = records[: config.max_samples]

        samples: list[EvalSample] = []
        for i, record in enumerate(records):
            input_text = _resolve_column(record, column_map.get("input", "input"))
            if not input_text:
                log.warning("Skipping record with empty input", index=i)
                continue

            reference = _resolve_column(record, column_map.get("reference", "reference"))
            contexts_raw = _resolve_column(record, column_map.get("contexts", "contexts"))
            contexts: list[str] = (
                [contexts_raw] if isinstance(contexts_raw, str)
                else (contexts_raw or [])
            )

            samples.append(
                EvalSample(
                    id=str(record.get("id", f"{dataset_name}_{i}")),
                    task_type=task_type,
                    input=str(input_text),
                    output=_resolve_column(record, column_map.get("output", "output")),
                    reference=str(reference) if reference else None,
                    contexts=contexts,
                    metadata={
                        k: v for k, v in record.items()
                        if k not in column_map.values()
                    },
                )
            )

        return samples

    def supports(self, format_name: str) -> bool:
        return format_name.lower() in ("hf", "huggingface", "hugging_face")


# ── CSV / Parquet loaders ─────────────────────────────────────────────────────

class CsvDatasetLoader(BaseDatasetLoader):
    """Loads samples from a CSV file using pandas."""

    def load(self, config: DatasetConfig) -> list[EvalSample]:
        from pathlib import Path
        import pandas as pd
        path = Path(config.path)
        if not path.exists():
            raise DatasetLoadError(f"CSV file not found: {path}")

        df = pd.read_csv(path)
        task_type = TaskType(config.extra.get("task_type", TaskType.GENERATION))

        if config.max_samples is not None:
            df = df.head(config.max_samples)

        samples: list[EvalSample] = []
        for _, row in df.iterrows():
            record = row.to_dict()
            input_text = record.get("input", "")
            if not input_text:
                continue
            samples.append(
                EvalSample(
                    id=str(record.get("id", "")),
                    task_type=task_type,
                    input=str(input_text),
                    output=str(record["output"]) if record.get("output") else None,
                    reference=str(record["reference"]) if record.get("reference") else None,
                    contexts=[record["contexts"]] if record.get("contexts") else [],
                    metadata={k: v for k, v in record.items()
                               if k not in ("id", "input", "output", "reference", "contexts")},
                )
            )
        return samples

    def supports(self, format_name: str) -> bool:
        return format_name.lower() == "csv"


class ParquetDatasetLoader(BaseDatasetLoader):
    """Loads samples from a Parquet file using pandas + pyarrow."""

    def load(self, config: DatasetConfig) -> list[EvalSample]:
        from pathlib import Path
        import pandas as pd
        path = Path(config.path)
        if not path.exists():
            raise DatasetLoadError(f"Parquet file not found: {path}")

        df = pd.read_parquet(path)
        # Reuse CSV logic via a fake CSV config
        csv_loader = CsvDatasetLoader()
        # Save to temp CSV … no, just iterate the df directly
        task_type = TaskType(config.extra.get("task_type", TaskType.GENERATION))
        if config.max_samples is not None:
            df = df.head(config.max_samples)

        samples: list[EvalSample] = []
        for _, row in df.iterrows():
            record = row.to_dict()
            input_text = record.get("input", "")
            if not input_text:
                continue
            samples.append(
                EvalSample(
                    id=str(record.get("id", "")),
                    task_type=task_type,
                    input=str(input_text),
                    output=str(record["output"]) if record.get("output") else None,
                    reference=str(record["reference"]) if record.get("reference") else None,
                    contexts=[record["contexts"]] if record.get("contexts") else [],
                    metadata={},
                )
            )
        return samples

    def supports(self, format_name: str) -> bool:
        return format_name.lower() in ("parquet", "parq")


# Register all at import time
DatasetLoaderRegistry.register(HuggingFaceDatasetLoader(), "hf", "huggingface", "hugging_face")
DatasetLoaderRegistry.register(CsvDatasetLoader(), "csv")
DatasetLoaderRegistry.register(ParquetDatasetLoader(), "parquet", "parq")
