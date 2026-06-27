"""
datasets/base.py
----------------
Dataset loading facade.

The DatasetManager is the single entry point for loading data.
It delegates to the appropriate registered loader based on config.format.

Usage:
    manager = DatasetManager()
    samples = manager.load(DatasetConfig(name="val", format="jsonl",
                                         path="data/processed/sft/val_conversations.jsonl"))
"""
from __future__ import annotations

from evaluation_platform.core.exceptions import DatasetLoadError
from evaluation_platform.core.protocols import BaseDatasetLoader
from evaluation_platform.core.registry import DatasetLoaderRegistry
from evaluation_platform.core.schemas import DatasetConfig, EvalSample
from evaluation_platform.logging_.structured import get_logger

log = get_logger(__name__)


class DatasetManager:
    """
    Facade that routes load() calls to the correct registered loader.

    Loaders are registered at import time via DatasetLoaderRegistry.
    """

    def load(self, config: DatasetConfig) -> list[EvalSample]:
        """
        Load samples using the loader registered for config.format.

        Raises:
            DatasetLoadError: if loading fails for any reason.
        """
        loader: BaseDatasetLoader = DatasetLoaderRegistry.get(config.format)

        log.info(
            "Loading dataset",
            name=config.name,
            format=config.format,
            path=config.path or config.hf_dataset_name,
            max_samples=config.max_samples,
        )
        try:
            samples = loader.load(config)
        except Exception as exc:
            raise DatasetLoadError(
                f"Failed to load dataset '{config.name}' ({config.format}): {exc}"
            ) from exc

        log.info(
            "Dataset loaded",
            name=config.name,
            sample_count=len(samples),
        )
        return samples
