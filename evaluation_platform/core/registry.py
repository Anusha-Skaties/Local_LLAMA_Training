"""
core/registry.py
----------------
Thread-safe evaluator and dataset-loader registries.

Design principle (Open/Closed):
  - Open for extension: call .register() to add any new evaluator/loader.
  - Closed for modification: no existing code needs to change.

Usage:
    # Register once at app startup (or use auto-discovery via decorators)
    EvaluatorRegistry.register(MyCustomEvaluator())

    # Retrieve by name anywhere in the codebase
    evaluator = EvaluatorRegistry.get("my_custom_evaluator")
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluation_platform.core.protocols import BaseDatasetLoader, BaseEvaluator

from evaluation_platform.core.exceptions import (
    EvaluatorNotFoundError,
    UnsupportedDatasetFormatError,
)


class EvaluatorRegistry:
    """
    Singleton registry of all BaseEvaluator implementations.

    Thread-safe: uses a module-level lock for writes; reads are safe because
    Python's GIL protects dict lookups.
    """

    _registry: dict[str, "BaseEvaluator"] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def register(cls, evaluator: "BaseEvaluator") -> None:
        """Register an evaluator.  Overwrites any previous registration."""
        with cls._lock:
            cls._registry[evaluator.name] = evaluator

    @classmethod
    def get(cls, name: str) -> "BaseEvaluator":
        try:
            return cls._registry[name]
        except KeyError:
            raise EvaluatorNotFoundError(name)

    @classmethod
    def all(cls) -> list["BaseEvaluator"]:
        return list(cls._registry.values())

    @classmethod
    def names(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """For testing only — clears all registrations."""
        with cls._lock:
            cls._registry.clear()


class DatasetLoaderRegistry:
    """Singleton registry of all BaseDatasetLoader implementations."""

    _registry: dict[str, "BaseDatasetLoader"] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def register(cls, loader: "BaseDatasetLoader", *formats: str) -> None:
        """Register a loader for one or more format strings."""
        with cls._lock:
            for fmt in formats:
                cls._registry[fmt.lower()] = loader

    @classmethod
    def get(cls, format_name: str) -> "BaseDatasetLoader":
        try:
            return cls._registry[format_name.lower()]
        except KeyError:
            raise UnsupportedDatasetFormatError(
                f"No dataset loader registered for format '{format_name}'. "
                f"Registered formats: {list(cls._registry.keys())}"
            )

    @classmethod
    def supported_formats(cls) -> list[str]:
        return list(cls._registry.keys())


def evaluator(name: str | None = None):
    """
    Class decorator that auto-registers an evaluator on import.

    Usage:
        @evaluator()
        class MyEvaluator(BaseEvaluator):
            ...
    """
    def decorator(cls):
        instance = cls()
        EvaluatorRegistry.register(instance)
        return cls
    return decorator
