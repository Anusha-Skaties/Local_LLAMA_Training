"""
core/protocols.py
-----------------
Abstract base classes (protocols) that define the contracts every component
must satisfy. Concrete implementations live in the feature subdirectories.

Design principle (SOLID – Interface Segregation + Dependency Inversion):
  - Every interface is small and focused.
  - High-level modules (pipelines, API) depend only on these abstractions.
  - Adding a new evaluator / dataset loader / reporter = implement the protocol,
    register it → zero changes to existing code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from evaluation_platform.core.schemas import (
    DatasetConfig,
    EvalRunResult,
    EvalSample,
    EvaluatorConfig,
    EvaluatorResult,
    ExperimentConfig,
)


class BaseDatasetLoader(ABC):
    """
    Contract for all dataset loaders.

    A loader is responsible for reading a data source and returning a
    flat list of EvalSample objects that the pipeline can iterate over.
    """

    @abstractmethod
    def load(self, config: DatasetConfig) -> list[EvalSample]:
        """Load samples from the configured source."""
        ...

    @abstractmethod
    def supports(self, format_name: str) -> bool:
        """Return True if this loader handles the given format string."""
        ...


class BaseEvaluator(ABC):
    """
    Contract for all evaluators (lexical, semantic, LLM-judge, RAG, safety…).

    Design notes:
      - evaluate_sample()  → synchronous single-sample evaluation
      - evaluate_batch()   → optional override for batched evaluation
        (default: loops over evaluate_sample; override for true batching)
      - name and version   → used in reports, registries, and tracing
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique evaluator identifier, e.g. 'rouge_lexical'."""
        ...

    @property
    def version(self) -> str:
        """Semantic version string.  Override when the evaluator logic changes."""
        return "1.0.0"

    @abstractmethod
    def evaluate_sample(
        self,
        sample: EvalSample,
        config: EvaluatorConfig,
    ) -> EvaluatorResult:
        """Evaluate one sample and return all metric results."""
        ...

    def evaluate_batch(
        self,
        samples: list[EvalSample],
        config: EvaluatorConfig,
    ) -> list[EvaluatorResult]:
        """
        Evaluate a list of samples.

        The default implementation is a simple loop; override this when the
        underlying library supports true batch evaluation (e.g. BERTScore).
        """
        return [self.evaluate_sample(s, config) for s in samples]

    def is_applicable(self, sample: EvalSample) -> bool:
        """
        Return True if this evaluator can be applied to the given sample.

        Default: True when output is non-empty.  Override for evaluators
        that require contexts, reference text, etc.
        """
        return bool(sample.output)


class BaseExperimentTracker(ABC):
    """Contract for experiment tracking backends (MLflow, W&B, etc.)."""

    @abstractmethod
    def start_run(self, config: ExperimentConfig) -> str:
        """Start a new tracking run; return backend-specific run ID."""
        ...

    @abstractmethod
    def log_metrics(self, run_id: str, metrics: dict[str, float], step: int = 0) -> None:
        ...

    @abstractmethod
    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def log_artifact(self, run_id: str, path: Path) -> None:
        ...

    @abstractmethod
    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        ...


class BaseReporter(ABC):
    """Contract for report generators."""

    @property
    @abstractmethod
    def format(self) -> str:
        """File format produced: 'json', 'csv', 'markdown', 'html'."""
        ...

    @abstractmethod
    def generate(self, result: EvalRunResult, output_dir: Path) -> Path:
        """Write the report to output_dir and return the created file path."""
        ...


class BasePipeline(ABC):
    """
    Contract for end-to-end evaluation pipelines.

    A pipeline owns the full flow:
      load dataset → run inference → evaluate → track → report
    """

    @abstractmethod
    def run(self, config: ExperimentConfig) -> EvalRunResult:
        """Execute the full evaluation pipeline and return the run result."""
        ...
