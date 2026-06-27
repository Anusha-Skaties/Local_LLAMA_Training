"""
core/exceptions.py
------------------
Typed exception hierarchy for the evaluation platform.

Using typed exceptions instead of generic RuntimeError lets callers catch
specific failures without parsing error strings.
"""
from __future__ import annotations


class EvalPlatformError(Exception):
    """Base class for all platform exceptions."""


# ── Dataset errors ─────────────────────────────────────────────────────────────
class DatasetLoadError(EvalPlatformError):
    """Raised when a dataset cannot be loaded (file missing, parse error, etc.)"""


class DatasetFormatError(EvalPlatformError):
    """Raised when a dataset record does not match the expected schema."""


class UnsupportedDatasetFormatError(EvalPlatformError):
    """Raised when no loader is registered for the requested format."""


# ── Evaluator errors ───────────────────────────────────────────────────────────
class EvaluatorError(EvalPlatformError):
    """Raised when an evaluator fails to compute metrics."""

    def __init__(self, evaluator_name: str, sample_id: str, cause: Exception) -> None:
        self.evaluator_name = evaluator_name
        self.sample_id = sample_id
        self.cause = cause
        super().__init__(
            f"Evaluator '{evaluator_name}' failed on sample '{sample_id}': {cause}"
        )


class EvaluatorTimeoutError(EvaluatorError):
    """Raised when an evaluator exceeds its configured timeout."""


class EvaluatorNotFoundError(EvalPlatformError):
    """Raised when a requested evaluator name is not in the registry."""

    def __init__(self, name: str) -> None:
        super().__init__(
            f"Evaluator '{name}' is not registered. "
            "Use EvaluatorRegistry.register() to add it."
        )


# ── Experiment / tracking errors ───────────────────────────────────────────────
class ExperimentTrackerError(EvalPlatformError):
    """Raised when an experiment tracking backend fails."""


class RegressionError(EvalPlatformError):
    """Raised by CI/CD regression check when a metric breaches its threshold."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__(
            "Evaluation regression detected:\n" + "\n".join(f"  • {v}" for v in violations)
        )


# ── Report errors ──────────────────────────────────────────────────────────────
class ReportGenerationError(EvalPlatformError):
    """Raised when a reporter fails to write its output."""


# ── Config errors ──────────────────────────────────────────────────────────────
class ConfigValidationError(EvalPlatformError):
    """Raised when a YAML/Pydantic config is invalid."""


# ── API errors ────────────────────────────────────────────────────────────────
class RunNotFoundError(EvalPlatformError):
    """Raised when a requested run_id does not exist."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Run '{run_id}' not found.")
