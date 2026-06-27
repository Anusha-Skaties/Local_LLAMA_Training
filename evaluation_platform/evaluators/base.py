"""
evaluators/base.py
------------------
BaseEvaluator with shared infrastructure: timing, error wrapping, span
emission, and Prometheus counter updates.

Every concrete evaluator inherits this class and implements:
    name       → property
    _evaluate  → core logic (called by evaluate_sample)

Concrete evaluators must NOT catch exceptions from _evaluate; they bubble up
to evaluate_sample, which wraps them into a failed EvaluatorResult so the
pipeline never crashes on a single bad metric.
"""
from __future__ import annotations

import time
import traceback
from abc import abstractmethod
from typing import TYPE_CHECKING

from evaluation_platform.core.exceptions import EvaluatorError
from evaluation_platform.core.protocols import BaseEvaluator
from evaluation_platform.core.schemas import (
    EvalSample,
    EvalStatus,
    EvaluatorConfig,
    EvaluatorResult,
    MetricResult,
    TaskType,
)
from evaluation_platform.logging_.structured import get_logger
from evaluation_platform.tracing.otel import eval_span

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


class Evaluator(BaseEvaluator):
    """
    Production base class for all evaluators.

    Subclasses implement _evaluate() and optionally _evaluate_batch().
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def _evaluate(
        self,
        sample: EvalSample,
        config: EvaluatorConfig,
    ) -> list[MetricResult]:
        """
        Core evaluation logic.  Return a list of MetricResult objects.
        Raise any exception on failure — the base class handles wrapping.
        """
        ...

    def evaluate_sample(
        self,
        sample: EvalSample,
        config: EvaluatorConfig,
    ) -> EvaluatorResult:
        """
        Evaluate one sample with full error handling, timing, and tracing.

        Returns a completed EvaluatorResult even if _evaluate() raises —
        the result will carry status=FAILED and the error message.
        """
        if not self.is_applicable(sample):
            return EvaluatorResult(
                evaluator_name=self.name,
                evaluator_version=self.version,
                sample_id=sample.id,
                task_type=sample.task_type,
                status=EvalStatus.SKIPPED,
                metrics=[],
            )

        bound_log = log.bind(evaluator=self.name, sample_id=sample.id)
        start = time.perf_counter()

        with eval_span(self.name, sample_id=sample.id, task_type=sample.task_type.value):
            try:
                metrics = self._evaluate(sample, config)
                latency_ms = (time.perf_counter() - start) * 1000

                self._emit_prometheus(metrics)

                bound_log.debug(
                    "Evaluation complete",
                    metric_count=len(metrics),
                    latency_ms=round(latency_ms, 2),
                )
                return EvaluatorResult(
                    evaluator_name=self.name,
                    evaluator_version=self.version,
                    sample_id=sample.id,
                    task_type=sample.task_type,
                    metrics=metrics,
                    status=EvalStatus.COMPLETED,
                    latency_ms=latency_ms,
                )

            except Exception as exc:
                latency_ms = (time.perf_counter() - start) * 1000
                bound_log.error(
                    "Evaluation failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                return EvaluatorResult(
                    evaluator_name=self.name,
                    evaluator_version=self.version,
                    sample_id=sample.id,
                    task_type=sample.task_type,
                    metrics=[],
                    status=EvalStatus.FAILED,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    latency_ms=latency_ms,
                )

    def evaluate_batch(
        self,
        samples: list[EvalSample],
        config: EvaluatorConfig,
    ) -> list[EvaluatorResult]:
        """
        Default: sequential loop.  Override for true batch evaluation.
        """
        return [self.evaluate_sample(s, config) for s in samples]

    def is_applicable(self, sample: EvalSample) -> bool:
        return bool(sample.output)

    def _emit_prometheus(self, metrics: list[MetricResult]) -> None:
        """Push metric values to Prometheus gauges if available."""
        try:
            from evaluation_platform.metrics.prometheus_metrics import (
                METRIC_VALUE_GAUGE,
                EVAL_COUNTER,
            )
            EVAL_COUNTER.labels(evaluator=self.name).inc()
            for m in metrics:
                if isinstance(m.value, (int, float)) and m.value is not None:
                    METRIC_VALUE_GAUGE.labels(
                        metric_name=m.name,
                        evaluator=self.name,
                    ).set(float(m.value))
        except Exception:
            pass  # Prometheus is optional; never let it break evaluation
