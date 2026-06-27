"""
evaluators/ragas/rag_evaluator.py
----------------------------------
Ragas-based RAG evaluation pipeline.

Computes all standard RAG metrics in one pass:
  - context_precision    → are retrieved chunks relevant and ranked correctly?
  - context_recall       → does retrieval cover all needed information?
  - faithfulness         → is the answer faithful to the retrieved context?
  - answer_relevancy     → is the answer relevant to the question?
  - context_utilization  → how well are retrieved chunks actually used?
  - noise_sensitivity    → how does the model handle irrelevant context?

All metrics are in [0, 1].

Reference: https://docs.ragas.io/en/stable/concepts/metrics/
"""
from __future__ import annotations

from typing import Any

from evaluation_platform.core.registry import EvaluatorRegistry
from evaluation_platform.core.schemas import (
    EvalSample,
    EvalStatus,
    EvaluatorConfig,
    EvaluatorResult,
    MetricCategory,
    MetricResult,
    TaskType,
)
from evaluation_platform.evaluators.base import Evaluator
from evaluation_platform.logging_.structured import get_logger

log = get_logger(__name__)


class RagasEvaluator(Evaluator):
    """
    Runs Ragas evaluation metrics on RAG samples.

    Requires:
      - sample.contexts (list of retrieved document strings)
      - sample.output   (generated answer)
      - sample.input    (the user question / query)
      - sample.reference (optional ground-truth, needed for recall)

    Configurable metrics via config.extra["metrics"]:
        ["context_precision", "context_recall", "faithfulness",
         "answer_relevancy", "context_utilization", "noise_sensitivity"]
    """

    @property
    def name(self) -> str:
        return "ragas"

    @property
    def version(self) -> str:
        return "1.0.0"

    def is_applicable(self, sample: EvalSample) -> bool:
        return (
            bool(sample.output)
            and bool(sample.contexts)
            and sample.task_type == TaskType.RAG
        )

    def _evaluate(
        self,
        sample: EvalSample,
        config: EvaluatorConfig,
    ) -> list[MetricResult]:
        # Single-sample → wrap in batch of one
        results = self.evaluate_batch([sample], config)
        return results[0].metrics if results else []

    def evaluate_batch(
        self,
        samples: list[EvalSample],
        config: EvaluatorConfig,
    ) -> list[EvaluatorResult]:
        """
        Ragas is designed for batch evaluation; we run all samples at once.
        """
        applicable = [s for s in samples if self.is_applicable(s)]
        non_applicable_ids = {s.id for s in samples if not self.is_applicable(s)}

        if not applicable:
            return [
                EvaluatorResult(
                    evaluator_name=self.name,
                    evaluator_version=self.version,
                    sample_id=s.id,
                    task_type=s.task_type,
                    status=EvalStatus.SKIPPED,
                )
                for s in samples
            ]

        enabled_metrics: list[str] = config.extra.get(
            "metrics",
            ["context_precision", "context_recall", "faithfulness", "answer_relevancy"],
        )

        try:
            results = self._run_ragas(applicable, enabled_metrics, config)
        except Exception as exc:
            log.error("Ragas batch evaluation failed", error=str(exc))
            results = [
                EvaluatorResult(
                    evaluator_name=self.name,
                    evaluator_version=self.version,
                    sample_id=s.id,
                    task_type=s.task_type,
                    status=EvalStatus.FAILED,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                for s in applicable
            ]

        # Add skipped
        for s in samples:
            if s.id in non_applicable_ids:
                results.append(
                    EvaluatorResult(
                        evaluator_name=self.name,
                        evaluator_version=self.version,
                        sample_id=s.id,
                        task_type=s.task_type,
                        status=EvalStatus.SKIPPED,
                    )
                )

        order = {s.id: i for i, s in enumerate(samples)}
        results.sort(key=lambda r: order.get(r.sample_id, 999))
        return results

    def _run_ragas(
        self,
        samples: list[EvalSample],
        enabled_metrics: list[str],
        config: EvaluatorConfig,
    ) -> list[EvaluatorResult]:
        from ragas import evaluate as ragas_evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset  # type: ignore

        # Build the metric list
        metric_map: dict[str, Any] = {
            "context_precision": context_precision,
            "context_recall": context_recall,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
        }

        # Optional metrics (may not be in all Ragas versions)
        try:
            from ragas.metrics import context_utilization  # type: ignore
            metric_map["context_utilization"] = context_utilization
        except ImportError:
            pass

        selected_metrics = [
            metric_map[m] for m in enabled_metrics if m in metric_map
        ]

        if not selected_metrics:
            raise ValueError(
                f"No valid Ragas metrics selected from: {enabled_metrics}. "
                f"Available: {list(metric_map.keys())}"
            )

        # Build HuggingFace Dataset (Ragas expects this format)
        data: dict[str, list[Any]] = {
            "question":    [s.input for s in samples],
            "answer":      [s.output or "" for s in samples],
            "contexts":    [s.contexts for s in samples],
            "ground_truth": [s.reference or "" for s in samples],
        }
        hf_dataset = Dataset.from_dict(data)

        log.info(
            "Running Ragas evaluation",
            metrics=enabled_metrics,
            sample_count=len(samples),
        )

        ragas_result = ragas_evaluate(
            hf_dataset,
            metrics=selected_metrics,
            raise_exceptions=False,
        )
        scores_df = ragas_result.to_pandas()

        # Convert to EvaluatorResult per sample
        results: list[EvaluatorResult] = []
        for i, sample in enumerate(samples):
            row = scores_df.iloc[i] if i < len(scores_df) else {}
            metrics: list[MetricResult] = []

            for metric_name in enabled_metrics:
                if metric_name in row:
                    raw_value = row[metric_name]
                    import math
                    value = None if (isinstance(raw_value, float) and math.isnan(raw_value)) else float(raw_value)
                    metrics.append(
                        MetricResult(
                            name=f"ragas_{metric_name}",
                            value=round(value, 4) if value is not None else None,
                            category=MetricCategory.RAG,
                            unit="score",
                            description=f"Ragas {metric_name.replace('_', ' ').title()}",
                            passing=(
                                value >= config.threshold
                                if (value is not None and config.threshold is not None)
                                else None
                            ),
                            threshold=config.threshold,
                        )
                    )

            results.append(
                EvaluatorResult(
                    evaluator_name=self.name,
                    evaluator_version=self.version,
                    sample_id=sample.id,
                    task_type=sample.task_type,
                    metrics=metrics,
                    status=EvalStatus.COMPLETED,
                )
            )

        return results


# Auto-register
EvaluatorRegistry.register(RagasEvaluator())
