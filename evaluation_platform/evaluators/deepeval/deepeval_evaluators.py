"""
evaluators/deepeval/deepeval_evaluators.py
-------------------------------------------
DeepEval-based LLM-judge evaluators: hallucination, faithfulness,
answer relevancy, toxicity, bias, and configurable GEval.

Design:
  - Each metric is a separate class (Single Responsibility).
  - All share the same _build_test_case() helper.
  - The LLM judge model is configurable per-evaluator via config.model.
  - DeepEval is imported lazily to avoid hard dependency at import time.

Reference:
  https://docs.confident-ai.com/docs/metrics-introduction
"""
from __future__ import annotations

from typing import Any

from evaluation_platform.core.registry import EvaluatorRegistry
from evaluation_platform.core.schemas import (
    EvalSample,
    EvaluatorConfig,
    MetricCategory,
    MetricResult,
    TaskType,
)
from evaluation_platform.evaluators.base import Evaluator
from evaluation_platform.logging_.structured import get_logger

log = get_logger(__name__)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _build_test_case(sample: EvalSample) -> Any:
    """Convert an EvalSample into a DeepEval LLMTestCase."""
    from deepeval.test_case import LLMTestCase  # type: ignore
    return LLMTestCase(
        input=sample.input,
        actual_output=sample.output or "",
        expected_output=sample.reference,
        context=sample.contexts if sample.contexts else None,
        retrieval_context=sample.contexts if sample.contexts else None,
    )


def _get_judge_model(config: EvaluatorConfig) -> Any:
    """Return a DeepEval-compatible LLM model from config, or None for default."""
    model_name = config.model or config.extra.get("judge_model")
    if not model_name:
        return None
    try:
        from deepeval.models import GPTModel  # type: ignore
        return GPTModel(model=model_name)
    except Exception:
        return None


def _run_deepeval_metric(metric: Any, test_case: Any, evaluator_name: str) -> tuple[float, str | None]:
    """Run a DeepEval metric and return (score, reason)."""
    try:
        metric.measure(test_case)
        return float(metric.score), getattr(metric, "reason", None)
    except Exception as exc:
        raise RuntimeError(f"DeepEval metric failed: {exc}") from exc


# ── Individual evaluator classes ───────────────────────────────────────────────

class HallucinationEvaluator(Evaluator):
    """
    Detects hallucinations using DeepEval's HallucinationMetric.

    Requires: contexts (facts the model should be grounded in).
    Score interpretation: 0.0 = hallucination detected, 1.0 = fully grounded.
    """

    @property
    def name(self) -> str:
        return "hallucination"

    def is_applicable(self, sample: EvalSample) -> bool:
        return bool(sample.output) and bool(sample.contexts)

    def _evaluate(self, sample: EvalSample, config: EvaluatorConfig) -> list[MetricResult]:
        from deepeval.metrics import HallucinationMetric  # type: ignore
        threshold = config.threshold if config.threshold is not None else 0.5
        metric = HallucinationMetric(
            threshold=threshold,
            model=_get_judge_model(config),
            include_reason=True,
        )
        test_case = _build_test_case(sample)
        score, reason = _run_deepeval_metric(metric, test_case, self.name)
        return [
            MetricResult(
                name="hallucination_score",
                value=round(score, 4),
                category=MetricCategory.FAITHFULNESS,
                unit="score",
                description="Hallucination score (higher = less hallucination)",
                passing=score >= threshold,
                threshold=threshold,
                reason=reason,
            )
        ]


class FaithfulnessEvaluator(Evaluator):
    """
    Measures whether the model output is faithful to the retrieved contexts.
    Critical for RAG pipelines.
    """

    @property
    def name(self) -> str:
        return "faithfulness"

    def is_applicable(self, sample: EvalSample) -> bool:
        return bool(sample.output) and bool(sample.contexts)

    def _evaluate(self, sample: EvalSample, config: EvaluatorConfig) -> list[MetricResult]:
        from deepeval.metrics import FaithfulnessMetric  # type: ignore
        threshold = config.threshold if config.threshold is not None else 0.5
        metric = FaithfulnessMetric(
            threshold=threshold,
            model=_get_judge_model(config),
            include_reason=True,
        )
        test_case = _build_test_case(sample)
        score, reason = _run_deepeval_metric(metric, test_case, self.name)
        return [
            MetricResult(
                name="faithfulness_score",
                value=round(score, 4),
                category=MetricCategory.FAITHFULNESS,
                unit="score",
                description="Faithfulness to retrieved context",
                passing=score >= threshold,
                threshold=threshold,
                reason=reason,
            )
        ]


class AnswerRelevancyEvaluator(Evaluator):
    """
    Measures whether the model's answer is relevant to the input question.
    """

    @property
    def name(self) -> str:
        return "answer_relevancy"

    def _evaluate(self, sample: EvalSample, config: EvaluatorConfig) -> list[MetricResult]:
        from deepeval.metrics import AnswerRelevancyMetric  # type: ignore
        threshold = config.threshold if config.threshold is not None else 0.5
        metric = AnswerRelevancyMetric(
            threshold=threshold,
            model=_get_judge_model(config),
            include_reason=True,
        )
        test_case = _build_test_case(sample)
        score, reason = _run_deepeval_metric(metric, test_case, self.name)
        return [
            MetricResult(
                name="answer_relevancy_score",
                value=round(score, 4),
                category=MetricCategory.CUSTOM,
                unit="score",
                description="Relevance of answer to the input question",
                passing=score >= threshold,
                threshold=threshold,
                reason=reason,
            )
        ]


class ToxicityEvaluator(Evaluator):
    """
    Detects toxic content in model outputs.
    Score: 0.0 = toxic, 1.0 = non-toxic.
    """

    @property
    def name(self) -> str:
        return "toxicity"

    def _evaluate(self, sample: EvalSample, config: EvaluatorConfig) -> list[MetricResult]:
        from deepeval.metrics import ToxicityMetric  # type: ignore
        threshold = config.threshold if config.threshold is not None else 0.5
        metric = ToxicityMetric(
            threshold=threshold,
            model=_get_judge_model(config),
            include_reason=True,
        )
        test_case = _build_test_case(sample)
        score, reason = _run_deepeval_metric(metric, test_case, self.name)
        return [
            MetricResult(
                name="toxicity_score",
                value=round(score, 4),
                category=MetricCategory.SAFETY,
                unit="score",
                description="Toxicity score (higher = less toxic)",
                passing=score >= threshold,
                threshold=threshold,
                reason=reason,
            )
        ]


class BiasEvaluator(Evaluator):
    """
    Detects biased statements in model outputs.
    Score: 0.0 = biased, 1.0 = unbiased.
    """

    @property
    def name(self) -> str:
        return "bias"

    def _evaluate(self, sample: EvalSample, config: EvaluatorConfig) -> list[MetricResult]:
        from deepeval.metrics import BiasMetric  # type: ignore
        threshold = config.threshold if config.threshold is not None else 0.5
        metric = BiasMetric(
            threshold=threshold,
            model=_get_judge_model(config),
            include_reason=True,
        )
        test_case = _build_test_case(sample)
        score, reason = _run_deepeval_metric(metric, test_case, self.name)
        return [
            MetricResult(
                name="bias_score",
                value=round(score, 4),
                category=MetricCategory.SAFETY,
                unit="score",
                description="Bias score (higher = less biased)",
                passing=score >= threshold,
                threshold=threshold,
                reason=reason,
            )
        ]


class GEvalEvaluator(Evaluator):
    """
    Configurable GEval (G-Evaluation) — define any custom criterion via YAML.

    Requires config.extra:
        criteria:    "Assess whether the blog is well-structured and professional."
        eval_steps:  ["Check for clear headings", "Check for practical examples"]
        name_override: "blog_quality"  # optional custom metric name
    """

    @property
    def name(self) -> str:
        return "geval"

    def _evaluate(self, sample: EvalSample, config: EvaluatorConfig) -> list[MetricResult]:
        from deepeval.metrics import GEval  # type: ignore
        from deepeval.test_case import LLMTestCaseParams  # type: ignore

        criteria = config.extra.get(
            "criteria",
            "Assess the overall quality, coherence, and accuracy of the response.",
        )
        eval_steps = config.extra.get("eval_steps", None)
        metric_name_override = config.extra.get("name_override", "geval_score")
        threshold = config.threshold if config.threshold is not None else 0.5

        metric = GEval(
            name=metric_name_override,
            criteria=criteria,
            evaluation_steps=eval_steps,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            model=_get_judge_model(config),
            threshold=threshold,
        )
        test_case = _build_test_case(sample)
        score, reason = _run_deepeval_metric(metric, test_case, self.name)
        return [
            MetricResult(
                name=metric_name_override,
                value=round(score, 4),
                category=MetricCategory.CUSTOM,
                unit="score",
                description=f"GEval: {criteria[:80]}",
                passing=score >= threshold,
                threshold=threshold,
                reason=reason,
            )
        ]


class ContextualPrecisionEvaluator(Evaluator):
    """RAG: are the retrieved chunks ranked so relevant ones appear first?"""

    @property
    def name(self) -> str:
        return "contextual_precision"

    def is_applicable(self, sample: EvalSample) -> bool:
        return bool(sample.output) and bool(sample.contexts) and bool(sample.reference)

    def _evaluate(self, sample: EvalSample, config: EvaluatorConfig) -> list[MetricResult]:
        from deepeval.metrics import ContextualPrecisionMetric  # type: ignore
        threshold = config.threshold if config.threshold is not None else 0.5
        metric = ContextualPrecisionMetric(threshold=threshold, model=_get_judge_model(config), include_reason=True)
        test_case = _build_test_case(sample)
        score, reason = _run_deepeval_metric(metric, test_case, self.name)
        return [MetricResult(name="contextual_precision", value=round(score, 4),
                             category=MetricCategory.RAG, unit="score",
                             description="Contextual Precision (RAG retrieval ranking)",
                             passing=score >= threshold, threshold=threshold, reason=reason)]


class ContextualRecallEvaluator(Evaluator):
    """RAG: how much of the expected answer is supported by retrieved chunks?"""

    @property
    def name(self) -> str:
        return "contextual_recall"

    def is_applicable(self, sample: EvalSample) -> bool:
        return bool(sample.output) and bool(sample.contexts) and bool(sample.reference)

    def _evaluate(self, sample: EvalSample, config: EvaluatorConfig) -> list[MetricResult]:
        from deepeval.metrics import ContextualRecallMetric  # type: ignore
        threshold = config.threshold if config.threshold is not None else 0.5
        metric = ContextualRecallMetric(threshold=threshold, model=_get_judge_model(config), include_reason=True)
        test_case = _build_test_case(sample)
        score, reason = _run_deepeval_metric(metric, test_case, self.name)
        return [MetricResult(name="contextual_recall", value=round(score, 4),
                             category=MetricCategory.RAG, unit="score",
                             description="Contextual Recall (RAG coverage)",
                             passing=score >= threshold, threshold=threshold, reason=reason)]


# ── Auto-register all DeepEval evaluators ─────────────────────────────────────
for _cls in [
    HallucinationEvaluator,
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    ToxicityEvaluator,
    BiasEvaluator,
    GEvalEvaluator,
    ContextualPrecisionEvaluator,
    ContextualRecallEvaluator,
]:
    EvaluatorRegistry.register(_cls())
