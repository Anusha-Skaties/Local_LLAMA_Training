"""
evaluators/performance/latency_evaluator.py
--------------------------------------------
Performance metrics evaluator: latency, throughput, token counts, TTFT,
cost estimation, and system resource usage.

These metrics are not computed FROM the text — they are recorded DURING
inference and stored in sample.metadata by the inference runner.

Expected metadata keys (set by the inference pipeline):
    latency_ms:          total generation latency in milliseconds
    ttft_ms:             time-to-first-token in milliseconds
    prompt_tokens:       number of input tokens
    completion_tokens:   number of output tokens
    tokens_per_second:   generation throughput
    gpu_memory_mb:       peak GPU memory during generation
    cpu_percent:         CPU utilisation during generation
    cost_usd:            estimated API cost (if using API-based model)
"""
from __future__ import annotations

from evaluation_platform.core.registry import EvaluatorRegistry
from evaluation_platform.core.schemas import (
    EvalSample,
    EvaluatorConfig,
    MetricCategory,
    MetricResult,
)
from evaluation_platform.evaluators.base import Evaluator


class PerformanceEvaluator(Evaluator):
    """
    Extracts and structures performance metrics from sample.metadata.

    This evaluator is always applicable — it gracefully returns None for
    any metadata key that was not recorded.
    """

    @property
    def name(self) -> str:
        return "performance"

    @property
    def version(self) -> str:
        return "1.0.0"

    def is_applicable(self, sample: EvalSample) -> bool:
        return bool(sample.output)

    def _evaluate(
        self,
        sample: EvalSample,
        config: EvaluatorConfig,
    ) -> list[MetricResult]:
        meta = sample.metadata
        results: list[MetricResult] = []

        # ── Latency ───────────────────────────────────────────────────────────
        if "latency_ms" in meta:
            results.append(
                MetricResult(
                    name="latency_ms",
                    value=round(float(meta["latency_ms"]), 2),
                    category=MetricCategory.PERFORMANCE,
                    unit="ms",
                    description="Total end-to-end generation latency",
                    passing=(
                        float(meta["latency_ms"]) <= config.threshold
                        if config.threshold else None
                    ),
                    threshold=config.threshold,
                )
            )

        if "ttft_ms" in meta:
            results.append(
                MetricResult(
                    name="ttft_ms",
                    value=round(float(meta["ttft_ms"]), 2),
                    category=MetricCategory.PERFORMANCE,
                    unit="ms",
                    description="Time-to-first-token",
                )
            )

        # ── Token counts ──────────────────────────────────────────────────────
        if "prompt_tokens" in meta:
            results.append(
                MetricResult(
                    name="prompt_tokens",
                    value=int(meta["prompt_tokens"]),
                    category=MetricCategory.PERFORMANCE,
                    unit="tokens",
                    description="Number of input prompt tokens",
                )
            )

        if "completion_tokens" in meta:
            results.append(
                MetricResult(
                    name="completion_tokens",
                    value=int(meta["completion_tokens"]),
                    category=MetricCategory.PERFORMANCE,
                    unit="tokens",
                    description="Number of generated completion tokens",
                )
            )

        total_tokens = meta.get("prompt_tokens", 0) + meta.get("completion_tokens", 0)
        if total_tokens > 0:
            results.append(
                MetricResult(
                    name="total_tokens",
                    value=int(total_tokens),
                    category=MetricCategory.PERFORMANCE,
                    unit="tokens",
                    description="Total tokens (prompt + completion)",
                )
            )

        # ── Throughput ────────────────────────────────────────────────────────
        if "tokens_per_second" in meta:
            results.append(
                MetricResult(
                    name="tokens_per_second",
                    value=round(float(meta["tokens_per_second"]), 2),
                    category=MetricCategory.PERFORMANCE,
                    unit="tokens/s",
                    description="Generation throughput",
                )
            )
        elif "completion_tokens" in meta and "latency_ms" in meta:
            # Derive throughput if not explicitly recorded
            latency_s = float(meta["latency_ms"]) / 1000.0
            if latency_s > 0:
                tps = float(meta["completion_tokens"]) / latency_s
                results.append(
                    MetricResult(
                        name="tokens_per_second",
                        value=round(tps, 2),
                        category=MetricCategory.PERFORMANCE,
                        unit="tokens/s",
                        description="Generation throughput (derived)",
                    )
                )

        # ── System resources ──────────────────────────────────────────────────
        if "gpu_memory_mb" in meta:
            results.append(
                MetricResult(
                    name="gpu_memory_mb",
                    value=round(float(meta["gpu_memory_mb"]), 1),
                    category=MetricCategory.PERFORMANCE,
                    unit="MB",
                    description="Peak GPU memory during generation",
                )
            )

        if "cpu_percent" in meta:
            results.append(
                MetricResult(
                    name="cpu_percent",
                    value=round(float(meta["cpu_percent"]), 1),
                    category=MetricCategory.PERFORMANCE,
                    unit="%",
                    description="Average CPU utilisation during generation",
                )
            )

        # ── Cost estimation ───────────────────────────────────────────────────
        if "cost_usd" in meta:
            results.append(
                MetricResult(
                    name="cost_usd",
                    value=round(float(meta["cost_usd"]), 6),
                    category=MetricCategory.COST,
                    unit="USD",
                    description="Estimated API cost for this request",
                )
            )
        elif "prompt_tokens" in meta and "completion_tokens" in meta:
            # Estimate cost for common models
            model = sample.metadata.get("model_name", "")
            cost = self._estimate_cost(
                int(meta["prompt_tokens"]),
                int(meta["completion_tokens"]),
                model,
            )
            if cost is not None:
                results.append(
                    MetricResult(
                        name="cost_usd_estimated",
                        value=round(cost, 6),
                        category=MetricCategory.COST,
                        unit="USD",
                        description="Estimated cost (based on token counts)",
                    )
                )

        return results

    @staticmethod
    def _estimate_cost(
        prompt_tokens: int,
        completion_tokens: int,
        model_name: str,
    ) -> float | None:
        """
        Rough cost estimates (USD per 1M tokens) for common models.
        Update these as pricing changes.
        """
        pricing: dict[str, tuple[float, float]] = {
            "gpt-4o":                    (5.00,  15.00),
            "gpt-4o-mini":               (0.15,   0.60),
            "gpt-4-turbo":               (10.00,  30.00),
            "gpt-3.5-turbo":             (0.50,   1.50),
            "claude-3-5-sonnet":         (3.00,  15.00),
            "claude-3-haiku":            (0.25,   1.25),
            "meta-llama/llama-3.2-3b":   (0.06,   0.06),  # self-hosted estimate
        }
        model_lower = model_name.lower()
        for key, (input_rate, output_rate) in pricing.items():
            if key in model_lower:
                return (
                    prompt_tokens * input_rate / 1_000_000
                    + completion_tokens * output_rate / 1_000_000
                )
        return None


# Auto-register
EvaluatorRegistry.register(PerformanceEvaluator())
