"""
metrics/prometheus_metrics.py
------------------------------
Prometheus metrics definitions for the evaluation platform.

All metrics are module-level singletons — safe to import from anywhere.
The HTTP metrics server is started once by the API or CLI.

Metrics exposed:
  eval_platform_evaluations_total          Counter
  eval_platform_evaluation_duration_seconds Histogram
  eval_platform_metric_value               Gauge
  eval_platform_run_duration_seconds       Histogram
  eval_platform_samples_evaluated_total    Counter
  eval_platform_evaluator_errors_total     Counter
"""
from __future__ import annotations

from evaluation_platform.logging_.structured import get_logger

log = get_logger(__name__)

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
        REGISTRY,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    log.warning("prometheus_client not installed; metrics server disabled.")

    # Provide no-op stubs so imports never fail
    class _NoOp:
        def labels(self, **kwargs):
            return self
        def inc(self, *a, **kw): pass
        def set(self, *a, **kw): pass
        def observe(self, *a, **kw): pass
        def time(self):
            import contextlib
            @contextlib.contextmanager
            def _ctx():
                yield
            return _ctx()

    Counter = Gauge = Histogram = _NoOp  # type: ignore
    def start_http_server(port: int) -> None:  # type: ignore
        log.warning("Prometheus not installed; metrics server not started.")


# ── Metric definitions ────────────────────────────────────────────────────────

EVAL_COUNTER = Counter(
    "eval_platform_evaluations_total",
    "Total number of individual evaluator runs",
    ["evaluator"],
) if _PROMETHEUS_AVAILABLE else Counter()

EVAL_DURATION_HISTOGRAM = Histogram(
    "eval_platform_evaluation_duration_seconds",
    "Duration of a single evaluator run in seconds",
    ["evaluator"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0],
) if _PROMETHEUS_AVAILABLE else Histogram()

METRIC_VALUE_GAUGE = Gauge(
    "eval_platform_metric_value",
    "Current value of an evaluation metric",
    ["metric_name", "evaluator"],
) if _PROMETHEUS_AVAILABLE else Gauge()

RUN_DURATION_HISTOGRAM = Histogram(
    "eval_platform_run_duration_seconds",
    "Duration of a full evaluation run in seconds",
    ["experiment_name", "task_type"],
    buckets=[1.0, 5.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0],
) if _PROMETHEUS_AVAILABLE else Histogram()

SAMPLES_COUNTER = Counter(
    "eval_platform_samples_evaluated_total",
    "Total number of samples evaluated",
    ["experiment_name", "task_type"],
) if _PROMETHEUS_AVAILABLE else Counter()

EVALUATOR_ERROR_COUNTER = Counter(
    "eval_platform_evaluator_errors_total",
    "Total evaluator failures",
    ["evaluator", "error_type"],
) if _PROMETHEUS_AVAILABLE else Counter()


def start_metrics_server(port: int = 9090) -> None:
    """Start the Prometheus HTTP metrics endpoint."""
    if not _PROMETHEUS_AVAILABLE:
        return
    try:
        start_http_server(port)
        log.info("Prometheus metrics server started", port=port)
    except Exception as exc:
        log.warning("Failed to start Prometheus metrics server", error=str(exc), port=port)
