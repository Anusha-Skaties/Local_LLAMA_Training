"""
tracing/otel.py
---------------
OpenTelemetry setup for the evaluation platform.

Provides:
  - TracerProvider configuration (OTLP or console exporter)
  - Convenience context manager eval_span() to instrument any evaluation block
  - Attributes helpers for consistent span annotation

Usage (in main.py / API startup):
    from evaluation_platform.tracing.otel import configure_tracing
    configure_tracing(endpoint="http://otel-collector:4317", service_name="eval-platform")

Usage in evaluators / pipelines:
    from evaluation_platform.tracing.otel import eval_span

    with eval_span("rouge_evaluator", sample_id=sample.id) as span:
        result = compute_rouge(...)
        span.set_attribute("rouge_l", result)
"""
from __future__ import annotations

import contextlib
import os
from typing import Any, Generator

from evaluation_platform.logging_.structured import get_logger

log = get_logger(__name__)

# ── Optional import guard ─────────────────────────────────────────────────────
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    log.warning("opentelemetry-sdk not installed; tracing is disabled.")

_tracer: Any = None          # global tracer instance
_provider: Any = None        # global provider (kept for shutdown)


def configure_tracing(
    endpoint: str | None = None,
    service_name: str = "evaluation-platform",
    service_version: str = "0.1.0",
    console_export: bool = False,
) -> None:
    """
    Initialise the global TracerProvider.

    Called once at application startup.  Idempotent — subsequent calls are
    no-ops to prevent double-initialisation when imported multiple times.

    Args:
        endpoint:        OTLP gRPC endpoint (e.g. "http://localhost:4317").
                         If None and console_export=False, tracing is skipped.
        service_name:    OTel resource service.name attribute.
        service_version: OTel resource service.version attribute.
        console_export:  Write spans to stdout (useful for local debugging).
    """
    global _tracer, _provider

    if not _OTEL_AVAILABLE:
        return

    if _provider is not None:
        return  # already initialised

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": service_version,
        }
    )
    provider = TracerProvider(resource=resource)

    if console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        log.info("OTel tracing → console exporter enabled")

    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            log.info("OTel tracing → OTLP exporter enabled", endpoint=endpoint)
        except ImportError:
            log.warning(
                "opentelemetry-exporter-otlp-proto-grpc not installed; "
                "falling back to no-op exporter."
            )

    trace.set_tracer_provider(provider)
    _provider = provider
    _tracer = trace.get_tracer(service_name, service_version)
    log.info("OpenTelemetry tracing configured", service=service_name)


def get_tracer() -> Any:
    """Return the global tracer (no-op if not configured)."""
    if not _OTEL_AVAILABLE or _tracer is None:
        return _NoOpTracer()
    return _tracer


@contextlib.contextmanager
def eval_span(
    operation_name: str,
    **attributes: Any,
) -> Generator[Any, None, None]:
    """
    Context manager that wraps an evaluation operation in an OTEL span.

    Automatically records exceptions and marks the span as ERROR on failure.

    Usage:
        with eval_span("bertscore", evaluator="bertscore", sample_id="abc") as span:
            score = compute_bertscore(...)
            span.set_attribute("bertscore_f1", score)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(operation_name) as span:
        for key, value in attributes.items():
            try:
                span.set_attribute(f"eval.{key}", str(value))
            except Exception:
                pass
        try:
            yield span
        except Exception as exc:
            if _OTEL_AVAILABLE:
                from opentelemetry.trace import Status, StatusCode
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
            raise


def shutdown_tracing() -> None:
    """Flush and shut down the TracerProvider.  Call on application exit."""
    global _provider
    if _provider is not None:
        _provider.shutdown()
        _provider = None
        log.info("OpenTelemetry tracing shut down")


# ── No-op fallback ────────────────────────────────────────────────────────────

class _NoOpSpan:
    def set_attribute(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    @contextlib.contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any) -> Generator[_NoOpSpan, None, None]:
        yield _NoOpSpan()
