"""
logging_/structured.py
----------------------
Structured JSON logging with correlation IDs and OpenTelemetry trace context.

Every log record is emitted as a single-line JSON object containing:
  - timestamp (ISO-8601 UTC)
  - level
  - logger name
  - message
  - correlation_id   (request-scoped UUID)
  - trace_id         (OTEL trace ID if available)
  - span_id          (OTEL span ID if available)
  - extra fields set via log.bind() or the 'extra' kwarg

Usage:
    from evaluation_platform.logging_.structured import get_logger

    log = get_logger(__name__)
    log.info("Evaluator started", evaluator="rouge_lexical", sample_id="abc")
"""
from __future__ import annotations

import json
import logging
import threading
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

# ── Correlation-ID context var ────────────────────────────────────────────────
_correlation_id: threading.local = threading.local()


def set_correlation_id(cid: str) -> None:
    _correlation_id.value = cid


def get_correlation_id() -> str:
    return getattr(_correlation_id, "value", "")


def new_correlation_id() -> str:
    cid = str(uuid.uuid4())
    set_correlation_id(cid)
    return cid


# ── JSON log formatter ────────────────────────────────────────────────────────

class StructuredFormatter(logging.Formatter):
    """Formats every log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        # Attempt to read OTEL trace/span IDs if tracing is active.
        trace_id: str = ""
        span_id: str = ""
        try:
            from opentelemetry import trace as otel_trace
            span = otel_trace.get_current_span()
            ctx = span.get_span_context()
            if ctx.is_valid:
                trace_id = format(ctx.trace_id, "032x")
                span_id = format(ctx.span_id, "016x")
        except Exception:  # opentelemetry not installed / not active
            pass

        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
            "trace_id": trace_id,
            "span_id": span_id,
        }

        # Attach any extra fields injected via log.info("msg", extra={...})
        for key, value in record.__dict__.items():
            if key not in _LOG_RECORD_BUILTIN_ATTRS:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(payload, default=str)


_LOG_RECORD_BUILTIN_ATTRS: frozenset[str] = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "id", "levelname", "levelno", "lineno", "module",
        "msecs", "message", "msg", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "thread", "threadName",
        "taskName",
    }
)


# ── Logger factory ────────────────────────────────────────────────────────────

def configure_root_logger(level: int = logging.INFO) -> None:
    """
    Replace the root logger's handlers with a structured JSON handler.

    Call once at application startup (e.g. in api/main.py or cli/main.py).
    """
    root = logging.getLogger()
    root.setLevel(level)
    # Remove existing handlers to avoid duplicate logs.
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> "BoundLogger":
    """Return a BoundLogger wrapping the standard Python logger."""
    return BoundLogger(logging.getLogger(name))


class BoundLogger:
    """
    Thin wrapper around logging.Logger that supports structured field binding.

    Usage:
        log = get_logger(__name__)
        log = log.bind(run_id="abc", evaluator="rouge")
        log.info("Starting evaluation")
    """

    def __init__(self, logger: logging.Logger, context: dict[str, Any] | None = None) -> None:
        self._logger = logger
        self._context: dict[str, Any] = context or {}

    def bind(self, **fields: Any) -> "BoundLogger":
        return BoundLogger(self._logger, {**self._context, **fields})

    def _emit(self, level: int, msg: str, **kwargs: Any) -> None:
        extra = {**self._context, **kwargs}
        self._logger.log(level, msg, extra=extra, stacklevel=3)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.ERROR, msg, **kwargs)

    def exception(self, msg: str, **kwargs: Any) -> None:
        self._logger.exception(msg, extra={**self._context, **kwargs}, stacklevel=2)
