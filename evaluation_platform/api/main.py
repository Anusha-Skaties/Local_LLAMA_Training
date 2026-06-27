"""
api/main.py
-----------
FastAPI application entry point for the evaluation platform.

Endpoints:
  GET  /health               → liveness check
  POST /evaluate             → submit evaluation run
  GET  /evaluate/runs        → list runs
  GET  /evaluate/runs/{id}   → get run result
  GET  /metrics/registry     → list registered evaluators
  GET  /metrics/prometheus   → Prometheus scrape endpoint (if enabled)

Production features:
  - Correlation ID middleware on every request
  - Structured JSON logging
  - OpenTelemetry FastAPI instrumentation
  - Global exception handler (never leaks stack traces)
  - CORS (configurable origins)
  - Prometheus metrics server on port 9090
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from evaluation_platform.__init__ import __version__
from evaluation_platform.api.middleware.correlation import CorrelationIdMiddleware
from evaluation_platform.api.routes.evaluate import router as evaluate_router
from evaluation_platform.core.registry import EvaluatorRegistry
from evaluation_platform.core.schemas import HealthResponse
from evaluation_platform.logging_.structured import configure_root_logger, get_logger
from evaluation_platform.metrics.prometheus_metrics import start_metrics_server
from evaluation_platform.tracing.otel import configure_tracing

configure_root_logger()
log = get_logger(__name__)


def create_app() -> FastAPI:
    """Application factory — returns a configured FastAPI instance."""

    app = FastAPI(
        title="AI Evaluation Platform",
        description=(
            "Production-grade LLM evaluation platform. "
            "Supports generation, QA, RAG, multi-agent, safety, and more."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── OpenTelemetry ──────────────────────────────────────────────────────────
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otel_endpoint:
        configure_tracing(endpoint=otel_endpoint, service_name="eval-platform")
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore
            FastAPIInstrumentor.instrument_app(app)
        except ImportError:
            pass

    # ── Routes ─────────────────────────────────────────────────────────────────
    app.include_router(evaluate_router)

    # ── Global exception handler ───────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled exception", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred.",
                # Do NOT include stack trace — security: never leak internals
            },
        )

    # ── Health / utility endpoints ─────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(version=__version__)

    @app.get("/metrics/registry", tags=["system"])
    async def list_evaluators() -> dict:
        return {"evaluators": EvaluatorRegistry.names()}

    @app.on_event("startup")
    async def startup() -> None:
        log.info("Evaluation platform API starting", version=__version__)
        # Start Prometheus metrics server on a separate port
        prometheus_port = int(os.getenv("PROMETHEUS_PORT", "9090"))
        start_metrics_server(port=prometheus_port)

        # Trigger evaluator auto-registration by importing modules
        _import_evaluators()
        log.info("Evaluators registered", names=EvaluatorRegistry.names())

    @app.on_event("shutdown")
    async def shutdown() -> None:
        from evaluation_platform.tracing.otel import shutdown_tracing
        shutdown_tracing()
        log.info("Evaluation platform API shut down")

    return app


def _import_evaluators() -> None:
    """Import all evaluator modules to trigger auto-registration."""
    try:
        from evaluation_platform.evaluators.lexical import lexical_evaluator  # noqa: F401
        from evaluation_platform.evaluators.semantic import bertscore_evaluator  # noqa: F401
        from evaluation_platform.evaluators.performance import latency_evaluator  # noqa: F401
    except Exception as exc:
        log.warning("Some evaluators failed to load", error=str(exc))
    try:
        from evaluation_platform.evaluators.deepeval import deepeval_evaluators  # noqa: F401
    except ImportError:
        log.warning("deepeval not installed; DeepEval evaluators not registered.")
    try:
        from evaluation_platform.evaluators.ragas import rag_evaluator  # noqa: F401
    except ImportError:
        log.warning("ragas not installed; Ragas evaluators not registered.")


# Module-level app instance (for `uvicorn evaluation_platform.api.main:app`)
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "evaluation_platform.api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_config=None,  # use our structured logger
    )
