"""
evaluation_platform
-------------------
Production AI Evaluation Platform for the Local LLAMA Training project.

Supports evaluating: summarization, QA, RAG, generation, multi-agent systems,
tool calling, classification, extraction, code generation, and chat assistants.

Architecture:
  core/          → Pydantic schemas, protocols, registry, exceptions
  tracing/       → OpenTelemetry + LangSmith tracing
  logging_/      → Structured JSON logging with correlation IDs
  datasets/      → Dataset loaders (JSONL, HF, CSV, Parquet)
  evaluators/    → DeepEval, Ragas, lexical, semantic, safety, performance
  experiments/   → MLflow + W&B experiment tracking
  metrics/       → Prometheus metrics server
  reports/       → JSON, CSV, Markdown, HTML report generation
  pipelines/     → End-to-end evaluation pipelines per task type
  api/           → FastAPI evaluation REST API
  cli/           → Typer CLI for local evaluation runs
  tests/         → pytest test suite
  configs/       → YAML-based configuration
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("evaluation_platform")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__: list[str] = []
