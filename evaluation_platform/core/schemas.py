"""
core/schemas.py
---------------
Central Pydantic models that flow through the entire evaluation platform.

Design principle: Every piece of data entering or leaving the system is
validated against one of these models. This gives us runtime type safety,
automatic API serialisation, and a single place to evolve the schema.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    """Supported evaluation task types.

    Adding a new task type here automatically makes it available throughout
    the entire platform (pipeline selection, metric routing, report labels).
    """
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    RAG = "rag"
    GENERATION = "generation"
    MULTI_AGENT = "multi_agent"
    TOOL_CALLING = "tool_calling"
    FUNCTION_CALLING = "function_calling"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    CODE_GENERATION = "code_generation"
    CHAT = "chat"


class MetricCategory(str, Enum):
    """High-level grouping of metrics for dashboard organisation."""
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    FAITHFULNESS = "faithfulness"
    SAFETY = "safety"
    PERFORMANCE = "performance"
    COST = "cost"
    RAG = "rag"
    CUSTOM = "custom"


class EvalStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ── Dataset / Sample schemas ──────────────────────────────────────────────────

class Message(BaseModel):
    """A single turn in a conversation (role + content)."""
    model_config = ConfigDict(frozen=True)

    role: str
    content: str


class EvalSample(BaseModel):
    """
    The atomic unit of evaluation.

    For generation/summarization:
        input      → user prompt
        output     → model-generated response (set after inference)
        reference  → ground-truth / reference response

    For RAG:
        contexts   → list of retrieved document chunks

    For chat / multi-turn:
        messages   → full conversation history
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskType = TaskType.GENERATION
    input: str
    output: str | None = None          # filled after model inference
    reference: str | None = None       # ground-truth answer
    contexts: list[str] = Field(default_factory=list)   # retrieved docs (RAG)
    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("id", mode="before")
    @classmethod
    def ensure_str_id(cls, v: Any) -> str:
        return str(v)


class RAGSample(EvalSample):
    """EvalSample specialised for RAG evaluation requiring non-empty contexts."""
    task_type: TaskType = TaskType.RAG
    query: str = ""          # the retrieval query (may differ from input)
    contexts: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def sync_query(self) -> "RAGSample":
        if not self.query:
            object.__setattr__(self, "query", self.input)
        return self


# ── Metric result schemas ─────────────────────────────────────────────────────

class MetricResult(BaseModel):
    """A single computed metric value."""
    model_config = ConfigDict(frozen=True)

    name: str
    value: float | str | None
    category: MetricCategory = MetricCategory.CUSTOM
    unit: str = ""                       # e.g. "ms", "tokens/s", "score"
    description: str = ""
    passing: bool | None = None          # None = no threshold configured
    threshold: float | None = None
    reason: str | None = None            # LLM-judge reason (DeepEval)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("value", mode="before")
    @classmethod
    def coerce_nan_to_none(cls, v: Any) -> Any:
        """Replace float NaN with None to keep JSON output valid."""
        import math
        if isinstance(v, float) and math.isnan(v):
            return None
        return v


class EvaluatorResult(BaseModel):
    """Output of running one evaluator on one sample."""
    evaluator_name: str
    evaluator_version: str = "unknown"
    sample_id: str
    task_type: TaskType
    metrics: list[MetricResult] = Field(default_factory=list)
    status: EvalStatus = EvalStatus.COMPLETED
    error: str | None = None
    error_type: str | None = None
    latency_ms: float = 0.0
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def get_metric(self, name: str) -> MetricResult | None:
        return next((m for m in self.metrics if m.name == name), None)

    def primary_score(self) -> float | None:
        """Return first numeric metric value; used for quick comparisons."""
        for m in self.metrics:
            if isinstance(m.value, (int, float)):
                return float(m.value)
        return None


# ── Aggregate / Run-level schemas ─────────────────────────────────────────────

class AggregateMetrics(BaseModel):
    """Per-metric aggregated statistics across all samples in a run."""
    metric_name: str
    category: MetricCategory
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    pass_rate: float | None = None     # fraction of samples that passed threshold
    sample_count: int = 0


class EvalRunResult(BaseModel):
    """Complete result of one evaluation run (all evaluators, all samples)."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    experiment_name: str
    model_name: str
    model_version: str | None = None
    dataset_id: str
    dataset_name: str
    task_type: TaskType
    samples: list[EvalSample] = Field(default_factory=list)
    evaluator_results: list[EvaluatorResult] = Field(default_factory=list)
    aggregate_metrics: list[AggregateMetrics] = Field(default_factory=list)
    status: EvalStatus = EvalStatus.COMPLETED
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    mlflow_run_id: str | None = None
    langsmith_run_id: str | None = None


# ── Configuration schemas ─────────────────────────────────────────────────────

class DatasetConfig(BaseModel):
    """How to load a dataset for evaluation."""
    name: str
    format: str = "jsonl"               # jsonl | csv | parquet | hf
    path: str = ""
    hf_dataset_name: str | None = None
    split: str = "validation"
    max_samples: int | None = None
    shuffle: bool = False
    shuffle_seed: int = 42
    filters: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


class EvaluatorConfig(BaseModel):
    """Configuration for one evaluator instance."""
    name: str
    enabled: bool = True
    threshold: float | None = None
    batch_size: int = 1
    timeout_seconds: float = 120.0
    model: str | None = None            # judge LLM override
    extra: dict[str, Any] = Field(default_factory=dict)


class TracingConfig(BaseModel):
    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "evaluation-platform"
    langsmith_enabled: bool = False
    langsmith_project: str = "llama-eval"


class ExperimentConfig(BaseModel):
    """Top-level configuration for an evaluation experiment."""
    name: str
    description: str = ""
    model_name: str
    model_version: str | None = None
    adapter_dir: str | None = None
    task_type: TaskType = TaskType.GENERATION
    dataset: DatasetConfig
    evaluators: list[EvaluatorConfig] = Field(default_factory=list)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    mlflow_experiment: str | None = None
    wandb_project: str | None = None
    output_dir: str = "outputs/eval_runs"
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("evaluators", mode="before")
    @classmethod
    def deduplicate_evaluators(cls, v: list[Any]) -> list[Any]:
        seen: set[str] = set()
        deduped = []
        for item in v:
            name = item["name"] if isinstance(item, dict) else item.name
            if name not in seen:
                seen.add(name)
                deduped.append(item)
        return deduped


# ── API request / response schemas ────────────────────────────────────────────

class EvalRequest(BaseModel):
    """POST /evaluate request body."""
    experiment_config: ExperimentConfig
    async_mode: bool = False


class EvalResponse(BaseModel):
    """POST /evaluate response body."""
    run_id: str
    status: EvalStatus
    message: str = ""


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RegressionThreshold(BaseModel):
    """Per-metric regression gate used in CI/CD."""
    metric_name: str
    min_value: float | None = None       # fail if mean falls below this
    max_value: float | None = None       # fail if mean rises above this (e.g. latency)
    max_regression_pct: float | None = None  # fail if regresses >X% vs baseline


class RegressionConfig(BaseModel):
    """CI/CD regression testing configuration."""
    baseline_run_id: str
    thresholds: list[RegressionThreshold] = Field(default_factory=list)
    fail_on_regression: bool = True
