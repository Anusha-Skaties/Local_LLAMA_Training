"""
pipelines/generation.py
------------------------
End-to-end evaluation pipeline for text generation tasks (blog generation,
summarization, chat).

Flow:
  1. Load dataset (JSONL → EvalSample list)
  2. Load model (base + QLoRA adapter OR merged model)
  3. Run inference: fill sample.output + record performance metadata
  4. Run all enabled evaluators
  5. Compute aggregate metrics
  6. Track to MLflow / W&B
  7. Generate reports (JSON, CSV, Markdown, HTML)
  8. (Optional) LangSmith tracing

Compatible with the existing scripts/evaluate_model.py output format.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation_platform.core.protocols import BasePipeline
from evaluation_platform.core.schemas import (
    EvalRunResult,
    EvalSample,
    EvalStatus,
    EvaluatorConfig,
    ExperimentConfig,
    TaskType,
)
from evaluation_platform.datasets.base import DatasetManager
from evaluation_platform.evaluators.base import Evaluator
from evaluation_platform.experiments.tracker import build_tracker
from evaluation_platform.logging_.structured import get_logger, new_correlation_id
from evaluation_platform.metrics.prometheus_metrics import (
    RUN_DURATION_HISTOGRAM,
    SAMPLES_COUNTER,
)
from evaluation_platform.reports.reporters import ReportManager, compute_aggregates
from evaluation_platform.tracing.langsmith_tracer import LangSmithTracer

log = get_logger(__name__)


class GenerationEvaluationPipeline(BasePipeline):
    """
    Full evaluation pipeline for local Llama / QLoRA adapter models.

    Handles model loading, batched inference, metric computation, tracking,
    and report generation.
    """

    def __init__(
        self,
        report_manager: ReportManager | None = None,
        langsmith_tracer: LangSmithTracer | None = None,
    ) -> None:
        self._report_manager = report_manager or ReportManager()
        self._langsmith = langsmith_tracer

    def run(self, config: ExperimentConfig) -> EvalRunResult:
        cid = new_correlation_id()
        pipeline_log = log.bind(
            correlation_id=cid,
            experiment=config.name,
            model=config.model_name,
        )
        pipeline_log.info("Pipeline started")

        run_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)
        tracker = build_tracker(config)
        mlflow_run_id = tracker.start_run(config)

        # Log experiment parameters
        tracker.log_params(
            mlflow_run_id,
            {
                "model_name": config.model_name,
                "adapter_dir": config.adapter_dir or "",
                "task_type": config.task_type.value,
                "dataset_name": config.dataset.name,
                "dataset_format": config.dataset.format,
                "max_samples": str(config.dataset.max_samples or "all"),
                "evaluators": ",".join(e.name for e in config.evaluators if e.enabled),
            },
        )

        result = EvalRunResult(
            run_id=run_id,
            experiment_name=config.name,
            model_name=config.model_name,
            model_version=config.model_version,
            dataset_id=config.dataset.name,
            dataset_name=config.dataset.name,
            task_type=config.task_type,
            status=EvalStatus.RUNNING,
            started_at=started_at,
            config_snapshot=config.model_dump(),
            tags=config.tags,
            mlflow_run_id=mlflow_run_id,
        )

        try:
            # ── 1. Load dataset ───────────────────────────────────────────────
            pipeline_log.info("Loading dataset")
            dataset_manager = DatasetManager()
            samples = dataset_manager.load(config.dataset)
            result.samples.extend(samples)
            pipeline_log.info("Dataset loaded", sample_count=len(samples))

            # ── 2. Load model ─────────────────────────────────────────────────
            pipeline_log.info("Loading model")
            model, tokenizer = self._load_model(config)

            # ── 3. Run inference ──────────────────────────────────────────────
            pipeline_log.info("Running inference", sample_count=len(samples))
            samples_with_output = self._run_inference(
                samples, model, tokenizer, config, pipeline_log
            )
            result.samples.clear()
            result.samples.extend(samples_with_output)

            # ── 4. Run evaluators ─────────────────────────────────────────────
            enabled_evaluators = self._get_evaluators(config)
            pipeline_log.info(
                "Running evaluators",
                evaluators=[e.name for e in enabled_evaluators],
            )
            all_eval_results = []
            for evaluator in enabled_evaluators:
                evaluator_config = self._get_evaluator_config(evaluator.name, config)
                eval_results = evaluator.evaluate_batch(
                    samples_with_output, evaluator_config
                )
                all_eval_results.extend(eval_results)
                pipeline_log.info(
                    "Evaluator complete",
                    evaluator=evaluator.name,
                    completed=sum(1 for r in eval_results if r.status == EvalStatus.COMPLETED),
                    failed=sum(1 for r in eval_results if r.status == EvalStatus.FAILED),
                )

            result.evaluator_results.extend(all_eval_results)

            # ── 5. Aggregate metrics ──────────────────────────────────────────
            aggregates = compute_aggregates(result)
            result.aggregate_metrics.extend(aggregates)

            # Log to tracker
            tracker.log_metrics(
                mlflow_run_id,
                {a.metric_name: a.mean for a in aggregates if a.mean is not None},
            )

            # Prometheus
            SAMPLES_COUNTER.labels(
                experiment_name=config.name,
                task_type=config.task_type.value,
            ).inc(len(samples_with_output))

            result.status = EvalStatus.COMPLETED

        except Exception as exc:
            pipeline_log.exception("Pipeline failed", error=str(exc))
            result.status = EvalStatus.FAILED
            result.error = str(exc)
            tracker.end_run(mlflow_run_id, status="FAILED")
            raise
        finally:
            completed_at = datetime.now(timezone.utc)
            result.completed_at = completed_at
            result.duration_seconds = (completed_at - started_at).total_seconds()

            RUN_DURATION_HISTOGRAM.labels(
                experiment_name=config.name,
                task_type=config.task_type.value,
            ).observe(result.duration_seconds)

            # ── 6. Generate reports ───────────────────────────────────────────
            if result.status == EvalStatus.COMPLETED:
                output_dir = Path(config.output_dir)
                written = self._report_manager.generate_all(result, output_dir)
                for path in written:
                    tracker.log_artifact(mlflow_run_id, path)
                pipeline_log.info(
                    "Reports written",
                    count=len(written),
                    paths=[str(p) for p in written],
                )

            tracker.end_run(mlflow_run_id, status="FINISHED" if result.status == EvalStatus.COMPLETED else "FAILED")
            pipeline_log.info(
                "Pipeline finished",
                status=result.status.value,
                duration_s=round(result.duration_seconds or 0, 1),
            )

        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_model(self, config: ExperimentConfig) -> tuple[Any, Any]:
        """Load the base model + adapter (or merged model)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_name = config.model_name
        adapter_dir = config.adapter_dir
        device_map = config.config_snapshot.get("device_map", "auto")

        log.info("Loading tokenizer", model=model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        log.info("Loading base model", model=model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config if torch.cuda.is_available() else None,
            device_map=device_map,
            trust_remote_code=True,
        )
        model.eval()

        if adapter_dir:
            log.info("Loading QLoRA adapter", adapter_dir=adapter_dir)
            from peft import PeftModel  # type: ignore
            model = PeftModel.from_pretrained(model, adapter_dir)
            model = model.merge_and_unload()
            log.info("Adapter merged")

        return model, tokenizer

    def _run_inference(
        self,
        samples: list[EvalSample],
        model: Any,
        tokenizer: Any,
        config: ExperimentConfig,
        pipeline_log: Any,
    ) -> list[EvalSample]:
        """Run model inference and attach outputs + performance metadata."""
        import torch
        import psutil

        max_new_tokens = config.config_snapshot.get("max_new_tokens", 1024)
        temperature = config.config_snapshot.get("temperature", 0.2)
        top_p = config.config_snapshot.get("top_p", 0.9)

        updated_samples: list[EvalSample] = []
        for i, sample in enumerate(samples):
            pipeline_log.debug("Inference", index=i + 1, total=len(samples), sample_id=sample.id)

            # Build prompt from messages (SFT chat template)
            prompt_messages = [m.model_dump() for m in sample.messages if m.role != "assistant"]
            if not prompt_messages:
                prompt_messages = [{"role": "user", "content": sample.input}]

            try:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt_text = sample.input

            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            prompt_token_count = inputs["input_ids"].shape[-1]

            # Performance tracking
            cpu_before = psutil.cpu_percent(interval=None)
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            latency_ms = (time.perf_counter() - start) * 1000
            cpu_after = psutil.cpu_percent(interval=None)

            completion_ids = output_ids[0][prompt_token_count:]
            generated_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            completion_tokens = len(completion_ids)

            perf_meta = {
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": int(prompt_token_count),
                "completion_tokens": int(completion_tokens),
                "tokens_per_second": round(completion_tokens / (latency_ms / 1000), 2) if latency_ms > 0 else 0,
                "cpu_percent": round((cpu_before + cpu_after) / 2, 1),
                "model_name": config.model_name,
            }

            if hasattr(torch, "cuda") and torch.cuda.is_available():
                gpu_mb = torch.cuda.max_memory_allocated() / 1e6
                perf_meta["gpu_memory_mb"] = round(gpu_mb, 1)

            updated_samples.append(
                sample.model_copy(
                    update={
                        "output": generated_text,
                        "metadata": {**sample.metadata, **perf_meta},
                    }
                )
            )

            if (i + 1) % 10 == 0 or (i + 1) == len(samples):
                pipeline_log.info(
                    "Inference progress",
                    done=i + 1,
                    total=len(samples),
                    avg_latency_ms=round(
                        sum(s.metadata.get("latency_ms", 0) for s in updated_samples) / len(updated_samples), 1
                    ),
                )

        return updated_samples

    def _get_evaluators(self, config: ExperimentConfig) -> list[Evaluator]:
        """Import and return all enabled evaluator instances."""
        # Trigger auto-registration by importing evaluator modules
        from evaluation_platform.evaluators.lexical import lexical_evaluator  # noqa: F401
        from evaluation_platform.evaluators.semantic import bertscore_evaluator  # noqa: F401
        from evaluation_platform.evaluators.performance import latency_evaluator  # noqa: F401
        try:
            from evaluation_platform.evaluators.deepeval import deepeval_evaluators  # noqa: F401
        except ImportError:
            log.warning("deepeval not installed; DeepEval evaluators disabled.")
        try:
            from evaluation_platform.evaluators.ragas import rag_evaluator  # noqa: F401
        except ImportError:
            log.warning("ragas not installed; Ragas evaluators disabled.")

        from evaluation_platform.core.registry import EvaluatorRegistry

        enabled_names = {e.name for e in config.evaluators if e.enabled}
        all_evaluators = EvaluatorRegistry.all()

        if not enabled_names:
            return all_evaluators  # run everything if none specified
        return [e for e in all_evaluators if e.name in enabled_names]

    def _get_evaluator_config(
        self, evaluator_name: str, config: ExperimentConfig
    ) -> EvaluatorConfig:
        """Find the EvaluatorConfig for a given name, or return defaults."""
        for ec in config.evaluators:
            if ec.name == evaluator_name:
                return ec
        return EvaluatorConfig(name=evaluator_name)
