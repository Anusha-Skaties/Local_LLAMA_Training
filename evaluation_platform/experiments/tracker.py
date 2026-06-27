"""
experiments/tracker.py
-----------------------
Unified experiment tracking facade supporting MLflow and W&B.

Design:
  - MLflowTracker and WandbTracker both implement BaseExperimentTracker.
  - CompositeTracker delegates to all enabled backends simultaneously.
  - The pipeline always uses CompositeTracker — it doesn't know which
    backends are active.
  - Both backends are optional: silently disabled if not installed.

Usage:
    tracker = build_tracker(config)
    run_id = tracker.start_run(config)
    tracker.log_metrics(run_id, {"rouge_l": 0.42, "bertscore_f1": 0.81})
    tracker.log_artifact(run_id, Path("outputs/eval_runs/my_run/report.html"))
    tracker.end_run(run_id)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation_platform.core.exceptions import ExperimentTrackerError
from evaluation_platform.core.protocols import BaseExperimentTracker
from evaluation_platform.core.schemas import ExperimentConfig
from evaluation_platform.logging_.structured import get_logger

log = get_logger(__name__)


# ── MLflow Tracker ─────────────────────────────────────────────────────────────

class MLflowTracker(BaseExperimentTracker):
    """Tracks experiments in MLflow."""

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._tracking_uri = tracking_uri
        self._run_id_map: dict[str, Any] = {}   # composite_key → mlflow run

        try:
            import mlflow  # type: ignore
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            self._mlflow = mlflow
            self._available = True
            log.info("MLflow tracker initialised", uri=tracking_uri or "default")
        except ImportError:
            self._available = False
            log.warning("mlflow not installed; MLflow tracking disabled.")

    def start_run(self, config: ExperimentConfig) -> str:
        if not self._available:
            return ""
        experiment_name = config.mlflow_experiment or config.name
        self._mlflow.set_experiment(experiment_name)
        run = self._mlflow.start_run(run_name=config.name, tags=config.tags)
        self._run_id_map[run.info.run_id] = run
        log.info("MLflow run started", run_id=run.info.run_id, experiment=experiment_name)
        return run.info.run_id

    def log_metrics(self, run_id: str, metrics: dict[str, float], step: int = 0) -> None:
        if not self._available or not run_id:
            return
        try:
            with self._mlflow.start_run(run_id=run_id):
                self._mlflow.log_metrics(metrics, step=step)
        except Exception as exc:
            log.warning("MLflow log_metrics failed", error=str(exc))

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        if not self._available or not run_id:
            return
        try:
            # MLflow params must be strings and ≤500 chars
            safe_params = {k: str(v)[:500] for k, v in params.items()}
            with self._mlflow.start_run(run_id=run_id):
                self._mlflow.log_params(safe_params)
        except Exception as exc:
            log.warning("MLflow log_params failed", error=str(exc))

    def log_artifact(self, run_id: str, path: Path) -> None:
        if not self._available or not run_id:
            return
        try:
            with self._mlflow.start_run(run_id=run_id):
                self._mlflow.log_artifact(str(path))
        except Exception as exc:
            log.warning("MLflow log_artifact failed", error=str(exc))

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        if not self._available or not run_id:
            return
        try:
            self._mlflow.end_run(status=status)
            log.info("MLflow run ended", run_id=run_id, status=status)
        except Exception as exc:
            log.warning("MLflow end_run failed", error=str(exc))


# ── W&B Tracker ───────────────────────────────────────────────────────────────

class WandbTracker(BaseExperimentTracker):
    """Tracks experiments in Weights & Biases."""

    def __init__(
        self,
        project: str = "llama-eval",
        entity: str | None = None,
    ) -> None:
        self._project = project
        self._entity = entity
        self._runs: dict[str, Any] = {}

        try:
            import wandb  # type: ignore
            self._wandb = wandb
            self._available = True
            log.info("W&B tracker initialised", project=project)
        except ImportError:
            self._available = False
            log.warning("wandb not installed; W&B tracking disabled.")

    def start_run(self, config: ExperimentConfig) -> str:
        if not self._available:
            return ""
        try:
            run = self._wandb.init(
                project=config.wandb_project or self._project,
                entity=self._entity,
                name=config.name,
                config=config.model_dump(),
                tags=list(config.tags.values()),
                reinit=True,
            )
            self._runs[run.id] = run
            log.info("W&B run started", run_id=run.id)
            return run.id
        except Exception as exc:
            log.warning("W&B start_run failed", error=str(exc))
            return ""

    def log_metrics(self, run_id: str, metrics: dict[str, float], step: int = 0) -> None:
        if not self._available or not run_id or run_id not in self._runs:
            return
        try:
            self._runs[run_id].log(metrics, step=step)
        except Exception as exc:
            log.warning("W&B log_metrics failed", error=str(exc))

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        if not self._available or not run_id or run_id not in self._runs:
            return
        try:
            self._runs[run_id].config.update(params)
        except Exception as exc:
            log.warning("W&B log_params failed", error=str(exc))

    def log_artifact(self, run_id: str, path: Path) -> None:
        if not self._available or not run_id or run_id not in self._runs:
            return
        try:
            artifact = self._wandb.Artifact(path.stem, type="evaluation_report")
            artifact.add_file(str(path))
            self._runs[run_id].log_artifact(artifact)
        except Exception as exc:
            log.warning("W&B log_artifact failed", error=str(exc))

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        if not self._available or not run_id or run_id not in self._runs:
            return
        try:
            self._runs[run_id].finish()
            log.info("W&B run finished", run_id=run_id)
        except Exception as exc:
            log.warning("W&B end_run failed", error=str(exc))


# ── Composite Tracker ─────────────────────────────────────────────────────────

class CompositeTracker(BaseExperimentTracker):
    """
    Delegates every call to all registered backend trackers.

    Returns the first non-empty run_id (MLflow takes precedence).
    """

    def __init__(self, trackers: list[BaseExperimentTracker]) -> None:
        self._trackers = trackers
        self._run_id_map: dict[str, list[str]] = {}

    def start_run(self, config: ExperimentConfig) -> str:
        run_ids = [t.start_run(config) for t in self._trackers]
        composite_id = next((r for r in run_ids if r), "")
        self._run_id_map[composite_id] = run_ids
        return composite_id

    def log_metrics(self, run_id: str, metrics: dict[str, float], step: int = 0) -> None:
        for tracker, tid in zip(self._trackers, self._run_id_map.get(run_id, [run_id] * len(self._trackers))):
            tracker.log_metrics(tid, metrics, step)

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        for tracker, tid in zip(self._trackers, self._run_id_map.get(run_id, [run_id] * len(self._trackers))):
            tracker.log_params(tid, params)

    def log_artifact(self, run_id: str, path: Path) -> None:
        for tracker, tid in zip(self._trackers, self._run_id_map.get(run_id, [run_id] * len(self._trackers))):
            tracker.log_artifact(tid, path)

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        for tracker, tid in zip(self._trackers, self._run_id_map.get(run_id, [run_id] * len(self._trackers))):
            tracker.end_run(tid, status)


# ── Factory ────────────────────────────────────────────────────────────────────

def build_tracker(config: ExperimentConfig) -> CompositeTracker:
    """
    Build a CompositeTracker from an ExperimentConfig.
    Enabled backends are determined by the config fields.
    """
    backends: list[BaseExperimentTracker] = []

    # MLflow is always attempted if configured or mlflow is installed
    backends.append(MLflowTracker())

    if config.wandb_project:
        backends.append(WandbTracker(project=config.wandb_project))

    return CompositeTracker(backends)
