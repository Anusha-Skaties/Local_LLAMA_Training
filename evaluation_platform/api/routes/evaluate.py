"""
api/routes/evaluate.py
-----------------------
POST /evaluate    → submit a synchronous or async evaluation run
GET  /runs/{id}   → get the status/result of a run
GET  /runs        → list all runs (paginated)
DELETE /runs/{id} → delete a run's results
"""
from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status

from evaluation_platform.core.exceptions import RunNotFoundError
from evaluation_platform.core.schemas import (
    EvalRequest,
    EvalResponse,
    EvalRunResult,
    EvalStatus,
    ExperimentConfig,
)
from evaluation_platform.logging_.structured import get_logger
from evaluation_platform.pipelines.generation import GenerationEvaluationPipeline

log = get_logger(__name__)
router = APIRouter(prefix="/evaluate", tags=["evaluate"])

# In-memory run store (replace with a database in production)
_runs: dict[str, EvalRunResult | dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=2)


def _run_in_background(run_id: str, config: ExperimentConfig) -> None:
    """Execute an evaluation pipeline in a background thread."""
    from evaluation_platform.core.schemas import EvalRunResult, EvalStatus
    import datetime

    log.info("Background evaluation started", run_id=run_id)
    pipeline = GenerationEvaluationPipeline()
    try:
        result = pipeline.run(config)
        _runs[run_id] = result
        log.info("Background evaluation completed", run_id=run_id)
    except Exception as exc:
        _runs[run_id] = {
            "run_id": run_id,
            "status": EvalStatus.FAILED.value,
            "error": str(exc),
        }
        log.error("Background evaluation failed", run_id=run_id, error=str(exc))


@router.post(
    "",
    response_model=EvalResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an evaluation run",
)
async def submit_evaluation(
    request: EvalRequest,
    background_tasks: BackgroundTasks,
) -> EvalResponse:
    """
    Submit an evaluation experiment.

    - **async_mode=true** → returns immediately with a run_id; poll GET /runs/{run_id}
    - **async_mode=false** → blocks until complete (suitable for small datasets / CI)
    """
    run_id = str(uuid.uuid4())
    config = request.experiment_config

    if request.async_mode:
        _runs[run_id] = {"run_id": run_id, "status": EvalStatus.RUNNING.value}
        background_tasks.add_task(_run_in_background, run_id, config)
        log.info("Async evaluation submitted", run_id=run_id)
        return EvalResponse(
            run_id=run_id,
            status=EvalStatus.RUNNING,
            message="Evaluation started. Poll GET /evaluate/runs/{run_id} for status.",
        )
    else:
        # Synchronous — block and return the result
        pipeline = GenerationEvaluationPipeline()
        try:
            result = pipeline.run(config)
            _runs[result.run_id] = result
            return EvalResponse(
                run_id=result.run_id,
                status=result.status,
                message=f"Evaluation completed. {len(result.samples)} samples, "
                        f"{len(result.aggregate_metrics)} metrics.",
            )
        except Exception as exc:
            log.exception("Synchronous evaluation failed", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            )


@router.get(
    "/runs/{run_id}",
    summary="Get evaluation run status and results",
)
async def get_run(run_id: str) -> dict[str, Any]:
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )
    if isinstance(run, EvalRunResult):
        return run.model_dump(mode="json")
    return run


@router.get(
    "/runs",
    summary="List all evaluation runs",
)
async def list_runs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    all_ids = list(_runs.keys())
    page = all_ids[offset : offset + limit]
    items = []
    for rid in page:
        run = _runs[rid]
        if isinstance(run, EvalRunResult):
            items.append(
                {
                    "run_id": run.run_id,
                    "experiment_name": run.experiment_name,
                    "model_name": run.model_name,
                    "status": run.status.value,
                    "started_at": run.started_at.isoformat(),
                    "duration_seconds": run.duration_seconds,
                    "sample_count": len(run.samples),
                }
            )
        else:
            items.append(run)
    return {"total": len(all_ids), "offset": offset, "limit": limit, "runs": items}


@router.delete(
    "/runs/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a run from the in-memory store",
)
async def delete_run(run_id: str) -> None:
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    del _runs[run_id]
