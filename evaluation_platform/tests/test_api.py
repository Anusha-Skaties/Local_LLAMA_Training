"""
tests/test_api.py
-----------------
Integration tests for the FastAPI evaluation platform.

Tests run without a real model — the pipeline is mocked so these tests
run instantly in CI and don't require a GPU.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from evaluation_platform.api.main import create_app
from evaluation_platform.core.schemas import (
    EvalRunResult,
    EvalStatus,
    TaskType,
)


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_health_sets_correlation_id(client):
    response = client.get("/health")
    assert "X-Correlation-ID" in response.headers


def test_health_accepts_caller_correlation_id(client):
    cid = "my-test-correlation-id-xyz"
    response = client.get("/health", headers={"X-Correlation-ID": cid})
    assert response.headers["X-Correlation-ID"] == cid


def test_list_evaluators(client):
    response = client.get("/metrics/registry")
    assert response.status_code == 200
    data = response.json()
    assert "evaluators" in data
    assert isinstance(data["evaluators"], list)


def test_list_runs_empty(client):
    response = client.get("/evaluate/runs")
    assert response.status_code == 200
    data = response.json()
    assert "runs" in data
    assert "total" in data


def test_get_unknown_run_returns_404(client):
    response = client.get("/evaluate/runs/does-not-exist-00000")
    assert response.status_code == 404


def _make_eval_request_payload() -> dict:
    return {
        "experiment_config": {
            "name": "test_experiment",
            "model_name": "mock-model",
            "task_type": "generation",
            "dataset": {
                "name": "mock_dataset",
                "format": "jsonl",
                "path": "data/processed/sft/val_conversations.jsonl",
                "max_samples": 2,
            },
            "evaluators": [
                {"name": "lexical", "enabled": True, "threshold": 0.3}
            ],
        },
        "async_mode": False,
    }


@pytest.fixture()
def mock_pipeline_result():
    """A fake EvalRunResult to return from the mocked pipeline."""
    from datetime import datetime, timezone
    result = EvalRunResult(
        run_id="mock-run-id-001",
        experiment_name="test_experiment",
        model_name="mock-model",
        dataset_id="mock_dataset",
        dataset_name="mock_dataset",
        task_type=TaskType.GENERATION,
        status=EvalStatus.COMPLETED,
        started_at=datetime.now(timezone.utc),
        duration_seconds=1.0,
    )
    return result


def test_submit_evaluation_sync_mocked(client, mock_pipeline_result):
    """POST /evaluate with mocked pipeline should return 202 and a run_id."""
    with patch(
        "evaluation_platform.api.routes.evaluate.GenerationEvaluationPipeline"
    ) as MockPipeline:
        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_pipeline_result
        MockPipeline.return_value = mock_instance

        response = client.post("/evaluate", json=_make_eval_request_payload())

    assert response.status_code == 202
    data = response.json()
    assert "run_id" in data
    assert data["status"] in ("completed", "running", "pending")


def test_submit_async_returns_immediately(client):
    """POST /evaluate with async_mode=true should return immediately."""
    payload = _make_eval_request_payload()
    payload["async_mode"] = True

    with patch(
        "evaluation_platform.api.routes.evaluate.GenerationEvaluationPipeline"
    ):
        response = client.post("/evaluate", json=payload)

    # Even if pipeline fails to start, we still get 202 with a run_id
    assert response.status_code == 202
    data = response.json()
    assert "run_id" in data


def test_get_run_after_submit(client, mock_pipeline_result):
    """After a synchronous run, GET /runs/{run_id} should return the result."""
    with patch(
        "evaluation_platform.api.routes.evaluate.GenerationEvaluationPipeline"
    ) as MockPipeline:
        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_pipeline_result
        MockPipeline.return_value = mock_instance

        post_response = client.post("/evaluate", json=_make_eval_request_payload())

    run_id = post_response.json()["run_id"]
    get_response = client.get(f"/evaluate/runs/{run_id}")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["run_id"] == run_id


def test_delete_run(client, mock_pipeline_result):
    """DELETE /runs/{run_id} should remove the run from the store."""
    with patch(
        "evaluation_platform.api.routes.evaluate.GenerationEvaluationPipeline"
    ) as MockPipeline:
        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_pipeline_result
        MockPipeline.return_value = mock_instance

        post_response = client.post("/evaluate", json=_make_eval_request_payload())

    run_id = post_response.json()["run_id"]
    del_response = client.delete(f"/evaluate/runs/{run_id}")
    assert del_response.status_code == 204

    get_response = client.get(f"/evaluate/runs/{run_id}")
    assert get_response.status_code == 404
