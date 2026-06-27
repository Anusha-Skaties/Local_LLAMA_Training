"""
tests/conftest.py
-----------------
Shared pytest fixtures for the evaluation platform test suite.
"""
from __future__ import annotations

import pytest

from evaluation_platform.core.schemas import (
    DatasetConfig,
    EvalSample,
    EvaluatorConfig,
    TaskType,
)


@pytest.fixture()
def generation_sample() -> EvalSample:
    return EvalSample(
        id="test_001",
        task_type=TaskType.GENERATION,
        input="Write a short blog about vector databases.",
        output=(
            "Vector databases store embeddings instead of raw text. "
            "They enable semantic search by finding vectors that are mathematically "
            "close to a query embedding. Pinecone and Weaviate are popular options."
        ),
        reference=(
            "Vector databases are specialised systems that store high-dimensional "
            "embedding vectors. Unlike traditional databases, they support similarity "
            "search — finding the closest vectors to a query. "
            "Common options include Pinecone, Weaviate, and pgvector."
        ),
    )


@pytest.fixture()
def rag_sample() -> EvalSample:
    return EvalSample(
        id="rag_001",
        task_type=TaskType.RAG,
        input="What is the capital of France?",
        output="The capital of France is Paris.",
        reference="Paris is the capital of France.",
        contexts=[
            "France is a country in Western Europe. Its capital is Paris.",
            "Paris has been the capital of France since the 10th century.",
        ],
    )


@pytest.fixture()
def sample_with_no_output() -> EvalSample:
    return EvalSample(
        id="empty_001",
        task_type=TaskType.GENERATION,
        input="Write something.",
        output=None,
    )


@pytest.fixture()
def default_evaluator_config() -> EvaluatorConfig:
    return EvaluatorConfig(name="test", threshold=0.3)


@pytest.fixture()
def val_dataset_config(tmp_path) -> DatasetConfig:
    """Creates a minimal JSONL file and returns a DatasetConfig pointing to it."""
    import json
    data = [
        {
            "id": "s1",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain embeddings."},
                {"role": "assistant", "content": "Embeddings map text to vectors."},
            ],
        }
    ]
    path = tmp_path / "val.jsonl"
    with path.open("w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

    return DatasetConfig(name="test_val", format="jsonl", path=str(path))
