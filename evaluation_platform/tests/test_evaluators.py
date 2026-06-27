"""
tests/test_evaluators.py
-------------------------
Unit tests for all evaluator implementations.

These tests run without a GPU and without API keys.
DeepEval / Ragas tests are skipped if the packages are not installed.
"""
from __future__ import annotations

import pytest

from evaluation_platform.core.registry import EvaluatorRegistry
from evaluation_platform.core.schemas import (
    EvalSample,
    EvalStatus,
    EvaluatorConfig,
    MetricCategory,
    TaskType,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _import_evaluators():
    """Force auto-registration of all evaluators."""
    from evaluation_platform.evaluators.lexical import lexical_evaluator  # noqa
    from evaluation_platform.evaluators.semantic import bertscore_evaluator  # noqa
    from evaluation_platform.evaluators.performance import latency_evaluator  # noqa


_import_evaluators()


# ── Registry tests ─────────────────────────────────────────────────────────────

def test_evaluator_registry_not_empty():
    assert len(EvaluatorRegistry.names()) > 0


def test_evaluator_registry_get_unknown_raises():
    from evaluation_platform.core.exceptions import EvaluatorNotFoundError
    with pytest.raises(EvaluatorNotFoundError):
        EvaluatorRegistry.get("does_not_exist_xyz")


# ── Lexical evaluator ──────────────────────────────────────────────────────────

class TestLexicalEvaluator:
    def test_rouge_scores_in_range(self, generation_sample, default_evaluator_config):
        evaluator = EvaluatorRegistry.get("lexical")
        result = evaluator.evaluate_sample(generation_sample, default_evaluator_config)

        assert result.status == EvalStatus.COMPLETED
        metric_names = {m.name for m in result.metrics}
        assert "rouge1_f1" in metric_names
        assert "rouge2_f1" in metric_names
        assert "rougeL_f1" in metric_names

        for m in result.metrics:
            if isinstance(m.value, float):
                assert 0.0 <= m.value <= 1.0, f"{m.name} = {m.value} out of range"

    def test_lexical_skipped_without_output(self, sample_with_no_output, default_evaluator_config):
        evaluator = EvaluatorRegistry.get("lexical")
        result = evaluator.evaluate_sample(sample_with_no_output, default_evaluator_config)
        assert result.status == EvalStatus.SKIPPED

    def test_lexical_skipped_without_reference(self, default_evaluator_config):
        sample = EvalSample(
            id="no_ref",
            task_type=TaskType.GENERATION,
            input="Tell me about AI.",
            output="AI is a field of computer science.",
            reference=None,
        )
        evaluator = EvaluatorRegistry.get("lexical")
        result = evaluator.evaluate_sample(sample, default_evaluator_config)
        assert result.status == EvalStatus.SKIPPED

    def test_perfect_match_gives_high_scores(self, default_evaluator_config):
        text = "The quick brown fox jumps over the lazy dog."
        sample = EvalSample(
            id="perfect",
            task_type=TaskType.GENERATION,
            input="Write a sentence.",
            output=text,
            reference=text,
        )
        evaluator = EvaluatorRegistry.get("lexical")
        result = evaluator.evaluate_sample(sample, default_evaluator_config)
        rouge_l = result.get_metric("rougeL_f1")
        assert rouge_l is not None
        assert rouge_l.value > 0.9


# ── Semantic evaluator ────────────────────────────────────────────────────────

class TestSemanticEvaluator:
    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("sentence_transformers"),
        reason="sentence-transformers not installed",
    )
    def test_cosine_similarity_in_range(self, generation_sample, default_evaluator_config):
        config = EvaluatorConfig(
            name="semantic",
            extra={"metrics": ["cosine_similarity"]},
        )
        evaluator = EvaluatorRegistry.get("semantic")
        result = evaluator.evaluate_sample(generation_sample, config)

        assert result.status == EvalStatus.COMPLETED
        cosine = result.get_metric("cosine_similarity")
        assert cosine is not None
        assert 0.0 <= cosine.value <= 1.0

    def test_semantic_skipped_without_output(self, sample_with_no_output, default_evaluator_config):
        evaluator = EvaluatorRegistry.get("semantic")
        result = evaluator.evaluate_sample(sample_with_no_output, default_evaluator_config)
        assert result.status == EvalStatus.SKIPPED


# ── Performance evaluator ─────────────────────────────────────────────────────

class TestPerformanceEvaluator:
    def test_extracts_latency_from_metadata(self, default_evaluator_config):
        sample = EvalSample(
            id="perf_001",
            task_type=TaskType.GENERATION,
            input="Hello",
            output="Hello back!",
            metadata={
                "latency_ms": 312.5,
                "prompt_tokens": 10,
                "completion_tokens": 50,
                "gpu_memory_mb": 2048.0,
            },
        )
        evaluator = EvaluatorRegistry.get("performance")
        result = evaluator.evaluate_sample(sample, default_evaluator_config)

        assert result.status == EvalStatus.COMPLETED
        metric_names = {m.name for m in result.metrics}
        assert "latency_ms" in metric_names
        assert "prompt_tokens" in metric_names
        assert "completion_tokens" in metric_names
        assert "total_tokens" in metric_names
        assert "tokens_per_second" in metric_names
        assert "gpu_memory_mb" in metric_names

        latency = result.get_metric("latency_ms")
        assert latency.value == 312.5

    def test_tokens_per_second_derived(self, default_evaluator_config):
        sample = EvalSample(
            id="tps_001",
            task_type=TaskType.GENERATION,
            input="x",
            output="y" * 100,
            metadata={"completion_tokens": 100, "latency_ms": 1000.0},
        )
        evaluator = EvaluatorRegistry.get("performance")
        result = evaluator.evaluate_sample(sample, default_evaluator_config)
        tps = result.get_metric("tokens_per_second")
        assert tps is not None
        assert abs(tps.value - 100.0) < 0.5

    def test_cost_estimation(self, default_evaluator_config):
        sample = EvalSample(
            id="cost_001",
            task_type=TaskType.GENERATION,
            input="x",
            output="y",
            metadata={
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "model_name": "gpt-4o-mini",
            },
        )
        evaluator = EvaluatorRegistry.get("performance")
        result = evaluator.evaluate_sample(sample, default_evaluator_config)
        cost = result.get_metric("cost_usd_estimated")
        assert cost is not None
        assert cost.value > 0


# ── Dataset loader ─────────────────────────────────────────────────────────────

class TestJsonlLoader:
    def test_loads_sft_format(self, val_dataset_config):
        from evaluation_platform.datasets.loaders.jsonl_loader import JsonlDatasetLoader
        loader = JsonlDatasetLoader()
        samples = loader.load(val_dataset_config)
        assert len(samples) == 1
        assert samples[0].input == "Explain embeddings."
        assert samples[0].reference == "Embeddings map text to vectors."
        assert samples[0].output is None  # not yet filled

    def test_max_samples_respected(self, tmp_path):
        import json
        from evaluation_platform.core.schemas import DatasetConfig

        path = tmp_path / "big.jsonl"
        with path.open("w") as f:
            for i in range(50):
                f.write(json.dumps({
                    "id": f"s{i}",
                    "messages": [
                        {"role": "user", "content": f"Question {i}"},
                        {"role": "assistant", "content": f"Answer {i}"},
                    ],
                }) + "\n")

        config = DatasetConfig(name="big", format="jsonl", path=str(path), max_samples=10)
        from evaluation_platform.datasets.loaders.jsonl_loader import JsonlDatasetLoader
        samples = JsonlDatasetLoader().load(config)
        assert len(samples) == 10


# ── Report generation ──────────────────────────────────────────────────────────

class TestReporters:
    def _make_run_result(self):
        from evaluation_platform.core.schemas import (
            EvalRunResult,
            EvaluatorResult,
            MetricResult,
            TaskType,
        )
        from datetime import datetime, timezone
        result = EvalRunResult(
            run_id="test_run_123",
            experiment_name="test_exp",
            model_name="test_model",
            dataset_id="test_ds",
            dataset_name="test_ds",
            task_type=TaskType.GENERATION,
            started_at=datetime.now(timezone.utc),
        )
        result.samples.append(
            EvalSample(id="s1", task_type=TaskType.GENERATION, input="q", output="a", reference="ref")
        )
        result.evaluator_results.append(
            EvaluatorResult(
                evaluator_name="lexical",
                evaluator_version="1.0.0",
                sample_id="s1",
                task_type=TaskType.GENERATION,
                metrics=[
                    MetricResult(name="rouge1_f1", value=0.72, category=MetricCategory.LEXICAL, unit="score"),
                    MetricResult(name="rougeL_f1", value=0.65, category=MetricCategory.LEXICAL, unit="score"),
                ],
                status=EvalStatus.COMPLETED,
            )
        )
        return result

    def test_json_reporter(self, tmp_path):
        from evaluation_platform.reports.reporters import JsonReporter
        result = self._make_run_result()
        reporter = JsonReporter()
        path = reporter.generate(result, tmp_path)
        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert data["run_id"] == "test_run_123"

    def test_csv_reporter(self, tmp_path):
        from evaluation_platform.reports.reporters import CsvReporter
        result = self._make_run_result()
        path = CsvReporter().generate(result, tmp_path)
        assert path.exists()
        content = path.read_text()
        assert "rouge1_f1" in content

    def test_markdown_reporter(self, tmp_path):
        from evaluation_platform.reports.reporters import MarkdownReporter
        result = self._make_run_result()
        path = MarkdownReporter().generate(result, tmp_path)
        assert path.exists()
        content = path.read_text()
        assert "# Evaluation Report" in content

    def test_html_reporter(self, tmp_path):
        from evaluation_platform.reports.reporters import HtmlReporter
        result = self._make_run_result()
        path = HtmlReporter().generate(result, tmp_path)
        assert path.exists()
        assert "<html" in path.read_text()
