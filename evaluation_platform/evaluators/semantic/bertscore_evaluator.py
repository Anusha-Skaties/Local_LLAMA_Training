"""
evaluators/semantic/bertscore_evaluator.py
-------------------------------------------
BERTScore + sentence-embedding cosine similarity evaluator.

BERTScore uses contextual BERT embeddings to measure semantic similarity
between hypothesis and reference — far superior to ROUGE for open-ended
generation tasks like blog writing.

Cosine similarity uses sentence-transformers (already in requirements.txt)
to compute semantic closeness using the same embedding model used during
training evaluation in the original evaluate_model.py.

Batch evaluation is supported for both metrics (BERTScore is slow per-sample;
calling it once for the whole batch is ~4× faster).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from evaluation_platform.core.registry import EvaluatorRegistry
from evaluation_platform.core.schemas import (
    EvalSample,
    EvalStatus,
    EvaluatorConfig,
    EvaluatorResult,
    MetricCategory,
    MetricResult,
)
from evaluation_platform.evaluators.base import Evaluator
from evaluation_platform.logging_.structured import get_logger

log = get_logger(__name__)

_DEFAULT_BERT_MODEL = "microsoft/deberta-xlarge-mnli"  # best BERTScore model
_DEFAULT_ST_MODEL = "all-MiniLM-L6-v2"                # fast, good cosine sim


class SemanticEvaluator(Evaluator):
    """
    Computes BERTScore (P/R/F1) and embedding cosine similarity.

    Override models via config.extra:
        bert_model: "microsoft/deberta-xlarge-mnli"
        st_model:   "all-MiniLM-L6-v2"
    """

    @property
    def name(self) -> str:
        return "semantic"

    @property
    def version(self) -> str:
        return "1.0.0"

    def is_applicable(self, sample: EvalSample) -> bool:
        return bool(sample.output) and bool(sample.reference)

    def _evaluate(
        self,
        sample: EvalSample,
        config: EvaluatorConfig,
    ) -> list[MetricResult]:
        # For single-sample calls we delegate to the batch path with one item.
        results = self.evaluate_batch([sample], config)
        return results[0].metrics

    def evaluate_batch(
        self,
        samples: list[EvalSample],
        config: EvaluatorConfig,
    ) -> list[EvaluatorResult]:
        """
        True batch evaluation — computes BERTScore for all samples in one call.
        This is significantly faster than per-sample calls.
        """
        applicable = [s for s in samples if self.is_applicable(s)]
        skipped_ids = {s.id for s in samples if not self.is_applicable(s)}

        results: list[EvaluatorResult] = []

        if not applicable:
            return [
                EvaluatorResult(
                    evaluator_name=self.name,
                    evaluator_version=self.version,
                    sample_id=s.id,
                    task_type=s.task_type,
                    status=EvalStatus.SKIPPED,
                )
                for s in samples
            ]

        hypotheses = [s.output for s in applicable]
        references = [s.reference for s in applicable]

        bert_model = config.extra.get("bert_model", _DEFAULT_BERT_MODEL)
        st_model = config.extra.get("st_model", _DEFAULT_ST_MODEL)
        enabled = set(config.extra.get("metrics", ["bertscore", "cosine_similarity"]))

        # ── BERTScore (batched) ───────────────────────────────────────────────
        bert_scores: dict[str, list[float]] = {"P": [], "R": [], "F1": []}
        if "bertscore" in enabled:
            try:
                import bert_score as bs  # type: ignore
                start = time.perf_counter()
                P, R, F1 = bs.score(
                    hypotheses,
                    references,
                    model_type=bert_model,
                    lang="en",
                    verbose=False,
                    batch_size=config.batch_size or 8,
                )
                elapsed = time.perf_counter() - start
                log.debug(
                    "BERTScore batch complete",
                    n=len(applicable),
                    elapsed_s=round(elapsed, 2),
                )
                bert_scores = {
                    "P": P.tolist(),
                    "R": R.tolist(),
                    "F1": F1.tolist(),
                }
            except ImportError:
                log.warning("bert-score not installed; BERTScore skipped.")
            except Exception as exc:
                log.error("BERTScore failed", error=str(exc))

        # ── Cosine similarity (batched with sentence-transformers) ─────────────
        cosine_scores: list[float] = []
        if "cosine_similarity" in enabled:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

                model = SentenceTransformer(st_model)
                hyp_embs = model.encode(hypotheses, convert_to_numpy=True, show_progress_bar=False)
                ref_embs = model.encode(references, convert_to_numpy=True, show_progress_bar=False)
                cosine_scores = [
                    float(cosine_similarity([h], [r])[0][0])
                    for h, r in zip(hyp_embs, ref_embs)
                ]
            except ImportError:
                log.warning("sentence-transformers not installed; cosine similarity skipped.")
            except Exception as exc:
                log.error("Cosine similarity failed", error=str(exc))

        # ── Assemble per-sample results ────────────────────────────────────────
        for i, sample in enumerate(applicable):
            metrics: list[MetricResult] = []

            if bert_scores["F1"]:
                metrics += [
                    MetricResult(
                        name="bertscore_precision",
                        value=round(bert_scores["P"][i], 4),
                        category=MetricCategory.SEMANTIC,
                        unit="score",
                        description="BERTScore Precision",
                    ),
                    MetricResult(
                        name="bertscore_recall",
                        value=round(bert_scores["R"][i], 4),
                        category=MetricCategory.SEMANTIC,
                        unit="score",
                        description="BERTScore Recall",
                    ),
                    MetricResult(
                        name="bertscore_f1",
                        value=round(bert_scores["F1"][i], 4),
                        category=MetricCategory.SEMANTIC,
                        unit="score",
                        description="BERTScore F1 (primary semantic metric)",
                        passing=(
                            bert_scores["F1"][i] >= config.threshold
                            if config.threshold is not None else None
                        ),
                        threshold=config.threshold,
                    ),
                ]

            if cosine_scores:
                metrics.append(
                    MetricResult(
                        name="cosine_similarity",
                        value=round(cosine_scores[i], 4),
                        category=MetricCategory.SEMANTIC,
                        unit="score",
                        description="Sentence embedding cosine similarity",
                    )
                )

            results.append(
                EvaluatorResult(
                    evaluator_name=self.name,
                    evaluator_version=self.version,
                    sample_id=sample.id,
                    task_type=sample.task_type,
                    metrics=metrics,
                    status=EvalStatus.COMPLETED,
                )
            )

        # Add skipped entries for non-applicable samples
        for sample in samples:
            if sample.id in skipped_ids:
                results.append(
                    EvaluatorResult(
                        evaluator_name=self.name,
                        evaluator_version=self.version,
                        sample_id=sample.id,
                        task_type=sample.task_type,
                        status=EvalStatus.SKIPPED,
                    )
                )

        # Return in original order
        order = {s.id: i for i, s in enumerate(samples)}
        results.sort(key=lambda r: order.get(r.sample_id, 999))
        return results


# Auto-register
EvaluatorRegistry.register(SemanticEvaluator())
