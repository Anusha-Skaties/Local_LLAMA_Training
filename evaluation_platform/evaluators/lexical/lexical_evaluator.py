"""
evaluators/lexical/lexical_evaluator.py
----------------------------------------
Lexical overlap metrics: ROUGE-1/2/L, BLEU, METEOR.

Bundles all lexical metrics into one evaluator to avoid loading the same
tokeniser multiple times.  Individual metrics are still individually
configurable via the YAML config (see configs/evaluators/lexical.yaml).

Metric categories:
  - ROUGE-1:  unigram overlap
  - ROUGE-2:  bigram overlap
  - ROUGE-L:  longest common subsequence
  - BLEU:     geometric mean of n-gram precisions (corpus-level quality signal)
  - METEOR:   harmonic mean of precision/recall with stemming and synonyms

All scores are in [0, 1].
"""
from __future__ import annotations

from evaluation_platform.core.registry import EvaluatorRegistry
from evaluation_platform.core.schemas import (
    EvalSample,
    EvaluatorConfig,
    MetricCategory,
    MetricResult,
)
from evaluation_platform.evaluators.base import Evaluator
from evaluation_platform.logging_.structured import get_logger

log = get_logger(__name__)


class LexicalEvaluator(Evaluator):
    """
    Computes ROUGE-1, ROUGE-2, ROUGE-L, BLEU, and METEOR in one pass.

    Requires: rouge-score, sacrebleu, nltk
    """

    @property
    def name(self) -> str:
        return "lexical"

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
        hypothesis = sample.output or ""
        reference = sample.reference or ""
        results: list[MetricResult] = []

        enabled_metrics: set[str] = set(
            config.extra.get(
                "metrics",
                ["rouge1", "rouge2", "rougeL", "bleu", "meteor"],
            )
        )

        # ── ROUGE ─────────────────────────────────────────────────────────────
        if enabled_metrics & {"rouge1", "rouge2", "rougeL"}:
            rouge_scores = self._compute_rouge(hypothesis, reference)
            if "rouge1" in enabled_metrics:
                results.append(
                    MetricResult(
                        name="rouge1_f1",
                        value=round(rouge_scores["rouge1"], 4),
                        category=MetricCategory.LEXICAL,
                        unit="score",
                        description="ROUGE-1 F1 (unigram overlap)",
                        passing=rouge_scores["rouge1"] >= (config.threshold or 0.0),
                        threshold=config.threshold,
                    )
                )
            if "rouge2" in enabled_metrics:
                results.append(
                    MetricResult(
                        name="rouge2_f1",
                        value=round(rouge_scores["rouge2"], 4),
                        category=MetricCategory.LEXICAL,
                        unit="score",
                        description="ROUGE-2 F1 (bigram overlap)",
                    )
                )
            if "rougeL" in enabled_metrics:
                results.append(
                    MetricResult(
                        name="rougeL_f1",
                        value=round(rouge_scores["rougeL"], 4),
                        category=MetricCategory.LEXICAL,
                        unit="score",
                        description="ROUGE-L F1 (longest common subsequence)",
                    )
                )

        # ── BLEU ──────────────────────────────────────────────────────────────
        if "bleu" in enabled_metrics:
            bleu_score = self._compute_bleu(hypothesis, reference)
            results.append(
                MetricResult(
                    name="bleu",
                    value=round(bleu_score, 4),
                    category=MetricCategory.LEXICAL,
                    unit="score",
                    description="SacreBLEU score (n-gram precision)",
                )
            )

        # ── METEOR ────────────────────────────────────────────────────────────
        if "meteor" in enabled_metrics:
            meteor_score = self._compute_meteor(hypothesis, reference)
            results.append(
                MetricResult(
                    name="meteor",
                    value=round(meteor_score, 4),
                    category=MetricCategory.LEXICAL,
                    unit="score",
                    description="METEOR score (precision/recall with stemming)",
                )
            )

        return results

    # ── Internal metric implementations ───────────────────────────────────────

    def _compute_rouge(self, hypothesis: str, reference: str) -> dict[str, float]:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
        )
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

    def _compute_bleu(self, hypothesis: str, reference: str) -> float:
        try:
            import sacrebleu
            result = sacrebleu.sentence_bleu(hypothesis, [reference])
            # sacrebleu returns 0-100; normalise to 0-1
            return result.score / 100.0
        except ImportError:
            log.warning("sacrebleu not installed; BLEU skipped.")
            return 0.0

    def _compute_meteor(self, hypothesis: str, reference: str) -> float:
        try:
            import nltk
            from nltk.translate.meteor_score import meteor_score as _meteor

            # Ensure NLTK data is available
            try:
                nltk.data.find("corpora/wordnet")
            except LookupError:
                nltk.download("wordnet", quiet=True)
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)

            hyp_tokens = hypothesis.split()
            ref_tokens = reference.split()
            if not hyp_tokens or not ref_tokens:
                return 0.0
            return float(_meteor([ref_tokens], hyp_tokens))
        except ImportError:
            log.warning("nltk not installed; METEOR skipped.")
            return 0.0


# Auto-register
EvaluatorRegistry.register(LexicalEvaluator())
