"""
tracing/langsmith_tracer.py
---------------------------
LangSmith tracing integration.

Wraps every LLM call and evaluation run as a LangSmith Run so that:
  - Full prompt / response pairs are stored with latency and token counts
  - Every evaluation score is attached to the originating LLM run
  - Experiments can be compared in the LangSmith UI
  - Regression trends are tracked across runs

Usage:
    tracer = LangSmithTracer(project="llama-eval")
    with tracer.trace_eval(run_name="rouge_eval", inputs={"prompt": "..."}) as run_id:
        result = evaluator.evaluate_sample(sample, config)
        tracer.log_feedback(run_id, "rouge_l", result.primary_score())
"""
from __future__ import annotations

import contextlib
import os
import time
import uuid
from typing import Any, Generator

from evaluation_platform.logging_.structured import get_logger

log = get_logger(__name__)

try:
    from langsmith import Client
    from langsmith.schemas import RunTypeEnum
    _LANGSMITH_AVAILABLE = True
except ImportError:
    _LANGSMITH_AVAILABLE = False
    log.warning("langsmith not installed; LangSmith tracing is disabled.")


class LangSmithTracer:
    """
    Facade over the LangSmith Python client.

    All methods are safe to call even when langsmith is not installed —
    they silently become no-ops, which means evaluator code never needs
    to guard against the absence of LangSmith.
    """

    def __init__(
        self,
        project: str = "llama-eval",
        api_key: str | None = None,
        api_url: str | None = None,
    ) -> None:
        self._project = project
        self._client: Any = None

        if not _LANGSMITH_AVAILABLE:
            return

        api_key = api_key or os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
        api_url = api_url or os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

        if not api_key:
            log.warning(
                "LANGCHAIN_API_KEY not set; LangSmith tracing is disabled."
            )
            return

        try:
            self._client = Client(api_url=api_url, api_key=api_key)
            log.info("LangSmith tracer initialised", project=project, url=api_url)
        except Exception as exc:
            log.warning("Failed to initialise LangSmith client", error=str(exc))

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextlib.contextmanager
    def trace_eval(
        self,
        run_name: str,
        inputs: dict[str, Any],
        run_type: str = "chain",
        tags: list[str] | None = None,
    ) -> Generator[str, None, None]:
        """
        Context manager that creates a LangSmith Run and yields its run_id.

        The run is patched with outputs / error on exit.

        Usage:
            with tracer.trace_eval("my_eval", inputs={"prompt": "..."}) as run_id:
                result = do_eval(...)
                tracer.log_feedback(run_id, "rouge_l", 0.72)
        """
        run_id = str(uuid.uuid4())
        start_time = time.time()
        outputs: dict[str, Any] = {}
        error: str | None = None

        if self.enabled:
            try:
                self._client.create_run(
                    id=run_id,
                    name=run_name,
                    run_type=run_type,
                    inputs=inputs,
                    project_name=self._project,
                    tags=tags or [],
                    start_time=start_time,
                )
            except Exception as exc:
                log.warning("LangSmith create_run failed", error=str(exc))

        try:
            yield run_id
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            end_time = time.time()
            if self.enabled:
                try:
                    self._client.update_run(
                        run_id,
                        outputs=outputs,
                        error=error,
                        end_time=end_time,
                    )
                except Exception as exc:
                    log.warning("LangSmith update_run failed", error=str(exc))

    def log_feedback(
        self,
        run_id: str,
        key: str,
        score: float | None,
        comment: str | None = None,
    ) -> None:
        """Attach an evaluation score (feedback) to a LangSmith run."""
        if not self.enabled:
            return
        try:
            self._client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                comment=comment,
                source_info={"source": "evaluation_platform"},
            )
        except Exception as exc:
            log.warning(
                "LangSmith log_feedback failed",
                run_id=run_id,
                key=key,
                error=str(exc),
            )

    def log_all_metrics(
        self,
        run_id: str,
        metrics: dict[str, float | None],
    ) -> None:
        """Log every metric in a dict as individual feedback entries."""
        for key, score in metrics.items():
            self.log_feedback(run_id, key, score)
