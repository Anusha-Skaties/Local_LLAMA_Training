"""
reports/base.py + all reporters: JSON, CSV, Markdown, HTML
-----------------------------------------------------------
All reporters inherit BaseReporter and implement generate().
The ReportManager runs all registered reporters in one call.

Reports written per run:
  <output_dir>/<run_id>/report.json
  <output_dir>/<run_id>/report.csv
  <output_dir>/<run_id>/report.md
  <output_dir>/<run_id>/report.html
"""
from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation_platform.core.exceptions import ReportGenerationError
from evaluation_platform.core.protocols import BaseReporter
from evaluation_platform.core.schemas import (
    AggregateMetrics,
    EvalRunResult,
    EvalStatus,
    MetricCategory,
    MetricResult,
)
from evaluation_platform.logging_.structured import get_logger

log = get_logger(__name__)


# ── Aggregate helpers ──────────────────────────────────────────────────────────

def compute_aggregates(result: EvalRunResult) -> list[AggregateMetrics]:
    """Compute per-metric statistics across all evaluator results."""
    # Collect all metric values grouped by metric name
    grouped: dict[str, dict[str, Any]] = {}
    for er in result.evaluator_results:
        if er.status != EvalStatus.COMPLETED:
            continue
        for m in er.metrics:
            if not isinstance(m.value, (int, float)) or m.value is None:
                continue
            key = m.name
            if key not in grouped:
                grouped[key] = {"values": [], "category": m.category, "threshold": m.threshold}
            grouped[key]["values"].append(float(m.value))

    aggregates: list[AggregateMetrics] = []
    for metric_name, data in grouped.items():
        vals = sorted(data["values"])
        n = len(vals)
        if n == 0:
            continue

        def pctile(p: float) -> float:
            idx = int(p * n)
            return vals[min(idx, n - 1)]

        pass_count = sum(1 for v in vals if data["threshold"] is not None and v >= data["threshold"])

        aggregates.append(
            AggregateMetrics(
                metric_name=metric_name,
                category=data["category"],
                mean=round(statistics.mean(vals), 4),
                median=round(statistics.median(vals), 4),
                std=round(statistics.stdev(vals), 4) if n > 1 else 0.0,
                min=round(min(vals), 4),
                max=round(max(vals), 4),
                p50=round(pctile(0.50), 4),
                p90=round(pctile(0.90), 4),
                p95=round(pctile(0.95), 4),
                p99=round(pctile(0.99), 4),
                pass_rate=round(pass_count / n, 4) if data["threshold"] else None,
                sample_count=n,
            )
        )

    return aggregates


# ── JSON Reporter ──────────────────────────────────────────────────────────────

class JsonReporter(BaseReporter):
    @property
    def format(self) -> str:
        return "json"

    def generate(self, result: EvalRunResult, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "report.json"
        try:
            payload = result.model_dump(mode="json")
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
            log.info("JSON report written", path=str(out_path))
            return out_path
        except Exception as exc:
            raise ReportGenerationError(f"JSON report failed: {exc}") from exc


# ── CSV Reporter ───────────────────────────────────────────────────────────────

class CsvReporter(BaseReporter):
    @property
    def format(self) -> str:
        return "csv"

    def generate(self, result: EvalRunResult, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "report.csv"
        try:
            # Flatten: one row per (sample_id, evaluator, metric)
            rows: list[dict[str, Any]] = []
            sample_map = {s.id: s for s in result.samples}
            for er in result.evaluator_results:
                sample = sample_map.get(er.sample_id)
                for m in er.metrics:
                    rows.append(
                        {
                            "run_id": result.run_id,
                            "experiment_name": result.experiment_name,
                            "model_name": result.model_name,
                            "dataset_name": result.dataset_name,
                            "task_type": result.task_type.value,
                            "sample_id": er.sample_id,
                            "evaluator": er.evaluator_name,
                            "metric_name": m.name,
                            "metric_value": m.value,
                            "metric_category": m.category.value,
                            "passing": m.passing,
                            "threshold": m.threshold,
                            "evaluator_status": er.status.value,
                            "evaluator_latency_ms": round(er.latency_ms, 2),
                            "input_preview": (sample.input[:100] if sample else ""),
                            "evaluated_at": er.evaluated_at.isoformat(),
                        }
                    )

            if not rows:
                rows = [{"run_id": result.run_id, "status": "no_metrics"}]

            with out_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            log.info("CSV report written", path=str(out_path), rows=len(rows))
            return out_path
        except Exception as exc:
            raise ReportGenerationError(f"CSV report failed: {exc}") from exc


# ── Markdown Reporter ──────────────────────────────────────────────────────────

class MarkdownReporter(BaseReporter):
    @property
    def format(self) -> str:
        return "markdown"

    def generate(self, result: EvalRunResult, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "report.md"
        try:
            aggregates = result.aggregate_metrics or compute_aggregates(result)
            lines: list[str] = [
                f"# Evaluation Report: {result.experiment_name}",
                "",
                f"**Run ID:** `{result.run_id}`  ",
                f"**Model:** `{result.model_name}`  ",
                f"**Dataset:** `{result.dataset_name}`  ",
                f"**Task:** `{result.task_type.value}`  ",
                f"**Status:** `{result.status.value}`  ",
                f"**Samples:** {len(result.samples)}  ",
                f"**Started:** {result.started_at.isoformat()}  ",
                "",
                "---",
                "",
                "## Aggregate Metrics",
                "",
                "| Metric | Mean | Median | Std | Min | Max | P90 | Pass Rate |",
                "|--------|------|--------|-----|-----|-----|-----|-----------|",
            ]
            for agg in sorted(aggregates, key=lambda a: a.metric_name):
                pass_rate_str = f"{agg.pass_rate:.1%}" if agg.pass_rate is not None else "N/A"
                lines.append(
                    f"| {agg.metric_name} | {agg.mean:.4f} | {agg.median:.4f} | "
                    f"{agg.std:.4f} | {agg.min:.4f} | {agg.max:.4f} | "
                    f"{agg.p90:.4f} | {pass_rate_str} |"
                )

            # Evaluator-level error summary
            errors = [er for er in result.evaluator_results if er.status == EvalStatus.FAILED]
            if errors:
                lines += [
                    "",
                    "## Evaluator Errors",
                    "",
                    f"**{len(errors)} evaluator runs failed.**",
                    "",
                    "| Sample ID | Evaluator | Error |",
                    "|-----------|-----------|-------|",
                ]
                for er in errors[:20]:  # cap at 20 for readability
                    lines.append(f"| {er.sample_id} | {er.evaluator_name} | {er.error} |")

            lines += [
                "",
                "---",
                f"*Generated by evaluation_platform at {datetime.now(timezone.utc).isoformat()}*",
            ]

            out_path.write_text("\n".join(lines), encoding="utf-8")
            log.info("Markdown report written", path=str(out_path))
            return out_path
        except Exception as exc:
            raise ReportGenerationError(f"Markdown report failed: {exc}") from exc


# ── HTML Reporter ──────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Eval Report: {experiment_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          max-width: 1200px; margin: 0 auto; padding: 2rem; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #3b82f6; padding-bottom: 0.5rem; }}
  h2 {{ color: #334155; margin-top: 2rem; }}
  .meta {{ background: #e2e8f0; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
  .meta span {{ margin-right: 1.5rem; font-size: 0.9rem; }}
  .badge {{ padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }}
  .badge-completed {{ background: #dcfce7; color: #166534; }}
  .badge-failed {{ background: #fee2e2; color: #991b1b; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th {{ background: #1e40af; color: white; padding: 12px 16px; text-align: left; font-size: 0.85rem; }}
  td {{ padding: 10px 16px; border-bottom: 1px solid #e2e8f0; font-size: 0.9rem; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f1f5f9; }}
  .score {{ font-weight: 600; }}
  .pass {{ color: #16a34a; }}
  .fail {{ color: #dc2626; }}
  footer {{ margin-top: 3rem; font-size: 0.8rem; color: #94a3b8; text-align: center; }}
</style>
</head>
<body>
<h1>Evaluation Report: {experiment_name}</h1>
<div class="meta">
  <span><strong>Run ID:</strong> {run_id}</span>
  <span><strong>Model:</strong> {model_name}</span>
  <span><strong>Dataset:</strong> {dataset_name}</span>
  <span><strong>Task:</strong> {task_type}</span>
  <span><strong>Samples:</strong> {sample_count}</span>
  <span class="badge badge-{status_class}">{status}</span>
</div>
<h2>Aggregate Metrics</h2>
<table>
<thead>
  <tr><th>Metric</th><th>Category</th><th>Mean</th><th>Median</th>
      <th>Std</th><th>Min</th><th>Max</th><th>P90</th><th>Pass Rate</th><th>Samples</th></tr>
</thead>
<tbody>
{metric_rows}
</tbody>
</table>
{error_section}
<footer>Generated by evaluation_platform &bull; {generated_at}</footer>
</body>
</html>"""


class HtmlReporter(BaseReporter):
    @property
    def format(self) -> str:
        return "html"

    def generate(self, result: EvalRunResult, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "report.html"
        try:
            aggregates = result.aggregate_metrics or compute_aggregates(result)

            metric_rows = ""
            for agg in sorted(aggregates, key=lambda a: a.metric_name):
                pass_rate_str = f"{agg.pass_rate:.1%}" if agg.pass_rate is not None else "—"
                metric_rows += (
                    f"<tr>"
                    f"<td class='score'>{agg.metric_name}</td>"
                    f"<td>{agg.category.value}</td>"
                    f"<td class='score'>{agg.mean:.4f}</td>"
                    f"<td>{agg.median:.4f}</td>"
                    f"<td>{agg.std:.4f}</td>"
                    f"<td>{agg.min:.4f}</td>"
                    f"<td>{agg.max:.4f}</td>"
                    f"<td>{agg.p90:.4f}</td>"
                    f"<td>{pass_rate_str}</td>"
                    f"<td>{agg.sample_count}</td>"
                    f"</tr>\n"
                )

            errors = [er for er in result.evaluator_results if er.status == EvalStatus.FAILED]
            error_section = ""
            if errors:
                error_rows = "".join(
                    f"<tr><td>{e.sample_id}</td><td>{e.evaluator_name}</td>"
                    f"<td class='fail'>{e.error}</td></tr>"
                    for e in errors[:20]
                )
                error_section = (
                    "<h2>Evaluator Errors</h2>"
                    "<table><thead><tr><th>Sample</th><th>Evaluator</th><th>Error</th></tr></thead>"
                    f"<tbody>{error_rows}</tbody></table>"
                )

            html = _HTML_TEMPLATE.format(
                experiment_name=result.experiment_name,
                run_id=result.run_id,
                model_name=result.model_name,
                dataset_name=result.dataset_name,
                task_type=result.task_type.value,
                sample_count=len(result.samples),
                status=result.status.value.upper(),
                status_class="completed" if result.status == EvalStatus.COMPLETED else "failed",
                metric_rows=metric_rows or "<tr><td colspan='10'>No metrics computed</td></tr>",
                error_section=error_section,
                generated_at=datetime.now(timezone.utc).isoformat(),
            )

            out_path.write_text(html, encoding="utf-8")
            log.info("HTML report written", path=str(out_path))
            return out_path
        except Exception as exc:
            raise ReportGenerationError(f"HTML report failed: {exc}") from exc


# ── Report Manager ─────────────────────────────────────────────────────────────

class ReportManager:
    """Runs all registered reporters and returns the list of written paths."""

    def __init__(self, reporters: list[BaseReporter] | None = None) -> None:
        self._reporters = reporters or [
            JsonReporter(),
            CsvReporter(),
            MarkdownReporter(),
            HtmlReporter(),
        ]

    def generate_all(self, result: EvalRunResult, output_dir: Path) -> list[Path]:
        """Write all reports; log but do not crash on individual reporter failure."""
        # Recompute aggregates once and attach
        if not result.aggregate_metrics:
            result.aggregate_metrics.extend(compute_aggregates(result))

        output_dir = output_dir / result.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        for reporter in self._reporters:
            try:
                path = reporter.generate(result, output_dir)
                written.append(path)
            except ReportGenerationError as exc:
                log.error(
                    "Reporter failed",
                    format=reporter.format,
                    error=str(exc),
                )
        return written
