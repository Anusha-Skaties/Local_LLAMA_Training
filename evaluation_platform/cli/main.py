"""
cli/main.py
-----------
Typer CLI for running evaluations locally without the API server.

Commands:
  eval run      → run a full evaluation pipeline from a YAML config
  eval list     → list evaluators available in the registry
  eval report   → regenerate reports from an existing JSON result

Usage:
  python -m evaluation_platform.cli.main run \\
      --config evaluation_platform/configs/experiments/blog_eval.yaml

  python -m evaluation_platform.cli.main list-evaluators

  python -m evaluation_platform.cli.main report \\
      --result-path outputs/eval_runs/<run_id>/report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from evaluation_platform.logging_.structured import configure_root_logger

configure_root_logger()
app = typer.Typer(
    name="eval",
    help="AI Evaluation Platform CLI",
    add_completion=False,
)
console = Console()


@app.command("run")
def run_evaluation(
    config_path: Path = typer.Option(
        ..., "--config", "-c",
        help="Path to the YAML experiment config file.",
        exists=True,
        file_okay=True,
        readable=True,
    ),
    max_samples: Optional[int] = typer.Option(
        None, "--max-samples", "-n",
        help="Override dataset max_samples (useful for quick smoke tests).",
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o",
        help="Override output directory for reports.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Load config and dataset only; skip inference and evaluation.",
    ),
) -> None:
    """Run a full evaluation pipeline from a YAML config file."""
    from evaluation_platform.core.schemas import ExperimentConfig
    from evaluation_platform.pipelines.generation import GenerationEvaluationPipeline

    console.print(f"[bold blue]Loading config:[/bold blue] {config_path}")

    try:
        config = _load_config(config_path)
    except Exception as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if max_samples is not None:
        config.dataset.max_samples = max_samples
        console.print(f"[yellow]max_samples overridden to {max_samples}[/yellow]")

    if output_dir is not None:
        config.output_dir = str(output_dir)

    _print_config_summary(config)

    if dry_run:
        console.print("[yellow]Dry run — skipping inference and evaluation.[/yellow]")
        from evaluation_platform.datasets.base import DatasetManager
        dm = DatasetManager()
        samples = dm.load(config.dataset)
        console.print(f"[green]Dataset loaded: {len(samples)} samples[/green]")
        return

    pipeline = GenerationEvaluationPipeline()
    with console.status("[bold green]Running evaluation pipeline...[/bold green]"):
        try:
            result = pipeline.run(config)
        except Exception as exc:
            console.print(f"[bold red]Pipeline failed:[/bold red] {exc}")
            raise typer.Exit(code=1)

    _print_results_summary(result)


@app.command("list-evaluators")
def list_evaluators() -> None:
    """List all evaluators registered in the platform."""
    # Import modules to trigger auto-registration
    try:
        from evaluation_platform.evaluators.lexical import lexical_evaluator  # noqa
        from evaluation_platform.evaluators.semantic import bertscore_evaluator  # noqa
        from evaluation_platform.evaluators.performance import latency_evaluator  # noqa
        from evaluation_platform.evaluators.deepeval import deepeval_evaluators  # noqa
        from evaluation_platform.evaluators.ragas import rag_evaluator  # noqa
    except ImportError:
        pass

    from evaluation_platform.core.registry import EvaluatorRegistry

    evaluators = EvaluatorRegistry.all()
    if not evaluators:
        console.print("[yellow]No evaluators registered.[/yellow]")
        return

    table = Table(title="Registered Evaluators", show_header=True, header_style="bold blue")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Class")

    for e in evaluators:
        table.add_row(e.name, e.version, type(e).__name__)

    console.print(table)


@app.command("report")
def regenerate_report(
    result_path: Path = typer.Option(
        ..., "--result-path", "-r",
        help="Path to an existing report.json file.",
        exists=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o",
    ),
) -> None:
    """Regenerate all report formats from an existing JSON result."""
    from evaluation_platform.core.schemas import EvalRunResult
    from evaluation_platform.reports.reporters import ReportManager

    console.print(f"[bold blue]Loading result:[/bold blue] {result_path}")
    try:
        with result_path.open() as fh:
            data = json.load(fh)
        result = EvalRunResult.model_validate(data)
    except Exception as exc:
        console.print(f"[bold red]Failed to load result:[/bold red] {exc}")
        raise typer.Exit(code=1)

    out = output_dir or result_path.parent
    manager = ReportManager()
    written = manager.generate_all(result, out)
    for p in written:
        console.print(f"  [green]✓[/green] {p}")


@app.command("regression-check")
def regression_check(
    baseline_path: Path = typer.Option(..., "--baseline", help="Path to baseline report.json"),
    current_path: Path = typer.Option(..., "--current", help="Path to current report.json"),
    threshold_pct: float = typer.Option(5.0, "--threshold", help="Max allowed regression %"),
) -> None:
    """
    Compare two evaluation runs and fail (exit code 1) if metrics regressed.
    Designed for use in CI/CD pipelines.
    """
    from evaluation_platform.core.schemas import EvalRunResult

    with baseline_path.open() as f:
        baseline = EvalRunResult.model_validate(json.load(f))
    with current_path.open() as f:
        current = EvalRunResult.model_validate(json.load(f))

    baseline_metrics = {a.metric_name: a.mean for a in baseline.aggregate_metrics if a.mean}
    current_metrics = {a.metric_name: a.mean for a in current.aggregate_metrics if a.mean}

    violations: list[str] = []
    table = Table(title="Regression Report", show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Baseline")
    table.add_column("Current")
    table.add_column("Δ%")
    table.add_column("Status")

    for metric, baseline_val in baseline_metrics.items():
        if metric not in current_metrics:
            continue
        current_val = current_metrics[metric]
        if baseline_val == 0:
            continue
        delta_pct = ((current_val - baseline_val) / abs(baseline_val)) * 100
        is_regression = delta_pct < -threshold_pct
        status_str = "[red]FAIL[/red]" if is_regression else "[green]PASS[/green]"
        table.add_row(
            metric,
            f"{baseline_val:.4f}",
            f"{current_val:.4f}",
            f"{delta_pct:+.2f}%",
            status_str,
        )
        if is_regression:
            violations.append(
                f"{metric}: {baseline_val:.4f} → {current_val:.4f} ({delta_pct:+.2f}%)"
            )

    console.print(table)

    if violations:
        console.print(
            f"\n[bold red]REGRESSION DETECTED ({len(violations)} metric(s)):[/bold red]"
        )
        for v in violations:
            console.print(f"  • {v}")
        raise typer.Exit(code=1)
    else:
        console.print("\n[bold green]No regressions detected.[/bold green]")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_config(path: Path):
    """Load an ExperimentConfig from a YAML file."""
    import yaml
    from evaluation_platform.core.schemas import ExperimentConfig

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(raw)


def _print_config_summary(config) -> None:
    table = Table(title="Experiment Config", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_row("Name", config.name)
    table.add_row("Model", config.model_name)
    table.add_row("Adapter", config.adapter_dir or "none")
    table.add_row("Dataset", f"{config.dataset.name} ({config.dataset.format})")
    table.add_row("Max Samples", str(config.dataset.max_samples or "all"))
    table.add_row("Task", config.task_type.value)
    table.add_row(
        "Evaluators",
        ", ".join(e.name for e in config.evaluators if e.enabled) or "all",
    )
    console.print(table)


def _print_results_summary(result) -> None:
    status_color = "green" if result.status.value == "completed" else "red"
    console.print(
        f"\n[bold {status_color}]Status: {result.status.value.upper()}[/bold {status_color}]"
        f"  |  Run ID: {result.run_id}"
        f"  |  Duration: {result.duration_seconds:.1f}s"
        f"  |  Samples: {len(result.samples)}"
    )

    if result.aggregate_metrics:
        table = Table(title="Aggregate Metrics", show_header=True, header_style="bold blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", justify="right")
        table.add_column("P90", justify="right")
        table.add_column("Pass Rate", justify="right")

        for agg in sorted(result.aggregate_metrics, key=lambda a: a.metric_name):
            pass_rate = f"{agg.pass_rate:.1%}" if agg.pass_rate is not None else "—"
            p90 = f"{agg.p90:.4f}" if agg.p90 is not None else "—"
            table.add_row(agg.metric_name, f"{agg.mean:.4f}", p90, pass_rate)
        console.print(table)


if __name__ == "__main__":
    app()
