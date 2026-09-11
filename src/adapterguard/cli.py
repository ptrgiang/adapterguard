from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .hf_semantic import MissingHFDependencies, run_semantic_checks
from .models import Status, VerificationReport
from .static_checks import run_static_checks

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Verify that PEFT adapters still behave correctly before you ship them.",
)
console = Console()


def _render(report: VerificationReport) -> None:
    table = Table(title=f"AdapterGuard v{__version__}", show_lines=False)
    table.add_column("Status", width=8)
    table.add_column("Check")
    table.add_column("Result")

    symbols = {
        Status.PASS: "[green]PASS[/green]",
        Status.WARN: "[yellow]WARN[/yellow]",
        Status.FAIL: "[red]FAIL[/red]",
        Status.SKIP: "[dim]SKIP[/dim]",
    }
    for check in report.checks:
        result = check.message
        if check.metrics:
            metric_text = ", ".join(
                f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in check.metrics.items()
            )
            result = f"{result} [dim]({metric_text})[/dim]"
        table.add_row(symbols[check.status], check.name, result)

    console.print(table)
    if report.safe_to_ship:
        console.print("\n[bold green]VERDICT: SAFE TO SHIP[/bold green]")
    else:
        console.print("\n[bold red]VERDICT: UNSAFE TO SHIP[/bold red]")


@app.command()
def verify(
    adapter: Annotated[
        Path,
        typer.Option("--adapter", "-a", exists=False, help="Local PEFT adapter directory."),
    ],
    base: Annotated[
        str | None,
        typer.Option("--base", "-b", help="Base model ID/path. Defaults to adapter_config.json."),
    ] = None,
    prompts: Annotated[
        Path | None,
        typer.Option("--prompts", "-p", help="JSONL prompts for semantic checks."),
    ] = None,
    merged: Annotated[
        str | None,
        typer.Option("--merged", help="Optional exported/merged model ID or local path."),
    ] = None,
    device: Annotated[str, typer.Option("--device", help="auto, cpu, cuda, cuda:0, mps...")] = "auto",
    dtype: Annotated[
        str,
        typer.Option("--dtype", help="auto, float32, float16, or bfloat16."),
    ] = "auto",
    max_prompts: Annotated[
        int, typer.Option("--max-prompts", min=1, help="Maximum prompts to evaluate.")
    ] = 8,
    max_merge_diff: Annotated[
        float,
        typer.Option(
            "--max-merge-diff",
            min=0.0,
            help="Maximum mean absolute logit difference allowed after merge/export.",
        ),
    ] = 1e-3,
    min_top1_agreement: Annotated[
        float,
        typer.Option(
            "--min-top1-agreement",
            min=0.0,
            max=1.0,
            help="Minimum token-level top-1 agreement after merge/export.",
        ),
    ] = 0.999,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON instead of a Rich table."),
    ] = False,
) -> None:
    """Run static checks and, when --prompts is provided, semantic model checks."""
    config, checks = run_static_checks(adapter, expected_base=base)

    resolved_base = base
    if not resolved_base and config:
        configured = config.get("base_model_name_or_path")
        if isinstance(configured, str) and configured:
            resolved_base = configured

    if prompts and not any(check.status == Status.FAIL for check in checks):
        if not resolved_base:
            console.print("[red]Cannot run semantic checks without a base model.[/red]")
            raise typer.Exit(code=2)
        try:
            checks.extend(
                run_semantic_checks(
                    base_model=resolved_base,
                    adapter_path=adapter,
                    prompts_path=prompts,
                    merged_model=merged,
                    device=device,
                    dtype=dtype,
                    max_prompts=max_prompts,
                    max_merge_diff=max_merge_diff,
                    min_top1_agreement=min_top1_agreement,
                )
            )
        except MissingHFDependencies as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=2) from exc
        except (OSError, ValueError, RuntimeError) as exc:
            console.print(f"[red]Semantic verification failed to run: {exc}[/red]")
            raise typer.Exit(code=2) from exc

    report = VerificationReport(checks)
    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        _render(report)

    if not report.safe_to_ship:
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the AdapterGuard version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
