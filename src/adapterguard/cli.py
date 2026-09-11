from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .fingerprints import fingerprint_artifact
from .hf_semantic import MissingHFDependencies, run_semantic_checks
from .models import ArtifactFingerprint, Status, VerificationReport
from .reports import write_json_report, write_markdown_report
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
                f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
                for key, value in check.metrics.items()
            )
            result = f"{result} [dim]({metric_text})[/dim]"
        table.add_row(symbols[check.status], check.name, result)

    console.print(table)
    if report.fingerprints:
        console.print("\n[bold]Artifact fingerprints[/bold]")
        for label, fingerprint in report.fingerprints.items():
            console.print(
                f"  {label}: {fingerprint.sha256[:16]}… "
                f"[dim]({fingerprint.mode})[/dim]"
            )

    if report.safe_to_ship:
        console.print("\n[bold green]VERDICT: SAFE TO SHIP[/bold green]")
    else:
        console.print("\n[bold red]VERDICT: UNSAFE TO SHIP[/bold red]")


def _collect_fingerprints(
    *,
    adapter: Path,
    base: str | None,
    merged: str | None,
    quantized: str | None,
    mode: str,
) -> dict[str, ArtifactFingerprint]:
    fingerprints: dict[str, ArtifactFingerprint] = {}
    sources: list[tuple[str, str | Path | None]] = [
        ("adapter", adapter),
        ("base", base),
        ("merged", merged),
        ("quantized", quantized),
    ]
    for label, source in sources:
        if source is None:
            continue
        fingerprint = fingerprint_artifact(label, source, mode=mode)
        if fingerprint is not None:
            fingerprints[label] = fingerprint
    return fingerprints


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
    quantized: Annotated[
        str | None,
        typer.Option(
            "--quantized",
            help="Optional quantized model ID/path to compare against the in-memory merge.",
        ),
    ] = None,
    endpoint: Annotated[
        str | None,
        typer.Option(
            "--endpoint",
            help="OpenAI-compatible API base URL, e.g. http://localhost:8000/v1.",
        ),
    ] = None,
    runtime_model: Annotated[
        str | None,
        typer.Option(
            "--runtime-model",
            help="Served model name sent to the endpoint. Defaults to --base.",
        ),
    ] = None,
    runtime_api_key_env: Annotated[
        str,
        typer.Option(
            "--runtime-api-key-env",
            help="Environment variable containing the endpoint API key.",
        ),
    ] = "ADAPTERGUARD_API_KEY",
    runtime_max_tokens: Annotated[
        int,
        typer.Option(
            "--runtime-max-tokens",
            min=1,
            help="Greedy completion length used for local-vs-runtime comparison.",
        ),
    ] = 8,
    runtime_timeout: Annotated[
        float,
        typer.Option(
            "--runtime-timeout",
            min=0.1,
            help="Per-request runtime endpoint timeout in seconds.",
        ),
    ] = 30.0,
    min_runtime_first_token_agreement: Annotated[
        float,
        typer.Option(
            "--min-runtime-first-token-agreement",
            min=0.0,
            max=1.0,
            help="Minimum first generated token agreement with the verified artifact.",
        ),
    ] = 1.0,
    min_runtime_exact_match_rate: Annotated[
        float,
        typer.Option(
            "--min-runtime-exact-match-rate",
            min=0.0,
            max=1.0,
            help="Minimum exact greedy completion match rate with the verified artifact.",
        ),
    ] = 1.0,
    device: Annotated[
        str, typer.Option("--device", help="auto, cpu, cuda, cuda:0, mps...")
    ] = "auto",
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
    max_quantized_diff: Annotated[
        float,
        typer.Option(
            "--max-quantized-diff",
            min=0.0,
            help="Maximum mean absolute logit drift allowed after quantization.",
        ),
    ] = 0.05,
    min_quantized_top1_agreement: Annotated[
        float,
        typer.Option(
            "--min-quantized-top1-agreement",
            min=0.0,
            max=1.0,
            help="Minimum token-level top-1 agreement allowed after quantization.",
        ),
    ] = 0.98,
    include_prompts: Annotated[
        bool,
        typer.Option(
            "--include-prompts",
            help="Include raw prompt/output text in evidence reports. Off by default.",
        ),
    ] = False,
    fingerprint_mode: Annotated[
        str,
        typer.Option(
            "--fingerprint-mode",
            help="Artifact fingerprint mode: sampled, full, or off.",
        ),
    ] = "sampled",
    report_json: Annotated[
        Path | None,
        typer.Option("--report-json", help="Write the complete evidence report as JSON."),
    ] = None,
    report_markdown: Annotated[
        Path | None,
        typer.Option("--report-markdown", help="Write a PR-friendly Markdown evidence report."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON instead of a Rich table."),
    ] = False,
) -> None:
    """Run static, semantic, quantization and optional serving-runtime checks."""
    if fingerprint_mode not in {"sampled", "full", "off"}:
        console.print("[red]--fingerprint-mode must be sampled, full, or off.[/red]")
        raise typer.Exit(code=2)
    if endpoint and prompts is None:
        console.print("[red]--endpoint requires --prompts for runtime comparison.[/red]")
        raise typer.Exit(code=2)

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
        served_model = runtime_model or resolved_base
        runtime_api_key = os.getenv(runtime_api_key_env) if runtime_api_key_env else None
        try:
            checks.extend(
                run_semantic_checks(
                    base_model=resolved_base,
                    adapter_path=adapter,
                    prompts_path=prompts,
                    merged_model=merged,
                    quantized_model=quantized,
                    runtime_endpoint=endpoint,
                    runtime_model=served_model,
                    runtime_api_key=runtime_api_key,
                    runtime_max_tokens=runtime_max_tokens,
                    runtime_timeout=runtime_timeout,
                    min_runtime_first_token_agreement=min_runtime_first_token_agreement,
                    min_runtime_exact_match_rate=min_runtime_exact_match_rate,
                    device=device,
                    dtype=dtype,
                    max_prompts=max_prompts,
                    max_merge_diff=max_merge_diff,
                    min_top1_agreement=min_top1_agreement,
                    max_quantized_diff=max_quantized_diff,
                    min_quantized_top1_agreement=min_quantized_top1_agreement,
                    include_prompts=include_prompts,
                )
            )
        except MissingHFDependencies as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=2) from exc
        except (OSError, ValueError, RuntimeError) as exc:
            console.print(f"[red]Semantic verification failed to run: {exc}[/red]")
            raise typer.Exit(code=2) from exc

    try:
        fingerprints = _collect_fingerprints(
            adapter=adapter,
            base=resolved_base,
            merged=merged,
            quantized=quantized,
            mode=fingerprint_mode,
        )
    except (OSError, ValueError) as exc:
        console.print(f"[red]Fingerprinting failed: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    report = VerificationReport(checks, fingerprints=fingerprints)
    if report_json:
        write_json_report(report, report_json)
    if report_markdown:
        write_markdown_report(report, report_markdown)

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
