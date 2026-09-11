from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys


_VALUE_OPTIONS = (
    ("AG_BASE", "--base"),
    ("AG_PROMPTS", "--prompts"),
    ("AG_MERGED", "--merged"),
    ("AG_QUANTIZED", "--quantized"),
    ("AG_ENDPOINT", "--endpoint"),
    ("AG_RUNTIME_MODEL", "--runtime-model"),
    ("AG_RUNTIME_MAX_TOKENS", "--runtime-max-tokens"),
    ("AG_RUNTIME_TIMEOUT", "--runtime-timeout"),
    ("AG_MIN_RUNTIME_FIRST_TOKEN_AGREEMENT", "--min-runtime-first-token-agreement"),
    ("AG_MIN_RUNTIME_EXACT_MATCH_RATE", "--min-runtime-exact-match-rate"),
    ("AG_DEVICE", "--device"),
    ("AG_DTYPE", "--dtype"),
    ("AG_MAX_PROMPTS", "--max-prompts"),
    ("AG_MAX_MERGE_DIFF", "--max-merge-diff"),
    ("AG_MIN_TOP1_AGREEMENT", "--min-top1-agreement"),
    ("AG_MAX_QUANTIZED_DIFF", "--max-quantized-diff"),
    ("AG_MIN_QUANTIZED_TOP1_AGREEMENT", "--min-quantized-top1-agreement"),
    ("AG_FINGERPRINT_MODE", "--fingerprint-mode"),
)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _value(env: dict[str, str], key: str) -> str:
    return env.get(key, "").strip()


def build_command(env: dict[str, str]) -> list[str]:
    adapter = _value(env, "AG_ADAPTER")
    if not adapter:
        raise ValueError("GitHub Action input `adapter` is required")

    report_json = _value(env, "AG_REPORT_JSON") or ".adapterguard/report.json"
    report_markdown = _value(env, "AG_REPORT_MARKDOWN") or ".adapterguard/report.md"

    command = [
        sys.executable,
        "-m",
        "adapterguard.cli",
        "verify",
        "--adapter",
        adapter,
    ]
    for env_name, option in _VALUE_OPTIONS:
        value = _value(env, env_name)
        if value:
            command.extend([option, value])

    if _truthy(env.get("AG_INCLUDE_PROMPTS")):
        command.append("--include-prompts")

    command.extend(
        [
            "--report-json",
            report_json,
            "--report-markdown",
            report_markdown,
        ]
    )
    return command


def _write_output(name: str, value: str, env: dict[str, str]) -> None:
    output_file = env.get("GITHUB_OUTPUT")
    if not output_file:
        return
    with pathlib.Path(output_file).open("a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def _append_summary(report_markdown: pathlib.Path, env: dict[str, str]) -> None:
    summary_file = env.get("GITHUB_STEP_SUMMARY")
    if not summary_file or not report_markdown.exists():
        return
    with pathlib.Path(summary_file).open("a", encoding="utf-8") as handle:
        handle.write(report_markdown.read_text(encoding="utf-8"))
        handle.write("\n")


def _publish_outputs(
    *,
    report_json: pathlib.Path,
    report_markdown: pathlib.Path,
    env: dict[str, str],
) -> None:
    verdict = "ERROR"
    safe_to_ship = "false"
    if report_json.exists():
        try:
            payload = json.loads(report_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        verdict = str(payload.get("verdict") or verdict)
        safe_to_ship = "true" if payload.get("safe_to_ship") is True else "false"

    _write_output("verdict", verdict, env)
    _write_output("safe_to_ship", safe_to_ship, env)
    _write_output("report_json", str(report_json.resolve()), env)
    _write_output("report_markdown", str(report_markdown.resolve()), env)
    _append_summary(report_markdown, env)


def main() -> int:
    env = dict(os.environ)
    api_key = _value(env, "AG_RUNTIME_API_KEY")
    if api_key:
        print(f"::add-mask::{api_key}")
        env["ADAPTERGUARD_API_KEY"] = api_key

    try:
        command = build_command(env)
    except ValueError as exc:
        print(f"AdapterGuard Action configuration error: {exc}", file=sys.stderr)
        return 2

    report_json = pathlib.Path(_value(env, "AG_REPORT_JSON") or ".adapterguard/report.json")
    report_markdown = pathlib.Path(
        _value(env, "AG_REPORT_MARKDOWN") or ".adapterguard/report.md"
    )

    result = subprocess.run(command, env=env, check=False)
    _publish_outputs(
        report_json=report_json,
        report_markdown=report_markdown,
        env=env,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
