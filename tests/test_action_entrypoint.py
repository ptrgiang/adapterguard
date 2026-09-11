import json
import sys

import pytest

from adapterguard.action_entrypoint import _publish_outputs, build_command


def test_build_command_maps_action_inputs_without_shell_parsing():
    env = {
        "AG_ADAPTER": "./adapter path",
        "AG_BASE": "Qwen/Qwen3-8B",
        "AG_PROMPTS": "tests/golden.jsonl",
        "AG_ENDPOINT": "http://localhost:8000/v1",
        "AG_RUNTIME_MODEL": "qwen-prod",
        "AG_REQUIRE_LEVEL": "runtime",
        "AG_INCLUDE_PROMPTS": "true",
        "AG_REPORT_JSON": "out/report.json",
        "AG_REPORT_MARKDOWN": "out/report.md",
    }

    command = build_command(env)

    assert command[:4] == [sys.executable, "-m", "adapterguard.cli", "verify"]
    assert command[command.index("--adapter") + 1] == "./adapter path"
    assert command[command.index("--base") + 1] == "Qwen/Qwen3-8B"
    assert command[command.index("--endpoint") + 1] == "http://localhost:8000/v1"
    assert command[command.index("--require-level") + 1] == "runtime"
    assert "--include-prompts" in command
    assert command[-4:] == [
        "--report-json",
        "out/report.json",
        "--report-markdown",
        "out/report.md",
    ]


def test_build_command_requires_adapter():
    with pytest.raises(ValueError, match="adapter"):
        build_command({})


def test_publish_outputs_writes_policy_and_coverage_outputs(tmp_path):
    report_json = tmp_path / "report.json"
    report_markdown = tmp_path / "report.md"
    github_output = tmp_path / "github-output.txt"
    step_summary = tmp_path / "summary.md"
    report_json.write_text(
        json.dumps(
            {
                "verdict": "STATIC CHECKS PASS",
                "safe_to_ship": False,
                "policy_passed": True,
                "verification_level": "static",
                "required_level": "static",
            }
        ),
        encoding="utf-8",
    )
    report_markdown.write_text("# AdapterGuard verification report\n", encoding="utf-8")

    _publish_outputs(
        report_json=report_json,
        report_markdown=report_markdown,
        env={
            "GITHUB_OUTPUT": str(github_output),
            "GITHUB_STEP_SUMMARY": str(step_summary),
        },
    )

    output = github_output.read_text(encoding="utf-8")
    assert "verdict=STATIC CHECKS PASS" in output
    assert "safe_to_ship=false" in output
    assert "policy_passed=true" in output
    assert "verification_level=static" in output
    assert "required_level=static" in output
    assert f"report_json={report_json.resolve()}" in output
    assert "AdapterGuard verification report" in step_summary.read_text(encoding="utf-8")


def test_publish_outputs_reports_error_when_report_is_missing(tmp_path):
    github_output = tmp_path / "github-output.txt"

    _publish_outputs(
        report_json=tmp_path / "missing.json",
        report_markdown=tmp_path / "missing.md",
        env={"GITHUB_OUTPUT": str(github_output)},
    )

    output = github_output.read_text(encoding="utf-8")
    assert "verdict=ERROR" in output
    assert "safe_to_ship=false" in output
    assert "policy_passed=false" in output
