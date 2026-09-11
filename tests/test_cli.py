import json
from pathlib import Path

from typer.testing import CliRunner

from adapterguard.cli import app

runner = CliRunner()


def _adapter(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "acme/base",
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj"],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "adapter_model.safetensors").write_bytes(b"fake")
    return tmp_path


def test_static_json_output_does_not_claim_safe_to_ship(tmp_path):
    adapter = _adapter(tmp_path)
    result = runner.invoke(app, ["verify", "--adapter", str(adapter), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["policy_passed"] is True
    assert payload["safe_to_ship"] is False
    assert payload["verdict"] == "STATIC CHECKS PASS"
    assert payload["verification_level"] == "static"
    assert payload["required_level"] == "static"
    assert payload["fingerprints"]["adapter"]["mode"] == "sampled"


def test_require_semantic_fails_when_only_static_checks_ran(tmp_path):
    adapter = _adapter(tmp_path)
    result = runner.invoke(
        app,
        [
            "verify",
            "--adapter",
            str(adapter),
            "--require-level",
            "semantic",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["policy_passed"] is False
    assert payload["coverage_satisfied"] is False
    assert payload["safe_to_ship"] is False
    assert payload["verdict"] == "UNSAFE TO SHIP"


def test_cli_writes_json_and_markdown_reports(tmp_path):
    adapter = _adapter(tmp_path / "adapter")
    json_path = tmp_path / "out" / "evidence.json"
    markdown_path = tmp_path / "out" / "evidence.md"
    result = runner.invoke(
        app,
        [
            "verify",
            "--adapter",
            str(adapter),
            "--report-json",
            str(json_path),
            "--report-markdown",
            str(markdown_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["safe_to_ship"] is False
    assert payload["policy_passed"] is True
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "AdapterGuard verification report" in markdown
    assert "Verification level:** `static`" in markdown


def test_quantized_artifact_is_only_fingerprinted_without_semantic_run(tmp_path):
    adapter = _adapter(tmp_path / "adapter")
    quantized = tmp_path / "quantized"
    quantized.mkdir()
    (quantized / "config.json").write_text("{}", encoding="utf-8")
    (quantized / "model.safetensors").write_bytes(b"quantized")

    result = runner.invoke(
        app,
        [
            "verify",
            "--adapter",
            str(adapter),
            "--quantized",
            str(quantized),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["fingerprints"]["quantized"]["mode"] == "sampled"
    assert payload["verification_level"] == "static"
    assert payload["safe_to_ship"] is False


def test_endpoint_requires_prompts(tmp_path):
    adapter = _adapter(tmp_path)
    result = runner.invoke(
        app,
        [
            "verify",
            "--adapter",
            str(adapter),
            "--endpoint",
            "http://localhost:8000/v1",
        ],
    )
    assert result.exit_code == 2
    assert "requires --prompts" in result.stdout


def test_invalid_fingerprint_mode_returns_usage_error(tmp_path):
    adapter = _adapter(tmp_path)
    result = runner.invoke(
        app,
        ["verify", "--adapter", str(adapter), "--fingerprint-mode", "wat"],
    )
    assert result.exit_code == 2


def test_invalid_required_level_returns_usage_error(tmp_path):
    adapter = _adapter(tmp_path)
    result = runner.invoke(
        app,
        ["verify", "--adapter", str(adapter), "--require-level", "wat"],
    )
    assert result.exit_code == 2
    assert "verification level must be one of" in result.stdout


def test_version_command_is_plain_text():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.5.0"
