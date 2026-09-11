import json
from pathlib import Path

from typer.testing import CliRunner

from adapterguard.cli import app

runner = CliRunner()


def _adapter(tmp_path: Path) -> Path:
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


def test_json_output_is_machine_readable(tmp_path):
    adapter = _adapter(tmp_path)
    result = runner.invoke(app, ["verify", "--adapter", str(adapter), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["safe_to_ship"] is True
    assert payload["verdict"] == "SAFE TO SHIP"


def test_version_command_is_plain_text():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"
