import json
from pathlib import Path

from adapterguard.models import Status
from adapterguard.static_checks import run_static_checks


def _write_adapter(tmp_path: Path, *, base: str = "acme/base") -> Path:
    config = {
        "base_model_name_or_path": base,
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"fake")
    return tmp_path


def test_valid_adapter_passes_core_static_checks(tmp_path):
    adapter = _write_adapter(tmp_path)
    config, checks = run_static_checks(adapter, expected_base="acme/base")

    assert config is not None
    assert not any(check.status == Status.FAIL for check in checks)


def test_base_model_mismatch_fails(tmp_path):
    adapter = _write_adapter(tmp_path, base="acme/base")
    _, checks = run_static_checks(adapter, expected_base="other/base")

    base_check = next(check for check in checks if check.name == "base model")
    assert base_check.status == Status.FAIL


def test_missing_weights_fails(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "acme/base",
                "peft_type": "LORA",
                "r": 8,
                "target_modules": ["q_proj"],
            }
        ),
        encoding="utf-8",
    )
    _, checks = run_static_checks(tmp_path)

    weights = next(check for check in checks if check.name == "adapter weights")
    assert weights.status == Status.FAIL


def test_missing_directory_returns_failure(tmp_path):
    _, checks = run_static_checks(tmp_path / "does-not-exist")
    assert checks[0].status == Status.FAIL
