from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import CheckResult, Status

_CONFIG = "adapter_config.json"
_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")


def load_adapter_config(adapter_path: str | Path) -> dict[str, Any]:
    config_path = Path(adapter_path) / _CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {_CONFIG}: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def run_static_checks(
    adapter_path: str | Path,
    *,
    expected_base: str | None = None,
) -> tuple[dict[str, Any] | None, list[CheckResult]]:
    adapter = Path(adapter_path)
    checks: list[CheckResult] = []

    if not adapter.is_dir():
        return None, [
            CheckResult("adapter directory", Status.FAIL, f"Directory does not exist: {adapter}")
        ]

    try:
        config = load_adapter_config(adapter)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return None, [CheckResult("adapter config", Status.FAIL, str(exc))]

    checks.append(CheckResult("adapter config", Status.PASS, f"Found {_CONFIG}"))

    peft_type = str(config.get("peft_type") or "").upper()
    if peft_type:
        checks.append(CheckResult("PEFT type", Status.PASS, peft_type))
    else:
        checks.append(CheckResult("PEFT type", Status.WARN, "peft_type is missing"))

    base = config.get("base_model_name_or_path")
    if base:
        status = Status.PASS
        message = str(base)
        if expected_base and str(base).rstrip("/") != expected_base.rstrip("/"):
            status = Status.FAIL
            message = f"adapter expects {base!r}, CLI requested {expected_base!r}"
        checks.append(CheckResult("base model", status, message))
    else:
        checks.append(
            CheckResult(
                "base model",
                Status.WARN,
                "base_model_name_or_path is missing; pass --base explicitly",
            )
        )

    targets = config.get("target_modules")
    if targets:
        target_count = len(targets) if isinstance(targets, list) else 1
        checks.append(
            CheckResult(
                "target modules",
                Status.PASS,
                f"{target_count} target module(s) declared",
                {"count": target_count},
            )
        )
    else:
        checks.append(CheckResult("target modules", Status.WARN, "target_modules is missing"))

    rank = config.get("r")
    alpha = config.get("lora_alpha")
    if peft_type in {"LORA", "ADALORA"}:
        if isinstance(rank, int) and rank > 0:
            metrics = {"r": rank}
            if alpha is not None:
                metrics["lora_alpha"] = alpha
            checks.append(
                CheckResult("LoRA hyperparameters", Status.PASS, f"rank={rank}", metrics)
            )
        else:
            checks.append(
                CheckResult("LoRA hyperparameters", Status.FAIL, "invalid or missing rank `r`")
            )

    weight_file = next(
        (adapter / name for name in _WEIGHT_NAMES if (adapter / name).is_file()),
        None,
    )
    if weight_file:
        checks.append(
            CheckResult(
                "adapter weights",
                Status.PASS,
                weight_file.name,
                {"bytes": weight_file.stat().st_size},
            )
        )
    else:
        checks.append(
            CheckResult(
                "adapter weights",
                Status.FAIL,
                f"expected one of: {', '.join(_WEIGHT_NAMES)}",
            )
        )

    return config, checks
