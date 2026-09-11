from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import CheckResult, Status


class MissingHFDependencies(RuntimeError):
    pass


def _imports():
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise MissingHFDependencies(
            "Semantic checks require Hugging Face dependencies. "
            "Install with: pip install 'adapterguard[hf]'"
        ) from exc
    return torch, PeftModel, AutoModelForCausalLM, AutoTokenizer


def load_prompts(path: str | Path, max_prompts: int) -> list[str]:
    prompts: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc

            if isinstance(value, str):
                prompt = value
            elif isinstance(value, dict) and isinstance(value.get("prompt"), str):
                prompt = value["prompt"]
            else:
                raise ValueError(
                    f"Line {line_number} must be a JSON string or an object with a string `prompt`"
                )

            prompts.append(prompt)
            if len(prompts) >= max_prompts:
                break

    if not prompts:
        raise ValueError("Prompt file contains no usable prompts")
    return prompts


def _resolve_dtype(torch, dtype: str):
    if dtype == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype}") from exc


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def _logits(model, tokenizer, prompts: list[str], torch):
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    device = _model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded)
    return output.logits.detach().float().cpu()


def _comparison_metrics(left, right, torch) -> dict[str, float]:
    diff = (left - right).abs()
    left_top = left.argmax(dim=-1)
    right_top = right.argmax(dim=-1)
    return {
        "mean_abs_logit_diff": float(diff.mean().item()),
        "max_abs_logit_diff": float(diff.max().item()),
        "top1_token_agreement": float((left_top == right_top).float().mean().item()),
    }


def run_semantic_checks(
    *,
    base_model: str,
    adapter_path: str | Path,
    prompts_path: str | Path,
    merged_model: str | None = None,
    device: str = "auto",
    dtype: str = "auto",
    max_prompts: int = 8,
    effect_threshold: float = 1e-6,
    max_merge_diff: float = 1e-3,
    min_top1_agreement: float = 0.999,
) -> list[CheckResult]:
    torch, PeftModel, AutoModelForCausalLM, AutoTokenizer = _imports()
    prompts = load_prompts(prompts_path, max_prompts)
    torch_dtype = _resolve_dtype(torch, dtype)

    model_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if device == "auto":
        model_kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if device != "auto":
        base = base.to(device)
    base.eval()
    base_logits = _logits(base, tokenizer, prompts, torch)

    adapted = PeftModel.from_pretrained(base, str(adapter_path))
    adapted.eval()
    adapted_logits = _logits(adapted, tokenizer, prompts, torch)

    checks: list[CheckResult] = []
    effect = _comparison_metrics(base_logits, adapted_logits, torch)
    if effect["mean_abs_logit_diff"] > effect_threshold:
        checks.append(
            CheckResult(
                "adapter changes model behavior",
                Status.PASS,
                "adapter produces a measurable logit delta",
                effect,
            )
        )
    else:
        checks.append(
            CheckResult(
                "adapter changes model behavior",
                Status.FAIL,
                "adapter output is effectively identical to the base model",
                effect,
            )
        )

    merged = adapted.merge_and_unload()
    merged.eval()
    merged_logits = _logits(merged, tokenizer, prompts, torch)
    merge_metrics = _comparison_metrics(adapted_logits, merged_logits, torch)
    merge_ok = (
        merge_metrics["mean_abs_logit_diff"] <= max_merge_diff
        and merge_metrics["top1_token_agreement"] >= min_top1_agreement
    )
    checks.append(
        CheckResult(
            "merge preserves adapter behavior",
            Status.PASS if merge_ok else Status.FAIL,
            "in-memory merge is semantically equivalent"
            if merge_ok
            else "in-memory merge diverges from the active adapter",
            merge_metrics,
        )
    )

    if merged_model:
        exported = AutoModelForCausalLM.from_pretrained(merged_model, **model_kwargs)
        if device != "auto":
            exported = exported.to(device)
        exported.eval()
        exported_logits = _logits(exported, tokenizer, prompts, torch)
        export_metrics = _comparison_metrics(adapted_logits, exported_logits, torch)
        export_ok = (
            export_metrics["mean_abs_logit_diff"] <= max_merge_diff
            and export_metrics["top1_token_agreement"] >= min_top1_agreement
        )
        checks.append(
            CheckResult(
                "exported model preserves adapter behavior",
                Status.PASS if export_ok else Status.FAIL,
                "exported model matches the active adapter"
                if export_ok
                else "exported model diverges from the active adapter",
                export_metrics,
            )
        )

    return checks
