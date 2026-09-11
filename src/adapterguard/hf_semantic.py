from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import CheckResult, PromptEvidence, Status


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


def _prompt_logits(model, tokenizer, prompts: list[str], torch):
    runs = []
    device = _model_device(model)
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded)
        runs.append(output.logits.detach().float().cpu())
    return runs


def _token_text(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except (TypeError, ValueError):
        return str(token_id)


def _comparison_metrics(
    left_runs,
    right_runs,
    *,
    torch,
    tokenizer,
    prompts: list[str],
    include_prompts: bool,
) -> tuple[dict[str, float], list[PromptEvidence]]:
    total_abs = 0.0
    total_values = 0
    max_abs = 0.0
    top1_matches = 0
    top1_total = 0
    evidence: list[PromptEvidence] = []

    for index, (prompt, left, right) in enumerate(
        zip(prompts, left_runs, right_runs, strict=True),
        start=1,
    ):
        if left.shape != right.shape:
            raise ValueError(
                "Logit shape mismatch for prompt "
                f"{index}: {tuple(left.shape)} != {tuple(right.shape)}"
            )

        diff = (left - right).abs()
        left_top = left.argmax(dim=-1)
        right_top = right.argmax(dim=-1)
        same = left_top == right_top

        value_count = diff.numel()
        token_count = same.numel()
        total_abs += float(diff.sum().item())
        total_values += value_count
        max_abs = max(max_abs, float(diff.max().item()))
        top1_matches += int(same.sum().item())
        top1_total += token_count

        first_position = None
        left_token_id = None
        right_token_id = None
        left_token = None
        right_token = None
        divergent = (~same).nonzero(as_tuple=False)
        if divergent.numel():
            first_position = int(divergent[0, -1].item())
            left_token_id = int(left_top[0, first_position].item())
            right_token_id = int(right_top[0, first_position].item())
            left_token = _token_text(tokenizer, left_token_id)
            right_token = _token_text(tokenizer, right_token_id)

        evidence.append(
            PromptEvidence(
                prompt_index=index,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                mean_abs_logit_diff=float(diff.mean().item()),
                max_abs_logit_diff=float(diff.max().item()),
                top1_token_agreement=float(same.float().mean().item()),
                first_divergent_position=first_position,
                left_token_id=left_token_id,
                right_token_id=right_token_id,
                left_token=left_token,
                right_token=right_token,
                prompt=prompt if include_prompts else None,
            )
        )

    metrics = {
        "mean_abs_logit_diff": total_abs / total_values if total_values else 0.0,
        "max_abs_logit_diff": max_abs,
        "top1_token_agreement": top1_matches / top1_total if top1_total else 1.0,
    }
    return metrics, evidence


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
    include_prompts: bool = False,
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
    base_logits = _prompt_logits(base, tokenizer, prompts, torch)

    adapted = PeftModel.from_pretrained(base, str(adapter_path))
    adapted.eval()
    adapted_logits = _prompt_logits(adapted, tokenizer, prompts, torch)

    checks: list[CheckResult] = []
    effect, effect_evidence = _comparison_metrics(
        base_logits,
        adapted_logits,
        torch=torch,
        tokenizer=tokenizer,
        prompts=prompts,
        include_prompts=include_prompts,
    )
    if effect["mean_abs_logit_diff"] > effect_threshold:
        checks.append(
            CheckResult(
                "adapter changes model behavior",
                Status.PASS,
                "adapter produces a measurable logit delta",
                effect,
                effect_evidence,
            )
        )
    else:
        checks.append(
            CheckResult(
                "adapter changes model behavior",
                Status.FAIL,
                "adapter output is effectively identical to the base model",
                effect,
                effect_evidence,
            )
        )

    merged = adapted.merge_and_unload()
    merged.eval()
    merged_logits = _prompt_logits(merged, tokenizer, prompts, torch)
    merge_metrics, merge_evidence = _comparison_metrics(
        adapted_logits,
        merged_logits,
        torch=torch,
        tokenizer=tokenizer,
        prompts=prompts,
        include_prompts=include_prompts,
    )
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
            merge_evidence,
        )
    )

    if merged_model:
        exported = AutoModelForCausalLM.from_pretrained(merged_model, **model_kwargs)
        if device != "auto":
            exported = exported.to(device)
        exported.eval()
        exported_logits = _prompt_logits(exported, tokenizer, prompts, torch)
        export_metrics, export_evidence = _comparison_metrics(
            adapted_logits,
            exported_logits,
            torch=torch,
            tokenizer=tokenizer,
            prompts=prompts,
            include_prompts=include_prompts,
        )
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
                export_evidence,
            )
        )

    return checks
