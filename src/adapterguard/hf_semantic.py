from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import CheckResult, PromptEvidence, Status
from .runtime_openai import run_openai_chat_runtime_check, run_openai_runtime_check


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


def _greedy_completion_reference(
    model,
    tokenizer,
    prompts: list[str],
    torch,
    *,
    max_tokens: int,
    use_chat_template: bool = False,
) -> tuple[list[str], list[str]]:
    completions: list[str] = []
    first_tokens: list[str] = []
    device = _model_device(model)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    for prompt in prompts:
        input_text = prompt
        if use_chat_template:
            if not getattr(tokenizer, "chat_template", None):
                raise ValueError(
                    "chat-completions runtime verification requires a tokenizer chat template"
                )
            input_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        encoded = tokenizer(input_text, return_tensors="pt", truncation=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_tokens,
                pad_token_id=pad_token_id,
            )
        new_tokens = generated[0, input_length:].detach().cpu()
        completion = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if new_tokens.numel():
            first = tokenizer.decode(
                [int(new_tokens[0].item())],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        else:
            first = ""
        completions.append(completion)
        first_tokens.append(first)
    return completions, first_tokens


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
) -> tuple[dict[str, Any], list[PromptEvidence]]:
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

    metrics: dict[str, Any] = {
        "mean_abs_logit_diff": total_abs / total_values if total_values else 0.0,
        "max_abs_logit_diff": max_abs,
        "top1_token_agreement": top1_matches / top1_total if top1_total else 1.0,
    }
    return metrics, evidence


def _within_drift_budget(
    metrics: dict[str, Any],
    *,
    max_mean_abs_logit_diff: float,
    min_top1_token_agreement: float,
) -> bool:
    return (
        float(metrics["mean_abs_logit_diff"]) <= max_mean_abs_logit_diff
        and float(metrics["top1_token_agreement"]) >= min_top1_token_agreement
    )


def _summarize_evidence(evidence: list[PromptEvidence]) -> dict[str, Any]:
    if not evidence:
        return {
            "prompt_count": 0,
            "divergent_prompt_count": 0,
            "worst_prompt_index": None,
            "worst_prompt_mean_abs_logit_diff": 0.0,
        }

    divergent = [item for item in evidence if item.first_divergent_position is not None]
    worst = max(evidence, key=lambda item: item.mean_abs_logit_diff)
    return {
        "prompt_count": len(evidence),
        "divergent_prompt_count": len(divergent),
        "worst_prompt_index": worst.prompt_index,
        "worst_prompt_mean_abs_logit_diff": worst.mean_abs_logit_diff,
    }


def _quantization_metadata(model) -> dict[str, Any]:
    config = getattr(getattr(model, "config", None), "quantization_config", None)
    if config is None:
        return {"quantization_declared": False}
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if not isinstance(config, dict):
        return {
            "quantization_declared": True,
            "quantization_config_type": type(config).__name__,
        }

    method = config.get("quant_method") or config.get("quantization_method")
    bits = config.get("bits")
    if bits is None:
        if config.get("load_in_4bit"):
            bits = 4
        elif config.get("load_in_8bit"):
            bits = 8

    metadata: dict[str, Any] = {"quantization_declared": True}
    if method is not None:
        metadata["quantization_method"] = str(method)
    if bits is not None:
        metadata["quantization_bits"] = int(bits)
    return metadata


def run_semantic_checks(
    *,
    base_model: str,
    adapter_path: str | Path,
    prompts_path: str | Path,
    merged_model: str | None = None,
    quantized_model: str | None = None,
    runtime_endpoint: str | None = None,
    runtime_model: str | None = None,
    runtime_api_key: str | None = None,
    runtime_max_tokens: int = 8,
    runtime_timeout: float = 30.0,
    min_runtime_first_token_agreement: float = 1.0,
    min_runtime_exact_match_rate: float = 1.0,
    device: str = "auto",
    dtype: str = "auto",
    max_prompts: int = 8,
    effect_threshold: float = 1e-6,
    max_merge_diff: float = 1e-3,
    min_top1_agreement: float = 0.999,
    max_quantized_diff: float = 0.05,
    min_quantized_top1_agreement: float = 0.98,
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
    checks.append(
        CheckResult(
            "adapter changes model behavior",
            Status.PASS if effect["mean_abs_logit_diff"] > effect_threshold else Status.FAIL,
            "adapter produces a measurable logit delta"
            if effect["mean_abs_logit_diff"] > effect_threshold
            else "adapter output is effectively identical to the base model",
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
    merge_ok = _within_drift_budget(
        merge_metrics,
        max_mean_abs_logit_diff=max_merge_diff,
        min_top1_token_agreement=min_top1_agreement,
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

    runtime_reference = merged
    runtime_reference_label = "in-memory-merged"

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
        export_ok = _within_drift_budget(
            export_metrics,
            max_mean_abs_logit_diff=max_merge_diff,
            min_top1_token_agreement=min_top1_agreement,
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
        runtime_reference = exported
        runtime_reference_label = "exported-merged"

    if quantized_model:
        quantized = AutoModelForCausalLM.from_pretrained(quantized_model, **model_kwargs)
        if device != "auto":
            quantized = quantized.to(device)
        quantized.eval()
        quantized_logits = _prompt_logits(quantized, tokenizer, prompts, torch)
        quant_metrics, quant_evidence = _comparison_metrics(
            merged_logits,
            quantized_logits,
            torch=torch,
            tokenizer=tokenizer,
            prompts=prompts,
            include_prompts=include_prompts,
        )
        quant_metrics.update(_summarize_evidence(quant_evidence))
        quant_metrics.update(_quantization_metadata(quantized))
        quant_metrics["budget_max_mean_abs_logit_diff"] = max_quantized_diff
        quant_metrics["budget_min_top1_token_agreement"] = min_quantized_top1_agreement
        quant_ok = _within_drift_budget(
            quant_metrics,
            max_mean_abs_logit_diff=max_quantized_diff,
            min_top1_token_agreement=min_quantized_top1_agreement,
        )
        checks.append(
            CheckResult(
                "quantized model preserves merged behavior",
                Status.PASS if quant_ok else Status.FAIL,
                "quantized artifact stays within the configured drift budget"
                if quant_ok
                else "quantized artifact exceeds the configured drift budget",
                quant_metrics,
                quant_evidence,
            )
        )
        runtime_reference = quantized
        runtime_reference_label = "quantized"

    if runtime_endpoint:
        if not runtime_model:
            raise ValueError("runtime verification requires a served model name")
        chat_runtime = runtime_endpoint.rstrip("/").endswith("/chat/completions")
        local_completions, local_first_tokens = _greedy_completion_reference(
            runtime_reference,
            tokenizer,
            prompts,
            torch,
            max_tokens=runtime_max_tokens,
            use_chat_template=chat_runtime,
        )
        runtime_checker = (
            run_openai_chat_runtime_check if chat_runtime else run_openai_runtime_check
        )
        runtime_check = runtime_checker(
            endpoint=runtime_endpoint,
            model=runtime_model,
            prompts=prompts,
            local_completions=local_completions,
            local_first_tokens=local_first_tokens,
            api_key=runtime_api_key,
            max_tokens=runtime_max_tokens,
            timeout=runtime_timeout,
            min_first_token_agreement=min_runtime_first_token_agreement,
            min_exact_match_rate=min_runtime_exact_match_rate,
            include_prompts=include_prompts,
        )
        if runtime_check.metrics is not None:
            runtime_check.metrics["reference_artifact"] = runtime_reference_label
        checks.append(runtime_check)

    return checks
