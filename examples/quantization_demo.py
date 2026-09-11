"""Self-contained quantization-drift proof using tiny local GPT-2 + LoRA.

The demo creates a tiny model, merges a real LoRA adapter, applies aggressive
symmetric fake quantization to the merged weights, saves the dequantized result as
a normal Hugging Face artifact, and verifies that AdapterGuard catches the drift.

This validates AdapterGuard's comparison path without requiring GPU-only
quantization libraries. Real GPTQ/AWQ/bitsandbytes artifacts can be passed to
`adapterguard verify --quantized ...` when their runtime dependencies are installed.
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path


def _require_hf():
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    except ImportError as exc:
        raise SystemExit(
            "This demo requires the Hugging Face extra.\n"
            "Install it with: pip install -e '.[hf]'"
        ) from exc

    return (
        torch,
        LoraConfig,
        get_peft_model,
        Tokenizer,
        WordLevel,
        Whitespace,
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
    )


def _build_tokenizer(Tokenizer, WordLevel, Whitespace, PreTrainedTokenizerFast):
    tokens = [
        "[PAD]",
        "[UNK]",
        "[EOS]",
        "hello",
        "world",
        "adapter",
        "guard",
        "quantized",
        "model",
        "safe",
        "drift",
        "test",
    ]
    vocab = {token: index for index, token in enumerate(tokens)}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )


def _fake_quantize_model(model, torch, *, bits: int) -> None:
    qmax = (2 ** (bits - 1)) - 1
    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.is_floating_point() or parameter.numel() == 0:
                continue
            peak = parameter.abs().max()
            if float(peak.item()) == 0.0:
                continue
            scale = peak / qmax
            parameter.copy_((parameter / scale).round().clamp(-qmax, qmax) * scale)


def main() -> int:
    (
        torch,
        LoraConfig,
        get_peft_model,
        Tokenizer,
        WordLevel,
        Whitespace,
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
    ) = _require_hf()

    from adapterguard.hf_semantic import run_semantic_checks
    from adapterguard.models import Status

    seed = 11
    random.seed(seed)
    torch.manual_seed(seed)

    with tempfile.TemporaryDirectory(prefix="adapterguard-quant-demo-") as tmp:
        root = Path(tmp)
        base_dir = root / "base"
        adapter_dir = root / "adapter"
        quantized_dir = root / "quantized"
        prompts_path = root / "prompts.jsonl"

        tokenizer = _build_tokenizer(
            Tokenizer,
            WordLevel,
            Whitespace,
            PreTrainedTokenizerFast,
        )
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=32,
            n_ctx=32,
            n_embd=32,
            n_layer=1,
            n_head=2,
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        base_model = GPT2LMHeadModel(config)
        base_model.save_pretrained(base_dir)
        tokenizer.save_pretrained(base_dir)

        adapted = get_peft_model(
            GPT2LMHeadModel.from_pretrained(base_dir),
            LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["c_attn"],
                lora_dropout=0.0,
                task_type="CAUSAL_LM",
            ),
        )
        with torch.no_grad():
            for name, parameter in adapted.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    parameter.normal_(mean=0.0, std=0.2)
        adapted.save_pretrained(adapter_dir)

        quantized = adapted.merge_and_unload()
        _fake_quantize_model(quantized, torch, bits=2)
        quantized.save_pretrained(quantized_dir)
        tokenizer.save_pretrained(quantized_dir)

        prompts = [
            {"prompt": "hello world adapter guard"},
            {"prompt": "quantized model drift test"},
            {"prompt": "adapter model safe test"},
        ]
        prompts_path.write_text(
            "\n".join(json.dumps(item) for item in prompts) + "\n",
            encoding="utf-8",
        )

        checks = run_semantic_checks(
            base_model=str(base_dir),
            adapter_path=adapter_dir,
            quantized_model=str(quantized_dir),
            prompts_path=prompts_path,
            device="cpu",
            dtype="float32",
            max_prompts=3,
            max_quantized_diff=1e-4,
            min_quantized_top1_agreement=0.999,
        )
        quant_check = next(
            check
            for check in checks
            if check.name == "quantized model preserves merged behavior"
        )

        print("AdapterGuard quantization demo")
        print(f"status: {quant_check.status.value.upper()}")
        for key, value in (quant_check.metrics or {}).items():
            print(f"{key}: {value}")

        if quant_check.status == Status.FAIL:
            print("\nDEMO PASS: AdapterGuard localized quantization-induced semantic drift.")
            return 0

        print("\nDEMO FAIL: expected quantization drift was not detected.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
