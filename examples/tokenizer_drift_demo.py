"""Self-contained proof that correct model weights can ship with a wrong tokenizer."""

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


def _build_tokenizer(
    Tokenizer,
    WordLevel,
    Whitespace,
    PreTrainedTokenizerFast,
    *,
    swap_content_ids: bool,
):
    tokens = [
        "[PAD]",
        "[UNK]",
        "[EOS]",
        "hello",
        "world",
        "adapter",
        "guard",
        "tokenizer",
        "drift",
        "model",
        "safe",
        "test",
    ]
    vocab = {token: index for index, token in enumerate(tokens)}
    if swap_content_ids:
        vocab["hello"], vocab["world"] = vocab["world"], vocab["hello"]

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )


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

    seed = 19
    random.seed(seed)
    torch.manual_seed(seed)

    with tempfile.TemporaryDirectory(prefix="adapterguard-tokenizer-demo-") as tmp:
        root = Path(tmp)
        base_dir = root / "base"
        adapter_dir = root / "adapter"
        exported_dir = root / "exported"
        prompts_path = root / "prompts.jsonl"

        base_tokenizer = _build_tokenizer(
            Tokenizer,
            WordLevel,
            Whitespace,
            PreTrainedTokenizerFast,
            swap_content_ids=False,
        )
        config = GPT2Config(
            vocab_size=len(base_tokenizer),
            n_positions=32,
            n_ctx=32,
            n_embd=32,
            n_layer=1,
            n_head=2,
            bos_token_id=base_tokenizer.eos_token_id,
            eos_token_id=base_tokenizer.eos_token_id,
            pad_token_id=base_tokenizer.pad_token_id,
        )
        base_model = GPT2LMHeadModel(config)
        base_model.save_pretrained(base_dir)
        base_tokenizer.save_pretrained(base_dir)

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

        merged = adapted.merge_and_unload()
        merged.save_pretrained(exported_dir)

        wrong_tokenizer = _build_tokenizer(
            Tokenizer,
            WordLevel,
            Whitespace,
            PreTrainedTokenizerFast,
            swap_content_ids=True,
        )
        wrong_tokenizer.save_pretrained(exported_dir)

        prompts = [
            {"prompt": "hello world adapter guard"},
            {"prompt": "tokenizer drift test"},
            {"prompt": "model safe test"},
        ]
        prompts_path.write_text(
            "\n".join(json.dumps(item) for item in prompts) + "\n",
            encoding="utf-8",
        )

        checks = run_semantic_checks(
            base_model=str(base_dir),
            adapter_path=adapter_dir,
            merged_model=str(exported_dir),
            prompts_path=prompts_path,
            device="cpu",
            dtype="float32",
            max_prompts=3,
            max_merge_diff=1e-4,
            min_top1_agreement=0.999,
        )
        tokenizer_check = next(
            check
            for check in checks
            if check.name == "exported model tokenizer preserves prompt encoding"
        )
        exported_check = next(
            check for check in checks if check.name == "exported model preserves adapter behavior"
        )

        print("AdapterGuard tokenizer drift demo")
        print(f"weights: {exported_check.status.value.upper()}")
        print(f"tokenizer: {tokenizer_check.status.value.upper()}")
        for key, value in (tokenizer_check.metrics or {}).items():
            print(f"{key}: {value}")

        if exported_check.status == Status.PASS and tokenizer_check.status == Status.FAIL:
            print(
                "\nDEMO PASS: AdapterGuard caught tokenizer drift while exported weights "
                "remained semantically correct."
            )
            return 0

        print("\nDEMO FAIL: expected isolated tokenizer drift was not detected.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
