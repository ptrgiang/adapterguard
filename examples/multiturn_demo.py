"""Self-contained proof for native multi-turn chat histories with a real LoRA adapter."""

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
        "system",
        "user",
        "assistant",
        "concise",
        "hello",
        "remember",
        "order",
        "A",
        "102",
        "understood",
        "which",
        "did",
        "mention",
        "adapter",
        "guard",
    ]
    vocab = {token: index for index, token in enumerate(tokens)}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    fast.chat_template = (
        "{% for message in messages %}"
        "<{{ message['role'] }}>{{ message['content'] }}"
        "{% endfor %}<assistant>"
    )
    return fast


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

    seed = 23
    random.seed(seed)
    torch.manual_seed(seed)

    with tempfile.TemporaryDirectory(prefix="adapterguard-multiturn-demo-") as tmp:
        root = Path(tmp)
        base_dir = root / "base"
        adapter_dir = root / "adapter"
        exported_dir = root / "exported"
        prompts_path = root / "messages.jsonl"

        tokenizer = _build_tokenizer(
            Tokenizer,
            WordLevel,
            Whitespace,
            PreTrainedTokenizerFast,
        )
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=64,
            n_ctx=64,
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

        merged = adapted.merge_and_unload()
        merged.save_pretrained(exported_dir)
        tokenizer.save_pretrained(exported_dir)

        cases = [
            {
                "messages": [
                    {"role": "system", "content": "concise"},
                    {"role": "user", "content": "hello remember order A 102"},
                    {"role": "assistant", "content": "understood"},
                    {"role": "user", "content": "which order did I mention"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "hello adapter guard"},
                    {"role": "assistant", "content": "understood"},
                    {"role": "user", "content": "remember adapter guard"},
                ]
            },
        ]
        prompts_path.write_text(
            "\n".join(json.dumps(item) for item in cases) + "\n",
            encoding="utf-8",
        )

        checks = run_semantic_checks(
            base_model=str(base_dir),
            adapter_path=adapter_dir,
            merged_model=str(exported_dir),
            prompts_path=prompts_path,
            device="cpu",
            dtype="float32",
            max_prompts=2,
            max_merge_diff=1e-4,
            min_top1_agreement=0.999,
        )
        adapter_effect = next(
            check for check in checks if check.name == "adapter changes model behavior"
        )
        merge_check = next(
            check for check in checks if check.name == "merge preserves adapter behavior"
        )
        exported_check = next(
            check for check in checks if check.name == "exported model preserves adapter behavior"
        )
        tokenizer_check = next(
            check
            for check in checks
            if check.name == "exported model tokenizer preserves prompt encoding"
        )

        print("AdapterGuard multi-turn demo")
        print(f"adapter effect: {adapter_effect.status.value.upper()}")
        print(f"merge: {merge_check.status.value.upper()}")
        print(f"exported weights: {exported_check.status.value.upper()}")
        print(f"chat tokenizer/template: {tokenizer_check.status.value.upper()}")
        print(f"chat prompts checked: {tokenizer_check.metrics['chat_prompt_count']}")

        passed = (
            adapter_effect.status == Status.PASS
            and merge_check.status == Status.PASS
            and exported_check.status == Status.PASS
            and tokenizer_check.status == Status.PASS
            and tokenizer_check.metrics["chat_prompt_count"] == 2
        )
        if passed:
            print(
                "\nDEMO PASS: AdapterGuard verified native multi-turn histories through "
                "real PEFT merge and chat-template integrity checks."
            )
            return 0

        print("\nDEMO FAIL: expected multi-turn verification to pass end to end.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
