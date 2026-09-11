"""Self-contained AdapterGuard proof using a tiny local GPT-2 + LoRA.

The demo creates:
1. a tiny random base model,
2. a real LoRA adapter that changes model behavior,
3. a deliberately corrupted merged artifact,
4. an AdapterGuard verification run that must catch the corruption.

No pretrained model download is required once the `hf` dependencies are installed.
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
        "fine",
        "tune",
        "merge",
        "safe",
        "broken",
        "model",
        "works",
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

    seed = 7
    random.seed(seed)
    torch.manual_seed(seed)

    with tempfile.TemporaryDirectory(prefix="adapterguard-demo-") as tmp:
        root = Path(tmp)
        base_dir = root / "base"
        adapter_dir = root / "adapter"
        corrupted_dir = root / "corrupted-merged"
        prompts_path = root / "prompts.jsonl"

        tokenizer = _build_tokenizer(
            Tokenizer, WordLevel, Whitespace, PreTrainedTokenizerFast
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

        # PEFT initializes LoRA B to zero so a fresh adapter is initially a no-op.
        # Give A/B deterministic non-zero values to create a genuine behavioral delta
        # without running a training loop.
        with torch.no_grad():
            for name, parameter in adapted.named_parameters():
                if "lora_A" in name:
                    parameter.normal_(mean=0.0, std=0.15)
                elif "lora_B" in name:
                    parameter.normal_(mean=0.0, std=0.15)

        adapted.save_pretrained(adapter_dir)

        merged = adapted.merge_and_unload()

        # Deliberately corrupt a core weight after a legitimate merge. This models
        # a broken export/conversion step that still produces a loadable model.
        corrupted_parameter = None
        with torch.no_grad():
            for name, parameter in merged.named_parameters():
                if parameter.ndim >= 2 and "wte" not in name:
                    parameter.zero_()
                    corrupted_parameter = name
                    break
        if corrupted_parameter is None:
            raise RuntimeError("Could not find a model parameter to corrupt")

        merged.save_pretrained(corrupted_dir)
        tokenizer.save_pretrained(corrupted_dir)

        prompts = [
            {"prompt": "hello world adapter guard"},
            {"prompt": "fine tune model works"},
            {"prompt": "merge safe model test"},
        ]
        prompts_path.write_text(
            "\n".join(json.dumps(item) for item in prompts) + "\n",
            encoding="utf-8",
        )

        print("AdapterGuard killer demo")
        print(f"Corrupted exported parameter: {corrupted_parameter}\n")

        checks = run_semantic_checks(
            base_model=str(base_dir),
            adapter_path=adapter_dir,
            merged_model=str(corrupted_dir),
            prompts_path=prompts_path,
            device="cpu",
            dtype="float32",
            max_prompts=3,
            max_merge_diff=1e-4,
            min_top1_agreement=0.999,
        )

        for check in checks:
            metrics = ", ".join(f"{key}={value:.6g}" for key, value in (check.metrics or {}).items())
            print(f"{check.status.value.upper():4}  {check.name}")
            if metrics:
                print(f"      {metrics}")

        in_memory_merge = next(
            check for check in checks if check.name == "merge preserves adapter behavior"
        )
        exported = next(
            check for check in checks if check.name == "exported model preserves adapter behavior"
        )
        adapter_effect = next(
            check for check in checks if check.name == "adapter changes model behavior"
        )

        detected = (
            adapter_effect.status == Status.PASS
            and in_memory_merge.status == Status.PASS
            and exported.status == Status.FAIL
        )

        if detected:
            print("\nDEMO PASS: AdapterGuard caught a loadable but semantically corrupted artifact.")
            return 0

        print("\nDEMO FAIL: expected corruption was not detected.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
