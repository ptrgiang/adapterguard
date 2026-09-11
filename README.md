# AdapterGuard

> **Know your adapter still works before you ship it.**

AdapterGuard is an independent semantic integrity checker for LoRA, QLoRA, DoRA and other
PEFT adapters. It targets the risky gap between **"training finished"** and
**"the artifact deployed to production still behaves like the trained adapter."**

Fine-tuning pipelines routinely transform the same model artifact:

```text
train -> save adapter -> reload -> merge -> quantize -> export -> serve
```

Each transformation can silently change behavior. A model can load successfully and still be
wrong. AdapterGuard makes those failures visible, reproducible and CI-friendly.

## 60-second proofs

### Corrupted export

AdapterGuard includes a self-contained demo that needs **no pretrained model download**. It builds
a tiny local GPT-2 model, attaches a real LoRA adapter, performs a valid merge, deliberately
corrupts the exported model while keeping it loadable, and proves that AdapterGuard catches the
semantic divergence.

```bash
pip install -e ".[hf]"
python examples/killer_demo.py
```

Expected final result:

```text
PASS  adapter changes model behavior
PASS  merge preserves adapter behavior
FAIL  exported model preserves adapter behavior

DEMO PASS: AdapterGuard caught a loadable but semantically corrupted artifact.
```

### Quantization drift

v0.3 adds a second self-contained proof. It creates a merged LoRA model, applies aggressive
symmetric fake quantization to its weights, saves a still-loadable Hugging Face artifact and asks
AdapterGuard to compare it against the in-memory merged reference.

```bash
python examples/quantization_demo.py
```

Expected final result:

```text
FAIL  quantized model preserves merged behavior

DEMO PASS: AdapterGuard localized quantization-induced semantic drift.
```

The fake-quantization demo validates the verifier without requiring GPU-only quantization
libraries. Real GPTQ, AWQ or bitsandbytes artifacts can be supplied through `--quantized` when the
corresponding runtime dependencies are installed.

## What v0.3 checks

**Static checks (fast, no model load):**

- `adapter_config.json` exists and is valid JSON
- adapter/base-model identity matches what you requested
- PEFT type and target modules are declared
- LoRA rank is valid
- adapter weights are present

**Semantic checks (optional Hugging Face extra):**

- the adapter measurably changes model logits vs. the base model
- `merge_and_unload()` preserves active-adapter behavior
- an exported/merged model preserves active-adapter behavior
- a quantized model stays within an explicit semantic drift budget vs. the merged reference
- mean/max logit drift and token-level top-1 agreement
- per-prompt localization of semantic drift
- first position where top-1 token predictions diverge
- worst prompt and count of prompts with top-1 divergence after quantization
- quantization metadata when exposed by the model config

**Evidence layer:**

- sampled or full SHA-256 fingerprints for local artifacts
- stable reference fingerprints for remote model IDs
- fingerprints for adapter, base, merged and quantized artifacts
- JSON report schema for machines
- Markdown report for pull requests and release evidence
- raw prompts excluded from reports by default for privacy

The goal is not to replace PEFT, Transformers, Unsloth, Axolotl, TRL or quantization runtimes.
AdapterGuard sits **after/between them as an independent verifier**.

## Install

Lightweight CLI and static checks:

```bash
pip install -e .
```

Semantic verification:

```bash
pip install -e ".[hf]"
```

Development:

```bash
pip install -e ".[dev]"
```

## Quick start

Static-only verification:

```bash
adapterguard verify --adapter ./my-adapter
```

Semantic verification against the adapter's configured base model:

```bash
adapterguard verify \
  --adapter ./my-adapter \
  --prompts examples/prompts.jsonl
```

Compare against a specific base and exported model:

```bash
adapterguard verify \
  --base Qwen/Qwen3-8B \
  --adapter ./my-adapter \
  --merged ./merged-model \
  --prompts examples/prompts.jsonl
```

A failed check exits with code `1`, so the command can gate a release in CI.

## Quantization integrity

Provide any quantized causal-LM artifact that `AutoModelForCausalLM.from_pretrained()` can load in
your environment:

```bash
adapterguard verify \
  --base Qwen/Qwen3-8B \
  --adapter ./my-adapter \
  --quantized ./qwen3-8b-awq \
  --prompts tests/golden.jsonl \
  --report-json artifacts/adapterguard.json \
  --report-markdown artifacts/adapterguard.md
```

AdapterGuard compares the quantized artifact against the **in-memory merged model** created from
the same base + adapter. This isolates the `merge -> quantize` transformation instead of mixing
quantization drift with adapter or merge drift.

The default quantization budget is intentionally configurable:

```text
max mean absolute logit drift: 0.05
minimum top-1 token agreement: 0.98
```

Override it for your model, quantizer and task:

```bash
adapterguard verify \
  ... \
  --quantized ./model-int4 \
  --max-quantized-diff 0.03 \
  --min-quantized-top1-agreement 0.99
```

These defaults are **release-gate starting points, not universal quality thresholds**. A useful
production budget should be calibrated against your own golden prompts and acceptable task-level
behavior.

Quantization evidence adds fields such as:

```text
prompt_count
divergent_prompt_count
worst_prompt_index
worst_prompt_mean_abs_logit_diff
quantization_method
quantization_bits
budget_max_mean_abs_logit_diff
budget_min_top1_token_agreement
```

## Generate evidence reports

Write both machine-readable JSON and a PR-friendly Markdown report:

```bash
adapterguard verify \
  --base Qwen/Qwen3-8B \
  --adapter ./my-adapter \
  --merged ./merged-model \
  --quantized ./quantized-model \
  --prompts examples/prompts.jsonl \
  --report-json artifacts/adapterguard.json \
  --report-markdown artifacts/adapterguard.md
```

The evidence report contains artifact fingerprints, aggregate comparison metrics, per-prompt
metrics and the first divergent top-1 prediction when one exists.

Raw prompt text is **not included by default**. If the prompts are safe to expose in CI artifacts:

```bash
adapterguard verify ... --include-prompts --report-markdown adapterguard.md
```

For stdout JSON:

```bash
adapterguard verify --adapter ./my-adapter --json
```

See [`docs/report-schema.md`](docs/report-schema.md) for the report contract.

## Artifact fingerprint modes

The default is optimized for large model directories:

```bash
adapterguard verify --adapter ./my-adapter --fingerprint-mode sampled
```

- `sampled`: hashes file identity, size, and the first/last 64 KiB of each local file
- `full`: streams every byte for release-grade SHA-256 evidence
- `off`: disables artifact fingerprints
- remote IDs such as `Qwen/Qwen3-8B` receive a stable reference fingerprint, not a content hash

Use `full` when the report is intended to prove the exact bytes shipped to production.

## Prompt format

`--prompts` accepts JSONL. Each line may be either a JSON string:

```json
"Explain gradient descent simply."
```

or an object:

```json
{"prompt": "Explain gradient descent simply."}
```

Start with prompts that exercise the exact behavior you fine-tuned. Quantization checks are only as
useful as the behavioral surface represented by those prompts.

## Current scope

AdapterGuard v0.3 intentionally stays narrow:

- causal language models supported by `AutoModelForCausalLM`
- local/Hugging Face model IDs
- PEFT adapters loadable through `PeftModel`
- semantic comparison at logits/top-1-token level
- local artifact evidence reports
- quantized artifacts loadable by the installed Transformers/runtime stack

AdapterGuard does **not** perform quantization itself. For GPTQ/AWQ/bitsandbytes or other formats,
install the dependencies required by Transformers to load that artifact, then point AdapterGuard at
it with `--quantized`.

Not yet covered: sequence-classification adapters, multimodal adapters, GGUF/llama.cpp native
loading, vLLM/TGI endpoint probing, cross-runtime comparison and signed provenance.

## Why this repo exists

Most tooling answers **"can I load/merge/quantize this model?"** AdapterGuard asks a different
question:

> **"Can I prove the artifact I am about to ship still behaves like the adapter I trained?"**

That distinction is the project.

## Roadmap

### v0.1 — semantic baseline

- [x] static adapter inspection
- [x] base-vs-adapter behavioral check
- [x] active-adapter-vs-merge comparison
- [x] exported model comparison
- [x] JSON output and CI-safe exit codes
- [x] self-contained corrupted-artifact proof

### v0.2 — reproducible evidence

- [x] adapter/model fingerprints
- [x] per-prompt failure localization
- [x] first divergent token prediction
- [x] machine-readable evidence schema
- [x] Markdown verification report
- [x] privacy-safe prompt hashing by default
- [ ] save -> reload equivalence against a captured pre-save reference

### v0.3 — quantization integrity

- [x] quantized-artifact comparison against in-memory merged reference
- [x] configurable quantization drift budgets
- [x] per-prompt quantization failure localization
- [x] quantized artifact fingerprinting
- [x] quantization metadata extraction when declared
- [x] self-contained fake-quantization proof

### v0.4 — runtime matrix

- [ ] vLLM endpoint verifier
- [ ] TGI endpoint verifier
- [ ] GGUF / llama.cpp comparison
- [ ] reusable GitHub Action for release gating
- [ ] runtime-to-runtime drift matrix

## Contributing

The most useful contributions are **real failure cases** where an adapter appeared to load, merge,
quantize or serve successfully but behavior changed. Open an issue with the smallest reproducible
example you can share.

## License

MIT
