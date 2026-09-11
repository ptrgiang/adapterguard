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

## 60-second proof

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

This is the failure class the project exists to catch: **syntactically valid artifact, wrong
behavior**.

## What v0.2 checks

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
- mean/max logit drift and token-level top-1 agreement
- per-prompt localization of semantic drift
- first position where top-1 token predictions diverge

**Evidence layer:**

- sampled or full SHA-256 fingerprints for local artifacts
- stable reference fingerprints for remote model IDs
- JSON report schema for machines
- Markdown report for pull requests and release evidence
- raw prompts excluded from reports by default for privacy

The goal is not to replace PEFT, Transformers, Unsloth, Axolotl, TRL or serving runtimes.
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

## Generate evidence reports

Write both machine-readable JSON and a PR-friendly Markdown report:

```bash
adapterguard verify \
  --base Qwen/Qwen3-8B \
  --adapter ./my-adapter \
  --merged ./merged-model \
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

Start with prompts that exercise the exact behavior you fine-tuned.

## Current scope

AdapterGuard v0.2 intentionally stays narrow:

- causal language models supported by `AutoModelForCausalLM`
- local/Hugging Face model IDs
- PEFT adapters loadable through `PeftModel`
- semantic comparison at logits/top-1-token level
- local artifact evidence reports

Not yet covered: sequence-classification adapters, multimodal adapters, GGUF, vLLM/TGI endpoint
probing, cross-runtime comparison, quantization-aware tolerance profiles and signed provenance.

## Why this repo exists

Most tooling answers **"can I load/merge this adapter?"** AdapterGuard asks a different question:

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

### v0.3 — runtime matrix

- [ ] quantization-aware verification
- [ ] vLLM endpoint verifier
- [ ] TGI endpoint verifier
- [ ] GGUF / llama.cpp comparison
- [ ] reusable GitHub Action for release gating

## Contributing

The most useful contributions are **real failure cases** where an adapter appeared to load or
merge successfully but behavior changed. Open an issue with the smallest reproducible example
you can share.

## License

MIT
