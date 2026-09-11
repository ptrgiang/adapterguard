# AdapterGuard

> **Know your adapter still works before you ship it.**

AdapterGuard is an independent semantic integrity checker for LoRA, QLoRA, DoRA and other PEFT
adapters. It verifies that behavior survives the full model-delivery path:

```text
train -> save -> reload -> merge -> quantize -> export -> serve
```

A model can load successfully, an inference server can return HTTP 200, and production can still be
serving the wrong behavior. AdapterGuard turns those silent regressions into reproducible evidence
and CI release gates.

## GitHub Action — release gate in a few lines

AdapterGuard ships as a reusable composite Action. Until the first versioned Action release is
published, use `@main`; pin a released tag or immutable commit SHA for production once available.

```yaml
name: Model integrity

on:
  pull_request:

jobs:
  adapterguard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7

      - name: Verify model artifact
        id: guard
        uses: ptrgiang/adapterguard@main
        with:
          adapter: ./adapter
          base: Qwen/Qwen3-8B
          quantized: ./qwen3-8b-awq
          prompts: ./tests/golden.jsonl

      - run: echo "${{ steps.guard.outputs.verdict }}"
```

For a deployed vLLM/OpenAI-compatible endpoint:

```yaml
- name: Verify deployed runtime
  uses: ptrgiang/adapterguard@main
  with:
    adapter: ./adapter
    base: Qwen/Qwen3-8B
    quantized: ./qwen3-8b-awq
    prompts: ./tests/golden.jsonl
    endpoint: ${{ secrets.VLLM_ENDPOINT }}
    runtime-model: qwen3-prod
    runtime-api-key: ${{ secrets.VLLM_API_KEY }}
```

The Action:

- fails the job when AdapterGuard returns an unsafe verdict
- writes `.adapterguard/report.json` and `.adapterguard/report.md`
- adds the Markdown evidence to the GitHub Job Summary
- uploads the reports as `adapterguard-evidence` by default
- exposes `verdict`, `safe-to-ship`, `report-json`, and `report-markdown` outputs
- masks runtime API keys and never serializes them into evidence

See [`docs/github-action.md`](docs/github-action.md) for all inputs, outputs and security notes.

## 60-second proofs

### 1. Loadable artifact, wrong behavior

The self-contained corrupted-artifact demo needs no pretrained model download. It builds a tiny
local GPT-2 model, attaches a real LoRA adapter, performs a valid merge, deliberately corrupts the
exported model while keeping it loadable, then proves AdapterGuard catches the divergence.

```bash
pip install -e ".[hf]"
python examples/killer_demo.py
```

Expected result:

```text
PASS  adapter changes model behavior
PASS  merge preserves adapter behavior
FAIL  exported model preserves adapter behavior

DEMO PASS: AdapterGuard caught a loadable but semantically corrupted artifact.
```

### 2. Quantization drift

```bash
python examples/quantization_demo.py
```

The demo applies aggressive fake quantization to a merged LoRA model and verifies that AdapterGuard
localizes the resulting drift without requiring GPU-only quantization libraries.

Real GPTQ, AWQ or bitsandbytes artifacts can be supplied through `--quantized` when their runtime
dependencies are installed.

## What v0.4.1 verifies

**Static integrity**

- valid `adapter_config.json`
- adapter/base identity
- PEFT type and target modules
- LoRA rank
- adapter weight presence

**Artifact semantic integrity**

- base vs active adapter behavior
- active adapter vs `merge_and_unload()`
- active adapter vs exported/merged artifact
- merged reference vs quantized artifact
- configurable drift budgets
- mean/max logit drift and token-level top-1 agreement
- per-prompt localization and first divergent token
- quantization metadata when declared

**Serving-runtime integrity**

- OpenAI-compatible `/v1/completions`, including vLLM-style servers
- deterministic greedy completion comparison
- first generated token agreement
- exact completion match rate
- local/served output hashes per prompt
- mean, p50, p95 and max endpoint latency
- requested and returned model identifiers
- automatic downstream reference selection: quantized -> merged/exported -> in-memory merge

**Evidence**

- sampled or full SHA-256 artifact fingerprints
- JSON machine-readable report
- Markdown PR/release report
- privacy-safe prompt and output hashing by default
- CI-safe exit codes

AdapterGuard does not replace PEFT, Transformers, Unsloth, Axolotl, TRL, quantizers or inference
servers. It sits after and between them as an independent verifier.

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

## CLI quick start

Static-only:

```bash
adapterguard verify --adapter ./my-adapter
```

Semantic adapter verification:

```bash
adapterguard verify \
  --adapter ./my-adapter \
  --prompts examples/prompts.jsonl
```

Full artifact chain:

```bash
adapterguard verify \
  --base Qwen/Qwen3-8B \
  --adapter ./my-adapter \
  --merged ./merged-model \
  --quantized ./quantized-model \
  --prompts tests/golden.jsonl \
  --report-json artifacts/adapterguard.json \
  --report-markdown artifacts/adapterguard.md
```

A failed verification exits with code `1`, so the command can gate a release or deployment.

## Quantization integrity

AdapterGuard compares `--quantized` against the in-memory merged model from the same base + adapter.
That isolates the `merge -> quantize` transformation instead of mixing quantization drift with
adapter/merge drift.

```bash
adapterguard verify \
  --base Qwen/Qwen3-8B \
  --adapter ./my-adapter \
  --quantized ./qwen3-8b-awq \
  --prompts tests/golden.jsonl \
  --max-quantized-diff 0.03 \
  --min-quantized-top1-agreement 0.99
```

The defaults are release-gate starting points, not universal quality thresholds. Calibrate them
against your model, task and golden prompts.

## Runtime integrity / vLLM

```bash
export ADAPTERGUARD_API_KEY=...

adapterguard verify \
  --base Qwen/Qwen3-8B \
  --adapter ./my-adapter \
  --quantized ./qwen3-8b-awq \
  --prompts tests/golden.jsonl \
  --endpoint http://localhost:8000/v1 \
  --runtime-model qwen3-prod
```

Runtime comparison uses the most downstream verified local artifact:

1. `--quantized` when supplied
2. otherwise `--merged`
3. otherwise the in-memory merged model

This choice is recorded as `reference_artifact`, so accepted quantization drift is not mistaken for
serving drift.

Default runtime budgets are intentionally strict:

```text
minimum first-token agreement: 1.0
minimum exact greedy completion match rate: 1.0
```

See [`docs/runtime-verification.md`](docs/runtime-verification.md).

## Evidence reports

```bash
adapterguard verify \
  --adapter ./my-adapter \
  --report-json .adapterguard/report.json \
  --report-markdown .adapterguard/report.md
```

Raw prompts and runtime outputs are excluded by default. Use `--include-prompts` only when the data
is safe to expose in CI artifacts.

Fingerprint modes:

- `sampled` — file identity, size and first/last 64 KiB; fast default for large models
- `full` — streams every byte for release-grade evidence
- `off` — disables artifact fingerprints

See [`docs/report-schema.md`](docs/report-schema.md).

## Prompt format

`--prompts` accepts JSONL. Each line can be a JSON string:

```json
"Explain gradient descent simply."
```

or an object:

```json
{"prompt": "Explain gradient descent simply."}
```

Golden prompts should cover the behavior the model was fine-tuned for and the behavior users rely
on in production.

## Current scope

AdapterGuard v0.4.1 supports:

- causal language models supported by `AutoModelForCausalLM`
- local/Hugging Face model IDs
- PEFT adapters loadable through `PeftModel`
- logit/top-1 semantic comparisons
- quantized artifacts loadable by the installed Transformers/runtime stack
- OpenAI-compatible text-completion endpoints
- reusable GitHub Action release/deployment gates
- JSON/Markdown evidence reports and GitHub Job Summary output

Not yet covered: sequence-classification adapters, multimodal adapters, native GGUF/llama.cpp
loading, chat-completions verification, TGI-specific verification and runtime-to-runtime matrices.

## Why this repo exists

Most tooling answers **"can I load, merge, quantize or serve this model?"** AdapterGuard asks:

> **"Can I prove the artifact and deployment I am about to ship still behave like the adapter I trained?"**

That distinction is the project.

## Roadmap

### v0.1 — semantic baseline ✅

- [x] static adapter inspection
- [x] base-vs-adapter behavioral check
- [x] active-adapter-vs-merge comparison
- [x] exported model comparison
- [x] JSON output and CI-safe exit codes
- [x] self-contained corrupted-artifact proof

### v0.2 — reproducible evidence ✅

- [x] artifact fingerprints
- [x] per-prompt failure localization
- [x] first divergent token prediction
- [x] machine-readable report schema
- [x] Markdown evidence report
- [x] privacy-safe hashing
- [ ] save -> reload equivalence against a captured pre-save reference

### v0.3 — quantization integrity ✅

- [x] quantized-artifact comparison
- [x] configurable quantization drift budgets
- [x] per-prompt quantization localization
- [x] quantized artifact fingerprinting
- [x] quantization metadata extraction
- [x] self-contained fake-quantization proof

### v0.4 — deployment integrity

- [x] OpenAI-compatible / vLLM endpoint verifier
- [x] downstream reference selection
- [x] per-prompt runtime divergence evidence
- [x] runtime latency evidence
- [x] reusable GitHub Action release gate
- [ ] chat-completions verifier
- [ ] TGI endpoint verifier
- [ ] GGUF / llama.cpp comparison
- [ ] runtime-to-runtime drift matrix

## Contributing

The most useful contributions are real failure cases where an adapter appeared to load, merge,
quantize or serve successfully but behavior changed. Open an issue with the smallest reproducible
example you can share.

## License

MIT
