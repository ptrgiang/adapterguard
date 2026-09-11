# AdapterGuard

[![CI](https://github.com/ptrgiang/adapterguard/actions/workflows/ci.yml/badge.svg)](https://github.com/ptrgiang/adapterguard/actions/workflows/ci.yml)
[![HF Semantic Smoke](https://github.com/ptrgiang/adapterguard/actions/workflows/hf-smoke.yml/badge.svg)](https://github.com/ptrgiang/adapterguard/actions/workflows/hf-smoke.yml)
[![Release](https://img.shields.io/github/v/release/ptrgiang/adapterguard)](https://github.com/ptrgiang/adapterguard/releases/latest)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/ptrgiang/adapterguard/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Know your adapter still works before you ship it.**

AdapterGuard is an independent semantic-integrity checker for LoRA, QLoRA, DoRA and other PEFT
adapters. It verifies that behavior survives the model-delivery path:

```text
train -> save -> reload -> merge -> quantize -> export -> serve
```

A model can load successfully, an inference server can return HTTP 200, and production can still be
serving the wrong behavior. AdapterGuard turns those silent regressions into reproducible evidence
and CI release gates.

**Stable:** `v0.4.1`  
**Development:** `main` / `0.5.0` — verification coverage policy, chat completions, native multi-turn
conversations, real HF smoke, and tokenizer/chat-template integrity.

## Why AdapterGuard exists

Most model tooling answers:

> **Can I load, merge, quantize or serve this model?**

AdapterGuard asks a different question:

> **Can I prove the artifact and deployment I am about to ship still behave like the adapter I trained?**

That distinction is the project.

## Release gate in a few lines

`v0.4.1` is the current stable reusable GitHub Action release. Pin a released tag for normal usage;
security-sensitive production workflows can pin the immutable release commit SHA.

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
        uses: ptrgiang/adapterguard@v0.4.1
        with:
          adapter: ./adapter
          base: Qwen/Qwen3-8B
          quantized: ./qwen3-8b-awq
          prompts: ./tests/golden.jsonl

      - run: echo "${{ steps.guard.outputs.verdict }}"
```

The upcoming v0.5 Action on `main` adds explicit verification coverage:

```yaml
- name: Require semantic release evidence
  id: guard
  uses: ptrgiang/adapterguard@main
  with:
    adapter: ./adapter
    base: Qwen/Qwen3-8B
    prompts: ./tests/golden.jsonl
    require-level: semantic
```

Coverage is ordered:

```text
static < semantic < quantization < runtime
```

A static-only run can pass a static policy, but it no longer claims release safety:

```text
verdict: STATIC CHECKS PASS
policy-passed: true
safe-to-ship: false
verification-level: static
```

Only semantic-or-stronger verification can return `SAFE TO SHIP`.

See [`docs/github-action.md`](docs/github-action.md) and
[`docs/report-schema.md`](docs/report-schema.md).

## Three self-contained failure proofs

The HF demos build tiny local GPT-2 models and real LoRA adapters. They do not need pretrained model
downloads.

```bash
pip install -e ".[hf]"
```

### 1. Loadable artifact, wrong weights

```bash
python examples/killer_demo.py
```

AdapterGuard performs a valid merge, then catches an exported model whose weights were deliberately
corrupted while the artifact still loads normally.

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
localizes the resulting behavior drift without requiring a GPU-only quantization stack.

### 3. Correct weights, wrong tokenizer

```bash
python examples/tokenizer_drift_demo.py
```

This demo keeps the exported model weights semantically correct but deliberately swaps tokenizer
IDs. AdapterGuard isolates the failure:

```text
weights: PASS
tokenizer: FAIL

DEMO PASS: AdapterGuard caught tokenizer drift while exported weights remained semantically correct.
```

All three proofs are exercised by the **HF Semantic Smoke** workflow on relevant pull requests,
relevant `main` changes, and a weekly schedule.

## What v0.5 verifies on `main`

**Static integrity**

- valid `adapter_config.json`
- adapter/base identity
- PEFT type and target modules
- LoRA rank
- adapter weight presence

**Behavioral integrity**

- base vs active adapter behavior
- active adapter vs `merge_and_unload()`
- active adapter vs exported/merged artifact
- merged reference vs quantized artifact
- configurable drift budgets
- mean/max logit drift and token-level top-1 agreement
- per-prompt localization and first divergent prediction

**Tokenizer integrity**

- exported tokenizer vs base tokenizer on golden-prompt token IDs
- quantized tokenizer vs base tokenizer
- mismatched prompt indices and exact encoding match rate
- chat-template rendering and chat token-ID equivalence for chat inputs
- explicit warning when an artifact tokenizer cannot be loaded and therefore was not verified

**Serving-runtime integrity**

- OpenAI-compatible `/v1/completions`
- OpenAI-compatible `/v1/chat/completions`
- native multi-turn OpenAI-style `messages` histories
- local chat reference rendered with the tokenizer's chat template
- deterministic greedy completion comparison
- first generated token agreement
- exact completion match rate
- local/served output hashes per prompt
- mean, p50, p95 and max endpoint latency
- requested and returned model identifiers
- automatic downstream reference selection: quantized -> exported/merged -> in-memory merge

**Evidence and policy**

- report schema `adapterguard.report.v2`
- `verification_level`, `required_level`, `policy_passed`, `safe_to_ship`
- sampled or full SHA-256 artifact fingerprints
- JSON machine-readable report
- Markdown PR/release report
- privacy-safe prompt/output hashing by default
- CI-safe exit codes
- reusable GitHub Action outputs and Job Summary

AdapterGuard does not replace PEFT, Transformers, Unsloth, Axolotl, TRL, quantizers, or inference
servers. It sits after and between them as an independent verifier.

## Install from source

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

PyPI publishing is prepared through Trusted Publishing but is not advertised here until the public
package has been published successfully.

## CLI quick start

Static-only inspection:

```bash
adapterguard verify --adapter ./my-adapter
```

Require semantic evidence:

```bash
adapterguard verify \
  --adapter ./my-adapter \
  --prompts examples/prompts.jsonl \
  --require-level semantic
```

Full artifact chain:

```bash
adapterguard verify \
  --base Qwen/Qwen3-8B \
  --adapter ./my-adapter \
  --merged ./merged-model \
  --quantized ./quantized-model \
  --prompts tests/golden.jsonl \
  --require-level quantization \
  --report-json artifacts/adapterguard.json \
  --report-markdown artifacts/adapterguard.md
```

A policy failure exits with code `1`, so the command can gate a release or deployment.

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
  --require-level quantization \
  --max-quantized-diff 0.03 \
  --min-quantized-top1-agreement 0.99
```

The defaults are release-gate starting points, not universal quality thresholds. Calibrate them
against your model, task, and golden prompts.

## Runtime integrity / vLLM

Text completions:

```bash
export ADAPTERGUARD_API_KEY=...

adapterguard verify \
  --base Qwen/Qwen3-8B \
  --adapter ./my-adapter \
  --quantized ./qwen3-8b-awq \
  --prompts tests/golden.jsonl \
  --endpoint http://localhost:8000/v1 \
  --runtime-model qwen3-prod \
  --require-level runtime
```

Chat completions can use the same single-turn prompt file or native multi-turn histories:

```bash
adapterguard verify \
  --base Qwen/Qwen3-8B \
  --adapter ./my-adapter \
  --prompts examples/messages.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --runtime-model qwen3-chat-prod \
  --require-level runtime
```

A multi-turn JSONL entry looks like:

```json
{"messages":[
  {"role":"system","content":"You are concise."},
  {"role":"user","content":"Remember order A-102."},
  {"role":"assistant","content":"Understood."},
  {"role":"user","content":"Which order did I mention?"}
]}
```

AdapterGuard applies the base tokenizer's configured chat template to the exact same history locally
with `add_generation_prompt=true`, then sends the original `messages` array to the runtime. String
content is supported in v0.5; multimodal content arrays fail explicitly until multimodal integrity
verification is implemented.

Runtime comparison uses the most downstream verified local artifact:

1. `--quantized` when supplied
2. otherwise `--merged`
3. otherwise the in-memory merged model

See [`docs/runtime-verification.md`](docs/runtime-verification.md).

## Evidence reports

```bash
adapterguard verify \
  --adapter ./my-adapter \
  --report-json .adapterguard/report.json \
  --report-markdown .adapterguard/report.md
```

Raw prompts and runtime outputs are excluded by default. Native message histories are canonicalized
and hashed before entering evidence. Use `--include-prompts` only when the data is safe to expose in
CI artifacts.

Fingerprint modes:

- `sampled` — file identity, size, and first/last 64 KiB; fast default for large models
- `full` — streams every byte for release-grade evidence
- `off` — disables artifact fingerprints

## Prompt format

`--prompts` accepts JSONL. Each line can be a JSON string:

```json
"Explain gradient descent simply."
```

an object with `prompt`:

```json
{"prompt":"Explain gradient descent simply."}
```

or an OpenAI-style conversation:

```json
{"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi"},{"role":"user","content":"Summarize our exchange."}]}
```

Golden inputs should cover the behavior the model was fine-tuned for and the behavior users rely on
in production.

## Current scope

Current `main` supports:

- causal language models supported by `AutoModelForCausalLM`
- local/Hugging Face model IDs
- PEFT adapters loadable through `PeftModel`
- logit/top-1 semantic comparisons
- quantized artifacts loadable by the installed Transformers/runtime stack
- exported/quantized tokenizer-integrity checks
- OpenAI-compatible text completions
- OpenAI-compatible single-turn and multi-turn chat completions
- reusable GitHub Action release/deployment gates
- JSON/Markdown evidence and GitHub Job Summary output
- real HF semantic smoke proofs in GitHub Actions

Not yet covered: sequence-classification adapters, multimodal adapters/messages, native GGUF/llama.cpp
loading, TGI-specific verification, and runtime-to-runtime drift matrices.

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

### v0.4 — deployment integrity ✅

- [x] OpenAI-compatible / vLLM text-completion verifier
- [x] downstream reference selection
- [x] per-prompt runtime divergence evidence
- [x] runtime latency evidence
- [x] reusable GitHub Action release gate

### v0.5 — release-policy and input-semantics hardening

- [x] report schema v2 verification levels
- [x] `--require-level` release policy
- [x] static-only verdict no longer claims `SAFE TO SHIP`
- [x] real localhost HTTP integration test
- [x] real Hugging Face semantic smoke workflow
- [x] HF smoke gate on semantic pull requests
- [x] OpenAI-compatible chat-completions verifier
- [x] tokenizer prompt-encoding drift detection
- [x] chat-template drift detection
- [x] self-contained tokenizer-drift proof
- [x] native multi-turn `messages` prompt format

### Later

- [ ] TGI endpoint verifier
- [ ] GGUF / llama.cpp comparison
- [ ] runtime-to-runtime drift matrix
- [ ] sequence-classification and multimodal adapter support

## Contributing

The most useful contributions are real failure cases where an adapter appeared to load, merge,
quantize or serve successfully but behavior changed. Open an issue with the smallest reproducible
example you can share.

## License

MIT
