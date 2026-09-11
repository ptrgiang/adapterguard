# AdapterGuard

> **Know your adapter still works before you ship it.**

AdapterGuard is an independent semantic integrity checker for LoRA, QLoRA, DoRA and other PEFT
adapters. It targets the risky gap between **"training finished"** and **"the model actually served
to users still behaves like the artifact I verified."**

```text
train -> save adapter -> reload -> merge -> quantize -> export -> serve
```

Each transformation can silently change behavior. A model can load successfully, a runtime can
answer requests successfully, and the deployed system can still be serving the wrong behavior.
AdapterGuard makes those failures visible, reproducible and CI-friendly.

## 60-second proofs

### Corrupted export

AdapterGuard includes a self-contained demo that needs no pretrained model download. It builds a
tiny local GPT-2 model, attaches a real LoRA adapter, performs a valid merge, deliberately corrupts
the exported model while keeping it loadable, and proves that AdapterGuard catches the semantic
divergence.

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

The quantization demo creates a merged LoRA model, applies aggressive symmetric fake quantization
to its weights, saves a still-loadable Hugging Face artifact and asks AdapterGuard to compare it
against the in-memory merged reference.

```bash
python examples/quantization_demo.py
```

Expected final result:

```text
FAIL  quantized model preserves merged behavior

DEMO PASS: AdapterGuard localized quantization-induced semantic drift.
```

Real GPTQ, AWQ or bitsandbytes artifacts can be supplied through `--quantized` when their runtime
dependencies are installed.

## What v0.4 checks

**Static checks (fast, no model load):**

- `adapter_config.json` exists and is valid JSON
- adapter/base-model identity matches what you requested
- PEFT type and target modules are declared
- LoRA rank is valid
- adapter weights are present

**Artifact semantic checks:**

- the adapter measurably changes model logits vs. the base model
- `merge_and_unload()` preserves active-adapter behavior
- an exported/merged model preserves active-adapter behavior
- a quantized model stays within an explicit semantic drift budget vs. the merged reference
- per-prompt mean/max logit drift and top-1 token agreement
- first position where top-1 token predictions diverge
- worst prompt and count of prompts with top-1 divergence after quantization
- quantization metadata when exposed by the model config

**Serving-runtime checks:**

- OpenAI-compatible `/v1/completions` endpoints, including vLLM-style servers
- deterministic greedy completion comparison against the verified local artifact
- first generated token agreement
- exact completion match rate
- per-prompt local/served output hashes
- mean, p50, p95 and max endpoint latency
- requested model and model identifiers returned by the server
- automatic selection of the most downstream local reference artifact

**Evidence layer:**

- sampled or full SHA-256 fingerprints for local artifacts
- stable reference fingerprints for remote model IDs
- fingerprints for adapter, base, merged and quantized artifacts
- JSON report schema for machines
- Markdown report for pull requests and release evidence
- raw prompts and runtime outputs excluded from reports by default for privacy

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

Compare against exported and quantized artifacts:

```bash
adapterguard verify \
  --base Qwen/Qwen3-8B \
  --adapter ./my-adapter \
  --merged ./merged-model \
  --quantized ./quantized-model \
  --prompts tests/golden.jsonl
```

A failed check exits with code `1`, so the command can gate a release or deployment.

## Quantization integrity

AdapterGuard compares `--quantized` against the in-memory merged model created from the same base +
adapter. This isolates the `merge -> quantize` transformation instead of mixing quantization drift
with adapter or merge drift.

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
against your own model, quantizer and golden prompts.

## Runtime integrity / vLLM

Once an artifact has been served, verify the last transition too:

```text
verified artifact -> serving runtime -> OpenAI-compatible API
```

Example:

```bash
export ADAPTERGUARD_API_KEY=...

adapterguard verify \
  --base Qwen/Qwen3-8B \
  --adapter ./my-adapter \
  --quantized ./qwen3-8b-awq \
  --prompts tests/golden.jsonl \
  --endpoint http://localhost:8000/v1 \
  --runtime-model qwen3-prod \
  --report-json artifacts/adapterguard.json \
  --report-markdown artifacts/adapterguard.md
```

The runtime reference is selected automatically:

1. `--quantized` if supplied
2. otherwise `--merged` if supplied
3. otherwise the in-memory merged base + adapter model

This is recorded as `reference_artifact` in the runtime metrics, so quantization drift is not
mistaken for serving drift.

Default runtime release budgets are strict:

```text
minimum first-token agreement: 1.0
minimum exact greedy completion match rate: 1.0
```

Relax them explicitly when your runtime is expected to introduce controlled numerical variation:

```bash
--min-runtime-first-token-agreement 0.99 \
--min-runtime-exact-match-rate 0.95
```

Endpoint authentication is read from `ADAPTERGUARD_API_KEY` by default. Use another variable with
`--runtime-api-key-env`. Secrets are never written into evidence reports.

See [`docs/runtime-verification.md`](docs/runtime-verification.md) for the full runtime contract.

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

Raw prompt and runtime output text is **not included by default**. If the data is safe to expose in
CI artifacts:

```bash
adapterguard verify ... --include-prompts --report-markdown adapterguard.md
```

For stdout JSON:

```bash
adapterguard verify --adapter ./my-adapter --json
```

See [`docs/report-schema.md`](docs/report-schema.md) for the report contract.

## Artifact fingerprint modes

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

Golden prompts should exercise the exact behavior you fine-tuned and the behavior users actually
rely on in production.

## Current scope

AdapterGuard v0.4 currently supports:

- causal language models supported by `AutoModelForCausalLM`
- local/Hugging Face model IDs
- PEFT adapters loadable through `PeftModel`
- semantic comparison at logits/top-1-token level
- quantized artifacts loadable by the installed Transformers/runtime stack
- OpenAI-compatible text completion endpoints
- local JSON/Markdown evidence reports

Not yet covered: sequence-classification adapters, multimodal adapters, native GGUF/llama.cpp
loading, chat-completions verification, TGI-specific verification and runtime-to-runtime matrices.

## Why this repo exists

Most tooling answers **"can I load, merge, quantize or serve this model?"** AdapterGuard asks a
different question:

> **"Can I prove the artifact and deployment I am about to ship still behave like the adapter I trained?"**

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

- [x] OpenAI-compatible / vLLM endpoint verifier
- [x] downstream reference selection
- [x] per-prompt runtime divergence evidence
- [x] runtime latency evidence
- [ ] chat-completions verifier
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
