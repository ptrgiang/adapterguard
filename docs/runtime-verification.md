# Runtime verification

AdapterGuard v0.4 can compare a verified local artifact with a model served through an OpenAI-compatible `/v1/completions` endpoint such as vLLM.

## Why this check exists

Artifact integrity and deployment integrity are different problems. A LoRA adapter can survive merge and quantization correctly while the serving layer still loads the wrong revision, wrong quantized artifact, wrong tokenizer, stale deployment, or an unexpected served model.

AdapterGuard therefore verifies the final transition:

```text
verified artifact -> serving runtime -> OpenAI-compatible API
```

## Reference selection

The runtime check automatically uses the most downstream local artifact available:

1. `--quantized` when supplied;
2. otherwise `--merged` when supplied;
3. otherwise the in-memory merged base + adapter model.

The selected source is recorded in `reference_artifact` inside the runtime check metrics. This prevents quantization drift from being incorrectly counted as serving-runtime drift.

## vLLM example

Start vLLM with the artifact you intend to deploy, then run:

```bash
adapterguard verify \
  --base Qwen/Qwen3-8B \
  --adapter ./adapter \
  --quantized ./qwen3-8b-awq \
  --prompts tests/golden.jsonl \
  --endpoint http://localhost:8000/v1 \
  --runtime-model qwen3-prod \
  --report-json artifacts/adapterguard.json \
  --report-markdown artifacts/adapterguard.md
```

`--endpoint` accepts a server root, `/v1`, or a complete `/v1/completions` URL.

## Authentication

API keys are read from an environment variable rather than the command line. The default variable is:

```bash
export ADAPTERGUARD_API_KEY=...
```

Use another variable with:

```bash
--runtime-api-key-env MY_VLLM_API_KEY
```

The key is never written to AdapterGuard reports.

## Comparison method

For each golden prompt AdapterGuard generates a deterministic greedy completion from the selected local reference and requests the same greedy completion from the endpoint with:

```text
temperature = 0
logprobs = 5
max_tokens = --runtime-max-tokens
```

It records:

- first generated token agreement;
- exact greedy completion match rate;
- per-prompt local and served completion SHA-256 values;
- mean, p50, p95 and max request latency;
- requested model and model identifiers returned by the endpoint;
- the local reference artifact used for comparison.

Raw prompts and completions are excluded by default. `--include-prompts` explicitly enables them for debugging.

## Release budgets

The initial defaults are strict because this check isolates serving from earlier transformations:

```text
minimum first-token agreement: 1.0
minimum exact completion match rate: 1.0
```

They can be relaxed when a runtime is expected to have controlled numerical differences:

```bash
--min-runtime-first-token-agreement 0.99 \
--min-runtime-exact-match-rate 0.95
```

A failed runtime check exits with code `1`, so it can gate a deployment or release pipeline.

## Current scope

v0.4 targets OpenAI-compatible text completion endpoints. Chat-completions, TGI-specific adapters, GGUF/llama.cpp and runtime-to-runtime matrices remain future work.
