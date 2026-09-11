# Runtime verification

AdapterGuard v0.5 can compare a verified local artifact with OpenAI-compatible text-completion and
single-turn chat-completion endpoints such as vLLM.

## Why this check exists

Artifact integrity and deployment integrity are different problems. A LoRA adapter can survive
merge and quantization correctly while the serving layer still loads the wrong revision, wrong
quantized artifact, wrong tokenizer, stale deployment, or an unexpected served model.

AdapterGuard verifies the final transition:

```text
verified artifact -> serving runtime -> OpenAI-compatible API
```

## Reference selection

The runtime check automatically uses the most downstream local artifact available:

1. `--quantized` when supplied;
2. otherwise `--merged` when supplied;
3. otherwise the in-memory merged base + adapter model.

The selected source is recorded in `reference_artifact` inside runtime metrics. This prevents
accepted quantization drift from being incorrectly counted as serving-runtime drift.

## Text completions

```bash
adapterguard verify \
  --base Qwen/Qwen3-8B \
  --adapter ./adapter \
  --quantized ./qwen3-8b-awq \
  --prompts tests/golden.jsonl \
  --endpoint http://localhost:8000/v1 \
  --runtime-model qwen3-prod \
  --require-level runtime
```

A server root, `/v1`, or complete `/v1/completions` URL selects the normal text-completions
verifier.

## Chat completions

To verify an OpenAI-compatible chat endpoint, pass the complete chat-completions URL:

```bash
adapterguard verify \
  --base Qwen/Qwen3-8B \
  --adapter ./adapter \
  --prompts tests/golden.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --runtime-model qwen3-chat-prod \
  --require-level runtime
```

For each JSONL `prompt`, AdapterGuard sends one user message:

```json
{"role": "user", "content": "...prompt..."}
```

The local reference applies the base tokenizer's configured chat template with
`add_generation_prompt=true` before greedy generation. This matters because comparing a raw local
prompt against a server-rendered chat prompt would not be the same inference input.

Chat verification currently requires the tokenizer to define a chat template. If it does not,
AdapterGuard fails explicitly instead of silently falling back to raw text.

Current chat support is intentionally **single-turn**. Native JSONL `messages` arrays and multi-turn
conversation verification remain future extensions.

## Authentication

API keys are read from an environment variable rather than the command line. The default variable
is:

```bash
export ADAPTERGUARD_API_KEY=...
```

Use another variable with:

```bash
--runtime-api-key-env MY_VLLM_API_KEY
```

The key is never written to AdapterGuard reports.

## Comparison method

For each golden prompt AdapterGuard generates a deterministic greedy completion from the selected
local reference and requests the corresponding greedy completion from the endpoint.

Text-completion requests use:

```text
temperature = 0
logprobs = 5
max_tokens = --runtime-max-tokens
```

Chat-completion requests use:

```text
temperature = 0
logprobs = true
top_logprobs = 5
max_tokens = --runtime-max-tokens
```

AdapterGuard records:

- runtime API type (`openai-completions` or `openai-chat-completions`);
- first generated token agreement;
- exact greedy completion match rate;
- per-prompt local and served completion SHA-256 values;
- mean, p50, p95 and max request latency;
- requested model and model identifiers returned by the endpoint;
- the local reference artifact used for comparison.

Raw prompts and completions are excluded by default. `--include-prompts` explicitly enables them
for debugging.

## Release budgets

The defaults are strict because runtime verification isolates serving from earlier transformations:

```text
minimum first-token agreement: 1.0
minimum exact completion match rate: 1.0
```

They can be relaxed when a runtime is expected to have controlled numerical differences:

```bash
--min-runtime-first-token-agreement 0.99 \
--min-runtime-exact-match-rate 0.95
```

A runtime policy failure exits with code `1`, so it can directly gate deployment.

## Current scope

v0.5 covers OpenAI-compatible `/v1/completions` and single-turn `/v1/chat/completions` verification.
TGI-specific adapters, native GGUF/llama.cpp comparison, multi-turn chat inputs, and
runtime-to-runtime drift matrices remain future work.
