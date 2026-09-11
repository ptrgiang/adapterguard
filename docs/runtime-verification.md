# Runtime verification

AdapterGuard v0.5 can compare a verified local artifact with OpenAI-compatible text-completion and
chat-completion endpoints such as vLLM.

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
verifier. Text-completion verification accepts only text prompt cases because the endpoint itself
has no chat-message semantics.

## Chat completions

To verify an OpenAI-compatible chat endpoint, pass the complete chat-completions URL:

```bash
adapterguard verify \
  --base Qwen/Qwen3-8B \
  --adapter ./adapter \
  --prompts examples/messages.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --runtime-model qwen3-chat-prod \
  --require-level runtime
```

Chat verification accepts both the original text format and native multi-turn `messages` cases.
A text prompt is treated as one user message:

```json
{"prompt":"Explain gradient descent simply."}
```

A native conversation keeps the supplied history intact:

```json
{"messages":[
  {"role":"system","content":"You are concise and factual."},
  {"role":"user","content":"My order ID is A-102. Remember it."},
  {"role":"assistant","content":"Understood."},
  {"role":"user","content":"What order ID did I give you?"}
]}
```

Each `messages` entry currently requires string `role` and string `content`. Multimodal content
arrays are rejected explicitly until multimodal verification is implemented.

For the local reference, AdapterGuard applies the base tokenizer's configured chat template to the
same message history with `add_generation_prompt=true`, then performs deterministic greedy
generation. The runtime request sends the original messages array to `/v1/chat/completions`.

This prevents a false comparison between raw local text and a server-rendered conversation. It also
means tokenizer integrity checks automatically compare chat-template rendering and resulting token
IDs for native `messages` cases, even when no runtime endpoint is supplied.

Chat verification requires the tokenizer to define a chat template. If it does not, AdapterGuard
fails explicitly instead of silently falling back to raw text.

## Prompt privacy and evidence

Text prompts are hashed directly. Native chat histories are converted to deterministic canonical
JSON before hashing, so the evidence fingerprint is stable without exposing the conversation.
`--include-prompts` stores that canonical representation in the report for debugging; leave it off
for sensitive production prompts.

Runtime metrics include `multi_turn_prompt_count`, making it visible whether the gate actually
covered conversation histories instead of only single user turns.

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

For each golden input AdapterGuard generates a deterministic greedy completion from the selected
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
- number of native multi-turn conversation cases;
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

v0.5 covers OpenAI-compatible `/v1/completions` plus single-turn and native multi-turn
`/v1/chat/completions` verification with text-only message content. TGI-specific adapters, native
GGUF/llama.cpp comparison, multimodal message content, and runtime-to-runtime drift matrices remain
future work.
