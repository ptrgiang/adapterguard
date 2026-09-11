# Changelog

All notable AdapterGuard changes are summarized here. GitHub Releases remain the source for signed
release tags and downloadable source archives.

## 0.5.0 — pending release

### Release-policy correctness

- add report schema `adapterguard.report.v2`
- distinguish `verification_level`, `required_level`, `policy_passed`, and `safe_to_ship`
- stop static-only verification from claiming `SAFE TO SHIP`
- add `--require-level static|semantic|quantization|runtime`
- expose verification coverage through the reusable GitHub Action

### Serving-runtime integrity

- add real localhost HTTP integration coverage for OpenAI-compatible endpoints
- support `/v1/chat/completions` in addition to `/v1/completions`
- support native multi-turn OpenAI-style `messages` JSONL histories
- apply the tokenizer chat template to the exact same conversation for the local reference
- reject native message histories on text-completion endpoints instead of flattening them
- record `multi_turn_prompt_count` in runtime evidence

### Tokenizer and input-semantics integrity

- compare exported and quantized tokenizer prompt encodings against the base tokenizer
- detect chat-template rendering and token-ID drift
- localize tokenizer mismatches to golden-input indices
- warn explicitly when an artifact tokenizer cannot be loaded
- add a self-contained proof where model weights are correct but the shipped tokenizer is wrong

### CI and reproducible proofs

- add real Transformers/PEFT semantic smoke testing on relevant pull requests and `main` changes
- run four self-contained proofs: corrupted weights, quantization drift, tokenizer drift, and native
  multi-turn histories
- use CPU-only PyTorch in HF smoke to avoid unnecessary CUDA dependency downloads
- add package build/install validation to normal CI
- harden PyPI publication so a release tag must pass lint, tests, all HF proofs, package validation,
  and exact wheel-version validation before OIDC publishing

### Documentation

- clarify stable `v0.4.1` versus development `v0.5.0`
- document static-vs-release-safe verdict semantics
- add native multi-turn prompt examples
- refresh runtime verification and project scope

## 0.4.1

- add reusable GitHub Action release gate
- publish JSON and Markdown evidence to GitHub Actions
- expose release-gate outputs and Job Summary evidence

## 0.4.0

- add OpenAI-compatible/vLLM runtime verification
- add runtime latency and divergence evidence
- select the most downstream verified local artifact as the runtime reference

## 0.3.0

- add quantized-artifact semantic drift verification
- add configurable quantization budgets and metadata evidence

## 0.2.0

- add reproducible artifact fingerprints
- add prompt-level evidence and Markdown reports

## 0.1.0

- initial static and semantic adapter-integrity verification
