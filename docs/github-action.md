# GitHub Action

AdapterGuard runs as a composite GitHub Action and can gate model releases with static, semantic,
quantization, and serving-runtime evidence. It writes JSON and Markdown reports, appends the report
to the GitHub Job Summary, and can upload the evidence as a workflow artifact.

## Version note

`v0.4.1` remains the current stable Action release. The verification-policy features documented
below are on `main` for the upcoming v0.5 release. Until v0.5 is tagged, use `@main` only when you
specifically want to test the development version. Production users should keep `@v0.4.1` or an
immutable release commit SHA until the next stable release is published.

## Verification coverage policy

Current `main` distinguishes the checks that passed from how much verification actually ran.
Coverage levels are ordered:

```text
static < semantic < quantization < runtime
```

Use `require-level` to state the minimum evidence required by the workflow.

A static-only check can pass a static policy, but it returns:

```text
verdict: STATIC CHECKS PASS
safe-to-ship: false
policy-passed: true
verification-level: static
```

This prevents structural validation from being mistaken for semantic release safety.

## Minimal static gate

```yaml
name: Model integrity

on:
  pull_request:

jobs:
  adapterguard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7

      - name: Verify adapter structure
        id: adapterguard
        uses: ptrgiang/adapterguard@main
        with:
          adapter: ./adapter
          install-hf: "false"
          require-level: static

      - run: echo "${{ steps.adapterguard.outputs.verdict }}"
```

`install-hf: "false"` keeps static-only checks lightweight.

## Semantic release gate

For a release gate that is allowed to claim semantic safety, require at least `semantic`:

```yaml
- name: Verify fine-tuned artifact
  id: adapterguard
  uses: ptrgiang/adapterguard@main
  with:
    adapter: ./adapter
    base: Qwen/Qwen3-8B
    merged: ./merged-model
    prompts: ./tests/golden.jsonl
    require-level: semantic
    fingerprint-mode: full
```

The default `install-hf: "true"` installs the Hugging Face dependencies required for semantic
checks.

## Quantized artifact gate

```yaml
- name: Verify quantized artifact
  uses: ptrgiang/adapterguard@main
  with:
    adapter: ./adapter
    base: Qwen/Qwen3-8B
    quantized: ./qwen3-8b-awq
    prompts: ./tests/golden.jsonl
    require-level: quantization
    max-quantized-diff: "0.05"
    min-quantized-top1-agreement: "0.98"
```

## Deployment gate for vLLM / OpenAI-compatible runtimes

```yaml
- name: Verify deployed runtime
  id: adapterguard
  uses: ptrgiang/adapterguard@main
  with:
    adapter: ./adapter
    base: Qwen/Qwen3-8B
    quantized: ./qwen3-8b-awq
    prompts: ./tests/golden.jsonl
    endpoint: ${{ secrets.VLLM_ENDPOINT }}
    runtime-model: qwen3-prod
    runtime-api-key: ${{ secrets.VLLM_API_KEY }}
    require-level: runtime
    min-runtime-first-token-agreement: "1.0"
    min-runtime-exact-match-rate: "1.0"
```

The API key is masked before verification and forwarded through `ADAPTERGUARD_API_KEY`. It is never
serialized into JSON or Markdown evidence.

## Evidence

By default the Action writes:

```text
.adapterguard/report.json
.adapterguard/report.md
```

and uploads them as the `adapterguard-evidence` workflow artifact. The Markdown report is also
appended to the job summary.

Customize these behaviors:

```yaml
with:
  report-json: artifacts/adapterguard.json
  report-markdown: artifacts/adapterguard.md
  evidence-name: model-integrity-evidence
  upload-evidence: "true"
```

Set `upload-evidence: "false"` when another workflow step owns artifact publication.

## Outputs on current `main`

| Output | Meaning |
| --- | --- |
| `verdict` | `STATIC CHECKS PASS`, `SAFE TO SHIP`, `UNSAFE TO SHIP`, or `ERROR` |
| `safe-to-ship` | `true` only when semantic-or-stronger verification passed policy |
| `policy-passed` | `true` when checks pass and requested coverage was reached |
| `verification-level` | Actual coverage reached: `static`, `semantic`, `quantization`, or `runtime` |
| `required-level` | Minimum coverage requested by `require-level` |
| `report-json` | Absolute path to JSON evidence |
| `report-markdown` | Absolute path to Markdown evidence |

A failed policy returns a non-zero Action exit code, so it can directly gate merge, release, or
deployment jobs.

## Common inputs on current `main`

| Input | Default | Purpose |
| --- | --- | --- |
| `adapter` | required | PEFT adapter directory |
| `base` | adapter config | Base model ID/path |
| `prompts` | empty | JSONL golden prompts |
| `merged` | empty | Exported merged model |
| `quantized` | empty | Quantized artifact |
| `endpoint` | empty | OpenAI-compatible API URL |
| `runtime-model` | base model | Model name sent to runtime |
| `runtime-api-key` | empty | Runtime secret |
| `require-level` | `static` | Minimum verification coverage required to pass |
| `install-hf` | `true` | Install semantic model dependencies |
| `fingerprint-mode` | `sampled` | `sampled`, `full`, or `off` |
| `upload-evidence` | `true` | Upload generated reports |

Drift budgets are also exposed:

```yaml
with:
  max-merge-diff: "0.001"
  min-top1-agreement: "0.999"
  max-quantized-diff: "0.05"
  min-quantized-top1-agreement: "0.98"
  min-runtime-first-token-agreement: "1.0"
  min-runtime-exact-match-rate: "1.0"
```

Treat these as release-policy inputs. Quantization/runtime thresholds should be calibrated against
the exact model, task, golden prompt set, and acceptable production behavior.

## Security notes

- Raw prompts and generated outputs remain excluded unless `include-prompts: "true"`.
- Runtime API keys are not passed as CLI arguments or serialized into reports.
- Use a released tag for normal version pinning and an immutable commit SHA for maximum supply-chain
  stability.
- Evidence is uploaded from explicit report paths only.
