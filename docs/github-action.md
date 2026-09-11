# GitHub Action

AdapterGuard can run as a composite GitHub Action and fail a job when model integrity checks fail.
The Action installs AdapterGuard, runs the same `verify` pipeline as the CLI, writes JSON and
Markdown evidence, adds the Markdown report to the GitHub Job Summary, and can upload both reports
as a workflow artifact.

`v0.4.1` is the first stable Action release. Use `ptrgiang/adapterguard@v0.4.1` for normal
version-pinned workflows. For security-sensitive production workflows, pin the immutable release
commit SHA instead.

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

      - name: Verify adapter
        id: adapterguard
        uses: ptrgiang/adapterguard@v0.4.1
        with:
          adapter: ./adapter
          install-hf: "false"

      - run: echo "Verdict: ${{ steps.adapterguard.outputs.verdict }}"
```

`install-hf: "false"` keeps static-only checks lightweight.

## Semantic artifact gate

```yaml
- name: Verify fine-tuned artifacts
  uses: ptrgiang/adapterguard@v0.4.1
  with:
    adapter: ./adapter
    base: Qwen/Qwen3-8B
    merged: ./merged-model
    quantized: ./quantized-model
    prompts: ./tests/golden.jsonl
    fingerprint-mode: full
```

The default `install-hf: "true"` installs the optional Hugging Face stack required for semantic
checks.

## Deployment gate for vLLM / OpenAI-compatible runtimes

```yaml
- name: Verify deployed runtime
  id: adapterguard
  uses: ptrgiang/adapterguard@v0.4.1
  with:
    adapter: ./adapter
    base: Qwen/Qwen3-8B
    quantized: ./qwen3-8b-awq
    prompts: ./tests/golden.jsonl
    endpoint: ${{ secrets.VLLM_ENDPOINT }}
    runtime-model: qwen3-prod
    runtime-api-key: ${{ secrets.VLLM_API_KEY }}
    min-runtime-first-token-agreement: "1.0"
    min-runtime-exact-match-rate: "1.0"
```

The API key is masked before verification and is forwarded to AdapterGuard through
`ADAPTERGUARD_API_KEY`. It is never written into JSON or Markdown evidence.

## Evidence

By default the Action writes:

```text
.adapterguard/report.json
.adapterguard/report.md
```

and uploads them as the `adapterguard-evidence` workflow artifact. The Markdown report is also
appended to the job summary, so reviewers can inspect the verdict without downloading anything.

Customize these behaviors:

```yaml
with:
  report-json: artifacts/adapterguard.json
  report-markdown: artifacts/adapterguard.md
  evidence-name: model-integrity-evidence
  upload-evidence: "true"
```

Set `upload-evidence: "false"` when another workflow step already owns artifact publication.

## Outputs

| Output | Meaning |
| --- | --- |
| `verdict` | `SAFE TO SHIP`, `UNSAFE TO SHIP`, or `ERROR` |
| `safe-to-ship` | `true` only when the verification report contains no failed check |
| `report-json` | Absolute path to the JSON evidence report |
| `report-markdown` | Absolute path to the Markdown evidence report |

A failed AdapterGuard verification still returns a non-zero Action exit code, so it can directly
gate merge, release, or deployment jobs.

## Inputs

The Action maps its inputs directly to the corresponding CLI verification controls. Common inputs:

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

- Raw prompts and generated outputs remain excluded from evidence unless `include-prompts: "true"`.
- Runtime API keys are not passed as CLI arguments and are not serialized into reports.
- Pin `@v0.4.1` for versioned usage or the immutable release commit SHA for maximum supply-chain stability.
- Evidence is uploaded from explicit report paths only.
