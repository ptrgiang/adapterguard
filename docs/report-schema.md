# AdapterGuard evidence report schema

AdapterGuard v0.5 uses report schema `adapterguard.report.v2`.

The v2 schema separates **verification coverage** from **release policy** so a lightweight static
check can pass without falsely claiming that a model is semantically safe to ship.

```json
{
  "schema_version": "adapterguard.report.v2",
  "verdict": "STATIC CHECKS PASS",
  "safe_to_ship": false,
  "policy_passed": true,
  "checks_passed": true,
  "coverage_satisfied": true,
  "verification_level": "static",
  "required_level": "static",
  "fingerprints": {
    "adapter": {
      "label": "adapter",
      "source": "./adapter",
      "mode": "sampled",
      "sha256": "...",
      "file_count": 3,
      "total_bytes": 123456
    }
  },
  "checks": []
}
```

## Verification levels

Coverage is ordered from weakest to strongest:

```text
static < semantic < quantization < runtime
```

- `static` validates adapter metadata and artifact structure.
- `semantic` verifies adapter behavior and merge/export equivalence with golden prompts.
- `quantization` additionally verifies the quantized artifact against the merged reference.
- `runtime` additionally verifies the deployed OpenAI-compatible endpoint against the most
  downstream local artifact.

`required_level` is the minimum coverage requested by the release policy. The CLI exposes it as
`--require-level`; the GitHub Action exposes `require-level`.

## Policy fields

`checks_passed` is true when no executed check has status `fail`.

`coverage_satisfied` is true when `verification_level >= required_level`.

`policy_passed` is true only when both of those conditions are true. CLI exit status follows
`policy_passed`, so a static-only gate can intentionally succeed when its policy only requires
`static`.

`safe_to_ship` is intentionally stricter. It is true only when the policy passed **and** at least
semantic verification actually ran. Therefore a static-only report can never claim
`safe_to_ship: true`.

Possible verdicts are:

- `STATIC CHECKS PASS` — static policy passed, but semantic verification did not run.
- `SAFE TO SHIP` — semantic-or-stronger coverage passed the requested policy.
- `UNSAFE TO SHIP` — a check failed or required coverage was not reached.

## Migrating from v1

Schema v1 treated `safe_to_ship` as effectively "no check failed." That made static-only validation
sound stronger than the evidence justified. Consumers should migrate to v2 as follows:

- use `policy_passed` to decide CI success under the configured coverage policy;
- use `safe_to_ship` only when a semantic release-safety claim is required;
- inspect `verification_level` and `required_level` when presenting evidence to humans.

This semantic change is why the report schema moved from v1 to v2 instead of adding fields to v1.

## Prompt evidence

Semantic checks include per-prompt evidence such as prompt hashes, logit drift, top-1 agreement and
the first divergent prediction. Runtime checks include exact-match/first-token agreement, latency,
and local/served completion hashes.

Raw prompts and generated outputs remain excluded unless `--include-prompts` is explicitly enabled.
`prompt_sha256` is included so failures can still be correlated with the source prompt set.

## Fingerprint semantics

- `sampled` reads artifact identity metadata plus the first/last 64 KiB of local files.
- `full` streams every byte of each local file and is preferred for release-grade evidence.
- `reference` means a source did not resolve locally; its SHA-256 identifies the model reference
  string, not the remote model contents.
- `off` disables artifact fingerprints.

Consumers that require byte-level supply-chain evidence should require `full` fingerprints for
local artifacts and pin/download remote model revisions before verification.
