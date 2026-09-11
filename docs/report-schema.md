# AdapterGuard evidence report schema

AdapterGuard v0.2 emits report schema `adapterguard.report.v1`.

The JSON contract is intentionally small so CI systems can gate deployments without parsing
human-readable console output.

```json
{
  "schema_version": "adapterguard.report.v1",
  "verdict": "UNSAFE TO SHIP",
  "safe_to_ship": false,
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
  "checks": [
    {
      "name": "exported model preserves adapter behavior",
      "status": "fail",
      "message": "exported model diverges from the active adapter",
      "metrics": {
        "mean_abs_logit_diff": 0.012,
        "max_abs_logit_diff": 1.9,
        "top1_token_agreement": 0.91
      },
      "evidence": [
        {
          "prompt_index": 1,
          "prompt_sha256": "...",
          "mean_abs_logit_diff": 0.014,
          "max_abs_logit_diff": 1.9,
          "top1_token_agreement": 0.88,
          "first_divergent_position": 7,
          "left_token_id": 123,
          "right_token_id": 456,
          "left_token": " blue",
          "right_token": " red",
          "prompt": null
        }
      ]
    }
  ]
}
```

## Stability

Fields under `adapterguard.report.v1` are additive: new optional fields may appear, while existing
field meanings will not intentionally change within the v1 schema. A breaking report change will
use a new schema version.

## Prompt privacy

`prompt_sha256` is always included so a failing prompt can be correlated with the source JSONL
without exposing its text. `prompt` is `null` unless the verifier runs with `--include-prompts`.

## Fingerprint semantics

- `sampled` is optimized for large local artifacts and reads the first/last 64 KiB of each file.
- `full` streams every local file byte.
- `reference` means the source did not resolve to a local path; the SHA-256 identifies the model
  reference string, not remote model contents.

Consumers that require byte-level supply-chain evidence should require `mode == "full"` for local
artifacts and should pin/download remote model revisions before verification.
