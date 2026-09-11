from adapterguard.hf_semantic import (
    _quantization_metadata,
    _summarize_evidence,
    _within_drift_budget,
)
from adapterguard.models import PromptEvidence


def _evidence(index: int, *, mean: float, divergent: bool) -> PromptEvidence:
    return PromptEvidence(
        prompt_index=index,
        prompt_sha256=f"hash-{index}",
        mean_abs_logit_diff=mean,
        max_abs_logit_diff=mean * 2,
        top1_token_agreement=0.8 if divergent else 1.0,
        first_divergent_position=2 if divergent else None,
    )


def test_quantized_drift_budget_requires_both_limits():
    good = {"mean_abs_logit_diff": 0.02, "top1_token_agreement": 0.99}
    too_different = {"mean_abs_logit_diff": 0.06, "top1_token_agreement": 0.99}
    too_many_token_changes = {"mean_abs_logit_diff": 0.02, "top1_token_agreement": 0.95}

    assert _within_drift_budget(
        good,
        max_mean_abs_logit_diff=0.05,
        min_top1_token_agreement=0.98,
    )
    assert not _within_drift_budget(
        too_different,
        max_mean_abs_logit_diff=0.05,
        min_top1_token_agreement=0.98,
    )
    assert not _within_drift_budget(
        too_many_token_changes,
        max_mean_abs_logit_diff=0.05,
        min_top1_token_agreement=0.98,
    )


def test_quantization_summary_localizes_worst_prompt():
    summary = _summarize_evidence(
        [
            _evidence(1, mean=0.01, divergent=False),
            _evidence(2, mean=0.08, divergent=True),
            _evidence(3, mean=0.03, divergent=True),
        ]
    )

    assert summary["prompt_count"] == 3
    assert summary["divergent_prompt_count"] == 2
    assert summary["worst_prompt_index"] == 2
    assert summary["worst_prompt_mean_abs_logit_diff"] == 0.08


def test_quantization_metadata_understands_common_config_shapes():
    class Config:
        quantization_config = {
            "quant_method": "gptq",
            "bits": 4,
        }

    class Model:
        config = Config()

    metadata = _quantization_metadata(Model())
    assert metadata == {
        "quantization_declared": True,
        "quantization_method": "gptq",
        "quantization_bits": 4,
    }


def test_quantization_metadata_handles_undeclared_artifact():
    class Config:
        pass

    class Model:
        config = Config()

    assert _quantization_metadata(Model()) == {"quantization_declared": False}
