from adapterguard.models import RuntimePromptEvidence, Status
from adapterguard.runtime_openai import (
    RuntimeCompletionResponse,
    _completions_url,
    run_openai_runtime_check,
)


def test_completions_url_accepts_base_v1_and_full_path():
    assert _completions_url("http://localhost:8000") == "http://localhost:8000/v1/completions"
    assert _completions_url("http://localhost:8000/v1") == "http://localhost:8000/v1/completions"
    assert (
        _completions_url("http://localhost:8000/v1/completions")
        == "http://localhost:8000/v1/completions"
    )


def test_runtime_check_passes_when_greedy_outputs_match():
    def requester(**kwargs):
        text = {"one": " alpha", "two": " beta"}[kwargs["prompt"]]
        return RuntimeCompletionResponse(
            text=text,
            tokens=[text],
            served_model="served/model",
            latency_ms=10.0,
        )

    result = run_openai_runtime_check(
        endpoint="http://runtime:8000/v1",
        model="served/model",
        prompts=["one", "two"],
        local_completions=[" alpha", " beta"],
        local_first_tokens=[" alpha", " beta"],
        requester=requester,
    )

    assert result.status == Status.PASS
    assert result.metrics["first_token_agreement"] == 1.0
    assert result.metrics["exact_completion_match_rate"] == 1.0
    assert result.metrics["served_models"] == ["served/model"]
    assert all(isinstance(item, RuntimePromptEvidence) for item in result.evidence)


def test_runtime_check_localizes_divergence_and_keeps_text_private_by_default():
    def requester(**kwargs):
        return RuntimeCompletionResponse(
            text=" wrong",
            tokens=[" wrong"],
            served_model="served/model",
            latency_ms=12.0,
        )

    result = run_openai_runtime_check(
        endpoint="http://runtime:8000/v1",
        model="served/model",
        prompts=["secret prompt"],
        local_completions=[" correct"],
        local_first_tokens=[" correct"],
        requester=requester,
    )

    assert result.status == Status.FAIL
    assert result.metrics["first_token_agreement"] == 0.0
    item = result.evidence[0]
    assert item.prompt is None
    assert item.local_completion is None
    assert item.served_completion is None
    assert item.local_completion_sha256 != item.served_completion_sha256


def test_runtime_check_can_include_debug_text_explicitly():
    def requester(**kwargs):
        return RuntimeCompletionResponse(
            text=" served",
            tokens=[" served"],
            served_model=None,
            latency_ms=1.0,
        )

    result = run_openai_runtime_check(
        endpoint="http://runtime:8000",
        model="model",
        prompts=["prompt"],
        local_completions=[" local"],
        local_first_tokens=[" local"],
        include_prompts=True,
        min_first_token_agreement=0.0,
        min_exact_match_rate=0.0,
        requester=requester,
    )

    item = result.evidence[0]
    assert item.prompt == "prompt"
    assert item.local_completion == " local"
    assert item.served_completion == " served"
