import pytest

from adapterguard.models import RuntimePromptEvidence, Status
from adapterguard.prompts import PromptCase
from adapterguard.runtime_openai import (
    RuntimeCompletionResponse,
    _chat_completions_url,
    _completions_url,
    run_openai_chat_runtime_check,
    run_openai_runtime_check,
)


def test_completions_url_accepts_base_v1_and_full_path():
    assert _completions_url("http://localhost:8000") == "http://localhost:8000/v1/completions"
    assert _completions_url("http://localhost:8000/v1") == "http://localhost:8000/v1/completions"
    assert (
        _completions_url("http://localhost:8000/v1/completions")
        == "http://localhost:8000/v1/completions"
    )


def test_chat_completions_url_accepts_base_v1_and_full_path():
    assert (
        _chat_completions_url("http://localhost:8000")
        == "http://localhost:8000/v1/chat/completions"
    )
    assert (
        _chat_completions_url("http://localhost:8000/v1")
        == "http://localhost:8000/v1/chat/completions"
    )
    assert (
        _chat_completions_url("http://localhost:8000/v1/chat/completions")
        == "http://localhost:8000/v1/chat/completions"
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
    assert result.metrics["runtime_api"] == "openai-completions"
    assert result.metrics["first_token_agreement"] == 1.0
    assert result.metrics["exact_completion_match_rate"] == 1.0
    assert result.metrics["served_models"] == ["served/model"]
    assert result.metrics["multi_turn_prompt_count"] == 0
    assert all(isinstance(item, RuntimePromptEvidence) for item in result.evidence)


def test_text_runtime_rejects_messages_prompt_cases():
    case = PromptCase(messages=({"role": "user", "content": "hello"},))

    with pytest.raises(ValueError, match="chat-completions endpoint"):
        run_openai_runtime_check(
            endpoint="http://runtime:8000/v1",
            model="served/model",
            prompts=[case],
            local_completions=[" answer"],
            local_first_tokens=[" answer"],
        )


def test_chat_runtime_check_uses_chat_api_metrics():
    def requester(**kwargs):
        return RuntimeCompletionResponse(
            text=" answer",
            tokens=[" answer"],
            served_model="served/chat-model",
            latency_ms=4.0,
        )

    result = run_openai_chat_runtime_check(
        endpoint="http://runtime:8000/v1/chat/completions",
        model="served/chat-model",
        prompts=["question"],
        local_completions=[" answer"],
        local_first_tokens=[" answer"],
        requester=requester,
    )

    assert result.status == Status.PASS
    assert result.metrics["runtime_api"] == "openai-chat-completions"
    assert result.metrics["endpoint"] == "http://runtime:8000/v1/chat/completions"
    assert result.metrics["exact_completion_match_rate"] == 1.0


def test_chat_runtime_preserves_multi_turn_case_for_requester_and_evidence():
    case = PromptCase(
        messages=(
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "What did I say first?"},
        )
    )
    captured = {}

    def requester(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return RuntimeCompletionResponse(
            text=" Hello",
            tokens=[" Hello"],
            served_model="served/chat-model",
            latency_ms=2.0,
        )

    result = run_openai_chat_runtime_check(
        endpoint="http://runtime:8000/v1/chat/completions",
        model="served/chat-model",
        prompts=[case],
        local_completions=[" Hello"],
        local_first_tokens=[" Hello"],
        include_prompts=True,
        requester=requester,
    )

    assert captured["prompt"] == case
    assert result.status == Status.PASS
    assert result.metrics["multi_turn_prompt_count"] == 1
    assert '"messages"' in result.evidence[0].prompt


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
