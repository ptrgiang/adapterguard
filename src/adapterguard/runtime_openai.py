from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import CheckResult, RuntimePromptEvidence, Status


class RuntimeEndpointError(RuntimeError):
    pass


@dataclass(slots=True)
class RuntimeCompletionResponse:
    text: str
    tokens: list[str]
    served_model: str | None
    latency_ms: float


def _completions_url(endpoint: str) -> str:
    normalized = endpoint.rstrip("/")
    if normalized.endswith("/completions") and not normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/completions"
    return f"{normalized}/v1/completions"


def _chat_completions_url(endpoint: str) -> str:
    normalized = endpoint.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _post_json(*, url: str, payload: dict[str, object], api_key: str | None, timeout: float):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeEndpointError(
            f"runtime endpoint returned HTTP {exc.code}: {body[:500]}"
        ) from exc
    except URLError as exc:
        raise RuntimeEndpointError(f"runtime endpoint request failed: {exc.reason}") from exc
    return raw, (time.perf_counter() - started) * 1000.0


def _request_completion(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    api_key: str | None,
    max_tokens: int,
    timeout: float,
) -> RuntimeCompletionResponse:
    payload: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": max_tokens,
        "logprobs": 5,
    }
    raw, latency_ms = _post_json(
        url=_completions_url(endpoint), payload=payload, api_key=api_key, timeout=timeout
    )

    try:
        data = json.loads(raw)
        choice = data["choices"][0]
        text = str(choice.get("text") or "")
        logprobs = choice.get("logprobs") or {}
        tokens = [str(token) for token in (logprobs.get("tokens") or [])]
        served_model = data.get("model")
    except (json.JSONDecodeError, KeyError, TypeError, IndexError) as exc:
        raise RuntimeEndpointError(
            "runtime endpoint returned an invalid completion payload"
        ) from exc

    return RuntimeCompletionResponse(
        text=text,
        tokens=tokens,
        served_model=str(served_model) if served_model is not None else None,
        latency_ms=latency_ms,
    )


def _request_chat_completion(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    api_key: str | None,
    max_tokens: int,
    timeout: float,
) -> RuntimeCompletionResponse:
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": 5,
    }
    raw, latency_ms = _post_json(
        url=_chat_completions_url(endpoint), payload=payload, api_key=api_key, timeout=timeout
    )

    try:
        data = json.loads(raw)
        choice = data["choices"][0]
        message = choice.get("message") or {}
        text = str(message.get("content") or "")
        logprobs = choice.get("logprobs") or {}
        content_logprobs = logprobs.get("content") or []
        tokens = [str(item.get("token") or "") for item in content_logprobs if isinstance(item, dict)]
        served_model = data.get("model")
    except (json.JSONDecodeError, KeyError, TypeError, IndexError) as exc:
        raise RuntimeEndpointError(
            "runtime endpoint returned an invalid chat completion payload"
        ) from exc

    return RuntimeCompletionResponse(
        text=text,
        tokens=tokens,
        served_model=str(served_model) if served_model is not None else None,
        latency_ms=latency_ms,
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _run_runtime_check(
    *,
    endpoint: str,
    endpoint_url: str,
    runtime_api: str,
    model: str,
    prompts: list[str],
    local_completions: list[str],
    local_first_tokens: list[str],
    api_key: str | None,
    max_tokens: int,
    timeout: float,
    min_first_token_agreement: float,
    min_exact_match_rate: float,
    include_prompts: bool,
    requester: Callable[..., RuntimeCompletionResponse],
) -> CheckResult:
    if not (len(prompts) == len(local_completions) == len(local_first_tokens)):
        raise ValueError("runtime reference lists must have the same length as prompts")
    if not prompts:
        raise ValueError("runtime verification requires at least one prompt")

    exact_matches = 0
    first_matches = 0
    latencies: list[float] = []
    served_models: set[str] = set()
    evidence: list[RuntimePromptEvidence] = []

    for index, (prompt, local_text, local_first) in enumerate(
        zip(prompts, local_completions, local_first_tokens, strict=True),
        start=1,
    ):
        response = requester(
            endpoint=endpoint,
            model=model,
            prompt=prompt,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        served_first = response.tokens[0] if response.tokens else response.text[: len(local_first)]
        exact_match = response.text == local_text
        first_match = served_first == local_first
        exact_matches += int(exact_match)
        first_matches += int(first_match)
        latencies.append(response.latency_ms)
        if response.served_model:
            served_models.add(response.served_model)

        evidence.append(
            RuntimePromptEvidence(
                prompt_index=index,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                exact_match=exact_match,
                first_token_match=first_match,
                latency_ms=response.latency_ms,
                local_completion_sha256=hashlib.sha256(local_text.encode()).hexdigest(),
                served_completion_sha256=hashlib.sha256(response.text.encode()).hexdigest(),
                local_first_token=local_first if include_prompts else None,
                served_first_token=served_first if include_prompts else None,
                prompt=prompt if include_prompts else None,
                local_completion=local_text if include_prompts else None,
                served_completion=response.text if include_prompts else None,
            )
        )

    prompt_count = len(prompts)
    first_rate = first_matches / prompt_count
    exact_rate = exact_matches / prompt_count
    metrics = {
        "runtime_api": runtime_api,
        "endpoint": endpoint_url,
        "requested_model": model,
        "served_models": sorted(served_models),
        "prompt_count": prompt_count,
        "first_token_agreement": first_rate,
        "exact_completion_match_rate": exact_rate,
        "mean_latency_ms": sum(latencies) / len(latencies),
        "p50_latency_ms": median(latencies),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "max_latency_ms": max(latencies),
        "runtime_max_tokens": max_tokens,
        "budget_min_first_token_agreement": min_first_token_agreement,
        "budget_min_exact_match_rate": min_exact_match_rate,
    }
    ok = first_rate >= min_first_token_agreement and exact_rate >= min_exact_match_rate
    return CheckResult(
        "served runtime preserves verified artifact behavior",
        Status.PASS if ok else Status.FAIL,
        "OpenAI-compatible runtime stays within the configured behavior budget"
        if ok
        else "OpenAI-compatible runtime diverges from the verified local artifact",
        metrics,
        evidence,
    )


def run_openai_runtime_check(
    *,
    endpoint: str,
    model: str,
    prompts: list[str],
    local_completions: list[str],
    local_first_tokens: list[str],
    api_key: str | None = None,
    max_tokens: int = 8,
    timeout: float = 30.0,
    min_first_token_agreement: float = 1.0,
    min_exact_match_rate: float = 1.0,
    include_prompts: bool = False,
    requester: Callable[..., RuntimeCompletionResponse] | None = None,
) -> CheckResult:
    return _run_runtime_check(
        endpoint=endpoint,
        endpoint_url=_completions_url(endpoint),
        runtime_api="openai-completions",
        model=model,
        prompts=prompts,
        local_completions=local_completions,
        local_first_tokens=local_first_tokens,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=timeout,
        min_first_token_agreement=min_first_token_agreement,
        min_exact_match_rate=min_exact_match_rate,
        include_prompts=include_prompts,
        requester=requester or _request_completion,
    )


def run_openai_chat_runtime_check(
    *,
    endpoint: str,
    model: str,
    prompts: list[str],
    local_completions: list[str],
    local_first_tokens: list[str],
    api_key: str | None = None,
    max_tokens: int = 8,
    timeout: float = 30.0,
    min_first_token_agreement: float = 1.0,
    min_exact_match_rate: float = 1.0,
    include_prompts: bool = False,
    requester: Callable[..., RuntimeCompletionResponse] | None = None,
) -> CheckResult:
    return _run_runtime_check(
        endpoint=endpoint,
        endpoint_url=_chat_completions_url(endpoint),
        runtime_api="openai-chat-completions",
        model=model,
        prompts=prompts,
        local_completions=local_completions,
        local_first_tokens=local_first_tokens,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=timeout,
        min_first_token_agreement=min_first_token_agreement,
        min_exact_match_rate=min_exact_match_rate,
        include_prompts=include_prompts,
        requester=requester or _request_chat_completion,
    )
