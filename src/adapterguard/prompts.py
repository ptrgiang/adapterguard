from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PromptCase:
    """One golden input, represented as raw text or an OpenAI-style chat history."""

    prompt: str | None = None
    messages: tuple[dict[str, str], ...] | None = None

    def __post_init__(self) -> None:
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("PromptCase requires exactly one of prompt or messages")
        if self.messages is not None and not self.messages:
            raise ValueError("messages must contain at least one message")

    @property
    def is_chat(self) -> bool:
        return self.messages is not None

    @property
    def evidence_text(self) -> str:
        if self.prompt is not None:
            return self.prompt
        return json.dumps(
            {"messages": list(self.messages or ())},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def as_messages(self) -> list[dict[str, str]]:
        if self.messages is not None:
            return [dict(message) for message in self.messages]
        return [{"role": "user", "content": self.prompt or ""}]


def _parse_messages(
    value: Any,
    *,
    line_number: int | None = None,
) -> tuple[dict[str, str], ...]:
    location = f" at line {line_number}" if line_number is not None else ""
    if not isinstance(value, list) or not value:
        raise ValueError(f"`messages`{location} must be a non-empty JSON array")

    messages: list[dict[str, str]] = []
    for index, message in enumerate(value, start=1):
        if not isinstance(message, dict):
            raise ValueError(f"message {index}{location} must be a JSON object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"message {index}{location} must have a non-empty string `role`")
        if not isinstance(content, str):
            raise ValueError(
                f"message {index}{location} must have string `content`; "
                "multimodal content is not yet supported"
            )
        messages.append({"role": role, "content": content})
    return tuple(messages)


def prompt_case_from_value(
    value: Any,
    *,
    line_number: int | None = None,
) -> PromptCase:
    location = f" at line {line_number}" if line_number is not None else ""
    if isinstance(value, PromptCase):
        return value
    if isinstance(value, str):
        return PromptCase(prompt=value)
    if isinstance(value, dict):
        has_prompt = "prompt" in value
        has_messages = "messages" in value
        if has_prompt and has_messages:
            raise ValueError(f"prompt entry{location} cannot contain both `prompt` and `messages`")
        if has_prompt:
            prompt = value.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError(f"`prompt`{location} must be a string")
            return PromptCase(prompt=prompt)
        if has_messages:
            return PromptCase(
                messages=_parse_messages(
                    value.get("messages"),
                    line_number=line_number,
                )
            )
    raise ValueError(
        f"prompt entry{location} must be a JSON string, an object with string `prompt`, "
        "or an object with a non-empty `messages` array"
    )


def coerce_prompt_cases(values: Iterable[PromptCase | str]) -> list[PromptCase]:
    return [prompt_case_from_value(value) for value in values]


def load_prompt_cases(path: str | Path, max_prompts: int) -> list[PromptCase]:
    prompts: list[PromptCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc

            prompts.append(prompt_case_from_value(value, line_number=line_number))
            if len(prompts) >= max_prompts:
                break

    if not prompts:
        raise ValueError("Prompt file contains no usable prompts")
    return prompts


def render_prompt_case(
    tokenizer,
    prompt: PromptCase | str,
    *,
    force_chat: bool = False,
) -> str:
    case = prompt_case_from_value(prompt)
    if not case.is_chat and not force_chat:
        return case.prompt or ""
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer does not define a chat template")
    return tokenizer.apply_chat_template(
        case.as_messages(),
        tokenize=False,
        add_generation_prompt=True,
    )
