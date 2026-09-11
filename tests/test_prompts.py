import json

import pytest

from adapterguard.prompts import PromptCase, load_prompt_cases, prompt_case_from_value


def test_prompt_case_accepts_multi_turn_messages():
    case = prompt_case_from_value(
        {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Summarize our exchange."},
            ]
        }
    )

    assert case.is_chat is True
    assert case.as_messages()[0] == {"role": "system", "content": "Be concise."}
    assert case.as_messages()[-1]["content"] == "Summarize our exchange."
    assert '"messages"' in case.evidence_text


def test_prompt_case_rejects_multimodal_content_for_now():
    with pytest.raises(ValueError, match="multimodal content is not yet supported"):
        prompt_case_from_value(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                ]
            }
        )


def test_load_prompt_cases_keeps_text_compatibility_and_messages(tmp_path):
    path = tmp_path / "prompts.jsonl"
    rows = [
        "plain prompt",
        {"prompt": "object prompt"},
        {
            "messages": [
                {"role": "user", "content": "first turn"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second turn"},
            ]
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    cases = load_prompt_cases(path, max_prompts=10)

    assert cases == [
        PromptCase(prompt="plain prompt"),
        PromptCase(prompt="object prompt"),
        PromptCase(
            messages=(
                {"role": "user", "content": "first turn"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second turn"},
            )
        ),
    ]
