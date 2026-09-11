from adapterguard.hf_semantic import _tokenizer_integrity_check
from adapterguard.models import Status


class FakeTokenizer:
    def __init__(self, *, offset: int = 0, chat_prefix: str = "<user>"):
        self.offset = offset
        self.chat_prefix = chat_prefix
        self.chat_template = "fake-template"

    def __call__(self, text: str, truncation: bool = True):
        del truncation
        return {"input_ids": [ord(char) + self.offset for char in text]}

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"{self.chat_prefix}{messages[0]['content']}<assistant>"


def test_tokenizer_integrity_passes_for_identical_encodings():
    result = _tokenizer_integrity_check(
        base_tokenizer=FakeTokenizer(),
        artifact_tokenizer=FakeTokenizer(),
        prompts=["hello", "world"],
        artifact_label="exported model",
        check_chat_template=False,
    )

    assert result.status == Status.PASS
    assert result.metrics["exact_encoding_match_rate"] == 1.0
    assert result.metrics["mismatched_prompt_count"] == 0


def test_tokenizer_integrity_fails_when_token_ids_drift():
    result = _tokenizer_integrity_check(
        base_tokenizer=FakeTokenizer(),
        artifact_tokenizer=FakeTokenizer(offset=1),
        prompts=["hello", "world"],
        artifact_label="quantized model",
        check_chat_template=False,
    )

    assert result.status == Status.FAIL
    assert result.metrics["exact_encoding_match_rate"] == 0.0
    assert result.metrics["mismatched_prompt_indices"] == [1, 2]


def test_tokenizer_integrity_fails_when_chat_template_drifts():
    result = _tokenizer_integrity_check(
        base_tokenizer=FakeTokenizer(chat_prefix="<user>"),
        artifact_tokenizer=FakeTokenizer(chat_prefix="[USER]"),
        prompts=["hello"],
        artifact_label="exported model",
        check_chat_template=True,
    )

    assert result.status == Status.FAIL
    assert result.metrics["exact_encoding_match_rate"] == 1.0
    assert result.metrics["chat_render_match_rate"] == 0.0
    assert result.metrics["chat_encoding_match_rate"] == 0.0
    assert result.metrics["mismatched_prompt_indices"] == [1]


def test_tokenizer_integrity_fails_when_artifact_has_no_chat_template():
    artifact = FakeTokenizer()
    artifact.chat_template = None

    result = _tokenizer_integrity_check(
        base_tokenizer=FakeTokenizer(),
        artifact_tokenizer=artifact,
        prompts=["hello"],
        artifact_label="exported model",
        check_chat_template=True,
    )

    assert result.status == Status.FAIL
    assert result.metrics["chat_render_match_rate"] == 0.0
    assert result.metrics["chat_encoding_match_rate"] == 0.0
