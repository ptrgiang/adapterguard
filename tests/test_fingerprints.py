from adapterguard.fingerprints import fingerprint_artifact


def test_full_fingerprint_changes_when_same_size_content_changes(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    weight = artifact / "weights.bin"
    weight.write_bytes(b"AAAA")
    before = fingerprint_artifact("adapter", artifact, mode="full")

    weight.write_bytes(b"BBBB")
    after = fingerprint_artifact("adapter", artifact, mode="full")

    assert before is not None
    assert after is not None
    assert before.sha256 != after.sha256
    assert after.file_count == 1
    assert after.total_bytes == 4


def test_remote_reference_gets_reference_fingerprint():
    result = fingerprint_artifact("base", "org/model", mode="sampled")
    assert result is not None
    assert result.mode == "reference"
    assert len(result.sha256) == 64


def test_fingerprint_can_be_disabled(tmp_path):
    assert fingerprint_artifact("adapter", tmp_path, mode="off") is None
