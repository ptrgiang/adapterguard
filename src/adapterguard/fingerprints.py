from __future__ import annotations

import hashlib
from pathlib import Path

from .models import ArtifactFingerprint

_SAMPLE_BYTES = 64 * 1024
_SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sampled_sha256(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode())
    with path.open("rb") as handle:
        digest.update(handle.read(_SAMPLE_BYTES))
        if size > _SAMPLE_BYTES:
            handle.seek(max(0, size - _SAMPLE_BYTES))
            digest.update(handle.read(_SAMPLE_BYTES))
    return digest.hexdigest()


def _iter_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not any(part in _SKIP_PARTS for part in path.relative_to(root).parts)
    )


def fingerprint_artifact(
    label: str,
    source: str | Path,
    *,
    mode: str = "sampled",
) -> ArtifactFingerprint | None:
    if mode == "off":
        return None
    if mode not in {"sampled", "full"}:
        raise ValueError("fingerprint mode must be one of: sampled, full, off")

    source_text = str(source)
    path = Path(source_text).expanduser()
    if not path.exists():
        return ArtifactFingerprint(
            label=label,
            source=source_text,
            mode="reference",
            sha256=hashlib.sha256(source_text.encode()).hexdigest(),
        )

    files = [path] if path.is_file() else _iter_files(path)
    digest = hashlib.sha256()
    total_bytes = 0
    hasher = _stream_sha256 if mode == "full" else _sampled_sha256

    for file_path in files:
        relative = file_path.name if path.is_file() else file_path.relative_to(path).as_posix()
        size = file_path.stat().st_size
        total_bytes += size
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        digest.update(hasher(file_path).encode())
        digest.update(b"\n")

    return ArtifactFingerprint(
        label=label,
        source=source_text,
        mode=mode,
        sha256=digest.hexdigest(),
        file_count=len(files),
        total_bytes=total_bytes,
    )
