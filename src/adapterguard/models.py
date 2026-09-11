from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(slots=True)
class PromptEvidence:
    prompt_index: int
    prompt_sha256: str
    mean_abs_logit_diff: float
    max_abs_logit_diff: float
    top1_token_agreement: float
    first_divergent_position: int | None = None
    left_token_id: int | None = None
    right_token_id: int | None = None
    left_token: str | None = None
    right_token: str | None = None
    prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ArtifactFingerprint:
    label: str
    source: str
    mode: str
    sha256: str
    file_count: int | None = None
    total_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CheckResult:
    name: str
    status: Status
    message: str
    metrics: dict[str, Any] | None = None
    evidence: list[PromptEvidence] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass(slots=True)
class VerificationReport:
    checks: list[CheckResult]
    fingerprints: dict[str, ArtifactFingerprint] = field(default_factory=dict)
    schema_version: str = "adapterguard.report.v1"

    @property
    def safe_to_ship(self) -> bool:
        return not any(check.status == Status.FAIL for check in self.checks)

    @property
    def verdict(self) -> str:
        return "SAFE TO SHIP" if self.safe_to_ship else "UNSAFE TO SHIP"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "verdict": self.verdict,
            "safe_to_ship": self.safe_to_ship,
            "fingerprints": {
                label: fingerprint.to_dict() for label, fingerprint in self.fingerprints.items()
            },
            "checks": [check.to_dict() for check in self.checks],
        }
