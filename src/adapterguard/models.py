from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class VerificationLevel(str, Enum):
    STATIC = "static"
    SEMANTIC = "semantic"
    QUANTIZATION = "quantization"
    RUNTIME = "runtime"

    @property
    def rank(self) -> int:
        return {
            VerificationLevel.STATIC: 0,
            VerificationLevel.SEMANTIC: 1,
            VerificationLevel.QUANTIZATION: 2,
            VerificationLevel.RUNTIME: 3,
        }[self]

    def satisfies(self, required: VerificationLevel) -> bool:
        return self.rank >= required.rank


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
class RuntimePromptEvidence:
    prompt_index: int
    prompt_sha256: str
    exact_match: bool
    first_token_match: bool
    latency_ms: float
    local_completion_sha256: str
    served_completion_sha256: str
    local_first_token: str | None = None
    served_first_token: str | None = None
    prompt: str | None = None
    local_completion: str | None = None
    served_completion: str | None = None

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


EvidenceItem = PromptEvidence | RuntimePromptEvidence


@dataclass(slots=True)
class CheckResult:
    name: str
    status: Status
    message: str
    metrics: dict[str, Any] | None = None
    evidence: list[EvidenceItem] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass(slots=True)
class VerificationReport:
    checks: list[CheckResult]
    fingerprints: dict[str, ArtifactFingerprint] = field(default_factory=dict)
    verification_level: VerificationLevel = VerificationLevel.STATIC
    required_level: VerificationLevel = VerificationLevel.STATIC
    schema_version: str = "adapterguard.report.v2"

    @property
    def checks_passed(self) -> bool:
        return not any(check.status == Status.FAIL for check in self.checks)

    @property
    def coverage_satisfied(self) -> bool:
        return self.verification_level.satisfies(self.required_level)

    @property
    def policy_passed(self) -> bool:
        return self.checks_passed and self.coverage_satisfied

    @property
    def safe_to_ship(self) -> bool:
        return self.policy_passed and self.verification_level.satisfies(
            VerificationLevel.SEMANTIC
        )

    @property
    def verdict(self) -> str:
        if not self.policy_passed:
            return "UNSAFE TO SHIP"
        if self.verification_level == VerificationLevel.STATIC:
            return "STATIC CHECKS PASS"
        return "SAFE TO SHIP"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "verdict": self.verdict,
            "safe_to_ship": self.safe_to_ship,
            "policy_passed": self.policy_passed,
            "checks_passed": self.checks_passed,
            "coverage_satisfied": self.coverage_satisfied,
            "verification_level": self.verification_level.value,
            "required_level": self.required_level.value,
            "fingerprints": {
                label: fingerprint.to_dict() for label, fingerprint in self.fingerprints.items()
            },
            "checks": [check.to_dict() for check in self.checks],
        }
