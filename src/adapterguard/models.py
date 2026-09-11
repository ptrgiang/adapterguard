from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(slots=True)
class CheckResult:
    name: str
    status: Status
    message: str
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass(slots=True)
class VerificationReport:
    checks: list[CheckResult]

    @property
    def safe_to_ship(self) -> bool:
        return not any(check.status == Status.FAIL for check in self.checks)

    @property
    def verdict(self) -> str:
        return "SAFE TO SHIP" if self.safe_to_ship else "UNSAFE TO SHIP"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "safe_to_ship": self.safe_to_ship,
            "checks": [check.to_dict() for check in self.checks],
        }
