from adapterguard.models import (
    ArtifactFingerprint,
    CheckResult,
    PromptEvidence,
    Status,
    VerificationLevel,
    VerificationReport,
)


def test_report_fails_when_any_check_fails():
    report = VerificationReport(
        [
            CheckResult("a", Status.PASS, "ok"),
            CheckResult("b", Status.FAIL, "bad"),
        ]
    )

    assert report.policy_passed is False
    assert report.safe_to_ship is False
    assert report.verdict == "UNSAFE TO SHIP"


def test_static_report_allows_warnings_without_claiming_safe_to_ship():
    report = VerificationReport(
        [
            CheckResult("a", Status.PASS, "ok"),
            CheckResult("b", Status.WARN, "heads up"),
        ]
    )

    assert report.policy_passed is True
    assert report.safe_to_ship is False
    assert report.verdict == "STATIC CHECKS PASS"


def test_semantic_report_can_be_safe_to_ship():
    report = VerificationReport(
        [CheckResult("semantic", Status.PASS, "equivalent")],
        verification_level=VerificationLevel.SEMANTIC,
        required_level=VerificationLevel.SEMANTIC,
    )

    assert report.policy_passed is True
    assert report.safe_to_ship is True
    assert report.verdict == "SAFE TO SHIP"


def test_report_fails_policy_when_coverage_is_too_weak():
    report = VerificationReport(
        [CheckResult("static", Status.PASS, "ok")],
        verification_level=VerificationLevel.STATIC,
        required_level=VerificationLevel.SEMANTIC,
    )

    assert report.checks_passed is True
    assert report.coverage_satisfied is False
    assert report.policy_passed is False
    assert report.safe_to_ship is False
    assert report.verdict == "UNSAFE TO SHIP"


def test_report_serializes_fingerprints_and_prompt_evidence():
    evidence = PromptEvidence(
        prompt_index=1,
        prompt_sha256="abc",
        mean_abs_logit_diff=0.1,
        max_abs_logit_diff=0.2,
        top1_token_agreement=0.5,
        first_divergent_position=3,
        left_token_id=10,
        right_token_id=11,
        left_token="A",
        right_token="B",
    )
    report = VerificationReport(
        [CheckResult("compare", Status.FAIL, "diverged", evidence=[evidence])],
        fingerprints={
            "adapter": ArtifactFingerprint(
                label="adapter",
                source="./adapter",
                mode="sampled",
                sha256="deadbeef",
                file_count=2,
                total_bytes=42,
            )
        },
    )

    payload = report.to_dict()
    assert payload["schema_version"] == "adapterguard.report.v2"
    assert payload["verification_level"] == "static"
    assert payload["required_level"] == "static"
    assert payload["fingerprints"]["adapter"]["sha256"] == "deadbeef"
    assert payload["checks"][0]["evidence"][0]["first_divergent_position"] == 3
