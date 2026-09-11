from adapterguard.models import CheckResult, Status, VerificationReport


def test_report_fails_when_any_check_fails():
    report = VerificationReport(
        [
            CheckResult("a", Status.PASS, "ok"),
            CheckResult("b", Status.FAIL, "bad"),
        ]
    )

    assert report.safe_to_ship is False
    assert report.verdict == "UNSAFE TO SHIP"


def test_report_allows_warnings():
    report = VerificationReport(
        [
            CheckResult("a", Status.PASS, "ok"),
            CheckResult("b", Status.WARN, "heads up"),
        ]
    )

    assert report.safe_to_ship is True
