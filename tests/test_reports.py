import json

from adapterguard.models import CheckResult, PromptEvidence, Status, VerificationReport
from adapterguard.reports import render_markdown, write_json_report, write_markdown_report


def _report():
    evidence = PromptEvidence(
        prompt_index=2,
        prompt_sha256="abc123",
        mean_abs_logit_diff=0.04,
        max_abs_logit_diff=0.9,
        top1_token_agreement=0.75,
        first_divergent_position=4,
        left_token_id=7,
        right_token_id=9,
        left_token=" cat",
        right_token=" dog",
        prompt="hello world",
    )
    return VerificationReport(
        [CheckResult("export", Status.FAIL, "diverged", evidence=[evidence])]
    )


def test_markdown_contains_first_divergent_token_evidence():
    markdown = render_markdown(_report())
    assert "First divergent position" in markdown
    assert "` cat` → ` dog`" in markdown
    assert "Prompt #2" in markdown


def test_report_writers_create_parent_directories(tmp_path):
    report = _report()
    json_path = write_json_report(report, tmp_path / "reports" / "report.json")
    md_path = write_markdown_report(report, tmp_path / "reports" / "report.md")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["safe_to_ship"] is False
    assert "UNSAFE TO SHIP" in md_path.read_text(encoding="utf-8")
