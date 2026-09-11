from pathlib import Path

import pytest

from adapterguard.release_validation import (
    expected_tag,
    project_version,
    validate_tag,
    validate_wheel,
    wheel_version,
)


def _write_pyproject(path: Path, version: str) -> None:
    path.write_text(
        '[project]\nname = "adapterguard"\nversion = "' + version + '"\n',
        encoding="utf-8",
    )


def test_project_version_and_expected_tag(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    _write_pyproject(pyproject, "0.5.0")

    assert project_version(pyproject) == "0.5.0"
    assert expected_tag("0.5.0") == "v0.5.0"


def test_validate_tag_accepts_exact_package_version(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    _write_pyproject(pyproject, "0.5.0")

    assert validate_tag("v0.5.0", pyproject=pyproject) == "0.5.0"


def test_validate_tag_rejects_mismatch(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    _write_pyproject(pyproject, "0.5.0")

    with pytest.raises(ValueError, match="does not match package version"):
        validate_tag("v0.4.1", pyproject=pyproject)


def test_wheel_version_parses_adapterguard_wheel():
    assert wheel_version("adapterguard-0.5.0-py3-none-any.whl") == "0.5.0"


def test_wheel_version_rejects_unexpected_filename():
    with pytest.raises(ValueError, match="unexpected AdapterGuard wheel filename"):
        wheel_version("other-0.5.0-py3-none-any.whl")


def test_validate_wheel_accepts_one_exact_match(tmp_path):
    wheel = tmp_path / "adapterguard-0.5.0-py3-none-any.whl"
    wheel.touch()

    assert validate_wheel("v0.5.0", dist_dir=tmp_path) == wheel


def test_validate_wheel_rejects_wrong_version(tmp_path):
    (tmp_path / "adapterguard-0.4.1-py3-none-any.whl").touch()

    with pytest.raises(ValueError, match="does not match release version"):
        validate_wheel("v0.5.0", dist_dir=tmp_path)


def test_validate_wheel_rejects_multiple_wheels(tmp_path):
    (tmp_path / "adapterguard-0.5.0-py3-none-any.whl").touch()
    (tmp_path / "adapterguard-0.5.1-py3-none-any.whl").touch()

    with pytest.raises(ValueError, match="expected exactly one wheel"):
        validate_wheel("v0.5.0", dist_dir=tmp_path)
