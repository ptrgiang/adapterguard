from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib


def project_version(pyproject: str | Path = "pyproject.toml") -> str:
    with Path(pyproject).open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def expected_tag(version: str) -> str:
    return f"v{version}"


def validate_tag(tag: str, *, pyproject: str | Path = "pyproject.toml") -> str:
    version = project_version(pyproject)
    expected = expected_tag(version)
    if tag != expected:
        raise ValueError(
            f"release tag {tag!r} does not match package version {expected!r}"
        )
    return version


def wheel_version(path: str | Path) -> str:
    name = Path(path).name
    match = re.fullmatch(r"adapterguard-(.+?)-py3-none-any\.whl", name)
    if match is None:
        raise ValueError(f"unexpected AdapterGuard wheel filename: {name!r}")
    return match.group(1)


def validate_wheel(
    tag: str,
    *,
    dist_dir: str | Path = "dist",
) -> Path:
    expected = tag.removeprefix("v")
    wheels = sorted(Path(dist_dir).glob("adapterguard-*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"expected exactly one wheel, found {len(wheels)}")
    actual = wheel_version(wheels[0])
    if actual != expected:
        raise ValueError(
            f"wheel {wheels[0].name!r} does not match release version {expected!r}"
        )
    return wheels[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate AdapterGuard release invariants")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tag_parser = subparsers.add_parser("tag", help="validate tag against pyproject version")
    tag_parser.add_argument("--tag", required=True)
    tag_parser.add_argument("--pyproject", default="pyproject.toml")

    wheel_parser = subparsers.add_parser("wheel", help="validate built wheel against release tag")
    wheel_parser.add_argument("--tag", required=True)
    wheel_parser.add_argument("--dist-dir", default="dist")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "tag":
            version = validate_tag(args.tag, pyproject=args.pyproject)
            print(f"Verified release tag {args.tag} for AdapterGuard {version}")
        else:
            wheel = validate_wheel(args.tag, dist_dir=args.dist_dir)
            print(f"Verified release wheel {wheel.name}")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
