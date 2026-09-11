# Releasing AdapterGuard

AdapterGuard publishes Python distributions from a GitHub release through PyPI Trusted Publishing.
No long-lived PyPI API token is required.

## Release invariants

A public release must satisfy all of these conditions:

1. the version in `pyproject.toml` and `src/adapterguard/__init__.py` is the intended release version;
2. the Git tag is exactly `v<package-version>`;
3. normal CI is green on the release commit;
4. HF Semantic Smoke is green on the release commit;
5. the release workflow independently reruns Ruff, pytest, all four HF proofs, distribution build,
   `twine check`, and wheel-version validation;
6. only the verified distributions produced by that workflow are passed to the PyPI publishing job.

The publish job uses GitHub OIDC with `id-token: write` and the `pypi` GitHub environment. The
workflow does not require or read a `PYPI_TOKEN` secret.

## One-time PyPI setup

Configure a GitHub Trusted Publisher on PyPI with these exact identity values:

```text
PyPI project: adapterguard
GitHub owner: ptrgiang
GitHub repository: adapterguard
Workflow filename: publish-pypi.yml
GitHub environment: pypi
```

If the `adapterguard` project does not exist on PyPI yet, configure it as a pending Trusted
Publisher. PyPI can create a new project on the first successful OIDC publication. A pending
publisher does not reserve the project name before that first publication.

The repository workflow already follows PyPI's recommended OIDC shape: the publishing job uses a
dedicated `pypi` environment, grants `id-token: write` only to that job, and uses
`pypa/gh-action-pypi-publish@release/v1`.

For stronger release control, configure required reviewers on the GitHub `pypi` environment. That
keeps package publication behind an explicit maintainer approval even after the automated release
verification job succeeds.

## v0.5.0 release procedure

Before tagging:

```text
package version: 0.5.0
release tag: v0.5.0
```

Confirm the final release commit contains:

- report schema v2 / verification-level semantics;
- chat-completions verification;
- native multi-turn `messages` support;
- tokenizer/chat-template integrity checks;
- real localhost runtime integration tests;
- four real HF semantic proofs;
- CPU-only HF smoke optimization;
- this guarded PyPI release workflow.

Then create the GitHub release from `main`:

```text
Tag: v0.5.0
Target: main
Title: AdapterGuard v0.5.0 — Input Semantics & Release Policy
Pre-release: no
Latest release: yes
```

Do not manually upload wheel or source-distribution files to PyPI. Publishing the GitHub release
triggers `.github/workflows/publish-pypi.yml`; that workflow checks out the immutable tag, verifies
it, builds fresh distributions, and publishes only those verified artifacts.

## Manual recovery

The workflow also supports `workflow_dispatch` with a required tag. Use it only to retry publication
of an existing immutable release tag, for example `v0.5.0`. It is not a way to publish arbitrary
untagged `main` state: the workflow checks out the supplied tag and rejects it when the tag does not
match the package version.

If PyPI rejects Trusted Publishing, first compare the configured publisher identity with the values
above, especially the repository owner/name, workflow filename, and environment name. A mismatch in
any of those claims prevents the OIDC identity from matching the trusted publisher.

## After publication

Verify that:

- PyPI shows version `0.5.0` for `adapterguard`;
- `pip install adapterguard==0.5.0` resolves from a clean environment;
- `adapterguard --version` reports `0.5.0`;
- the GitHub release is marked latest;
- README installation examples can safely move from source installation to `pip install`.

Only after those checks should the README advertise PyPI as the default installation path.
