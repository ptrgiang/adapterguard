# Publishing AdapterGuard to PyPI

AdapterGuard publishes Python distributions through GitHub Actions and PyPI Trusted Publishing. No long-lived PyPI API token is stored in GitHub.

## Release model

A Python package release is published from an immutable Git tag. The workflow validates that the tag matches the version in `pyproject.toml` before building.

For example:

```text
pyproject.toml version = 0.4.1
Git tag                = v0.4.1
```

The workflow builds both a wheel and source distribution, runs `twine check`, uploads the build as a short-lived GitHub Actions artifact, then publishes through OpenID Connect to PyPI.

## One-time PyPI Trusted Publisher setup

Create or claim the `adapterguard` project on PyPI through Trusted Publishing, then configure these values:

- PyPI project: `adapterguard`
- GitHub owner: `ptrgiang`
- GitHub repository: `adapterguard`
- Workflow filename: `publish-pypi.yml`
- Environment name: `pypi`

The GitHub workflow requests only `id-token: write` in the publish job. Do not create a long-lived PyPI API token for this workflow.

## GitHub environment

Create a GitHub Actions environment named exactly:

```text
pypi
```

For a public repository, deployment protection rules are optional. If maintainers want a manual approval gate before package publication, add required reviewers to this environment.

## Publishing the existing v0.4.1 tag

The `v0.4.1` GitHub Release predates the PyPI workflow, so its `release.published` event will not be replayed automatically.

After Trusted Publishing is configured:

1. Open **Actions** in GitHub.
2. Select **Publish to PyPI**.
3. Choose **Run workflow**.
4. Keep branch `main` selected.
5. Enter `v0.4.1` in the `tag` input.
6. Run the workflow.

The workflow checks out the immutable `v0.4.1` tag, validates `v0.4.1 == project.version 0.4.1`, builds the package, and publishes that exact tagged source.

## Future releases

For subsequent releases:

1. Update the version in `pyproject.toml` and `src/adapterguard/__init__.py`.
2. Ensure CI is green.
3. Create and publish a GitHub Release with matching tag, for example `v0.5.0`.
4. The `Publish to PyPI` workflow starts automatically from the `release.published` event.

Never move an already published release tag. If a package needs a fix, bump the version and publish a new release.

## Verification after publish

Confirm all of the following:

```bash
python -m pip install --upgrade adapterguard
adapterguard version
```

The CLI version should match the PyPI release and the GitHub tag.
