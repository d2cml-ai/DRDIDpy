# Automated PyPI releases

Merging a version update into `main` starts `Estimator tests`. After that workflow
succeeds, `Publish to PyPI` builds a wheel and source distribution from the exact
tested commit and publishes them to the existing `drdid` PyPI project. Merges
without a version bump safely skip files already on PyPI. A release version must
still be chosen and reviewed in `drdid/_version_.py`; the publisher does not invent
versions or update package names.

## One-time configuration

The `drdid` project owner must add this GitHub Actions Trusted Publisher in
[PyPI's publishing settings](https://pypi.org/manage/project/drdid/settings/publishing/):

| Field | Value |
|---|---|
| Owner | `d2cml-ai` |
| Repository | `DRDIDpy` |
| Workflow filename | `pypi_upload.yml` |
| Environment | `pypi` |

Create the matching `pypi` GitHub environment and restrict deployment branches to
`main`. Leave required reviewers and wait timers disabled for unattended releases.
Keep the four estimator checks required for merging. This workflow grants OIDC
permission only to its publish job; building runs in a separate job without that
permission. No PyPI username, password or long-lived API token is used. The old
`PYPI_USER` and `PYPI_PASSWORD` secrets can be removed once Trusted Publishing is
confirmed working.

See [PyPI's setup instructions](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
and the [PyPA publishing action](https://github.com/pypa/gh-action-pypi-publish).

## Release checks and retries

Both automatic publication and a manual **Run workflow** on `main` require all
four successful GitHub Actions checks from the latest `Estimator tests` main-push
workflow for the same commit: Python 3.8, Python 3.12, CSDID integration, and current
R DRDID parity. A pull request result, fork, unrelated workflow, incomplete rerun,
or older main revision cannot authorize publication. The gate checks repository
IDs, workflow identity, check-suite identity, source app and exact commit.

The publisher checks again immediately before upload, after the build or any
environment delay. If main has advanced or a newer test attempt is unfinished,
it skips publication. The new main test completion starts another attempt.
`Run workflow` on `main` provides a safe retry after fixing PyPI configuration or
a transient upload failure; it cannot bypass CI. Rerun `Estimator tests` first if
the latest test attempt failed or was cancelled.

Build artifacts belong to the current workflow run and tested SHA, and are retained
for three days. A manual retry rebuilds them from that SHA. Publication is serialized
across all main revisions. Before skipping an existing file, the publisher checks
its PyPI digest and compares its installable package source, build configuration
(for source distributions), version, and dependency metadata with the new artifact.
Different source or a yanked artifact under the same version fails with an
instruction to bump the version. Matching existing files are skipped, allowing a
partially completed upload to resume. Archive timestamps and documentation changes
do not require a release.

After upload, success also requires both public distribution files to match the
tested build and both filenames, URLs and SHA256 digests to appear in PyPI's pip
index. A cached 404 or temporarily missing file is retried up to six times, with
5, 10, 20, 40 and 60 seconds between attempts and a 30-second timeout per request.
Visible source, metadata, digest or yank mismatches fail immediately. Exhausting
the visibility retries fails the run; a successful upload alone is not reported
as a verified release. The complete publish job has a 20-minute timeout.

PyPI uploads the wheel and source distribution separately. Verification cannot
roll back a partial upload, prevent a concurrent installer from observing it, or
replace an immutable file. A matching partial release can be retried; different
published source requires a new version.

Offline verification of the release gate runs with the ordinary package tests:

```sh
python -m pytest tests/test_release_gate.py -q
```
