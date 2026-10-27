"""Read-only PyPI release gate; uses only the Python standard library."""
import argparse
import email.parser
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile


ACTION_APP_ID = 15368
WORKFLOW = "Estimator tests"
WORKFLOW_PATH = ".github/workflows/tests.yml"
REQUIRED = {
    "Estimators (Python 3.8)",
    "Estimators (Python 3.12)",
    "CSDID integration",
    "Current R DRDID parity",
}


class GateError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise GateError(message)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def same_repo(value, repository, repository_id):
    return (isinstance(value, dict) and value.get("full_name", "").lower() == repository.lower()
            and str(value.get("id")) == str(repository_id))


def trusted_run(run, repository, repository_id, sha):
    require(same_repo(run.get("repository"), repository, repository_id)
            and same_repo(run.get("head_repository"), repository, repository_id),
            "The test workflow must come from this repository")
    require(run.get("name") == WORKFLOW and run.get("path", "").split("@", 1)[0] == WORKFLOW_PATH,
            "Unexpected test workflow")
    require(run.get("event") == "push" and run.get("head_branch") == "main",
            "The test workflow must be a main push")
    require(run.get("head_sha") == sha, "The test workflow belongs to another commit")


def candidate(event_name, event, repository, repository_id, ref, sha):
    require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository),
            "Invalid repository identity")
    require(same_repo(event.get("repository"), repository, repository_id),
            "The event belongs to another repository")
    if event_name == "workflow_dispatch":
        require(ref == "refs/heads/main" and event.get("ref") in ("main", "refs/heads/main"),
                "Manual publication is allowed only on main")
    elif event_name == "workflow_run":
        run = event.get("workflow_run", {})
        sha = run.get("head_sha")
        trusted_run(run, repository, repository_id, sha)
        require(run.get("status") == "completed" and run.get("conclusion") == "success",
                "The triggering test workflow has not succeeded")
        require(type(run.get("id")) is int and run["id"] > 0, "Invalid triggering workflow ID")
    else:
        raise GateError("Unsupported publication event")
    require(valid_sha(sha), "A complete tested commit SHA is required")
    return sha


def ci_ready(repository, repository_id, sha, api):
    """Require the newest main-push workflow and all four exact-SHA checks."""
    require(valid_sha(sha), "A complete tested commit SHA is required")
    if api("/branches/main")["commit"]["sha"] != sha:
        return False, "Skipped: the tested commit is no longer the main head."

    # Check the workflow itself, including queued reruns with no check runs yet.
    # Filtering by SHA avoids accidentally using a success from a different commit.
    payload = api("/actions/workflows/tests.yml/runs?head_sha=%s&event=push&branch=main&per_page=100" % sha)
    runs = payload["workflow_runs"]
    if not runs:
        return False, "Skipped: no main-push estimator workflow exists for the tested commit."
    latest = max(runs, key=lambda run: run["id"])
    run = api("/actions/runs/" + str(latest["id"]))
    trusted_run(run, repository, repository_id, sha)
    require(run.get("id") == latest["id"], "Unexpected workflow run ID")
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        return False, "Skipped: the latest main-push estimator workflow has not succeeded."

    suite_id = run["check_suite_id"]
    suite = api("/check-suites/" + str(suite_id))
    require(suite.get("id") == suite_id and suite.get("head_sha") == sha,
            "The test suite belongs to another commit or workflow")
    require(suite.get("app", {}).get("id") == ACTION_APP_ID
            and suite.get("app", {}).get("slug") == "github-actions", "Untrusted test suite app")
    if suite.get("status") != "completed" or suite.get("conclusion") != "success":
        return False, "Skipped: the trusted test suite has not succeeded."

    latest_checks = {}
    count = 0
    for page in range(1, 101):
        payload = api("/check-suites/%s/check-runs?filter=all&per_page=100&page=%d" % (suite_id, page))
        batch = payload["check_runs"]
        count += len(batch)
        for check in batch:
            name = check.get("name")
            if name not in REQUIRED:
                continue
            require(check.get("head_sha") == sha and check.get("check_suite", {}).get("id") == suite_id,
                    "A required check belongs to another commit or suite")
            require(check.get("app", {}).get("id") == ACTION_APP_ID
                    and check.get("app", {}).get("slug") == "github-actions", "Untrusted check app")
            details = urlparse(check.get("details_url", ""))
            expected_path = (r"/" + re.escape(repository) + r"/actions/runs/"
                             + str(run["id"]) + r"/job/[0-9]+")
            require(details.scheme == "https" and details.netloc == "github.com"
                    and re.fullmatch(expected_path, details.path, flags=re.IGNORECASE),
                    "Cannot verify the workflow source of " + name)
            if name not in latest_checks or check["id"] > latest_checks[name]["id"]:
                latest_checks[name] = check
        if count >= payload["total_count"]:
            break
        require(batch, "Incomplete check-run pagination")
    else:
        raise GateError("Too many check runs to establish release readiness")

    missing = sorted(REQUIRED - set(latest_checks))
    if missing:
        return False, "Skipped: required checks are missing: " + ", ".join(missing)
    blocked = sorted(name for name, check in latest_checks.items()
                     if check.get("status") != "completed" or check.get("conclusion") != "success")
    if blocked:
        return False, "Skipped: required checks have not all succeeded: " + ", ".join(blocked)
    if api("/branches/main")["commit"]["sha"] != sha:
        return False, "Skipped: main advanced while verifying the tested commit."
    return True, "All four trusted main-push GitHub Actions checks succeeded for " + sha


def release_ready(event_name, event, repository, repository_id, ref, sha, api, expected_sha=None):
    sha = candidate(event_name, event, repository, repository_id, ref, sha)
    require(expected_sha is None or sha == expected_sha, "The release revision changed after building")
    if event_name == "workflow_run":
        # Re-fetch the triggering run rather than trusting an old event payload.
        run = api("/actions/runs/" + str(event["workflow_run"]["id"]))
        require(run.get("id") == event["workflow_run"]["id"], "Unexpected triggering workflow ID")
        trusted_run(run, repository, repository_id, sha)
    ready, message = ci_ready(repository, repository_id, sha, api)
    return ready, sha, message


def artifact_identity(content, filename):
    """Compare installable source and metadata, ignoring archive timestamps/docs."""
    wheel_identity = {}
    if filename.endswith(".whl"):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), "Duplicate wheel entries")
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
            require(len(metadata_names) == 1, "Expected one wheel metadata file")
            metadata = archive.read(metadata_names[0])
            wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
            require(len(wheel_names) == 1
                    and wheel_names[0].rsplit("/", 1)[0] == metadata_names[0].rsplit("/", 1)[0],
                    "Expected one matching WHEEL metadata file")
            wheel = email.parser.BytesParser().parsebytes(archive.read(wheel_names[0]))
            wheel_identity = {"Wheel-" + name: sorted(wheel.get_all(name, []))
                              for name in ("Root-Is-Purelib", "Tag")}
            require(wheel_identity["Wheel-Root-Is-Purelib"] in (["true"], ["false"])
                    and wheel_identity["Wheel-Tag"], "Missing wheel installation metadata")
            files = {name: archive.read(name) for name in names
                     if not name.endswith("/") and ".dist-info/" not in name}
            entrypoints = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
            require(len(entrypoints) <= 1, "Duplicate entry point metadata")
            if entrypoints:
                files["entry_points.txt"] = archive.read(entrypoints[0])
    elif filename.endswith(".tar.gz"):
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
            members = archive.getmembers()
            names, roots = [], set()
            # Validate every member before filtering or reading. Links, devices,
            # duplicate paths and traversal entries must never be silently ignored.
            for member in members:
                require(member.isfile() or member.isdir(), "Unsupported source archive member type")
                name = member.name[:-1] if member.isdir() and member.name.endswith("/") else member.name
                parts = name.split("/")
                require("\\" not in name and all(part not in ("", ".", "..") for part in parts),
                        "Unsafe source archive path")
                require(member.isdir() or len(parts) > 1, "Source files must be below the archive root")
                names.append(name)
                roots.add(parts[0])
            require(len(names) == len(set(names)), "Duplicate source archive entries")
            require(len(roots) == 1, "Expected one source archive root")
            files = {}
            metadata = None
            for member in members:
                if member.isdir():
                    continue
                name = member.name.split("/", 1)[-1]
                if name == "PKG-INFO":
                    metadata = archive.extractfile(member).read()
                elif name.startswith("drdid/") or name in ("setup.py", "setup.cfg", "pyproject.toml"):
                    files[name] = archive.extractfile(member).read()
            require(metadata is not None, "Missing source archive metadata")
    else:
        raise GateError("Unexpected distribution format: " + filename)
    require(any(name.startswith("drdid/") for name in files), "Missing drdid package source")
    parsed = email.parser.BytesParser().parsebytes(metadata)
    identity = {name: sorted(parsed.get_all(name, [])) for name in
                ("Name", "Version", "Requires-Python", "Requires-Dist", "Provides-Extra")}
    identity.update(wheel_identity)
    require(identity["Name"] == ["drdid"] and len(identity["Version"]) == 1,
            "Unexpected package identity")
    return identity, files


def fetch_bytes(url):
    with urlopen(Request(url, headers={"User-Agent": "drdid-release-check"}), timeout=30) as response:
        return response.read()


def check_pypi_artifacts(directory):
    paths = sorted(Path(directory).iterdir())
    require(len(paths) == 2 and sum(path.name.endswith(".whl") for path in paths) == 1
            and sum(path.name.endswith(".tar.gz") for path in paths) == 1,
            "Expected exactly one wheel and one source distribution")
    candidate_files = {path.name: artifact_identity(path.read_bytes(), path.name) for path in paths}
    versions = {identity[0]["Version"][0] for identity in candidate_files.values()}
    require(len(versions) == 1, "Distribution versions disagree")
    version = versions.pop()
    require(re.fullmatch(r"[0-9A-Za-z!+_.-]+", version), "Invalid distribution version")
    try:
        release = json.loads(fetch_bytes("https://pypi.org/pypi/drdid/%s/json" % version))
    except HTTPError as error:
        if error.code == 404:
            return "drdid %s is new on PyPI; publication may proceed." % version
        raise
    require(release.get("info", {}).get("version") == version, "Wrong PyPI release metadata")
    require(not release.get("info", {}).get("yanked"),
            "drdid %s is yanked on PyPI; bump the version" % version)
    published = release.get("urls", [])
    require(published, "The existing PyPI release has no verifiable files")
    seen = set()
    for artifact in published:
        require(not artifact.get("yanked"),
                "drdid %s contains a yanked PyPI artifact; bump the version" % version)
        name = artifact.get("filename")
        require(name in candidate_files and name not in seen, "Unexpected existing PyPI artifact")
        seen.add(name)
        url = urlparse(artifact.get("url", ""))
        require(url.scheme == "https" and url.netloc == "files.pythonhosted.org",
                "Unexpected PyPI artifact host")
        content = fetch_bytes(artifact["url"])
        require(hashlib.sha256(content).hexdigest() == artifact.get("digests", {}).get("sha256"),
                "Published artifact does not match its PyPI digest")
        require(artifact_identity(content, name) == candidate_files[name],
                "drdid %s already contains different source or metadata; bump the version" % version)
    return "Verified %d existing drdid %s artifact(s); identical files may be skipped." % (len(seen), version)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", help="Compare built distributions with any existing PyPI release")
    args = parser.parse_args(argv)
    if args.artifacts:
        try:
            print(check_pypi_artifacts(args.artifacts))
        except GateError as error:
            raise SystemExit("Release blocked: " + str(error))
        return
    repository = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GH_TOKEN"]

    def api(path):
        request = Request("https://api.github.com/repos/" + repository + path,
                          headers={"Authorization": "Bearer " + token,
                                   "Accept": "application/vnd.github+json",
                                   "X-GitHub-Api-Version": "2022-11-28"})
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    try:
        ready, sha, message = release_ready(
            os.environ["GITHUB_EVENT_NAME"], json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()),
            repository, os.environ["GITHUB_REPOSITORY_ID"], os.environ["GITHUB_REF"],
            os.environ["GITHUB_SHA"], api, os.environ.get("RELEASE_SHA"))
    except GateError as error:
        raise SystemExit("Release blocked: " + str(error))
    print(message)
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
        stream.write(message + "\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        stream.write("ready=%s\nsha=%s\n" % (str(ready).lower(), sha))


if __name__ == "__main__":
    main()
