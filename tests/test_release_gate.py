"""Release failure contracts. All GitHub responses are offline fixtures."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
import zipfile


SPEC = importlib.util.spec_from_file_location(
    "check_release_ci", Path(__file__).parents[1] / "scripts" / "check_release_ci.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
SHA = "a" * 40
OTHER_SHA = "b" * 40
REPO = "d2cml-ai/DRDIDpy"
IDENTITY = {"full_name": REPO, "id": 123}


class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.run = dict(id=10, name="Estimator tests", path=".github/workflows/tests.yml",
                        event="push", head_branch="main", head_sha=SHA, status="completed",
                        conclusion="success", repository=IDENTITY, head_repository=IDENTITY,
                        check_suite_id=100)
        self.runs = {10: self.run}
        self.suite = dict(id=100, head_sha=SHA, status="completed", conclusion="success",
                          app={"id": 15368, "slug": "github-actions"})
        self.checks = [dict(id=index, name=name, head_sha=SHA, status="completed",
                            conclusion="success", app={"id": 15368, "slug": "github-actions"},
                            check_suite={"id": 100},
                            details_url="https://github.com/%s/actions/runs/10/job/%d" % (REPO, index))
                       for index, name in enumerate(sorted(gate.REQUIRED), 1)]
        self.main_sha = SHA
        self.event = dict(repository=IDENTITY, workflow_run=copy.deepcopy(self.run))
        self.queries = []

    def api(self, path):
        self.queries.append(path)
        if path == "/branches/main":
            return {"commit": {"sha": self.main_sha}}
        if path.startswith("/actions/workflows/tests.yml/runs?"):
            self.assertIn("head_sha=" + SHA, path)
            self.assertIn("event=push&branch=main", path)
            return {"workflow_runs": list(self.runs.values())}
        if path.startswith("/actions/runs/"):
            return self.runs[int(path.rsplit("/", 1)[1])]
        if path == "/check-suites/100":
            return self.suite
        if path.startswith("/check-suites/100/check-runs?"):
            page = int(path.rsplit("=", 1)[1])
            return {"check_runs": self.checks[(page - 1) * 100:page * 100],
                    "total_count": len(self.checks)}
        raise AssertionError("Unexpected API request: " + path)

    def ready(self, **kwargs):
        params = dict(event_name="workflow_run", event=self.event, repository=REPO,
                      repository_id=123, ref="refs/heads/main", sha=OTHER_SHA, api=self.api)
        params.update(kwargs)
        return gate.release_ready(**params)[0]

    def test_main_workflow_uses_its_tested_commit_not_the_default_branch_context(self):
        self.assertTrue(self.ready())
        self.assertIn("/actions/runs/10", self.queries)

    def test_manual_retry_requires_main_and_all_the_same_ci_checks(self):
        manual = dict(repository=IDENTITY, ref="main")
        self.assertTrue(self.ready(event_name="workflow_dispatch", event=manual, sha=SHA))
        for ref, event_ref in [("refs/heads/feature", "feature"), ("refs/tags/v1", "v1"),
                               ("refs/heads/main", "feature")]:
            with self.subTest(ref=ref, event_ref=event_ref), self.assertRaises(gate.GateError):
                self.ready(event_name="workflow_dispatch", event=dict(manual, ref=event_ref), ref=ref, sha=SHA)
        self.checks.pop()
        self.assertFalse(self.ready(event_name="workflow_dispatch", event=manual, sha=SHA))

    def test_unsupported_events_and_malformed_or_changed_release_sha_are_blocked(self):
        for event_name in ("push", "pull_request", "repository_dispatch"):
            with self.subTest(event_name=event_name), self.assertRaises(gate.GateError):
                self.ready(event_name=event_name)
        for sha in ("main", "abc123", None, "g" * 40):
            event = copy.deepcopy(self.event)
            event["workflow_run"]["head_sha"] = sha
            with self.subTest(sha=sha), self.assertRaises(gate.GateError):
                self.ready(event=event)
        with self.assertRaisesRegex(gate.GateError, "changed after building"):
            self.ready(expected_sha=OTHER_SHA)

    def test_untrusted_trigger_or_live_workflow_never_uses_successful_checks(self):
        mutations = [("event", "pull_request"), ("head_branch", "feature"),
                     ("path", ".github/workflows/other.yml"), ("name", "Other tests"),
                     ("repository", {"full_name": REPO, "id": 999}),
                     ("head_repository", {"full_name": "fork/DRDIDpy", "id": 9}),
                     ("head_sha", OTHER_SHA)]
        for field, value in mutations:
            with self.subTest(field=field):
                saved = copy.deepcopy(self.run)
                self.run[field] = value
                with self.assertRaises(gate.GateError):
                    self.ready()
                self.run.clear()
                self.run.update(saved)
        for field, value in [("conclusion", "failure"), ("status", "in_progress"), ("id", 0)]:
            event = copy.deepcopy(self.event)
            event["workflow_run"][field] = value
            with self.subTest(field=field), self.assertRaises(gate.GateError):
                self.ready(event=event)
        with self.assertRaises(gate.GateError):
            self.ready(event=dict(self.event, repository={"full_name": REPO, "id": 999}))

    def test_main_can_be_superseded_before_or_during_verification(self):
        self.main_sha = OTHER_SHA
        self.assertFalse(self.ready())
        self.main_sha = SHA
        original = self.api
        reads = []
        def moved(path):
            if path == "/branches/main":
                reads.append(path)
                return {"commit": {"sha": SHA if len(reads) == 1 else OTHER_SHA}}
            return original(path)
        self.assertFalse(self.ready(api=moved))

    def test_missing_latest_pending_failed_or_cancelled_workflow_blocks_old_success(self):
        for status, conclusion in [("queued", None), ("in_progress", None),
                                   ("completed", "failure"), ("completed", "cancelled"),
                                   ("completed", "skipped"), ("completed", "neutral")]:
            with self.subTest(status=status, conclusion=conclusion):
                self.runs[20] = dict(self.run, id=20, status=status, conclusion=conclusion)
                self.assertFalse(self.ready())
        self.runs.clear()
        self.assertFalse(self.ready(event_name="workflow_dispatch", sha=SHA,
                                    event=dict(repository=IDENTITY, ref="main")))

    def test_rerun_of_the_same_workflow_id_cannot_reuse_old_event_success(self):
        self.run.update(status="queued", conclusion=None, run_attempt=2)
        self.assertFalse(self.ready())

    def test_suite_identity_and_github_actions_app_are_required(self):
        for field, value in [("head_sha", OTHER_SHA), ("id", 999),
                             ("app", {"id": 7, "slug": "github-actions"}),
                             ("app", {"id": 15368, "slug": "other-app"})]:
            with self.subTest(field=field):
                original = copy.deepcopy(self.suite)
                self.suite[field] = value
                with self.assertRaises(gate.GateError):
                    self.ready()
                self.suite = original
        self.suite["conclusion"] = "failure"
        self.assertFalse(self.ready())

    def test_every_required_job_must_have_a_successful_latest_attempt(self):
        for check in self.checks:
            for status, conclusion in [("queued", None), ("completed", "failure"),
                                       ("completed", "skipped"), ("completed", "cancelled")]:
                with self.subTest(name=check["name"], status=status, conclusion=conclusion):
                    newer = dict(check, id=1000, status=status, conclusion=conclusion)
                    self.checks.append(newer)
                    self.assertFalse(self.ready())
                    self.checks.pop()
        missing = self.checks.pop()
        self.assertFalse(self.ready())
        self.checks.append(missing)

    def test_checks_cannot_be_forged_by_another_app_suite_sha_workflow_or_url(self):
        mutations = [("head_sha", OTHER_SHA), ("check_suite", {"id": 999}),
                     ("app", {"id": 7, "slug": "github-actions"}),
                     ("app", {"id": 15368, "slug": "other-app"}),
                     ("details_url", "https://attacker.invalid/actions/runs/10/job/1"),
                     ("details_url", "https://github.com/fork/DRDIDpy/actions/runs/10/job/1"),
                     ("details_url", "https://github.com/%s/actions/runs/999/job/1" % REPO)]
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                saved = copy.deepcopy(self.checks)
                self.checks[0][field] = value
                with self.assertRaises(gate.GateError):
                    self.ready()
                self.checks = saved

    def test_check_pagination_is_complete_and_fails_closed_on_missing_page(self):
        self.checks = [{"name": "unrelated"}] * 100 + self.checks
        self.assertTrue(self.ready())
        self.assertTrue(any("page=2" in query for query in self.queries))
        original = self.api
        def truncated(path):
            if path.startswith("/check-suites/100/check-runs?") and path.endswith("page=2"):
                return {"check_runs": [], "total_count": len(self.checks)}
            return original(path)
        with self.assertRaisesRegex(gate.GateError, "Incomplete check-run pagination"):
            self.ready(api=truncated)

    def test_cli_outputs_cannot_claim_ready_when_main_is_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event_path, output, summary = [root / name for name in ("event.json", "output", "summary")]
            event_path.write_text(json.dumps(self.event))
            env = dict(GITHUB_REPOSITORY=REPO, GITHUB_REPOSITORY_ID="123", GH_TOKEN="unused",
                       GITHUB_EVENT_NAME="workflow_run", GITHUB_EVENT_PATH=str(event_path),
                       GITHUB_REF="refs/heads/main", GITHUB_SHA=OTHER_SHA,
                       GITHUB_OUTPUT=str(output), GITHUB_STEP_SUMMARY=str(summary))
            self.main_sha = OTHER_SHA
            with patch.dict(os.environ, env, clear=True), patch.object(gate, "urlopen") as open_url:
                open_url.return_value.__enter__.return_value.read.side_effect = [
                    json.dumps(self.run).encode(), json.dumps({"commit": {"sha": OTHER_SHA}}).encode()]
                gate.main([])
            self.assertEqual(output.read_text(), "ready=false\nsha=%s\n" % SHA)
            self.assertIn("no longer the main head", summary.read_text())


class ExistingReleaseTests(unittest.TestCase):
    WHEEL = "drdid-1.1.7-py3-none-any.whl"
    SDIST = "drdid-1.1.7.tar.gz"

    def wheel(self, source=b"reviewed\n", requirements="numpy", name="drdid", version="1.1.7",
              pure="true", tags=("py3-none-any",), generator="build-tool"):
        result = io.BytesIO()
        with zipfile.ZipFile(result, "w") as archive:
            archive.writestr("drdid/__init__.py", source)
            archive.writestr("drdid-1.1.7.dist-info/METADATA",
                             "Name: %s\nVersion: %s\nRequires-Dist: %s\n" % (name, version, requirements))
            archive.writestr("drdid-1.1.7.dist-info/WHEEL",
                             "Wheel-Version: 1.0\nGenerator: %s\nRoot-Is-Purelib: %s\n%s" %
                             (generator, pure, "".join("Tag: " + tag + "\n" for tag in tags)))
        return result.getvalue()

    def sdist(self, source=b"reviewed\n", setup=b"reviewed build\n", docs=b"original docs\n", extra_members=()):
        result = io.BytesIO()
        with tarfile.open(fileobj=result, mode="w:gz") as archive:
            files = {"drdid/__init__.py": source, "setup.py": setup, "README.md": docs,
                     "PKG-INFO": b"Name: drdid\nVersion: 1.1.7\nRequires-Dist: numpy\n"}
            for name, content in files.items():
                member = tarfile.TarInfo("drdid-1.1.7/" + name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
            for member in extra_members:
                archive.addfile(member, io.BytesIO(b"") if member.isfile() else None)
        return result.getvalue()

    def preflight(self, existing, candidate=None, mutation=None):
        candidate = candidate or {self.WHEEL: self.wheel(), self.SDIST: self.sdist()}
        urls = []
        content_by_url = {}
        for name, content in existing.items():
            url = "https://files.pythonhosted.org/packages/" + name
            content_by_url[url] = content
            urls.append(dict(filename=name, url=url, digests={"sha256": hashlib.sha256(content).hexdigest()}))
        release = {"info": {"version": "1.1.7"}, "urls": urls}
        if mutation:
            mutation(release)
        def fetch(url):
            if url == "https://pypi.org/pypi/drdid/1.1.7/json":
                return json.dumps(release).encode()
            return content_by_url[url]
        with tempfile.TemporaryDirectory() as directory:
            for name, content in candidate.items():
                (Path(directory) / name).write_bytes(content)
            with patch.object(gate, "fetch_bytes", side_effect=fetch):
                return gate.check_pypi_artifacts(directory)

    def publication(self, states, delays=(5, 10, 20, 40, 60)):
        candidate = {self.WHEEL: self.wheel(), self.SDIST: self.sdist()}
        self.publication_attempts, self.publication_delays = 0, []
        state, urls, content_by_url = {}, [], {}

        def fetch(url, accept=None):
            nonlocal state, urls, content_by_url
            if url == "https://pypi.org/pypi/drdid/1.1.7/json":
                state = states[min(self.publication_attempts, len(states) - 1)]
                self.publication_attempts += 1
                if state.get("version404"):
                    raise HTTPError(url, 404, "Not visible", {}, None)
                urls, content_by_url = [], {}
                for name, content in state.get("files", candidate).items():
                    file_url = "https://files.pythonhosted.org/packages/" + name
                    content_by_url[file_url] = content
                    urls.append(dict(filename=name, url=file_url,
                                     digests={"sha256": hashlib.sha256(content).hexdigest()}))
                return json.dumps({"info": {"version": "1.1.7"}, "urls": urls}).encode()
            if url == "https://pypi.org/simple/drdid/":
                self.assertEqual(accept, "application/vnd.pypi.simple.v1+json")
                if state.get("simple404"):
                    raise HTTPError(url, 404, "Not visible", {}, None)
                files = [dict(filename=item["filename"], url=item["url"], hashes=item["digests"])
                         for item in urls if item["filename"] not in state.get("simple_missing", [])]
                if state.get("simple_bad_digest"):
                    files[0]["hashes"] = {"sha256": "0" * 64}
                if state.get("simple_yanked"):
                    files[0]["yanked"] = True
                if state.get("simple_bad_url"):
                    files[0]["url"] = "https://attacker.invalid/file.whl"
                return json.dumps({"name": "drdid", "files": files}).encode()
            if url.rsplit("/", 1)[-1] in state.get("download404", []):
                raise HTTPError(url, 404, "Not visible", {}, None)
            return content_by_url[url]

        with tempfile.TemporaryDirectory() as directory:
            for name, content in candidate.items():
                (Path(directory) / name).write_bytes(content)
            with patch.object(gate, "fetch_bytes", side_effect=fetch), \
                    patch.object(gate.time, "sleep", side_effect=self.publication_delays.append):
                return gate.verify_publication(directory, delays=delays)

    def test_post_upload_cached_404_must_become_a_complete_matching_release(self):
        message = self.publication([{"version404": True}, {}])
        self.assertIn("Confirmed both drdid 1.1.7 distributions", message)
        self.assertEqual(self.publication_attempts, 2)
        self.assertEqual(self.publication_delays, [5])

    def test_post_upload_partial_artifacts_and_partial_pip_index_wait_for_both_files(self):
        message = self.publication([{"files": {self.WHEEL: self.wheel()}},
                                    {"simple_missing": [self.SDIST]}, {}])
        self.assertIn("available from public PyPI", message)
        self.assertEqual(self.publication_attempts, 3)
        self.assertEqual(self.publication_delays, [5, 10])

    def test_post_upload_missing_download_or_simple_index_can_propagate(self):
        for missing in ({"download404": [self.SDIST]}, {"simple404": True}):
            with self.subTest(missing=missing):
                self.assertIn("Confirmed both", self.publication([missing, {}]))
                self.assertEqual(self.publication_attempts, 2)

    def test_post_upload_stale_404_cannot_hide_old_source_accepted_by_skip_existing(self):
        with self.assertRaisesRegex(gate.GateError, "bump the version"):
            self.publication([{"version404": True},
                              {"files": {self.WHEEL: self.wheel(source=b"old broken behavior\n")}}])
        self.assertEqual(self.publication_attempts, 2)
        self.assertEqual(self.publication_delays, [5])

    def test_post_upload_visible_simple_digest_url_or_yank_mismatch_fails_without_retry(self):
        for state in ({"simple_bad_digest": True}, {"simple_yanked": True}, {"simple_bad_url": True}):
            with self.subTest(state=state), self.assertRaises(gate.GateError):
                self.publication([state, {}])
            self.assertEqual(self.publication_attempts, 1)
            self.assertEqual(self.publication_delays, [])

    def test_post_upload_missing_release_or_partial_index_exhausts_a_bounded_retry_budget(self):
        for pending in ({"version404": True}, {"files": {}}, {"simple_missing": [self.SDIST]}):
            with self.subTest(pending=pending), self.assertRaisesRegex(gate.GateError, "after 6 attempts"):
                self.publication([pending])
            self.assertEqual(self.publication_attempts, 6)
            self.assertEqual(self.publication_delays, [5, 10, 20, 40, 60])

    def test_existing_complete_or_partial_release_can_be_safely_retried(self):
        wheel, sdist = self.wheel(), self.sdist()
        for existing in ({self.WHEEL: wheel}, {self.SDIST: sdist}, {self.WHEEL: wheel, self.SDIST: sdist}):
            with self.subTest(files=sorted(existing)):
                self.assertIn("identical files may be skipped", self.preflight(existing))

    def test_same_version_different_runtime_build_config_or_dependencies_is_blocked(self):
        for existing in ({self.WHEEL: self.wheel(source=b"old broken behavior\n")},
                         {self.WHEEL: self.wheel(requirements="numpy<2")},
                         {self.SDIST: self.sdist(source=b"old broken behavior\n")},
                         {self.SDIST: self.sdist(setup=b"old build\n")}):
            with self.subTest(files=sorted(existing)), self.assertRaisesRegex(gate.GateError, "bump the version"):
                self.preflight(existing)

    def test_documentation_only_change_does_not_require_a_new_release(self):
        self.assertIn("identical files may be skipped",
                      self.preflight({self.SDIST: self.sdist(docs=b"older docs\n")}))

    def test_source_archive_links_and_other_nonregular_members_are_rejected(self):
        for member_type in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE,
                            tarfile.CHRTYPE, tarfile.BLKTYPE):
            for name in ("drdid-1.1.7/extra", "drdid-1.1.7/drdid/__init__.py"):
                member = tarfile.TarInfo(name)
                member.type = member_type
                member.linkname = "drdid/__init__.py"
                with self.subTest(member_type=member_type, name=name), self.assertRaises(gate.GateError):
                    self.preflight({self.SDIST: self.sdist(extra_members=[member])})

    def test_source_archive_all_paths_are_unique_below_one_safe_root(self):
        bad_names = ("/drdid-1.1.7/absolute", "drdid-1.1.7/../escape", "drdid-1.1.7/drdid/../extra",
                     "drdid-1.1.7/./extra", "drdid-1.1.7//extra", "drdid-1.1.7\\extra",
                     "other-root/extra", "drdid-1.1.7/drdid/__init__.py")
        for name in bad_names:
            member = tarfile.TarInfo(name)
            with self.subTest(name=name), self.assertRaises(gate.GateError):
                self.preflight({self.SDIST: self.sdist(extra_members=[member])})
        first = tarfile.TarInfo("drdid-1.1.7")
        first.type = tarfile.DIRTYPE
        duplicate = tarfile.TarInfo("drdid-1.1.7/")
        duplicate.type = tarfile.DIRTYPE
        with self.assertRaisesRegex(gate.GateError, "Duplicate"):
            self.preflight({self.SDIST: self.sdist(extra_members=[first, duplicate])})
        second_root = tarfile.TarInfo("other-root")
        second_root.type = tarfile.DIRTYPE
        with self.assertRaisesRegex(gate.GateError, "one source archive root"):
            self.preflight({self.SDIST: self.sdist(extra_members=[second_root])})
        self.assertIn("identical files", self.preflight({self.SDIST: self.sdist(extra_members=[first])}))

    def test_wheel_installation_semantics_must_match_but_generator_can_change(self):
        for wheel in (self.wheel(pure="false"), self.wheel(tags=("cp312-none-any",))):
            with self.subTest(), self.assertRaisesRegex(gate.GateError, "bump the version"):
                self.preflight({self.WHEEL: wheel})
        self.assertIn("identical files", self.preflight({self.WHEEL: self.wheel(generator="other-tool")}))

    def test_yanked_files_require_a_new_version_even_if_source_matches(self):
        for mutation in (lambda release: release["urls"][0].update(yanked=True),
                         lambda release: release["info"].update(yanked=True)):
            with self.subTest(), self.assertRaisesRegex(gate.GateError, "yanked.*bump the version"):
                self.preflight({self.WHEEL: self.wheel()}, mutation=mutation)

    def test_digest_url_release_version_and_unknown_artifacts_fail_closed(self):
        mutations = [lambda release: release["urls"][0]["digests"].update(sha256="0" * 64),
                     lambda release: release["urls"][0].update(url="https://attacker.invalid/file.whl"),
                     lambda release: release["info"].update(version="1.1.6"),
                     lambda release: release["urls"][0].update(filename="unexpected.whl"),
                     lambda release: release.update(urls=[])]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(gate.GateError):
                self.preflight({self.WHEEL: self.wheel()}, mutation=mutation)

    def test_candidate_package_identity_and_distribution_versions_must_agree(self):
        for wheel in (self.wheel(name="csdid"), self.wheel(version="1.1.8")):
            with self.subTest(), self.assertRaises(gate.GateError):
                self.preflight({}, candidate={self.WHEEL: wheel, self.SDIST: self.sdist()})

    def test_only_a_missing_pypi_version_allows_a_new_release(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / self.WHEEL).write_bytes(self.wheel())
            (Path(directory) / self.SDIST).write_bytes(self.sdist())
            for status in (404, 403, 500):
                error = HTTPError("url", status, "response", {}, None)
                with self.subTest(status=status), patch.object(gate, "fetch_bytes", side_effect=error):
                    if status == 404:
                        self.assertIn("is new on PyPI", gate.check_pypi_artifacts(directory))
                    else:
                        with self.assertRaises(HTTPError):
                            gate.check_pypi_artifacts(directory)


if __name__ == "__main__":
    unittest.main()
