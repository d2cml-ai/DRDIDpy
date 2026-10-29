"""Separately reviewed automation cannot hide any other upstream drift."""
import copy
import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "check_r_source", Path(__file__).parents[1] / "scripts/check_r_source.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
PATH = ".github/workflows/python-downstream.yml"
SHA = "b" * 64


class AutomationReviewTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"r_revision": "a" * 40,
                         "r_source_hashes": {"R/kernel.R": "c" * 64,
                                             "tests/test-kernel.R": "d" * 64}}
        self.new = dict(self.manifest["r_source_hashes"], **{PATH: SHA})
        self.modes = {PATH: "100644"}
        self.review = {"baseline_revision": self.manifest["r_revision"],
                       "added_workflows": {PATH: {"sha256": SHA, "mode": "100644"}}}

    def result(self, review=True):
        return gate.compare_sources(self.manifest, self.new, self.modes,
                                    self.review if review else None)

    def test_unchanged_numerical_baseline_passes_before_workflow_is_added(self):
        del self.new[PATH]
        report, status = self.result()
        self.assertEqual(status, 0)
        self.assertEqual(report["changed_source_or_test_files"], [])
        self.assertEqual(report["reviewed_automation_files"], [])

    def test_exact_added_workflow_is_reported_but_needs_no_second_review(self):
        report, status = self.result()
        self.assertEqual(status, 0)
        self.assertEqual(report["changed_source_or_test_files"], [PATH])
        self.assertEqual(report["reviewed_automation_files"], [PATH])
        self.assertEqual(report["unreviewed_source_or_test_files"], [])

    def test_missing_record_cannot_accept_an_added_workflow(self):
        report, status = self.result(review=False)
        self.assertEqual(status, 3)
        self.assertEqual(report["unreviewed_source_or_test_files"], [PATH])
        self.review["added_workflows"] = {}
        self.assertEqual(self.result()[1], 3)

    def test_baseline_or_workflow_hash_mismatch_requires_review(self):
        self.review["baseline_revision"] = "e" * 40
        report, status = self.result()
        self.assertEqual(status, 3)
        self.assertIn("baseline", report["automation_review_error"])
        self.review["baseline_revision"] = self.manifest["r_revision"]
        self.new[PATH] = "f" * 64
        self.assertEqual(self.result()[1], 3)

    def test_symlink_or_changed_executable_mode_cannot_match_regular_file(self):
        for mode in ("120000", "100755", "160000"):
            with self.subTest(mode=mode):
                self.modes[PATH] = mode
                self.assertEqual(self.result()[1], 3)

    def test_runtime_tests_docs_existing_workflows_and_removals_still_fail(self):
        for path in ("R/kernel.R", "tests/test-kernel.R", "README.md",
                     ".github/workflows/existing.yml"):
            with self.subTest(path=path):
                new = dict(self.new, **{path: "e" * 64})
                report, status = gate.compare_sources(self.manifest, new, self.modes, self.review)
                self.assertEqual(status, 3)
                self.assertEqual(report["reviewed_automation_files"], [PATH])
                self.assertIn(path, report["unreviewed_source_or_test_files"])
        del self.new["R/kernel.R"]
        self.assertEqual(self.result()[1], 3)

    def test_record_cannot_approve_a_change_to_a_file_in_the_numerical_baseline(self):
        self.manifest["r_source_hashes"][PATH] = "e" * 64
        report, status = self.result()
        self.assertEqual(status, 3)
        self.assertEqual(report["reviewed_automation_files"], [])

    def test_baseline_mode_only_change_still_requires_review(self):
        baseline_modes = {name: "100644" for name in self.manifest["r_source_hashes"]}
        modes = dict(baseline_modes, **self.modes)
        modes["R/kernel.R"] = "100755"
        report, status = gate.compare_sources(self.manifest, self.new, modes,
                                              self.review, baseline_modes)
        self.assertEqual(status, 3)
        self.assertIn("R/kernel.R", report["changed_mode_files"])
        self.assertEqual(report["reviewed_automation_files"], [PATH])
        self.assertEqual(report["unreviewed_source_or_test_files"], ["R/kernel.R"])

    def test_malformed_or_overbroad_records_fail_closed(self):
        variants = [[], {}, dict(self.review, unexpected=True)]
        for path in ("R/kernel.R", ".github/workflows/../kernel.yml", "/tmp/workflow.yml"):
            variants.append({"baseline_revision": self.manifest["r_revision"],
                             "added_workflows": {path: {"sha256": SHA, "mode": "100644"}}})
        for field, value in (("mode", "120000"), ("mode", "regular"),
                             ("sha256", "invalid"), ("sha256", None)):
            variant = copy.deepcopy(self.review)
            variant["added_workflows"][PATH][field] = value
            variants.append(variant)
        for review in variants:
            with self.subTest(review=review):
                report, status = gate.compare_sources(self.manifest, self.new, self.modes, review)
                self.assertEqual(status, 3)
                self.assertEqual(report["reviewed_automation_files"], [])


if __name__ == "__main__":
    unittest.main()
