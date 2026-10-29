"""Require review of every tracked R-repository change, even if numbers pass."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def source_hashes(root):
    paths = subprocess.check_output(["git", "-C", str(root), "ls-files"],
                                    universal_newlines=True).splitlines()
    # Bind the entire tracked tree, including native code, namespace declarations
    # and build hooks. Documentation-only changes deliberately require review.
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in paths}


def source_modes(root):
    entries = subprocess.check_output(["git", "-C", str(root), "ls-files", "--stage", "-z"],
                                      universal_newlines=True).split("\0")
    return {entry.split("\t", 1)[1]: entry.split(" ", 1)[0]
            for entry in entries if entry}


def reviewed_automation_changes(manifest, new, modes, review):
    """Accept only exact, separately reviewed additions to the workflow tree."""
    if review is None:
        return []
    if not isinstance(review, dict) or set(review) != {"baseline_revision", "added_workflows"}:
        raise ValueError("Invalid automation review record")
    if review["baseline_revision"] != manifest["r_revision"]:
        raise ValueError("Automation review belongs to a different numerical baseline")
    records = review["added_workflows"]
    if not isinstance(records, dict):
        raise ValueError("Automation review must map added workflow paths to exact identities")
    for path, identity in records.items():
        if not isinstance(path, str) or not re.fullmatch(r"\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml", path):
            raise ValueError("Automation reviews may cover only workflow YAML files")
        if (not isinstance(identity, dict) or set(identity) != {"sha256", "mode"}
                or not isinstance(identity["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", identity["sha256"])
                or identity["mode"] not in ("100644", "100755")):
            raise ValueError("Automation reviews require a SHA256 and regular Git file mode")
    return sorted(path for path, identity in records.items()
                  if path not in manifest["r_source_hashes"]
                  and new.get(path) == identity["sha256"]
                  and modes.get(path) == identity["mode"])


def compare_sources(manifest, new, modes, review=None):
    old = manifest["r_source_hashes"]
    changed = sorted(name for name in set(old) | set(new) if old.get(name) != new.get(name))
    report = {"reviewed_revision": manifest["r_revision"],
              "changed_source_or_test_files": changed}
    try:
        reviewed = reviewed_automation_changes(manifest, new, modes, review)
    except ValueError as error:
        reviewed = []
        report["automation_review_error"] = str(error)
    remaining = sorted(set(changed) - set(reviewed))
    report.update(reviewed_automation_files=reviewed,
                  unreviewed_source_or_test_files=remaining)
    return report, 3 if remaining else 0


def main():
    root = Path(sys.argv[1]).resolve()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
                                     "--untracked-files=all"], universal_newlines=True)
    if dirty.strip():
        print("R checkout must be clean, including untracked files")
        return 2
    package_root = Path(__file__).resolve().parents[1]
    manifest = json.loads((package_root / "tests/r_ref/manifest.json").read_text())
    try:
        review = json.loads((package_root / "scripts/reviewed_r_automation.json").read_text())
    except FileNotFoundError:
        review = None
    except json.JSONDecodeError:
        review = {}  # Invalid records cannot accept any source changes.
    report, status = compare_sources(manifest, source_hashes(root), source_modes(root), review)
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       universal_newlines=True).strip()
    report["selected_revision"] = revision
    print(json.dumps(report, indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())
