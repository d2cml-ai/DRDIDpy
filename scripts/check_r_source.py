"""Require review of changes to the R source and test trees, even if numbers pass."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def source_hashes(root):
    paths = subprocess.check_output(["git", "-C", str(root), "ls-files"],
                                    universal_newlines=True).splitlines()
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in paths if name == "DESCRIPTION" or name.startswith(("R/", "tests/"))}


def main():
    root = Path(sys.argv[1]).resolve()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
                                     "--untracked-files=all"], universal_newlines=True)
    if dirty.strip():
        print("R checkout must be clean, including untracked files")
        return 2
    manifest = json.loads((Path(__file__).resolve().parents[1] /
                           "tests/r_ref/manifest.json").read_text())
    old, new = manifest["r_source_hashes"], source_hashes(root)
    changed = sorted(name for name in set(old) | set(new) if old.get(name) != new.get(name))
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       universal_newlines=True).strip()
    print(json.dumps({"reviewed_revision": manifest["r_revision"],
                      "selected_revision": revision, "changed_source_or_test_files": changed}, indent=2))
    return 3 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
