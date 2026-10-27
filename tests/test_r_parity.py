"""Full numerical contracts for the six traditional/normalized/OR kernels."""
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from drdid import drdid, ipwd_did, reg_did


ROOT = Path(__file__).parent / "r_ref"
CASES = json.loads((ROOT / "cases.json").read_text())
REFERENCE_PATH = Path(os.environ.get("DRDID_R_REFERENCE", ROOT / "references.json.gz"))
if REFERENCE_PATH.suffix == ".gz":
    with gzip.open(str(REFERENCE_PATH), "rt") as stream:
        PAYLOAD = json.load(stream)
else:
    PAYLOAD = json.loads(REFERENCE_PATH.read_text())
REFERENCES = PAYLOAD["results"]


def _assert_both_units(actual, expected, scale):
    # Preserve the strict raw-output comparison as well as the declared unit
    # conversion. The latter makes small outcomes informative under atol.
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(actual) / scale, np.asarray(expected) / scale,
                               rtol=1e-6, atol=1e-7)


def test_reviewed_reference_integrity():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert manifest["r_revision"] == "eaac3b49cd225d0d7706c633858555a5f204e21e"
    assert manifest["configurations"] == len(CASES) == len({case["id"] for case in CASES})
    assert PAYLOAD["schema_version"] == 1
    assert PAYLOAD["configurations"] == len(CASES)
    expected_revision = os.environ.get("DRDID_R_REVISION", manifest["r_revision"])
    assert PAYLOAD["source"]["revision"] == expected_revision
    if expected_revision == manifest["r_revision"]:
        assert PAYLOAD["source"]["version"] == manifest["r_version"]
        assert PAYLOAD["source"]["hashes"] == manifest["r_source_hashes"]
    assert set(PAYLOAD["inputs"]) == {"ordinary.csv", "trim.csv", "cases.json",
                                       "export.R", "generate_inputs.py"}
    for name, expected in PAYLOAD["inputs"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assert set(manifest["artifacts"]) == {"ordinary.csv", "trim.csv", "cases.json",
                                        "references.json.gz", "export.R", "generate_inputs.py"}
    for name, expected in manifest["artifacts"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assert set(REFERENCES) == {case["id"] for case in CASES}


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_complete_output_matches_r(case):
    data = np.genfromtxt(str(ROOT / (case["dataset"] + ".csv")), names=True, delimiter=",")
    n = len(data)
    weights = None if case["weights"] == "none" else (
        np.ones(n) if case["weights"] == "ones" else data["w"].copy())
    if case["weights"] == "zero":
        weights[::11] = 0
    design = np.column_stack((np.ones(n), data["x"], data["z"])) if case["covariates"] else None
    name, scale = case["method"], case["scale"]
    module = reg_did if name.startswith("reg_") else ipwd_did if name.startswith("std_") else drdid
    arguments = (data["y"] * scale, data["post"], data["D"]) if name.endswith("_rc") else (
        data["y1"] * scale, data["y0"] * scale, data["D"])
    options = {"covariates": design, "i_weights": weights}
    if not name.startswith("reg_"):
        options["trim_level"] = case["trim_level"]
    att, influence = getattr(module, name)(*arguments, **options)
    assert np.isfinite(att) and np.isfinite(influence).all()
    assert influence.shape == (n,)
    expected = REFERENCES[case["id"]]
    assert set(expected) == {"att", "influence", "se", "lci", "uci"}
    assert np.asarray(expected["influence"]).shape == (n,)
    # Same fixed raw-output tolerances as CSDID's current-R gate, plus the
    # predeclared conversion in cases.json (never fitted to a discrepancy).
    _assert_both_units(att, expected["att"], scale)
    _assert_both_units(influence, expected["influence"], scale)
    # R analytical SE is sd(IF)*sqrt(n-1)/n, i.e. the centered L2 norm/n.
    se = np.linalg.norm(influence - influence.mean()) / n
    _assert_both_units(se, expected["se"], scale)
    _assert_both_units([att - 1.96*se, att + 1.96*se],
                       [expected["lci"], expected["uci"]], scale)
