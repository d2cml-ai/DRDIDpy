"""Compare CSDID's actual dispatched kernels on the fixed common inputs.

Run with the proposed drdid package installed and the chosen CSDID checkout or
installation on PYTHONPATH. This checks kernels, not public group-time sample
selection and overlap guards; those require CSDID's complete suite as well.
"""
import json
from pathlib import Path

import numpy as np

from csdid.attgt_fnc.compute_att_gt_shared import select_estimators
from drdid import drdid, ipwd_did, reg_did


def main():
    root = Path(__file__).resolve().parents[1] / "tests/r_ref"
    cases = json.loads((root / "cases.json").read_text())
    # CSDID exposes the default .995 threshold; .90 standalone cases are checked
    # against R directly, preserving the public CSDID method specification.
    cases = [case for case in cases if case["trim_level"] == .995]
    for case in cases:
        data = np.genfromtxt(str(root / (case["dataset"] + ".csv")), names=True, delimiter=",")
        name, scale, n = case["method"], case["scale"], len(data)
        module = reg_did if name.startswith("reg_") else ipwd_did if name.startswith("std_") else drdid
        method = "reg" if name.startswith("reg_") else "ipw" if name.startswith("std_") else "dr"
        panel = name.endswith("_panel")
        design = np.column_stack((np.ones(n), data["x"], data["z"])) if case["covariates"] else None
        weights = None if case["weights"] == "none" else (
            np.ones(n) if case["weights"] == "ones" else data["w"].copy())
        if case["weights"] == "zero":
            weights[::11] = 0
        args = (data["y1"]*scale, data["y0"]*scale, data["D"]) if panel else (
            data["y"]*scale, data["post"], data["D"])
        standalone = getattr(module, name)(*args, covariates=design, i_weights=weights)
        provider = select_estimators(method)[0 if panel else 1]
        integrated = provider(*args, covariates=design, i_weights=weights)
        # Keep the strict raw-output contract as well as the predeclared unit
        # conversion that makes the small-outcome cases informative.
        for direct, dispatched in zip(standalone, integrated):
            np.testing.assert_allclose(direct, dispatched, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(np.asarray(direct)/scale,
                                       np.asarray(dispatched)/scale,
                                       rtol=1e-6, atol=1e-7)
    print("%d common-input CSDID dispatch comparisons passed" % len(cases))


if __name__ == "__main__":
    main()
