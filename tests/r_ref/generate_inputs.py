"""Deterministic common inputs; run separately from the R reference exporter."""
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
METHODS = ("drdid_panel", "drdid_rc", "std_ipw_did_panel", "std_ipw_did_rc",
           "reg_did_panel", "reg_did_rc")


def main():
    for dataset, seed, n in (("ordinary", 8201, 600), ("trim", 902, 2000)):
        rng = np.random.default_rng(seed)
        x, z = rng.normal(size=(2, n))
        logit = -.3 + .5*x - .2*z + .4*np.sin(2*x) if dataset == "ordinary" else 6*x
        treated = (rng.random(n) < 1/(1+np.exp(-logit))).astype(float)
        if dataset == "trim":
            x[:4], treated[:4] = [2.5, 2.8, 3.1, 3.3], 0
        post = (rng.random(n) < .55).astype(float)
        if dataset == "trim":
            post[:4] = [0, 1, 0, 1]
        y0 = .8 + .3*x - .2*z + rng.normal(size=n) if dataset == "ordinary" else .4+x+.5*z+rng.normal(size=n)
        y1 = y0 + .6 + .4*x + 1.1*treated + rng.normal(size=n) if dataset == "ordinary" else y0+.7+.8*x+1.2*treated+rng.normal(size=n)
        weights = np.exp(.2*x-.15*z) if dataset == "ordinary" else np.exp(.1*x-.1*z)
        with (ROOT / (dataset + ".csv")).open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("y0", "y1", "y", "D", "post", "x", "z", "w"))
            writer.writerows(zip(y0, y1, post*y1+(1-post)*y0, treated, post, x, z, weights))

    cases = []
    for covariates in (True, False):
        for weights in ("none", "ones", "weights", "zero"):
            for method in METHODS:
                cases.append(dict(dataset="ordinary", covariates=covariates,
                                  weights=weights, method=method, scale=1., trim_level=.995))
    for weights in ("weights", "zero"):
        for method in METHODS:
            for trim in ((.995, .90) if not method.startswith("reg_") else (.995,)):
                cases.append(dict(dataset="trim", covariates=True, weights=weights,
                                  method=method, scale=1., trim_level=trim))
    for scale in (1e-10, 1e10):
        for method in METHODS:
            cases.append(dict(dataset="ordinary", covariates=True, weights="weights",
                              method=method, scale=scale, trim_level=.995))
    for index, case in enumerate(cases):
        case["id"] = "case-%03d" % index
    (ROOT / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")


if __name__ == "__main__":
    main()
