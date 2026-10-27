"""Estimator equivariance across independent samples with zero-weight rows.

These transformations have known effects on both the ATT and every influence
contribution. Ten samples and six kernels exercise 390 transformations without
using a reference implementation or regenerating expected numerical fixtures.
"""

import numpy as np
import pytest

from drdid import drdid, ipwd_did, reg_did


ESTIMATORS = {
    "drdid_panel": drdid.drdid_panel,
    "drdid_rc": drdid.drdid_rc,
    "std_ipw_did_panel": ipwd_did.std_ipw_did_panel,
    "std_ipw_did_rc": ipwd_did.std_ipw_did_rc,
    "reg_did_panel": reg_did.reg_did_panel,
    "reg_did_rc": reg_did.reg_did_rc,
}


def _sample(seed):
    rng = np.random.default_rng(7000 + seed)
    n = 320
    design = np.column_stack((np.ones(n), rng.normal(size=(n, 2))))
    probability = 1 / (1 + np.exp(.2 - .5 * design[:, 1] + .1 * design[:, 2]))
    treated = (rng.random(n) < probability).astype(float)
    post = (rng.random(n) < .55).astype(float)
    y0 = design[:, 1] + rng.normal(size=n)
    y1 = (y0 + .6 + .3 * design[:, 2]
          + treated * (1 + .1 * design[:, 1]) + rng.normal(size=n))
    y = post * y1 + (1 - post) * y0
    weights = np.exp(.3 * design[:, 1])
    weights[::13] = 0
    # Preserve a separate deterministic row permutation for each kernel.
    permutations = {name: rng.permutation(n) for name in ESTIMATORS}
    return y1, y0, y, post, treated, design, weights, permutations


def _assert_same(actual, expected, transformation):
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-9, rtol=1e-9,
                               err_msg=transformation + ": ATT")
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-8, rtol=1e-8,
                               err_msg=transformation + ": influence function")


@pytest.mark.parametrize("name", tuple(ESTIMATORS))
@pytest.mark.parametrize("seed", range(10))
def test_known_transformations_preserve_att_and_full_influence(name, seed):
    y1, y0, y, post, treated, design, weights, permutations = _sample(seed)
    estimator = ESTIMATORS[name]
    repeated_cross_section = name.endswith("_rc")
    args = (y, post, treated) if repeated_cross_section else (y1, y0, treated)
    expected = estimator(*args, covariates=design, i_weights=weights)

    for scale in (1e-6, 1e6):
        actual = estimator(*args, covariates=design, i_weights=weights * scale)
        _assert_same(actual, expected, "sampling-weight scale %g" % scale)

    permutation = permutations[name]
    att, influence = estimator(
        *(values[permutation] for values in args),
        covariates=design[permutation], i_weights=weights[permutation])
    _assert_same((att, influence[np.argsort(permutation)]), expected,
                 "row permutation with influence contributions restored")

    for scale in (1e-5, 1e5):
        scaled = ((y * scale, post, treated) if repeated_cross_section
                  else (y1 * scale, y0 * scale, treated))
        att, influence = estimator(*scaled, covariates=design, i_weights=weights)
        _assert_same((att / scale, influence / scale), expected,
                     "outcome-unit scale %g" % scale)

    shifted = ((y + 5, post, treated) if repeated_cross_section
               else (y1 + 5, y0 + 5, treated))
    _assert_same(estimator(*shifted, covariates=design, i_weights=weights),
                 expected, "common outcome-location shift")

    if repeated_cross_section:
        att, influence = estimator(
            y, 1 - post, treated, covariates=design, i_weights=weights)
        _assert_same((-att, -influence), expected, "reversed pre/post labels")
