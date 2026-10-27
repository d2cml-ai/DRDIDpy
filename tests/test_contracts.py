"""Independent contracts for the six shared two-period DiD estimators.

Expected values below come from weighted difference-of-means identities and
refitted empirical derivatives, rather than a second copy of estimator code.
The fixed-R numerical fixtures live in a separate test module.
"""

import numpy as np
import pytest

from drdid import drdid, ipwd_did, reg_did


ESTIMATORS = {
    "drdid_panel": drdid.drdid_panel,
    "std_ipw_did_panel": ipwd_did.std_ipw_did_panel,
    "reg_did_panel": reg_did.reg_did_panel,
    "drdid_rc": drdid.drdid_rc,
    "std_ipw_did_rc": ipwd_did.std_ipw_did_rc,
    "reg_did_rc": reg_did.reg_did_rc,
}
NAMES = tuple(ESTIMATORS)
RC_NAMES = tuple(name for name in NAMES if name.endswith("_rc"))


def _sample(zero_weights=False):
    rng = np.random.default_rng(8201)
    n = 240
    x, z = rng.normal(size=(2, n))
    cov = np.column_stack((np.ones(n), x, z))
    probability = 1 / (1 + np.exp(.3 - .5 * x + .2 * z - .4 * np.sin(2 * x)))
    treated = (rng.random(n) < probability).astype(float)
    post = (rng.random(n) < .55).astype(float)
    y0 = .8 + .3 * x - .2 * z + rng.normal(size=n)
    y1 = y0 + .6 + .4 * x + 1.1 * treated + rng.normal(size=n)
    weights = np.exp(.2 * x - .15 * z)
    if zero_weights:
        weights[::11] = 0
    return y1, y0, treated, post, cov, weights


def _arguments(name, sample):
    y1, y0, treated, post, cov, weights = sample
    outcomes = ((post * y1 + (1 - post) * y0, post)
                if name.endswith("_rc") else (y1, y0))
    return (*outcomes, treated), cov, weights


def _closed_form(args, weights, repeated_cross_section):
    first, second, treated = args
    n = len(treated)
    weights = np.ones(n) if weights is None else np.asarray(weights, float)
    influence = np.zeros(n)
    att = 0.
    if repeated_cross_section:
        cells = ((1, 1, 1), (1, 0, -1), (0, 1, -1), (0, 0, 1))
        for d, t, sign in cells:
            cell_weights = weights * ((treated == d) & (second == t))
            cell_mean = np.dot(cell_weights, first) / cell_weights.sum()
            att += sign * cell_mean
            influence += sign * cell_weights / cell_weights.mean() * (first - cell_mean)
    else:
        change = first - second
        for d, sign in ((1, 1), (0, -1)):
            cell_weights = weights * (treated == d)
            cell_mean = np.dot(cell_weights, change) / cell_weights.sum()
            att += sign * cell_mean
            influence += sign * cell_weights / cell_weights.mean() * (change - cell_mean)
    return att, influence


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("weight_kind", ["default", "varying", "zero"])
def test_intercept_only_att_influence_and_se_match_closed_form(name, weight_kind):
    sample = _sample(zero_weights=weight_kind == "zero")
    args, _, weights = _arguments(name, sample)
    if weight_kind == "default":
        weights = None
    actual_att, actual_if = ESTIMATORS[name](*args, i_weights=weights)
    expected_att, expected_if = _closed_form(args, weights, name.endswith("_rc"))
    np.testing.assert_allclose(actual_att, expected_att, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(actual_if, expected_if, atol=1e-10, rtol=1e-10)
    n = len(expected_if)
    # R uses sd(IF) * sqrt(n - 1) / n: the centered L2 norm divided by n.
    np.testing.assert_allclose(np.linalg.norm(actual_if - actual_if.mean()) / n,
                               np.linalg.norm(expected_if - expected_if.mean()) / n,
                               atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("name", NAMES)
def test_default_weights_match_explicit_ones_with_covariates(name):
    args, cov, weights = _arguments(name, _sample())
    default = ESTIMATORS[name](*args, covariates=cov)
    explicit = ESTIMATORS[name](*args, covariates=cov, i_weights=np.ones_like(weights))
    np.testing.assert_allclose(default[0], explicit[0], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(default[1], explicit[1], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("zero_weights", [False, True])
def test_weight_scale_preserves_att_and_full_influence(name, zero_weights):
    args, cov, weights = _arguments(name, _sample(zero_weights))
    att, influence = ESTIMATORS[name](*args, covariates=cov, i_weights=weights)
    assert np.isfinite(att) and np.isfinite(influence).all()
    for multiplier in (.001, 17.3):
        scaled_att, scaled_if = ESTIMATORS[name](
            *args, covariates=cov, i_weights=multiplier * weights)
        np.testing.assert_allclose(scaled_att, att, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(scaled_if, influence, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("weight_kind", ["equal", "varying", "zero"])
def test_finite_large_weight_scale_avoids_intermediate_mean_overflow(name, weight_kind):
    """R accepts these finite weights although their naive sum overflows."""
    args, cov, weights = _arguments(name, _sample(weight_kind == "zero"))
    args = tuple(np.resize(values, 600) for values in args)
    cov = np.vstack((cov, cov, cov[:120]))
    weights = (np.ones(600) if weight_kind == "equal"
               else np.resize(weights, 600))
    expected_att, expected_if = ESTIMATORS[name](
        *args, covariates=cov, i_weights=weights)
    large_weights = weights * 1e307
    assert np.isfinite(large_weights).all()
    with np.errstate(over="ignore"):
        assert np.isinf(np.mean(large_weights))
    actual_att, actual_if = ESTIMATORS[name](
        *args, covariates=cov, i_weights=large_weights)
    np.testing.assert_allclose(actual_att, expected_att, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(actual_if, expected_if, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("name", NAMES)
def test_weight_mean_that_rounds_to_zero_is_rejected(name):
    """R's positive-mean check also rejects an underflowed final double mean."""
    args, cov, weights = _arguments(name, _sample())
    weights[:] = 0
    weights[0] = np.nextafter(0., 1.)
    assert np.mean(weights) == 0
    with pytest.raises(ValueError, match="positive finite mean"):
        ESTIMATORS[name](*args, covariates=cov, i_weights=weights)


@pytest.mark.parametrize("name", NAMES)
def test_appending_zero_mass_preserves_contributions_and_se(name):
    args, cov, weights = _arguments(name, _sample(zero_weights=True))
    att, influence = ESTIMATORS[name](*args, covariates=cov, i_weights=weights)
    n = len(weights)
    extended_args = tuple(np.r_[values, [0., 1., 0., 1.]] for values in args)
    extended_cov = np.vstack((cov, [[1., -4., 2.], [1., 3., -1.],
                                    [1., .2, .1], [1., -1., 3.]]))
    new_att, new_if = ESTIMATORS[name](
        *extended_args, covariates=extended_cov,
        i_weights=np.r_[weights, np.zeros(4)])
    np.testing.assert_allclose(new_att, att, atol=1e-11, rtol=1e-11)
    np.testing.assert_array_equal(new_if[n:], np.zeros(4))
    np.testing.assert_allclose(new_if[:n] / (n + 4), influence / n,
                               atol=1e-10, rtol=1e-10)
    # R's analytical SE is the centered norm of the IF/n contributions.
    np.testing.assert_allclose(np.linalg.norm(new_if - new_if.mean()) / (n + 4),
                               np.linalg.norm(influence - influence.mean()) / n,
                               atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("zero_weights", [False, True])
def test_full_influence_matches_refitted_empirical_weight_derivative(name, zero_weights):
    """Include each RC cell and large contributions to catch pre/post sign bugs."""
    sample = _sample(zero_weights)
    args, cov, weights = _arguments(name, sample)
    _, influence = ESTIMATORS[name](*args, covariates=cov, i_weights=weights)
    treated, post = sample[2:4]
    indices = []
    for d in (0, 1):
        for t in (0, 1):
            cell = np.flatnonzero((treated == d) & (post == t))
            indices.extend(cell[np.argsort(np.abs(influence[cell]))[-2:]])
    indices.extend(np.flatnonzero(weights == 0)[:2])
    h, n = .001, len(weights)
    derivatives = []
    for index in indices:
        plus, minus = weights.copy(), weights.copy()
        plus[index] *= 1 + h
        minus[index] *= 1 - h
        att_plus = ESTIMATORS[name](*args, covariates=cov, i_weights=plus)[0]
        att_minus = ESTIMATORS[name](*args, covariates=cov, i_weights=minus)[0]
        derivatives.append((att_plus - att_minus) / (2 * h / n))
    # Central-difference error is O(h**2); models are refitted in every call.
    np.testing.assert_allclose(influence[indices], derivatives, atol=2e-5, rtol=2e-6)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("representation", ["column", "list"])
def test_vector_column_and_list_inputs_preserve_results(name, representation):
    args, cov, weights = _arguments(name, _sample())
    expected = ESTIMATORS[name](*args, covariates=cov, i_weights=weights)
    if representation == "column":
        changed_args = tuple(values[:, None] for values in args)
        changed_cov, changed_weights = cov, weights[:, None]
    else:
        changed_args = tuple(values.tolist() for values in args)
        changed_cov, changed_weights = cov.tolist(), weights.tolist()
    actual = ESTIMATORS[name](
        *changed_args, covariates=changed_cov, i_weights=changed_weights)
    assert np.asarray(actual[1]).shape == weights.shape
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("name", NAMES)
def test_one_covariate_accepts_vector_and_adds_one_intercept(name):
    args, cov, weights = _arguments(name, _sample())
    expected = ESTIMATORS[name](
        *args, covariates=cov[:, :2], i_weights=weights)
    for one_covariate in (cov[:, 1], cov[:, 1:2], cov[:, 1].tolist()):
        actual = ESTIMATORS[name](
            *args, covariates=one_covariate, i_weights=weights)
        np.testing.assert_allclose(actual[0], expected[0], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(actual[1], expected[1], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("bad_value", [-1., .5, 2., np.nan, np.inf])
def test_nonbinary_or_nonfinite_treatment_is_rejected(name, bad_value):
    args, cov, weights = _arguments(name, _sample())
    invalid = list(args)
    invalid[2] = invalid[2].copy()
    invalid[2][0] = bad_value
    with pytest.raises(ValueError):
        ESTIMATORS[name](*invalid, covariates=cov, i_weights=weights)


@pytest.mark.parametrize("name", RC_NAMES)
@pytest.mark.parametrize("bad_value", [-1., .5, 2., np.nan, np.inf])
def test_nonbinary_or_nonfinite_post_is_rejected(name, bad_value):
    args, cov, weights = _arguments(name, _sample())
    invalid = list(args)
    invalid[1] = invalid[1].copy()
    invalid[1][0] = bad_value
    with pytest.raises(ValueError):
        ESTIMATORS[name](*invalid, covariates=cov, i_weights=weights)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("bad_kind", ["negative", "nan", "inf", "zero", "short", "matrix"])
def test_invalid_sampling_weights_are_rejected_before_fitting(name, bad_kind):
    args, cov, weights = _arguments(name, _sample())
    if bad_kind == "negative":
        weights[0] = -1
    elif bad_kind == "nan":
        weights[0] = np.nan
    elif bad_kind == "inf":
        weights[0] = np.inf
    elif bad_kind == "zero":
        weights[:] = 0
    elif bad_kind == "short":
        weights = weights[:-1]
    else:
        weights = np.column_stack((weights, weights))
    with pytest.raises(ValueError):
        ESTIMATORS[name](*args, covariates=cov, i_weights=weights)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("bad_kind", ["outcome_length", "outcome_nan", "outcome_matrix",
                                       "design_length", "design_nan", "design_ndim"])
def test_invalid_outcome_and_covariate_shapes_are_rejected(name, bad_kind):
    args, cov, weights = _arguments(name, _sample())
    invalid = list(args)
    if bad_kind == "outcome_length":
        invalid[0] = invalid[0][:-1]
    elif bad_kind == "outcome_nan":
        invalid[0] = invalid[0].copy()
        invalid[0][0] = np.nan
    elif bad_kind == "outcome_matrix":
        invalid[0] = np.column_stack((invalid[0], invalid[0]))
    elif bad_kind == "design_length":
        cov = cov[:-1]
    elif bad_kind == "design_nan":
        cov[0, 1] = np.nan
    else:
        cov = cov[:, :, None]
    with pytest.raises(ValueError):
        ESTIMATORS[name](*invalid, covariates=cov, i_weights=weights)


@pytest.mark.parametrize("name", NAMES)
def test_singular_design_is_rejected(name):
    args, cov, weights = _arguments(name, _sample())
    duplicated = np.column_stack((cov, cov[:, 1]))
    with pytest.raises(ValueError, match="singular|rank|variation|covariate"):
        ESTIMATORS[name](*args, covariates=duplicated, i_weights=weights)


@pytest.mark.parametrize("name", NAMES)
def test_positive_weight_support_is_required_in_both_groups(name):
    args, cov, weights = _arguments(name, _sample())
    for group in (0, 1):
        no_mass = weights.copy()
        no_mass[args[2] == group] = 0
        with pytest.raises(ValueError):
            ESTIMATORS[name](*args, covariates=cov, i_weights=no_mass)


@pytest.mark.parametrize("name", RC_NAMES)
def test_positive_weight_support_is_required_in_each_rc_cell(name):
    args, cov, weights = _arguments(name, _sample())
    for d in (0, 1):
        for t in (0, 1):
            no_mass = weights.copy()
            no_mass[(args[2] == d) & (args[1] == t)] = 0
            with pytest.raises(ValueError):
                ESTIMATORS[name](*args, covariates=cov, i_weights=no_mass)


@pytest.mark.parametrize("name", ["drdid_panel", "reg_did_panel", "drdid_rc", "reg_did_rc"])
def test_zero_weights_do_not_make_an_undersized_outcome_design_estimable(name):
    args, cov, weights = _arguments(name, _sample())
    control = args[2] == 0
    if name.endswith("_rc"):
        control &= args[1] == 0
    # Three columns need at least three positive-weight observations. Other
    # groups remain large, isolating the outcome-regression support failure.
    indices = np.flatnonzero(control)
    weights[indices[2:]] = 0
    with pytest.raises(ValueError):
        ESTIMATORS[name](*args, covariates=cov, i_weights=weights)


@pytest.mark.parametrize("name", NAMES)
def test_inputs_are_not_mutated(name):
    args, cov, weights = _arguments(name, _sample())
    before = tuple(values.copy() for values in (*args, cov, weights))
    ESTIMATORS[name](*args, covariates=cov, i_weights=weights)
    for actual, expected in zip((*args, cov, weights), before):
        np.testing.assert_array_equal(actual, expected)
