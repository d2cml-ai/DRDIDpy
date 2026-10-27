"""Shared input and fitted-design checks for the six supported DiD kernels."""

import warnings

import numpy as np
import statsmodels.api as sm


def _numeric(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
        raise ValueError("{} must contain finite numeric values".format(name))
    return np.asarray(array, dtype=float)


def _vector(values, name, n=None):
    array = _numeric(values, name)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or (n is not None and len(array) != n):
        raise ValueError("{} must contain one value per observation".format(name))
    if not len(array):
        raise ValueError("{} must not be empty".format(name))
    return array


def _inputs(D, covariates, i_weights):
    treated = _vector(D, "D")
    if not np.all((treated == 0) | (treated == 1)):
        raise ValueError("D must contain only 0 (comparison) and 1 (treated)")
    n = len(treated)
    if covariates is None:
        design = np.ones((n, 1))
    else:
        design = _numeric(covariates, "covariates")
        if design.ndim == 1:
            design = design[:, None]
        if design.ndim != 2 or design.shape[0] != n:
            raise ValueError("covariates must have one row per observation")
        # Retain Python's historical intercept convention. R's low-level
        # functions instead require callers to supply the complete design.
        if design.shape[1] == 0 or not np.all(design[:, 0] == 1):
            design = np.column_stack((np.ones(n), design))
    if i_weights is None:
        weights = np.ones(n)
    else:
        weights = _vector(i_weights, "i_weights", n)
    if np.any(weights < 0):
        raise ValueError("i_weights must be non-negative")
    with np.errstate(over="ignore", invalid="ignore"):
        mean_weight = np.mean(weights)
    if not np.isfinite(mean_weight):
        # R's mean avoids an overflowing intermediate sum. Every input is
        # finite and non-negative here, so the actual mean cannot exceed the
        # largest weight. Keep the ordinary path unchanged when it succeeds.
        largest = np.max(weights)
        mean_weight = largest * np.mean(weights / largest)
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError(
            "i_weights must be finite, non-negative, have positive finite mean, "
            "and contain one value per observation")
    return treated, design, weights / mean_weight


def panel_inputs(y1, y0, D, covariates, i_weights):
    treated, design, weights = _inputs(D, covariates, i_weights)
    return (_vector(y1, "y1", len(treated)), _vector(y0, "y0", len(treated)),
            treated, design, weights)


def rc_inputs(y, post, D, covariates, i_weights):
    treated, design, weights = _inputs(D, covariates, i_weights)
    post = _vector(post, "post", len(treated))
    if not np.all((post == 0) | (post == 1)):
        raise ValueError("post must contain only 0 (pre) and 1 (post)")
    return _vector(y, "y", len(treated)), post, treated, design, weights


def validate_trim_level(trim_level):
    value = _numeric(trim_level, "trim_level")
    if value.ndim != 0 or not value > 0:
        raise ValueError("trim_level must be a positive finite scalar")
    return float(value)


def checked_inverse(matrix, label):
    """Apply R's reciprocal-condition guard before an ordinary inverse."""
    try:
        rcond = 1.0 / np.linalg.cond(matrix, 1)
        if not np.isfinite(rcond) or rcond < np.finfo(float).eps:
            raise np.linalg.LinAlgError("singular design")
        return np.linalg.solve(matrix, np.eye(matrix.shape[0]))
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "The {} design matrix is singular. Consider removing some covariates."
            .format(label)) from error


def check_design(design, weights, mask, label):
    """Check the positive normalized-weight support actually used in a fit."""
    support = np.asarray(mask, dtype=bool) & (weights > 0)
    if np.count_nonzero(support) < design.shape[1]:
        raise ValueError("The {} design has too few positive-weight observations"
                         .format(label))
    x, w = design[support], weights[support]
    with np.errstate(over="ignore", invalid="ignore"):
        gram = (x * w[:, None]).T @ x / len(design)
    return checked_inverse(gram, label)


def fit_outcome(y, design, weights, mask):
    check_design(design, weights, mask, "regression")
    result = sm.WLS(y[mask], design[mask], weights=weights[mask]).fit()
    if not np.all(np.isfinite(result.params)):
        raise ValueError("Outcome regression model coefficients are not finite")
    return result


def fit_pscore(D, design, weights):
    check_design(design, weights, np.ones(len(D), dtype=bool), "propensity score")
    result = sm.GLM(D, design, family=sm.families.Binomial(),
                    freq_weights=weights).fit()
    if not result.converged:
        warnings.warn("glm algorithm did not converge", RuntimeWarning)
    if not np.all(np.isfinite(result.params)):
        raise ValueError("Propensity score model coefficients are not finite")
    return result


def require_positive_mass(*components):
    if any(not np.isfinite(np.mean(weights)) or np.mean(weights) <= 0
           for weights in components):
        raise ValueError("Each required treatment/period group must have positive weight after trimming")
