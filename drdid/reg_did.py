"""Outcome regression DiD kernels for panel and repeated cross sections.

Both functions return ``(ATT, observation_influence_function)``. Covariates
retain Python's automatic intercept; supply that augmented design explicitly
when comparing with R's low-level functions.
"""
import numpy as np

from ._estimators import reg_did_panel, reg_did_rc
from ._validation import checked_inverse

__all__ = ["reg_did_panel", "reg_did_rc", "asy_lin_wols"]


def asy_lin_wols(d, post, y, out_y, int_cov, i_w, n):
    """Retain the historical weighted-OLS influence helper and its signature."""
    weights_ols = (np.asarray(i_w).reshape(-1)
                   * np.asarray(d).reshape(-1) * np.asarray(post).reshape(-1))
    design = np.asarray(int_cov)
    residual = np.asarray(y).reshape(-1) - np.asarray(out_y).reshape(-1)
    wols_x = weights_ols[:, None] * design
    wols_ex = (weights_ols * residual)[:, None] * design
    inverse = checked_inverse(wols_x.T @ design / n, "regression")
    return wols_ex @ inverse
