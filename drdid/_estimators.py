"""Traditional DR and normalized IPW equations aligned with R DRDID 1.4.0.

The public entry points remain in drdid.py and ipwd_did.py. The calculations
retain the independently reviewed csdid kernels; input and design checks also
make these kernels safe when called outside csdid's group-time dispatch.
"""
import numpy as np

from ._validation import (panel_inputs, rc_inputs, validate_trim_level,
                          checked_inverse, fit_outcome, fit_pscore,
                          require_positive_mass)


def _trim_pscore(ps_fit, D, trim_level):
    ps_fit = np.minimum(ps_fit, 1 - 1e-6)
    trim_ps = ps_fit < 1.01
    controls = D == 0
    trim_ps[controls] = ps_fit[controls] < trim_level
    return ps_fit, trim_ps


def _pscore_hessian(int_cov, i_weights, ps_fit):
    # Rebuild information from the final capped probabilities, as R does.
    # Statsmodels' stored covariance may use the preceding IRLS iteration.
    weights = i_weights * ps_fit * (1 - ps_fit)
    information = int_cov.T @ (weights[:, None] * int_cov)
    checked_inverse(information, "propensity score")
    try:
        cholesky = np.linalg.cholesky(information)
        inverse_cholesky = np.linalg.solve(cholesky, np.eye(information.shape[0]))
    except np.linalg.LinAlgError as error:
        raise ValueError("The propensity score design matrix is singular. "
                         "Consider removing some covariates.") from error
    return (inverse_cholesky.T @ inverse_cholesky) * len(ps_fit)


def std_ipw_did_panel(y1, y0, D, covariates=None, i_weights=None, trim_level=0.995):
    y1, y0, D, int_cov, i_weights = panel_inputs(
        y1, y0, D, covariates, i_weights)
    n = len(D)
    delta_y = y1 - y0
    trim_level = validate_trim_level(trim_level)
    pscore_results = fit_pscore(D, int_cov, i_weights)
    ps_fit, trim_ps = _trim_pscore(pscore_results.predict(), D, trim_level)

    w_treat = trim_ps * i_weights * D
    w_cont = trim_ps * i_weights * ps_fit * (1 - D) / (1 - ps_fit)
    require_positive_mass(w_treat, w_cont)
    att_treat = w_treat * delta_y
    att_cont = w_cont * delta_y

    eta_treat = np.mean(att_treat) / np.mean(w_treat)
    eta_cont = np.mean(att_cont) / np.mean(w_cont)
    ipw_att = eta_treat - eta_cont

    score_ps = i_weights[:, np.newaxis] * (D - ps_fit)[:, np.newaxis] * int_cov
    Hessian_ps = _pscore_hessian(int_cov, i_weights, ps_fit)
    asy_lin_rep_ps = np.dot(score_ps, Hessian_ps)

    inf_treat = (att_treat - w_treat * eta_treat) / np.mean(w_treat)
    inf_cont_1 = att_cont - w_cont * eta_cont
    pre_m2 = w_cont * (delta_y - eta_cont)
    M2 = np.mean(pre_m2[:, np.newaxis] * int_cov, axis=0)
    inf_cont_2 = np.dot(asy_lin_rep_ps, M2)

    inf_control = (inf_cont_1 + inf_cont_2) / np.mean(w_cont)
    att_inf_func = inf_treat - inf_control
    return ipw_att, att_inf_func


def drdid_panel(y1, y0, D, covariates=None, i_weights=None, trim_level=0.995):
    y1, y0, D, int_cov, i_weights = panel_inputs(
        y1, y0, D, covariates, i_weights)
    n = len(D)
    deltaY = y1 - y0
    trim_level = validate_trim_level(trim_level)
    pscore_results = fit_pscore(D, int_cov, i_weights)
    ps_fit, trim_ps = _trim_pscore(pscore_results.predict(), D, trim_level)

    mask = D == 0
    reg_results = fit_outcome(deltaY, int_cov, i_weights, mask)
    if np.any(np.isnan(reg_results.params)):
        raise ValueError(
            "Outcome regression model coefficients have NA components. \n"
            "Multicollinearity (or lack of variation) of covariates is a likely reason."
        )
    out_delta = np.dot(int_cov, reg_results.params)

    w_treat = trim_ps * i_weights * D
    w_cont = trim_ps * i_weights * ps_fit * (1 - D) / (1 - ps_fit)
    require_positive_mass(w_treat, w_cont)
    dr_att_treat = w_treat * (deltaY - out_delta)
    dr_att_cont = w_cont * (deltaY - out_delta)

    eta_treat = np.mean(dr_att_treat) / np.mean(w_treat)
    eta_cont = np.mean(dr_att_cont) / np.mean(w_cont)
    dr_att = eta_treat - eta_cont

    weights_ols = i_weights * (1 - D)
    wols_x = weights_ols[:, np.newaxis] * int_cov
    wols_eX = weights_ols[:, np.newaxis] * (deltaY - out_delta)[:, np.newaxis] * int_cov
    XpX_inv = checked_inverse(np.dot(wols_x.T, int_cov) / n, "regression")
    asy_lin_rep_wols = np.dot(wols_eX, XpX_inv)

    score_ps = i_weights[:, np.newaxis] * (D - ps_fit)[:, np.newaxis] * int_cov
    Hessian_ps = _pscore_hessian(int_cov, i_weights, ps_fit)
    asy_lin_rep_ps = np.dot(score_ps, Hessian_ps)

    inf_treat_1 = dr_att_treat - w_treat * eta_treat
    M1 = np.mean(w_treat[:, np.newaxis] * int_cov, axis=0)
    inf_treat_2 = np.dot(asy_lin_rep_wols, M1)
    inf_treat = (inf_treat_1 - inf_treat_2) / np.mean(w_treat)

    inf_cont_1 = dr_att_cont - w_cont * eta_cont
    M2 = np.mean(w_cont[:, np.newaxis] * (deltaY - out_delta - eta_cont)[:, np.newaxis] * int_cov, axis=0)
    inf_cont_2 = np.dot(asy_lin_rep_ps, M2)
    M3 = np.mean(w_cont[:, np.newaxis] * int_cov, axis=0)
    inf_cont_3 = np.dot(asy_lin_rep_wols, M3)
    inf_control = (inf_cont_1 + inf_cont_2 - inf_cont_3) / np.mean(w_cont)

    dr_att_inf_func = inf_treat - inf_control
    return dr_att, dr_att_inf_func


def reg_did_panel(y1, y0, D, covariates=None, i_weights=None):
    y1, y0, D, int_cov, i_weights = panel_inputs(
        y1, y0, D, covariates, i_weights)
    n = len(D)
    delta_y = y1 - y0
    result = fit_outcome(delta_y, int_cov, i_weights, D == 0)
    out_delta = int_cov @ result.params

    w_treat = i_weights * D
    require_positive_mass(w_treat)
    reg_att_treat = w_treat * delta_y
    reg_att_cont = w_treat * out_delta
    eta_treat = np.mean(reg_att_treat) / np.mean(w_treat)
    eta_cont = np.mean(reg_att_cont) / np.mean(w_treat)
    reg_att = eta_treat - eta_cont

    weights_ols = i_weights * (1 - D)
    wols_x = weights_ols[:, None] * int_cov
    wols_ex = weights_ols[:, None] * (delta_y - out_delta)[:, None] * int_cov
    inverse = checked_inverse(wols_x.T @ int_cov / n, "regression")
    asy_lin_rep = wols_ex @ inverse
    inf_treat = (reg_att_treat - w_treat * eta_treat) / np.mean(w_treat)
    inf_cont_1 = reg_att_cont - w_treat * eta_cont
    moment = np.mean(w_treat[:, None] * int_cov, axis=0)
    inf_control = (inf_cont_1 + asy_lin_rep @ moment) / np.mean(w_treat)
    return reg_att, inf_treat - inf_control


def reg_did_rc(y, post, D, covariates=None, i_weights=None):
    y, post, D, int_cov, i_weights = rc_inputs(
        y, post, D, covariates, i_weights)
    n = len(D)
    mask_pre = (D == 0) & (post == 0)
    mask_post = (D == 0) & (post == 1)
    result_pre = fit_outcome(y, int_cov, i_weights, mask_pre)
    result_post = fit_outcome(y, int_cov, i_weights, mask_post)
    out_pre = int_cov @ result_pre.params
    out_post = int_cov @ result_post.params

    w_treat_pre = i_weights * D * (1 - post)
    w_treat_post = i_weights * D * post
    w_treat = i_weights * D
    require_positive_mass(w_treat_pre, w_treat_post)
    reg_att_treat_pre = w_treat_pre * y
    reg_att_treat_post = w_treat_post * y
    reg_att_cont = w_treat * (out_post - out_pre)
    eta_treat_pre = np.mean(reg_att_treat_pre) / np.mean(w_treat_pre)
    eta_treat_post = np.mean(reg_att_treat_post) / np.mean(w_treat_post)
    eta_cont = np.mean(reg_att_cont) / np.mean(w_treat)
    reg_att = eta_treat_post - eta_treat_pre - eta_cont

    asy_lin_reps = []
    for mask, out in ((mask_pre, out_pre), (mask_post, out_post)):
        weights_ols = i_weights * mask
        wols_x = weights_ols[:, None] * int_cov
        wols_ex = weights_ols[:, None] * (y - out)[:, None] * int_cov
        inverse = checked_inverse(wols_x.T @ int_cov / n, "regression")
        asy_lin_reps.append(wols_ex @ inverse)
    inf_treat_pre = (reg_att_treat_pre - w_treat_pre * eta_treat_pre) / np.mean(w_treat_pre)
    inf_treat_post = (reg_att_treat_post - w_treat_post * eta_treat_post) / np.mean(w_treat_post)
    inf_cont_1 = reg_att_cont - w_treat * eta_cont
    moment = np.mean(w_treat[:, None] * int_cov, axis=0)
    inf_cont_est = asy_lin_reps[1] @ moment - asy_lin_reps[0] @ moment
    inf_control = (inf_cont_1 + inf_cont_est) / np.mean(w_treat)
    return reg_att, inf_treat_post - inf_treat_pre - inf_control


def std_ipw_did_rc(y, post, D, covariates=None, i_weights=None, trim_level=0.995):
    y, post, D, int_cov, i_weights = rc_inputs(
        y, post, D, covariates, i_weights)
    n = len(D)
    trim_level = validate_trim_level(trim_level)
    pscore_results = fit_pscore(D, int_cov, i_weights)
    ps_fit, trim_ps = _trim_pscore(pscore_results.predict(), D, trim_level)

    w_treat_pre = trim_ps * i_weights * D * (1 - post)
    w_treat_post = trim_ps * i_weights * D * post
    w_cont_pre = trim_ps * i_weights * ps_fit * (1 - D) * (1 - post) / (1 - ps_fit)
    w_cont_post = trim_ps * i_weights * ps_fit * (1 - D) * post / (1 - ps_fit)

    require_positive_mass(w_treat_pre, w_treat_post, w_cont_pre, w_cont_post)
    eta_treat_pre = w_treat_pre * y / np.mean(w_treat_pre)
    eta_treat_post = w_treat_post * y / np.mean(w_treat_post)
    eta_cont_pre = w_cont_pre * y / np.mean(w_cont_pre)
    eta_cont_post = w_cont_post * y / np.mean(w_cont_post)

    att_treat_pre = np.mean(eta_treat_pre)
    att_treat_post = np.mean(eta_treat_post)
    att_cont_pre = np.mean(eta_cont_pre)
    att_cont_post = np.mean(eta_cont_post)
    ipw_att = (att_treat_post - att_treat_pre) - (att_cont_post - att_cont_pre)

    score_ps = (i_weights * (D - ps_fit))[:, np.newaxis] * int_cov
    Hessian_ps = _pscore_hessian(int_cov, i_weights, ps_fit)
    asy_lin_rep_ps = np.dot(score_ps, Hessian_ps)

    inf_treat_pre = eta_treat_pre - w_treat_pre * att_treat_pre / np.mean(w_treat_pre)
    inf_treat_post = eta_treat_post - w_treat_post * att_treat_post / np.mean(w_treat_post)
    inf_treat = inf_treat_post - inf_treat_pre

    inf_cont_pre = eta_cont_pre - w_cont_pre * att_cont_pre / np.mean(w_cont_pre)
    inf_cont_post = eta_cont_post - w_cont_post * att_cont_post / np.mean(w_cont_post)
    inf_cont = inf_cont_post - inf_cont_pre

    M2_pre = np.mean((w_cont_pre * (y - att_cont_pre))[:, np.newaxis] * int_cov, axis=0) / np.mean(w_cont_pre)
    M2_post = np.mean((w_cont_post * (y - att_cont_post))[:, np.newaxis] * int_cov, axis=0) / np.mean(w_cont_post)
    inf_cont_ps = np.dot(asy_lin_rep_ps, (M2_post - M2_pre))

    inf_cont = inf_cont + inf_cont_ps
    att_inf_func = inf_treat - inf_cont
    return ipw_att, att_inf_func


def drdid_rc(y, post, D, covariates=None, i_weights=None, trim_level=0.995):
    y, post, D, int_cov, i_weights = rc_inputs(
        y, post, D, covariates, i_weights)
    n = len(D)
    trim_level = validate_trim_level(trim_level)
    pscore_results = fit_pscore(D, int_cov, i_weights)
    ps_fit, trim_ps = _trim_pscore(pscore_results.predict(), D, trim_level)

    mask_cont_pre = (D == 0) & (post == 0)
    reg_cont_pre = fit_outcome(y, int_cov, i_weights, mask_cont_pre)
    if np.any(np.isnan(reg_cont_pre.params)):
        raise ValueError(
            "Outcome regression model coefficients have NA components. \n"
            "Multicollinearity (or lack of variation) of covariates is a likely reason."
        )
    out_y_cont_pre = np.dot(int_cov, reg_cont_pre.params)

    mask_cont_post = (D == 0) & (post == 1)
    reg_cont_post = fit_outcome(y, int_cov, i_weights, mask_cont_post)
    if np.any(np.isnan(reg_cont_post.params)):
        raise ValueError(
            "Outcome regression model coefficients have NA components. \n"
            "Multicollinearity (or lack of variation) of covariates is a likely reason."
        )
    out_y_cont_post = np.dot(int_cov, reg_cont_post.params)

    out_y_cont = post * out_y_cont_post + (1 - post) * out_y_cont_pre

    mask_treat_pre = (D == 1) & (post == 0)
    reg_treat_pre = fit_outcome(y, int_cov, i_weights, mask_treat_pre)
    out_y_treat_pre = np.dot(int_cov, reg_treat_pre.params)

    mask_treat_post = (D == 1) & (post == 1)
    reg_treat_post = fit_outcome(y, int_cov, i_weights, mask_treat_post)
    out_y_treat_post = np.dot(int_cov, reg_treat_post.params)

    w_treat_pre = trim_ps * i_weights * D * (1 - post)
    w_treat_post = trim_ps * i_weights * D * post
    w_cont_pre = trim_ps * i_weights * ps_fit * (1 - D) * (1 - post) / (1 - ps_fit)
    w_cont_post = trim_ps * i_weights * ps_fit * (1 - D) * post / (1 - ps_fit)
    w_d = trim_ps * i_weights * D
    w_dt1 = trim_ps * i_weights * D * post
    w_dt0 = trim_ps * i_weights * D * (1 - post)

    require_positive_mass(w_treat_pre, w_treat_post, w_cont_pre, w_cont_post)
    eta_treat_pre = w_treat_pre * (y - out_y_cont) / np.mean(w_treat_pre)
    eta_treat_post = w_treat_post * (y - out_y_cont) / np.mean(w_treat_post)
    eta_cont_pre = w_cont_pre * (y - out_y_cont) / np.mean(w_cont_pre)
    eta_cont_post = w_cont_post * (y - out_y_cont) / np.mean(w_cont_post)

    eta_d_post = w_d * (out_y_treat_post - out_y_cont_post) / np.mean(w_d)
    eta_dt1_post = w_dt1 * (out_y_treat_post - out_y_cont_post) / np.mean(w_dt1)
    eta_d_pre = w_d * (out_y_treat_pre - out_y_cont_pre) / np.mean(w_d)
    eta_dt0_pre = w_dt0 * (out_y_treat_pre - out_y_cont_pre) / np.mean(w_dt0)

    att_treat_pre = np.mean(eta_treat_pre)
    att_treat_post = np.mean(eta_treat_post)
    att_cont_pre = np.mean(eta_cont_pre)
    att_cont_post = np.mean(eta_cont_post)
    att_d_post = np.mean(eta_d_post)
    att_dt1_post = np.mean(eta_dt1_post)
    att_d_pre = np.mean(eta_d_pre)
    att_dt0_pre = np.mean(eta_dt0_pre)

    dr_att = (att_treat_post - att_treat_pre) - (att_cont_post - att_cont_pre) + \
        (att_d_post - att_dt1_post) - (att_d_pre - att_dt0_pre)

    weights_ols_pre = i_weights * (1 - D) * (1 - post)
    wols_x_pre = weights_ols_pre[:, np.newaxis] * int_cov
    wols_eX_pre = weights_ols_pre[:, np.newaxis] * (y - out_y_cont_pre)[:, np.newaxis] * int_cov
    XpX_inv_pre = checked_inverse(np.dot(wols_x_pre.T, int_cov) / n, "regression")
    asy_lin_rep_ols_pre = np.dot(wols_eX_pre, XpX_inv_pre)

    weights_ols_post = i_weights * (1 - D) * post
    wols_x_post = weights_ols_post[:, np.newaxis] * int_cov
    wols_eX_post = weights_ols_post[:, np.newaxis] * (y - out_y_cont_post)[:, np.newaxis] * int_cov
    XpX_inv_post = checked_inverse(np.dot(wols_x_post.T, int_cov) / n, "regression")
    asy_lin_rep_ols_post = np.dot(wols_eX_post, XpX_inv_post)

    weights_ols_pre_treat = i_weights * D * (1 - post)
    wols_x_pre_treat = weights_ols_pre_treat[:, np.newaxis] * int_cov
    wols_eX_pre_treat = weights_ols_pre_treat[:, np.newaxis] * (y - out_y_treat_pre)[:, np.newaxis] * int_cov
    XpX_inv_pre_treat = checked_inverse(np.dot(wols_x_pre_treat.T, int_cov) / n, "regression")
    asy_lin_rep_ols_pre_treat = np.dot(wols_eX_pre_treat, XpX_inv_pre_treat)

    weights_ols_post_treat = i_weights * D * post
    wols_x_post_treat = weights_ols_post_treat[:, np.newaxis] * int_cov
    wols_eX_post_treat = weights_ols_post_treat[:, np.newaxis] * (y - out_y_treat_post)[:, np.newaxis] * int_cov
    XpX_inv_post_treat = checked_inverse(np.dot(wols_x_post_treat.T, int_cov) / n, "regression")
    asy_lin_rep_ols_post_treat = np.dot(wols_eX_post_treat, XpX_inv_post_treat)

    score_ps = i_weights[:, np.newaxis] * (D - ps_fit)[:, np.newaxis] * int_cov
    Hessian_ps = _pscore_hessian(int_cov, i_weights, ps_fit)
    asy_lin_rep_ps = np.dot(score_ps, Hessian_ps)

    inf_treat_pre = eta_treat_pre - w_treat_pre * att_treat_pre / np.mean(w_treat_pre)
    inf_treat_post = eta_treat_post - w_treat_post * att_treat_post / np.mean(w_treat_post)

    M1_post = -np.mean(w_treat_post[:, np.newaxis] * post[:, np.newaxis] * int_cov, axis=0) / np.mean(w_treat_post)
    M1_pre = -np.mean(w_treat_pre[:, np.newaxis] * (1 - post)[:, np.newaxis] * int_cov, axis=0) / np.mean(w_treat_pre)

    inf_treat_or_post = np.dot(asy_lin_rep_ols_post, M1_post)
    inf_treat_or_pre = np.dot(asy_lin_rep_ols_pre, M1_pre)

    inf_cont_pre = eta_cont_pre - w_cont_pre * att_cont_pre / np.mean(w_cont_pre)
    inf_cont_post = eta_cont_post - w_cont_post * att_cont_post / np.mean(w_cont_post)

    M2_pre = np.mean(w_cont_pre[:, np.newaxis] * (y - out_y_cont - att_cont_pre)[:, np.newaxis] * int_cov, axis=0) / np.mean(w_cont_pre)
    M2_post = np.mean(w_cont_post[:, np.newaxis] * (y - out_y_cont - att_cont_post)[:, np.newaxis] * int_cov, axis=0) / np.mean(w_cont_post)
    inf_cont_ps = np.dot(asy_lin_rep_ps, M2_post - M2_pre)

    M3_post = -np.mean(w_cont_post[:, np.newaxis] * post[:, np.newaxis] * int_cov, axis=0) / np.mean(w_cont_post)
    M3_pre = -np.mean(w_cont_pre[:, np.newaxis] * (1 - post)[:, np.newaxis] * int_cov, axis=0) / np.mean(w_cont_pre)
    inf_cont_or_post = np.dot(asy_lin_rep_ols_post, M3_post)
    inf_cont_or_pre = np.dot(asy_lin_rep_ols_pre, M3_pre)

    inf_eff1 = eta_d_post - w_d * att_d_post / np.mean(w_d)
    inf_eff2 = eta_dt1_post - w_dt1 * att_dt1_post / np.mean(w_dt1)
    inf_eff3 = eta_d_pre - w_d * att_d_pre / np.mean(w_d)
    inf_eff4 = eta_dt0_pre - w_dt0 * att_dt0_pre / np.mean(w_dt0)
    inf_eff = (inf_eff1 - inf_eff2) - (inf_eff3 - inf_eff4)

    mom_post = np.mean((w_d[:, np.newaxis] / np.mean(w_d) - w_dt1[:, np.newaxis] / np.mean(w_dt1)) * int_cov, axis=0)
    mom_pre = np.mean((w_d[:, np.newaxis] / np.mean(w_d) - w_dt0[:, np.newaxis] / np.mean(w_dt0)) * int_cov, axis=0)
    inf_or_post = np.dot(asy_lin_rep_ols_post_treat - asy_lin_rep_ols_post, mom_post)
    inf_or_pre = np.dot(asy_lin_rep_ols_pre_treat - asy_lin_rep_ols_pre, mom_pre)

    inf_treat_or = inf_treat_or_post - inf_treat_or_pre
    inf_cont_or = inf_cont_or_post - inf_cont_or_pre
    inf_or = inf_or_post - inf_or_pre

    inf_treat = inf_treat_post - inf_treat_pre + inf_treat_or
    inf_cont = inf_cont_post - inf_cont_pre + inf_cont_ps + inf_cont_or
    dr_att_inf_func1 = inf_treat - inf_cont
    dr_att_inf_func = dr_att_inf_func1 + inf_eff + inf_or

    return dr_att, dr_att_inf_func
