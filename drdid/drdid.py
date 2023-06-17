import numpy as np
from numpy import ndarray
import statsmodels.api as sm
from .utils import *


def drdid_rc(y: ndarray, post: ndarray, D: ndarray, covariates = None, i_weights = None):
  
  n = len(D)

  int_cov = np.ones(n)
  if covariates is not None:
    if np.all(covariates[:, 0] == int_cov):
      int_cov = covariates
    else:
      int_cov = np.concatenate((np.ones((n, 1)), covariates), axis=1)
  pscore_tr = glm(D, int_cov, family=binomial, freq_weights=i_weights)\
    .fit()

  _, w_cont_pre, _,\
    w_cont_post, _, asy_lin_rep_ps = fit_ps(D, int_cov, i_weights, post)
  
  def reg_out_y(d, p, y = y, int_cov = int_cov, wg = i_weights):
    rows_ = (D == d) & (post == p)
    reg_cont = lm(y[rows_], int_cov[rows_], weights=wg[rows_])\
      .fit().params
    out_y = np.dot(reg_cont, int_cov.T)
    return out_y


  out_y_cont_pre = reg_out_y(d = 0, p = 0)
  out_y_cont_post = reg_out_y(d = 0, p = 1)
  out_y_cont = post * out_y_cont_post + (1 - post) * out_y_cont_pre 

  out_y_treat_pre = reg_out_y(d = 1, p = 0)
  out_y_treat_post = reg_out_y(d = 1, p = 1)

  w_treat_pre = i_weights * D * (1 - post)
  w_treat_post = i_weights * D * post
  rest_cont = i_weights * ps_fit * (1 - D) 

  w_d = i_weights * D
  w_dt1 = w_d * post
  w_dt0 = w_d * (1 - post)

  def eta_treat(a, b = out_y_cont, y = y):
    return a * (y - b) / np.mean(a)

  eta_treat_pre = eta_treat(w_treat_pre)
  eta_treat_post = eta_treat(w_treat_post)
  eta_cont_pre = eta_treat(w_cont_pre)
  eta_cont_post = eta_treat(w_cont_post)

  eta_d_post = eta_treat(w_d, out_y_cont_post, out_y_treat_post)
  eta_d_pre = eta_treat(w_d, out_y_cont_pre, out_y_treat_pre)
  eta_dt1_post = eta_treat(w_dt1, out_y_cont_post, out_y_treat_post)
  eta_dt0_pre = eta_treat(w_dt0, out_y_cont_pre, out_y_treat_pre)

  att_treat_pre = np.mean(eta_treat_pre)
  att_treat_post = np.mean(eta_treat_post)
  att_cont_pre = np.mean(eta_cont_pre)
  att_cont_post = np.mean(eta_cont_post)

  att_d_post = np.mean(eta_d_post)
  att_dt1_post = np.mean(eta_dt1_post)
  att_d_pre = np.mean(eta_d_pre)
  att_dt0_pre = np.mean(eta_dt0_pre)


  dr_att = (att_treat_post - att_treat_pre) - \
    (att_cont_post - att_cont_pre) -\
    (att_d_post - att_dt1_post) -\
    (att_d_pre - att_dt0_pre)
  def asy_lin_wols(d, post, out_y, int_cov = int_cov):
    weigths_ols = i_weights * d * post
    # weigths_ols_pre
    wols_x = weigths_ols[:, n_x] * int_cov
    wols_ex = (weigths_ols * (y - out_y))[:, n_x] * int_cov
    cr = np.dot(wols_x.T, int_cov) / n
    xpx_inv = qr_solver(cr)
    asy_lin_rep_ols = np.dot(wols_ex, xpx_inv)
    return asy_lin_rep_ols

  asy_lin_rep_ols_pre = asy_lin_wols((1 - D), (1 - post), out_y_cont_pre)
  asy_lin_rep_ols_post = asy_lin_wols((1 - D), post, out_y_cont_post)

  asy_lin_rep_ols_pre_treat = asy_lin_wols(
    D, (1 - post), out_y_treat_pre
  )
  asy_lin_rep_ols_post_treat = asy_lin_wols(D, post, out_y_treat_post)


  inf_treat_pre = eta_treat_pre - w_treat_pre * att_treat_pre\
    / np.mean(w_treat_pre)
  inf_treat_post = eta_treat_post - w_treat_post * att_treat_post\
    / np.mean(w_treat_post)

  M1_post = np.mean((w_treat_post * post)[:, n_x] * int_cov, axis=0) / np.mean(w_treat_post)
  M1_pre = np.mean((w_treat_pre * (1 - post))[:, n_x] * int_cov, axis=0) / np.mean(w_treat_pre)

  inf_treat_or_post = np.dot(asy_lin_rep_ols_post, M1_post)
  inf_treat_or_pre = np.dot(asy_lin_rep_ols_pre, M1_pre)
  inf_treat_or = inf_treat_or_post - inf_treat_or_pre

  inf_treat = inf_treat_post - inf_treat_pre + inf_treat_or

  inf_cont_pre = eta_cont_pre - w_cont_pre * att_cont_pre / np.mean(w_cont_pre)
  inf_cont_post = eta_cont_post - w_cont_post * att_cont_post / np.mean(w_cont_post)

  M2_pre = np.mean(
    ((w_cont_pre * (y - out_y_cont - att_cont_pre))[:, n_x] * int_cov), 
    axis=0
  ) / np.mean(w_cont_pre)
  M2_post = np.mean(
    ((w_cont_post * (y - out_y_cont - att_cont_post))[:, n_x] * int_cov),
    axis=0
  ) / np.mean(w_cont_post)

  inf_cont_ps = np.dot(asy_lin_rep_ps, M2_post - M2_pre)

  M3_post = np.mean(
    (w_cont_post * post)[:, n_x] * int_cov, axis=0
  ) / np.mean(w_cont_post)
  M3_pre = np.mean(
    (w_cont_pre * (1 - post))[:, n_x] * int_cov, axis=0
  ) / np.mean(w_cont_pre)

  inf_cont_or_post = np.dot(asy_lin_rep_ols_post, M3_post)
  inf_cont_or_pre = np.dot(asy_lin_rep_ols_pre, M3_pre)
  inf_cont_or = inf_cont_or_post - inf_cont_or_pre

  inf_cont = inf_cont_post - inf_cont_pre + inf_cont_ps + inf_cont_or

  dr_att_inf_func1 = inf_treat - inf_cont

  def inf_eff_f(a, b, c):
    return a - b * c / np.mean(b)

  inf_eff1 = inf_eff_f(eta_d_post, w_d, att_d_post)
  inf_eff2 = inf_eff_f(eta_dt1_post, w_dt1, att_dt1_post)
  inf_eff3 = inf_eff_f(eta_d_pre, w_d, att_d_pre)
  inf_eff4 = inf_eff_f(eta_dt0_pre, w_dt0, att_dt0_pre)
  inf_eff = inf_eff1 - inf_eff2 - (inf_eff3 - inf_eff4)

  def mom_f(a, b, int_cov = int_cov):
    left = a / np.mean(a) - b / np.mean(b)
    return np.mean(left[:, n_x] * int_cov, axis=0) 

  mom_post = mom_f(w_d, w_dt1)
  mom_pre = mom_f(w_d, w_dt0)

  inf_or_post = np.dot(
    asy_lin_rep_ols_post_treat - asy_lin_rep_ols_post, 
    mom_post
  )
  inf_or_pre = np.dot(
    asy_lin_rep_ols_pre_treat - asy_lin_rep_ols_pre, 
    mom_pre
  )
  inf_or = inf_or_post - inf_or_pre
  dr_att_inf_func = dr_att_inf_func1 + inf_eff + inf_or

  return (dr_att, dr_att_inf_func)

def drdid_panel(
  y1: ndarray, y0: ndarray, D: ndarray,
  covariates, i_weights = None, boot = False, inf_function = True):

  n = len(D)
  delta_y = y1 - y0

  int_cov = has_intercept(covariates, n)
  i_weights = has_weights(i_weights, n)

  ref_rows = D == 0
  w_d = i_weights * D
  d1 = (1 - D)

  out_delta, asy_lin_rep_wols = \
    out_wols(delta_y, d1, x, ref_rows, i_weights)


  _, _, w_cont\
    , _, _, asy_lin_rep_ps = fit_ps(D, int_cov, i_weights, post)

  w_treat = w_d
  y_out = delta_y - out_delta

  dr_att_treat = w_treat * y_out
  dr_att_cont = w_cont * y_out

  eta_treat = eta_val(dr_att_treat, w_treat)
  eta_cont  = eta_val(dr_att_cont, w_cont)
  
  dr_att = eta_treat - eta_cont

  inf_treat_1 = dr_att_treat - w_treat * eta_treat

  M1 = np.mean(w_treat[:, n_x] * int_cov, axis=0)
  inf_treat_2 = np.dot(asy_lin_rep_wols, M1) 

  inf_treat = (inf_treat_1 - inf_treat_2) / np.mean(w_treat)

  inf_cont_1 = (dr_att_cont - w_cont) * eta_cont

  w_ref = w_cont * (delta_y - out_delta - eta_cont)

  M2 = np.mean(w_ref[:, n_x] * int_cov, axis=0)

  inf_cont_2 = np.dot(asy_lin_rep_ps, M2) 

  M3 = np.mean(w_count[:, n_x] * int_cov)

  inf_cont_3 = np.mean(asy_lin_rep_wols, M3)

  inf_control = (inf_cont_1 + inf_cont_2 - inf_cont_3) / np.mean(w_cont)

  dr_att_inf_func = inf_treat - inf_control

  if not boot:
    se_att = np.std(dr_att_inf_func) / np.sqrt(n)

  return (dr_att, dr_att_inf_func, se_att)

  