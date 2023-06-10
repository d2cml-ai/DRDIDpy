import numpy as np
from numpy import ndarray
import statsmodels.api as sm

lm = sm.WLS
n_x = np.newaxis
qr_solver = np.linalg.pinv

def asy_lin_wols(d, post, y, out_y, int_cov, i_w):
  weigths_ols = i_weights * d * post
  # weigths_ols_pre
  wols_x = weigths_ols[:, n_x] * int_cov
  wols_ex = (weigths_ols * (y - out_y))[:, n_x] * int_cov
  cr = np.dot(wols_x.T, int_cov) / n
  xpx_inv = qr_solver(cr)
  asy_lin_rep_ols = np.dot(wols_ex, xpx_inv)
  return asy_lin_rep_ols

def reg_did_panel(
  y1: ndarray, y0: ndarray, D: ndarray,
  covariates, i_weights = None, boot = False, inf_function = True):

  n = len(D)
  delta_y = y1 - y0
  int_cov = np.ones(n)

  if covariates is not None:
    if np.all(covariates[:, 0] == int_cov):
      int_cov = covariates
    else:
      int_cov = np.concatenate((np.ones((n, 1)), covariates), axis=1)

  if i_weights is not None:
    i_weights = np.ones(n)
  elif np.min(i_weights < 0):
    raise "I_weights mus be non-negative"

  rows = D == 0
  int_cov = covariates
  reg_coeff = lm(delta_y[rows], int_cov[rows], weights=i_weights[rows]).fit().params

  if np.any(np.isnan(reg_coeff)):
    raise "Outcome regression model coefficients have NA components. \n Multicollinearity (or lack of variation) of covariates is probably the reason for it."

  out_delta = np.dot(reg_coeff, int_cov.T)
  w_treat = i_weights * D
  w_cont = w_treat.copy()

  reg_att_treat = w_treat * delta_y
  reg_att_cont = w_cont * out_delta

  eta_treat = np.mean(reg_att_treat) / np.mean(w_treat)
  eta_cont = np.mean(reg_att_cont) / np.mean(w_cont)

  reg_att = eta_treat - eta_cont

  d = 1 - D
  post = 1

  asy_lin_rep_ols = asy_lin_wols(d, post, delta_y, out_delta, int_cov)
  inf_treat = (reg_att_treat - w_treat * eta_treat) / np.mean(w_treat)
  inf_cont_1 = (reg_att_cont - w_cont * eta_cont)

  M1 = np.mean(w_cont[:, n_x] * int_cov, axis = 0)
  inf_cont_2 = np.dot(asy_lin_rep_ols, M1)
  inf_control = (inf_cont_1 + inf_cont_2) / np.mean(w_cont)

  reg_att_inf_func = inf_treat - inf_control
  
  return (reg_att, reg_att_inf_func)


def reg_did_rc(
    y: ndarray, post: ndarray, D: ndarray, covariates = None, i_weights = None):

  n = len(D)
  int_cov = np.ones(n)

  if covariates is not None:
    col_ones = np.ones(n)
    covariates = np.asarray(covariates)
    if np.all(covariates[:, 0] == col_ones):
      int_cov = covariates
    else:
      int_cov = np.column_stack((col_ones, covariates))

  if i_weights is not None:
    i_weights = np.ones(n)
  elif np.min(i_weights < 0):
    raise "I_weights mus be non-negative"

  rows_eval = (D == 0) & (post == 0)
  reg_pre = lm(y[rows_eval], int_cov[rows_eval], weights=i_weights[rows_eval]).fit()
  out_y_pre = np.dot(reg_pre.params, int_cov)

  if np.any(np.isnan(reg_pre.params)):
    raise "Outcome regression model coefficients have NA components. \n Multicollinearity (or lack of variation) of covariates is probably the reason for it."


  rows_eval = ((D == 0) & (post == 1))
  reg_post = lm(y[rows_eval], int_cov[rows_eval], weights=i_weights[rows_eval]).fit()
  out_y_post = np.dot(reg_post.params, int_cov)

  if np.any(np.isnan(reg_post.params)):
    raise "Outcome regression model coefficients have NA components. \n Multicollinearity (or lack of variation) of covariates is probably the reason for it."

  w_treat_pre = i_weights * D * (1 - post)
  w_treat_post = i_weights * D * post
  w_cont = i_weights * D

  reg_att_treat_pre = w_treat_pre * y
  reg_att_treat_post = w_treat_post * y
  reg_att_cont = w_cont * (out_y_post - out_y_pre)

  eta_treat_pre = np.mean(reg_att_treat_pre) / np.mean(w_treat_pre)
  eta_treat_post = np.mean(reg_att_treat_post) / np.mean(w_treat_post)
  eta_cont = np.mean(reg_att_cont) / np.mean(w_cont)

  reg_att = (eta_treat_post - eta_treat_pre) - eta_cont

  weights_ols_pre = i_weights * (1 - D) * (1 - post)
  wols_x_pre = weights_ols_pre[:, n_x] * int_cov
  wols_ex_pre = (weights_ols_pre * (y - out_y_pre))[:, n_x] * int_cov
  cr = np.dot(wols_ex_pre, int_cov) / n
  xpx_inv_pre = qr_solver(cr)
  asy_lin_rep_ols_pre = np.dot(wols_ex_pre, xpx_inv_pre)

  weights_ols_post = i_weights * (1 - D) * (post)
  wols_x_post = weights_ols_post[:, n_x] * int_cov
  wols_ex_post = (weights_ols_post * (y - out_y_post))[:, n_x] * int_cov
  cr = np.dot(wols_x_post, int_cov) / n
  xpx_inv_post = qr_solver(cr)
  asy_lin_rep_ols_post = np.dot(wols_ex_post, xpx_inv_post)

  inf_treat_pre = (reg_att_treat_pre - w_treat_pre * eta_treat_pre) \
    / np.mean(w_treat_pre)
  inf_treat_post = (reg_att_treat_post - w_treat_post * eta_treat_post) \
    / np.mean(w_treat_post)
  inf_treat = inf_treat_post - inf_treat_pre
  
  inf_cont_1 = reg_att_cont - w_cont * eta_cont

  M1 = np.mean(w_cont[:, n_x], int_cov, axis = 0)

  inf_cont_2_post = np.dot(asy_lin_rep_ols_post, M1)
  inf_cont_2_pre = np.dot(asy_lin_rep_ols_pre, M1)

  inf_control = (ifn_cont_1 - inf_cont_2_post - inf_cont_2_pre) / np.mean(w_cont)

  reg_att_inf_func = inf_treat - inf_control

  if not boot:
    se_reg_att = np.std(reg_att_inf_func) / np.sqrt(n)
  
  return(reg_att, reg_att_inf_func, se_reg_att)