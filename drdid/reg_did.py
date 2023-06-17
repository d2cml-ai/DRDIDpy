import numpy as np
from numpy import ndarray

from .utils import * 

def asy_lin_wols(d, post, y, out_y, int_cov, i_w, n):
  weigths_ols = i_w * d * post
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

  d1 = 1 - D
  
  int_cov = has_intercept(covariates, n)
  i_weights = has_weights(i_weights, n)

  rows = D == 0

  out_delta, asy_lin_rep_ols = \
    out_wols(delta_y, d1, int_cov, rows, i_weights)

  w_treat = i_weights * D
  w_cont = w_treat.copy()

  reg_att_treat = w_treat * delta_y
  reg_att_cont = w_cont * out_delta

  eta_treat = np.mean(reg_att_treat) / np.mean(w_treat)
  eta_cont = np.mean(reg_att_cont) / np.mean(w_cont)

  reg_att = eta_treat - eta_cont


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
  
  int_cov = has_intercept(covariates, n)
  i_weights = has_weights(i_weights, n)

  d1, post1 = 1 - D, 1 - post

  rows_pre = (D == 0) & (post == 0)
  out_y_pre, asy_lin_rep_ols_pre =\
    out_wols(y, d1, int_cov, rows_pre, i_weights, post1)

  rows_post = (D == 0) & (post == 1)
  out_y_post, asy_lin_rep_ols_post =\
    out_wols(y, d1, int_cov, rows_post, i_weights, post)

  w_cont = i_weights * D

  w_treat_pre  = w_cont * post1
  w_treat_post = w_cont * post

  reg_att_treat_pre = w_treat_pre * y
  reg_att_treat_post = w_treat_post * y
  reg_att_cont = w_cont * (out_y_post - out_y_pre)

  eta_treat_pre = np.mean(reg_att_treat_pre) / np.mean(w_treat_pre)
  eta_treat_post = np.mean(reg_att_treat_post) / np.mean(w_treat_post)
  eta_cont = np.mean(reg_att_cont) / np.mean(w_cont)
  
  reg_att = (eta_treat_post - eta_treat_pre) - eta_cont

  inf_treat_pre = (reg_att_treat_pre - w_treat_pre * eta_treat_pre) \
    / np.mean(w_treat_pre)
  inf_treat_post = (reg_att_treat_post - w_treat_post * eta_treat_post) \
    / np.mean(w_treat_post)
  inf_treat = inf_treat_post - inf_treat_pre
  
  inf_cont_1 = reg_att_cont - w_cont * eta_cont

  M1 = np.mean(w_cont[:, n_x], int_cov, axis = 0)

  inf_cont_2_post = np.dot(asy_lin_rep_ols_post, M1)
  inf_cont_2_pre = np.dot(asy_lin_rep_ols_pre, M1)

  inf_control = (ifn_cont_1 - inf_cont_2_post - inf_cont_2_pre) \
    / np.mean(w_cont)

  reg_att_inf_func = inf_treat - inf_control

  if not boot:
    se_reg_att = np.std(reg_att_inf_func) / np.sqrt(n)
  
  return(reg_att, reg_att_inf_func, se_reg_att)