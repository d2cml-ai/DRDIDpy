from numpy import ndarray
import numpy as np
from numpy.linalg import qr, solve
from sklearn.linear_model import LogisticRegression as glm
# from sklearn
from sklearn.linear_model import LinearRegression as lm
# ---------------- Panel

def reg_did_panel():
  pass
def drdid_panel(
  y1: ndarray, y0: ndarray, D: ndarray,
  covariates, i_weights, boot = False, inf_function = True):

  n = len(D)
  delta_y = y1 - y0
  int_cov = np.ones(n)

  if covariates is not None:
    covariates = np.asarray(covariates)
    if np.all(covariates[:, 0] == np.ones(n)):
      int_cov = covariates
    else:
      int_cov = np.concatenate((inv_cov, covariates), axis = True)

  if i_weights is not None:
    i_weights = np.ones(n)
  elif np.min(i_weights < 0):
    raise "I_weights mus be non-negative"
  
  reg = lm(fit_intercept = False)
  reg.fit(int_cov[D==0], deltaY[D==0], sample_weight=i.weights[D==0])
  reg_coeff = reg.coef_

  if np.any(np.isnan(reg_coeff)):
    raise "Outcome regression model coefficients have NA components. \n Multicollinearity (or lack of variation) of covariates is probably the reason for it."

  out_delta = np.dot(reg_coef, int_cov.T)
  out_delta = np.squeeze(out_delta)

  w_treat, w_cont = i_weights * D, i_weights * D

  reg_att_treat = np.dot(w_treat, delta_y)
  reg_att_cont = np.dot(w_cont, out_delta)

  eta_treat = np.mean(reg_att_treat) / np.mean(w_treat)
  eta_cont = np.mean(reg_att_cont) / np.mean(w_cont)

  reg_att = eta_treat - eta_cont

  weights_ols = i_weights * (1 - D)
  wols_x = weights_ols * int_cov
  wols_ex = weights_ols * (delta_y - out_delta) * int_cov
  xpx_inv = solve(np.dot(wols_x.T, wols_x) / n)
  asy_lin_rep_ols = np.dot(wols_ex, xpx_inv)

  inf_treat = (reg_att_treat - w_treat * eta_treat) / np.mean(w_treat)
  inf_cont_1 = reg_att_cont - w_cont * eta_cont

  M1 = np.mean(w_cont * int_cov)

  inf_cont_2 = np.dot(asy_lin_rep_ols, int_cov) 
  inf_control = (inf_cont_1 + inf_cont_2) / np.mean(w_cont)

  reg_att_inf_func = inf_treat - inf_control

  if not boot:
    se_reg_att = np.std(reg_att_inf_func) / sqrt(n)
  
  return (reg_att, reg_att_inf_func, se_reg_att)


  pass
def std_ipw_did_panel(
  y1: ndarray, y0: ndarray, D: ndarray,
  covariates, i_weights, boot = False, inf_function = True):

  n = len(D)
  delta_y = y1 - y0
  int_cov = np.ones(n)

  if covariates is not None:
    covariates = np.asarray(covariates)
    if np.all(covariates[:, 0] == np.ones(n)):
      int_cov = covariates
    else:
      int_cov = np.concatenate((inv_cov, covariates), axis = True)

  if i_weights is not None:
    i_weights = np.ones(n)
  elif np.min(i_weights < 0):
    raise "I_weights mus be non-negative"


  ps = glm(fit_intercept=False)
  ps.fit(int_cov, D, sample_weight=i_weights)

  ps_fit = ps.predict_proba(int_cov)[:, 1]
  ps_fit = np.minimum(ps_fit, 1 - 1e-16)

  w_treat = i_weights * D
  w_cont = i_weights * ps_fit * (1 - D) / (1 - ps_fit)

  att_treat = w_treat * delta_y
  att_cont = w_cont * delta_y

  eta_treat = np.mean(att_treat) / np.mean(w_treat)
  eta_cont = np.mean(att_cond) / np.mean(w_cont)

  ipw_att = eta_treat - eta_cont

  score_ps = i_weights * (D - ps_fit) * int_cov

  Hessian_ps = np.linalg.inv(np.dot(int_cov.T * ps_fit * (1 - ps_fit), int_cov))
  asy_lin_rep_ps = np.dot(score_ps, Hessian_ps)

  inf_treat = (att_treat - w_treat * eta_treat) / np.mean(w_treat)

  inf_cont_1 = att_cont - w_cont * eta_cont

  M2 = np.mean(w_cont * (delta_y - eta_cont) * int_cov)

  inf_control = asy_lin_rep_ps * M2

  att_inf_func = inf_treat - inf_control

  if not boot:
    se_att = np.std(att_inf_func) / sqrt(n)
  
  return (ipw_att, att_inf_func, se_att)

  

  pass
# ----------------- RC

def drdid_rc():
  pass
def std_ipw_did_rc():
  pass
def reg_did_rc():
  pass





