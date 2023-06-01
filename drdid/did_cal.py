from numpy import ndarray
import numpy as np
from numpy.linalg import qr, solve
from sklearn.linear_model import LogisticRegression as glm
# from sklearn
from sklearn.linear_model import LinearRegression as lm
# ---------------- Panel

def drdid_panel(
  y1: ndarray, y0: ndarray, D: ndarray,
  covariates, i_weights = None, boot = False, inf_function = True):

  n = len(D)
  delta_y = y1 - y0
  int_cov = np.ones(n)

  if covariates is not None:
    covariates = np.asarray(covariates)
    if np.all(covariates[:, 0] == np.ones(n)):
      int_cov = covariates
    else:
      int_cov = np.concatenate((inv_cov, covariates), axis = True)

  if i_weights is None:
    i_weights = np.ones(n)
  elif np.min(i_weights < 0):
    raise "I_weights mus be non-negative"

  pscore_tr = glm(fit_intercept= False)
  pscore_tr.fit(int_cov, D, sample_weight=i_weights)

  if not pscore_tr.converged_:
      print("Warning: glm algorithm did not converge")

  if np.any(np.isnan(pscore_tr.coef_)):
      raise ValueError("Propensity score model coefficients have NA components. \n Multicollinearity (or lack of variation) of covariates is a likely reason.")

  ps_fit = pscore_tr.predict_proba(int_cov)[:, 1]

  ps_fit = np.squeeze(ps_fit)

  reg_coeff = lm(fit_intercept=False)
  reg_coeff.fit(int_cov[D == 0], delta_y[D == 0], sample_weight=i_weights)
  if np.any(np.isnan(reg_coeff)):
    raise "Outcome regression model coefficients have NA components. \n Multicollinearity (or lack of variation) of covariates is probably the reason for it."

  out_delta = np.dot(reg_coeff, int_cov.T)
  out_delta = np.squeeze(out_delta)

  w_treat = i_weights * D
  w_cont = i_weights * ps_fit * (1 - D) / (1 - ps_fit)

  dr_att_treat = w_treat * (delta_y - out_delta)
  dr_att_cont = w_cont * (delta_y - out_delta)

  eta_treat = np.mean(dr_att_treat) / np.mean(w_treat)
  eta_cont = np.mean(dr_att_cont) / np.mean(w_cont)
  
  dr_att = eta_treat - eta_cont

  weights_ols = i_weights * (1 - D)
  wols_x = weights_ols * int_cov
  wols_ex = weights_ols * (delta_y - out_delta) * int_cov
  xpx_inv = solve(np.dot(wols_x.T, wols_x) / n)
  asy_lin_rep_wols = np.dot(wols_ex, xpx_inv)

  score_ps = i_weights * (D - ps_fit) * int_cov
  hessian_ps = np.linalg.inv(np.dot(int_cov.T * ps_fit) * (1 - ps_fit), inv_cov)
  asy_lin_rep_ps = np.dot(score_ps, hessian_ps)

  inf_treat_1 = dr_att_treat - w_treat * eta_treat

  M1 = np.mean(w_treat * int_cov)
  inf_treat_2 = np.dot(asy_lin_rep_wols, M1) 

  inf_treat = (inf_treat_1 - inf_treat_2) / np.mean(w_treat)

  inf_cont_1 = (dr_att_cont - w_cont) * eta_cont

  M2 = np.mean(w_cont * (
    delta_y - out_delta - eta_cont
  ) * int_cov
  )

  inf_cont_2 = np.dot(asy_lin_rep_ps, M2) 

  M3 = np.mean(w_count * int_cov)

  inf_cont_3 = np.mean(asy_lin_rep_wols, M3)

  inf_control = (inf_cont_1 + inf_cont_2 - inf_cont_3) / np.mean(w_cont)

  dr_att_inf_func = inf_treat - inf_control

  if not boot:
    se_att = np.std(dr_att_inf_func) / np.sqrt(n)

  return (dr_att, dr_att_inf_func, se_att)

  



  pass
def reg_did_panel(
  y1: ndarray, y0: ndarray, D: ndarray,
  covariates, i_weights = None, boot = False, inf_function = True):

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
      int_cov = np.concatenate((np.ones((n, 1)), covariates), axis=1)
  


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
    se_att = np.std(att_inf_func) / np.sqrt(n)
  
  return (ipw_att, att_inf_func, se_att)

  

  pass
# ----------------- RC

def std_ipw_did_rc(
  y: ndarray, post: ndarray, D: ndarray, covariates = None, i_weights = None):
  n = len(D)
  int_cov = np.ones(n)

  if covariates is not None:
    if np.all(covariates[:, 1] == int_cov):
      int_cov = covariates
    else:
      int_cov = np.concatenate((np.ones((n, 1)), covariates), axis=1)
  
  ps = glm(fit_intercept=False)
  ps.fit(int_cov, D, sample_weight=i_weights)

  ps_fit = ps.predict(int_cov)
  ps_fit = np.minimum(ps_fit, 1 - 1e-16)

  w_treat_pre = i_weights * D * (1 - post)
  w_treat_post = i_weights * D * post
  diff_w =  i_weights * ps_fit * (1 - D)
  w_cont_pre = diff_w *  (1 - post) / (1 - ps_fit)
  w_cont_post = diff_w * post / (1 - ps_fit)

  def eta_form(vect, y_ = y):
    return vect * y_ / np.mean(vect)

  eta_treat_pre = eta_form(w_treat_pre)
  eta_treat_post = eta_form(w_treat_post)
  eta_cont_pre = eta_form(w_cont_pre)
  eta_cont_post = eta_form(w_cont_post)

  att_treat_pre = np.mean(eta_treat_pre)
  att_treat_post = np.mean(eta_treat_post)
  att_cont_pre = np.mean(eta_cont_pre)
  att_cont_post = np.mean(eta_cont_post)

  ipw_att = att_treat_post - att_treat_pre - (att_cont_post - att_cont_pre)

  score_ps = np.dot(i_weights * (D - ps_fit), int_cov)
  hessian = ps.cov_params() * n
  asy_lin_rep_ps = np.dot(score_ps, hessian)

  inf_treat_pre = eta_treat_pre - \
    w_treat_pre * att_treat_pre / np.mean(w_treat_pre)
  inf_treat_post = eta_treat_post -\
    w_treat_post * att_treat_post / np.mean(w_treat_post)
  inf_treat = inf_treat_post - inf_treat_pre

  inf_cont_pre = eta_cont_pre - \
    w_cont_pre * att_cont_pre / np.mean(w_cont_pre)
  inf_cont_post = eta_cont_post -\
    w_cont_post * att_cont_post / np.mean(w_cont_post)
  
  def simple_rep(a, b, y_ = y, cov = int_cov):
    return a * (y - b) * cov / np.mean(a)

  M2_pre = np.mean(simple_rep(w_cont_pre, att_cont_pre), axis=0)
  M2_post = np.mean(simple_rep(w_cont_post, att_treat_post), axis=0)

  inf_cont_ps = np.dot(asy_lin_rep_ps, M2_post - M2_pre)
  inf_cont = inf_cont + inf_cont_ps
  att_inf_func = inf_treat - inf_cont

  se_att = np.std(att_inf_func) / np.sqrt(n)

  return(ipw_att, att_inf_func, se_att)

def reg_did_rc(
    y: ndarray, post: ndarray, D: ndarray, covariates = None, i_weights = None):
  n = len(D)
  int_cov = np.ones(n)

  if covariates is not None:
    if np.all(covariates[:, 0] == int_cov):
      int_cov = covariates
    else:
      int_cov = np.concatenate((np.ones((n, 1)), covariates), axis=1)

  reg_pre = lm(fit_intercept=False)
  rows_eval = ((D == 0) & (post == 0))
  reg_pre.fit(int_cov[rows_eval, :], y[rows_eval, :], sample_weight=i_weights)
  out_y_pre = np.dot(int_cov, reg_pre.params)

  reg_post = lm(fit_intercept=False)
  rows_eval = ((D == 0) & (post == 1))
  reg_post.fit(int_cov[rows_eval, :], y[rows_eval, :], sample_weight=i_weights)
  out_y_post = np.dot(int_cov, reg_post.params)

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
  wols_x_pre = weights_ols_pre * int_cov
  wols_ex_pre = weights_ols_pre * (y - out_y_pre) * int_cov
  xpx_inv_pre = np.linalg.solve(np.dot(wolx))


  pass

def drdid_rc():
  pass




