import numpy as np
from numpy import ndarray
import statsmodels.api as sm

lm = sm.WLS
glm = sm.GLM
n_x = np.newaxis
qr_solver = np.linalg.pinv
binomial = sm.families.Binomial()


def drdid_rc(
  y: ndarray, post: ndarray, D: ndarray, covariates = None, i_weights = None):
  
  n = len(D)
  int_cov = np.ones(n)
  if covariates is not None: 
    cov_ones = np.ones(n)
    if np.all(covariates[:, 0] == cov_ones):
      int_cov = covariates
    else:
      int_cov = np.column_stack((cov_ones, covariates))

  pscore_tr = glm(D, int_cov, family=binomial, freq_weights=i_weights)\
    .fit()
  ps_fit = pscore_tr.fittedvalues
  ps_fit = np.minimum(ps_fit, 1 - 1e-16)

  row_pre = (D == 0) & (post == 0)
  reg_cont_coef_pre = lm(y[row_pre], int_cov[row_pre], weights=i_weights[row_pre])\
    .fit().params
  out_y_cont_pre = np.dot(reg_cont_coef_pre, int_cov.T)

  row_post = (D == 0) & (post == 1)
  reg_cont_coef_post = lm(y[row_post], int_cov[row_post], weights=i_weights[row_post])\
    .fit().params
  out_y_cont_post = np.dot(reg_cont_coef_post, int_cov.T)

  out_y_cont = post * out_y_cont_post + (1 - post) * out_y_cont_pre 

  row_pre_d = (D == 1) * (post == 0)
  reg_treat_coef_pre = lm(y[row_pre_d], int_cov[row_pre_d], weights=i_weights[row_pre_d])\
    .fit().params
  out_y_treat_pre = np.dot(reg_treat_coef_pre, int_cov.T)

  row_post_d = (D == 1) * (post == 1)
  reg_treat_coef_post = lm(y[row_post_d], int_cov[row_post_d], weights=i_weights[row_pre_d])\
    .fit().params
  out_y_treat_post = np.dot(reg_treat_coef_post, int_cov.T)

  w_treat_pre = i_weights * D * (1 - post)
  w_treat_post = i_weights * D * post
  rest_cont = i_weights * ps_fit * (1 - D) 
  w_cont_pre = rest_cont * (1 - post) / (1 - ps_fit)
  w_cont_post = rest_cont * post / (1 - ps_fit)

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

  ############# inf_func
  weigths_ols_pre = i_weights * (1 - D) * (1 - post)
  wols_x_pre = weigts_ols_pre[:, n_x] * int_cov
  wols_ex_pre = (weigths_ols_pre * (y - out_y_cont_post))[:, n_x] * int_cov
  cr = np.dot(wols_ex_pre, int_cov) / n
  xpx_inv_pre = qr_solver(cr)
  asy_lin_rep_ols_pre = np.dot(wols_ex_pre, xpx_inv_pre)


  weigths_ols_post = i_weights * (1 - D) * post
  wols_x_post = weigths_ols_post[:, n_x] * int_cov
  wols_ex_post = (weigths_ols_post * (y - out_y_cont_post))[:, n_x] * int_cov
  cr = np.dot(wols_x_post, int_cov) / n
  xpx_inv_post = qr_solver(cr)
  asy_lin_rep_ols_post = np.dot(wols_ex_post, xpx_inv_post)

  weigths_ols_pre_treat = i_weights * D * (1 - post)
  wols_x_pre_treat = weigths_ols_pre_treat[:, n_x] * int_cov 
  wols_ex_pre_treat = (weigths_ols_pre_treat * (y - out_y_treat_pre))[:, n_x] * int_cov
  cr = np.dot(wols_x_pre_treat, int_cov) / n
  xpx_inv_pre_treat = qr_solver(cr)
  asy_lin_rep_ols_pre_treat = np.dot(wols_ex_pre_treat, xpx_inv_pre_treat)

  weigths_ols_post_treat = i_weights * D * post
  wols_x_post_treat = weigths_ols_post_treat[:, n_x] * int_cov
  wols_ex_post_treat = (weigths_ols_post_treat * (y - out_y_treat_post))[:, n_x] * int_cov
  cr = np.dot(wols_x_post_treat, int_cov) / n
  xpx_inv_post_treat = qr_solver(cr)
  asy_lin_rep_ols_post_treat = np.dot(wols_ex_post_treat, xpx_inv_post_treat)

  score_ps = (i_weights * (D - ps_fit))[:, n_x] * int_cov
  hessian_ps = pscore_tr.cov_params() * n 
  asy_lin_rep_ps = np.dot(score_ps, hessian_ps)

  inf_treat_pre = eta_treat_pre - w_treat_pre * att_treat_pre\
    / w_treat_pre
  inf_treat_post = eta_treat_post - w_treat_post * att_treat_post\
    / w_treat_post
  
  M1_post = np.mean((w_treat_post * post)[:, n_x] * int_cov, axis=0) / np.mean(w_treat_post)
  M1_pre = np.mean((w_treat_pre * (1 - post))[:, n_x] * int_cov, axis=0) / np.mean(w_treat_pre)

  inf_treat_or_post = np.dot(asy_lin_rep_ols_post, M1_post)
  inf_treat_or_pre = np.dot(asy_lin_rep_ols_pre, M1_pre)
  inf_treat_or = inf_treat_or_post - inf_treat_or_pre

  inf_treat = inf_treat_post - inf_treat_pre + inf_treat_or

  inf_cont_pre = eta_cont_pre - w_cont_pre * att_cont_pre / np.mean(w_cont_pre)
  inf_cont_post = eta_cont_post - w_cont_post * att_cont_post / np.mean(w_cont_post)

  M2_pre = np.mean(
    (w_cont_pre * (y - out_y_cont - att_cont_pre)[:, n_x] * int_cov), 
    axis=0
  ) / np.mean(w_cont_pre)
  M2_post = np.mean(
    (w_cont_post * (y - out_y_cont - att_cont_post)[:, n_x] * int_cov),
    axis=0
  ) / np.mean(w_cont_post)

  inf_cont_ps = np.dot(asy_lin_rep_ps, M2_post - M2_pre)

  M3_post = np.mean(
    (w_cont_post * post)[:, n_x] * int_cov, axis=0
  ) / np.mean(w_cont_post)
  M3_pre = np.mean(
    (w_cont_pre * (1 - post))[:, n_x] * int_cov, axis=0
  ) / np.mean(w_cont_pre)

  inf_cont_or_post = np.mean(asy_lin_rep_ols_post, M3_post)
  inf_cont_or_pre = np.mean(asy_lin_rep_ols_pre, M3_pre)
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
    np.mean(left[:, n_x] * int_cov, axis=0) 

  mom_post = mom_f(w_d, w_dt1)
  mom_pre = mom_f(wd, w_dt0)

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

  se_inf = np.std(dr_att_inf_func) / np.sqrt(n)

  return (dr_att, dr_att_inf_func, se_inf)
  
  pass


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

  