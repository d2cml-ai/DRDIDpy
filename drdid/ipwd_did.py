from .utils import *

def std_ipw_did_panel(
  y1: ndarray, y0: ndarray, D: ndarray,
  covariates, i_weights, boot = False):

  n = len(D)
  delta_y = y1 - y0

  int_cov = has_intercept(int_cov)
  i_weights = has_weights(i_weights)

  _, _, w_cont, _, _, asy_lin_rep_ps =\
    fit_ps(D, int_cov, i_weights)

  w_treat = i_weights * D

  att_treat = w_treat * delta_y
  att_cont = w_cont * delta_y

  eta_treat = np.mean(att_treat) / np.mean(w_treat)
  eta_cont = np.mean(att_cond) / np.mean(w_cont)

  ipw_att = eta_treat - eta_cont

  inf_treat = (att_treat - w_treat * eta_treat) / np.mean(w_treat)
  inf_cont_1 = att_cont - w_cont * eta_cont
  w_ref = w_cont * (delta_y - eta_cont)
  M2 = np.mean(w_ref[:, n_x] * int_cov, axis=0)
  inf_control = asy_lin_rep_ps * M2

  att_inf_func = inf_treat - inf_control

  se_att = None
  if not boot:
    se_att = np.std(att_inf_func) / np.sqrt(n)
  
  return (ipw_att, att_inf_func, se_att)
# ----------------- RC

def std_ipw_did_rc(
  y: ndarray, post: ndarray, D: ndarray, covariates = None, i_weights = None
  , boot = False):

  n = len(D)
  post1 = (1 - post)
  d1 = (1 - D)

  int_cov = has_intercept(covariates, n)
  i_weights = has_weights(i_weights, n)
 
  ps_fit, w_cont_pre, _, w_cont_post, _, asy_lin_rep_ps = \
    fit_ps(D, int_cov, i_weights, post)

  w_i = i_weights * D
  w_treat_pre  = w_i * post1
  w_treat_post = w_i * post

  diff_w =  i_weights * ps_fit * (1 - D)

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
    return (a * (y - b))[:, n_x] * cov / np.mean(a)

  M2_pre = np.mean(simple_rep(w_cont_pre, att_cont_pre), axis=0)
  M2_post = np.mean(simple_rep(w_cont_post, att_treat_post), axis=0)

  inf_cont_ps = np.dot(asy_lin_rep_ps, M2_post - M2_pre)
  inf_cont = inf_cont + inf_cont_ps
  att_inf_func = inf_treat - inf_cont

  if not boot:
    se_att = np.std(att_inf_func) / np.sqrt(n)

  return(ipw_att, att_inf_func, se_att)






