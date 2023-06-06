
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






