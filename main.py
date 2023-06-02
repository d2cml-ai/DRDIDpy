
#  ####################### d 
#   reg.coeff.pre <- stats::coef(stats::lm(y ~ -1 + int.cov,
#                                          subset = ((D==0) & (post==0)),
#                                          weights = i.weights))
#   if(anyNA(reg.coeff.pre)){
#     stop("Outcome regression model coefficients have NA components. \n Multicollinearity of covariates is probably the reason for it.")
#   }
#   out.y.pre <-   as.vector(tcrossprod(reg.coeff.pre, int.cov))
#   #-----------------------------------------------------------------------------
#   #Compute the Outcome regression for the control group at the pre-treatment period, using ols.
#   reg.coeff.post <- stats::coef(stats::lm(y ~ -1 + int.cov,
#                                           subset = ((D==0) & (post==1)),
#                                           weights = i.weights))
#   if(anyNA(reg.coeff.post)){
#     stop("Outcome regression model coefficients have NA components. \n Multicollinearity (or lack of variation) of covariates is probably the reason for it.")
#   }
#   out.y.post <-   as.vector(tcrossprod(reg.coeff.post, int.cov))

#     M1 <- base::colMeans(w.cont * int.cov)





#  ####################### b 
#   PS <- suppressWarnings(stats::glm(D ~ -1 + int.cov, family = "binomial", weights = i.weights))
#  ####################### b 


#  ####################### c 
#   ps.fit <- as.vector(PS$fitted.values)
#   # Do not divide by zero
#   ps.fit <- pmin(ps.fit, 1 - 1e-16)
#  ####################### c 

#   score.ps <- i.weights * (D - ps.fit) * int.cov
#   Hessian.ps <- stats::vcov(PS) * n
#   asy.lin.rep.ps <-  score.ps %*% Hessian.ps

#     M2.pre <- base::colMeans(w.cont.pre *(y - att.cont.pre) * int.cov)/mean(w.cont.pre)
#   M2.post <- base::colMeans(w.cont.post *(y - att.cont.post) * int.cov)/mean(w.cont.post)
