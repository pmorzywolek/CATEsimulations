
#################################################
# R-code to replicate the simulation set-up 3 from the paper
# "On a General Class of Orthogonal Learners for the Estimation of Heterogeneous Treatment Effects"
#################################################

library(MASS)
library(ggpubr)
library(ggplot2)
library(reshape2)
library(SuperLearner)

# Set initial parameters (number of individuals in each of the data sets: train A, train B and test, and number of simulations)
n <- 500
nsim <- 2

# Simulation set-up 3 is based on the simulation set-up from (Belloni et al., 2017)
# Parameters of the simulation
dimx <- 50
theta <- 0
R2y <- 0.5
R2d <- 0.5
rho <- 0.5

beta <- 1/c(1:dimx)^2
Sigma <- toeplitz(rho^(0:(dimx-1)))

cy <- as.numeric(sqrt( R2y / ((1-R2y)*beta%*%Sigma%*%beta) ))
cd <- as.numeric(sqrt( ((pi^2/3)*R2d) / ((1-R2d)*beta%*%Sigma%*%beta)))

# Select list of models to be included in the SuperLearner
SL.library = c("SL.glm", "SL.ranger", "SL.glmnet", "SL.xgboost")

# Data frame for the MSE of each method. "T", "IPW", "DR", "R" and "psDR" mean T-, IPW-, DR-, R- and propensity-score-weighted DR-Learner, respectively. "MSE"
# is mean-squared error, "psMSE" is mean-squared error in treated and "powMSE" is propensity-overlap-weighted mean-squared error.

res <- data.frame(matrix(nrow=nsim,ncol=13))
colnames(res) <- c("T_MSE", "IPW_MSE", "DR_MSE", "R_MSE", "T_powMSE", "IPW_powMSE", "DR_powMSE", "R_powMSE", "T_psMSE", "IPW_psMSE", "DR_psMSE", "R_psMSE", "psDR_psMSE")

for(i in 1:nsim) {
  
  #####################################
  # Generate the training data sets A and B, and the test data set.
  # Y is the outcome, X are covariates and A is the exposure.
  #####################################
  
  zeta_train_A <- rnorm(n = n, mean = 0, sd = 1)
  zeta_train_B <- rnorm(n = n, mean = 0, sd = 1)
  zeta_test <- rnorm(n = n, mean = 0, sd = 1)
  
  v_train_A <- runif(n = n, min = 0, max = 1)
  v_train_B <- runif(n = n, min = 0, max = 1)
  v_test <- runif(n = n, min = 0, max = 1)
  
  # Simulate the covariates in train A, train B and test data sets
  X_train_A <- as.data.frame(mvrnorm(n = n, mu = rep(0, dimx), Sigma = Sigma))
  X_train_B <- as.data.frame(mvrnorm(n = n, mu = rep(0, dimx), Sigma = Sigma))
  X_test <- as.data.frame(mvrnorm(n = n, mu = rep(0, dimx), Sigma = Sigma))
  
  # Compute true propensity score
  ps_train_A <- exp(cd * as.matrix(X_train_A) %*% beta) / (1 + exp(cd * as.matrix(X_train_A) %*% beta))
  ps_train_B <- exp(cd * as.matrix(X_train_B) %*% beta) / (1 + exp(cd * as.matrix(X_train_B) %*% beta))
  ps_test <- exp(cd * as.matrix(X_test) %*% beta) / (1 + exp(cd * as.matrix(X_test) %*% beta))
  
  # Compute the exposure variable
  A_train_A <- as.numeric(ps_train_A > v_train_A)
  A_train_B <- as.numeric(ps_train_B > v_train_B)
  A_test <- as.numeric(ps_test > v_test)
  
  # Compute the outcome variable
  Y_train_A <- theta * A_train_A + cy * as.matrix(X_train_A) %*% beta * A_train_A + zeta_train_A
  Y_train_B <- theta * A_train_B + cy * as.matrix(X_train_B) %*% beta * A_train_B + zeta_train_B
  Y_test <- theta * A_test + cy * as.matrix(X_test) %*% beta * A_test + zeta_test
  
  # Compute true CATE
  CATE_train_A <- theta + cy * as.matrix(X_train_A) %*% beta
  CATE_train_B <- theta + cy * as.matrix(X_train_B) %*% beta
  CATE_test <- theta + cy * as.matrix(X_test) %*% beta
  
  data_train_A <- as.data.frame(cbind(X_train_A, A_train_A, Y_train_A, ps_train_A, CATE_train_A))
  data_train_B <- as.data.frame(cbind(X_train_B, A_train_B, Y_train_B, ps_train_B, CATE_train_B))
  data_test <- as.data.frame(cbind(X_test, A_test, Y_test, ps_test, CATE_test))
  
  colnames(data_train_A)[c(51:54)] <- c("a", "y", "ps", "CATE")
  colnames(data_train_B)[c(51:54)] <- c("a", "y", "ps", "CATE")
  colnames(data_test)[c(51:54)] <- c("a", "y", "ps", "CATE")

  # Combine the covariates and the exposure into a single data frame 
  Z_train_A <- data_train_A[, c(1:51)]
  Z_train_B <- data_train_B[, c(1:51)]
  Z_test <- data_test[, c(1:51)]
  
  #####################################
  # Estimate nuisance parameters
  #####################################
  
  ### Propensity score (PS) model
  # Fit the propensity score model on the training data
  model_ps_A = SuperLearner(Y = A_train_A,
                            X = X_train_A,
                            family = binomial(),
                            SL.library = SL.library)
  model_ps_B = SuperLearner(Y = A_train_B,
                            X = X_train_B,
                            family = binomial(),
                            SL.library = SL.library)
  
  # Compute PS on the training sets
  data_train_A$ps_SL <- as.vector(predict(model_ps_B, X_train_A, onlySL = TRUE)$pred)
  data_train_B$ps_SL <- as.vector(predict(model_ps_A, X_train_B, onlySL = TRUE)$pred)
  
  ### Outcome prediction model
  model_Q_A <- SuperLearner(Y = Y_train_A,
                            X = Z_train_A,
                            SL.library = SL.library)
  model_Q_B <- SuperLearner(Y = Y_train_B,
                            X = Z_train_B,
                            SL.library = SL.library)
  
  # Construct data sets were everyone/no one is treated.
  Z_train_A_a1 <- Z_train_A
  Z_train_A_a1$a <- 1
  
  Z_train_A_a0 <- Z_train_A
  Z_train_A_a0$a <- 0
  
  Z_train_B_a1 <- Z_train_B
  Z_train_B_a1$a <- 1
  
  Z_train_B_a0 <- Z_train_B
  Z_train_B_a0$a <- 0
  
  Z_test_a1 <- Z_test
  Z_test_a1$a <- 1
  
  Z_test_a0 <- Z_test
  Z_test_a0$a <- 0
  
  # Compute predictions from the outcome model when everyone/no one is treated
  Z_train_A_a1$Q_pred_a1 <- as.vector(predict(model_Q_B, newdata = Z_train_A_a1, onlySL = TRUE)$pred)
  Z_train_A_a0$Q_pred_a0 <- as.vector(predict(model_Q_B, newdata = Z_train_A_a0, onlySL = TRUE)$pred)
  
  Z_train_B_a1$Q_pred_a1 <- as.vector(predict(model_Q_A, newdata = Z_train_B_a1, onlySL = TRUE)$pred)
  Z_train_B_a0$Q_pred_a0 <- as.vector(predict(model_Q_A, newdata = Z_train_B_a0, onlySL = TRUE)$pred)
  
  # Add predicted counterfactual to the training sets
  data_train_A$Q1 <- Z_train_A_a1$Q_pred_a1
  data_train_A$Q0 <- Z_train_A_a0$Q_pred_a0
  
  data_train_B$Q1 <- Z_train_B_a1$Q_pred_a1
  data_train_B$Q0 <- Z_train_B_a0$Q_pred_a0
  
  #####################################
  # Compute the pseudo-outcomes in the training sets
  #####################################
  
  # IPW-Learner
  data_train_A$pseudooutcome_IPW <-
    data_train_A$a * data_train_A$y / data_train_A$ps_SL  - (1 - data_train_A$a) * data_train_A$y / (1 - data_train_A$ps_SL)
  data_train_B$pseudooutcome_IPW <-
    data_train_B$a * data_train_B$y / data_train_B$ps_SL  - (1 - data_train_B$a) * data_train_B$y / (1 - data_train_B$ps_SL)
  
  # DR-Learner
  data_train_A$pseudooutcome_DR <-
    data_train_A$a / data_train_A$ps_SL * (data_train_A$y - data_train_A$Q1) - (1 - data_train_A$a) /
    (1 - data_train_A$ps_SL) * (data_train_A$y - data_train_A$Q0) + data_train_A$Q1 - data_train_A$Q0
  data_train_B$pseudooutcome_DR <-
    data_train_B$a / data_train_B$ps_SL * (data_train_B$y - data_train_B$Q1) - (1 - data_train_B$a) /
    (1 - data_train_B$ps_SL) * (data_train_B$y - data_train_B$Q0) + data_train_B$Q1 - data_train_B$Q0
  
  # Propensity-overlap-weighted DR-Learner
  data_train_A$pseudooutcome_R <-
    (data_train_A$y - data_train_A$ps_SL * data_train_A$Q1 - (1 - data_train_A$ps_SL) * data_train_A$Q0) / 
    (data_train_A$a - data_train_A$ps_SL)
  data_train_B$pseudooutcome_R <-
    (data_train_B$y - data_train_B$ps_SL * data_train_B$Q1 - (1 - data_train_B$ps_SL) * data_train_B$Q0) / 
    (data_train_B$a - data_train_B$ps_SL)
  
  # Propensity-score-weighted DR-Learner
  data_train_A$pseudooutcome_psDR[data_train_A$a == 1] <-
    (data_train_A$y[data_train_A$a == 1] - data_train_A$Q0[data_train_A$a == 1]) * (data_train_A$a[data_train_A$a == 1] - data_train_A$ps_SL[data_train_A$a == 1]) /
    ((1 - data_train_A$ps_SL[data_train_A$a == 1]) * data_train_A$a[data_train_A$a == 1])
  data_train_B$pseudooutcome_psDR[data_train_B$a == 1] <-
    (data_train_B$y[data_train_B$a == 1] - data_train_B$Q0[data_train_B$a == 1]) * (data_train_B$a[data_train_B$a == 1] - data_train_B$ps_SL[data_train_B$a == 1]) /
    ((1 - data_train_B$ps_SL[data_train_B$a == 1]) * data_train_B$a[data_train_B$a == 1])
  
  #####################################
  # Combine both training data sets
  #####################################
  
  data_train_AB <- rbind(data_train_A, data_train_B)
  X_train_AB <- rbind(X_train_A, X_train_B)
  Z_train_AB <- rbind(Z_train_A, Z_train_B)
  A_train_AB <- rbind(A_train_A, A_train_B)
  
  #####################################
  # Regress the pseudo-outcomes in the combined training data set
  #####################################
  
  # IPW-Learner
  model_pseudooutcome_IPW = SuperLearner(Y = data_train_AB$pseudooutcome_IPW,
                                         X = X_train_AB,
                                         SL.library = SL.library)
  
  # DR-Learner
  model_pseudooutcome_DR = SuperLearner(Y = data_train_AB$pseudooutcome_DR,
                                        X = X_train_AB,
                                        SL.library = SL.library)
  
  # Propensity-overlap-weighted DR-Learner
  # Compute weights in the set used for regressing propensity-overlap-weighted pseudo-outcome
  wt <- as.vector((data_train_AB$a - data_train_AB$ps_SL) ^ 2)
  model_pseudooutcome_R = SuperLearner(Y = data_train_AB$pseudooutcome_R,
                                       X = X_train_AB, 
                                       SL.library = SL.library, 
                                       obsWeights = wt)
  
  # Propensity-score-weighted DR-Learner
  model_pseudooutcome_psDR = SuperLearner(Y = data_train_AB$pseudooutcome_psDR[data_train_AB$a == 1],
                                          X = X_train_AB[data_train_AB$a == 1, ],
                                          SL.library = SL.library)
  
  #####################################
  # T-Learner 
  #####################################
  
  # We train T-Learner on combined training data set A and B to make sure that we use the same amount of data for training different methods
  model_Q_T <- SuperLearner(Y = data_train_AB$y,
                            X = Z_train_AB,
                            SL.library = SL.library)
  
  Z_test_a1$Q_T_pred_a1 <- as.vector(predict(model_Q_T, newdata = Z_test_a1, onlySL = TRUE)$pred)
  Z_test_a0$Q_T_pred_a0 <- as.vector(predict(model_Q_T, newdata = Z_test_a0, onlySL = TRUE)$pred)
  
  data_test$Q1_T <- Z_test_a1$Q_T_pred_a1
  data_test$Q0_T <- Z_test_a0$Q_T_pred_a0
  
  #####################################
  # Predict CATE on the test set
  #####################################
  
  # T-Learner
  data_test$pred_CATE_T <- data_test$Q1_T - data_test$Q0_T
  
  # IPW-Learner
  data_test$pred_CATE_IPW <- as.vector(predict(model_pseudooutcome_IPW, newdata = X_test, onlySL = TRUE)$pred)
  
  # DR-Learner
  data_test$pred_CATE_DR <- as.vector(predict(model_pseudooutcome_DR, newdata = X_test, onlySL = TRUE)$pred)
  
  # Propensity-overlap-weighted DR-Learner
  data_test$pred_CATE_R <- as.vector(predict(model_pseudooutcome_R, newdata = X_test, onlySL = TRUE)$pred)
  
  # Propensity-score-weighted DR-Learner (only in treated)
  data_test$pred_CATE_psDR[data_test$a == 1] <- as.vector(predict(model_pseudooutcome_psDR, newdata = X_test[data_test$a == 1, ], onlySL = TRUE)$pred)
  
  #####################################
  # MSE calculation on the test set
  #####################################
  
  # Calculate propensity-overlap-weights (using true propensity scores) on the test set (for pow-MSE evaluation)
  data_test$pow <- data_test$ps * (1 - data_test$ps)
  
  # Calculate MSE between predictions of T-Learner and CATE
  res$T_MSE[i] <- mean((data_test$CATE - data_test$pred_CATE_T)^2)
  res$T_powMSE[i] <- sum(data_test$pow * (data_test$CATE - data_test$pred_CATE_T)^2) / sum(data_test$pow)
  res$T_psMSE[i] <- mean((data_test$CATE[data_test$a == 1] - data_test$pred_CATE_T[data_test$a == 1])^2)
  
  # Calculate MSE between predictions of IPW-Learner and CATE
  res$IPW_MSE[i] <- mean((data_test$CATE - data_test$pred_CATE_IPW)^2)
  res$IPW_powMSE[i] <- sum(data_test$pow * (data_test$CATE - data_test$pred_CATE_IPW)^2) / sum(data_test$pow)
  res$IPW_psMSE[i] <- mean((data_test$CATE[data_test$a == 1] - data_test$pred_CATE_IPW[data_test$a == 1])^2)
  
  # Calculate MSE between predictions of DR-Learner and CATE
  res$DR_MSE[i] <- mean((data_test$CATE - data_test$pred_CATE_DR)^2)
  res$DR_powMSE[i] <- sum(data_test$pow * (data_test$CATE - data_test$pred_CATE_DR)^2) / sum(data_test$pow)
  res$DR_psMSE[i] <- mean((data_test$CATE[data_test$a == 1] - data_test$pred_CATE_DR[data_test$a == 1])^2)
  
  # Calculate MSE between predictions of propensity-overlap-weighted DR-Learner and CATE
  res$R_MSE[i] <- mean((data_test$CATE - data_test$pred_CATE_R)^2)
  res$R_powMSE[i] <- sum(data_test$pow * (data_test$CATE - data_test$pred_CATE_R)^2) / sum(data_test$pow)
  res$R_psMSE[i] <- mean((data_test$CATE[data_test$a == 1] - data_test$pred_CATE_R[data_test$a == 1])^2)
  
  # Calculate MSE between predictions of propensity-score-weighted DR-Learner and CATE (in treated)
  res$psDR_psMSE[i] <- mean((data_test$CATE[data_test$a == 1] - data_test$pred_CATE_psDR[data_test$a == 1])^2)
  
  }

################################################################################
### MSE, MSE in treated and propensity-overlap-weighted MSE plots (computed on the test set)
################################################################################

res0 <- res[, 1:4]
names(res0) <- c("T", "IPW", "DR", "R")
res_pow <- res[, 5:8]
names(res_pow) <- c("T", "IPW", "DR", "R")
res_ps <- res[, 9:13]
names(res_ps) <- c("T", "IPW", "DR", "R", "psDR")

# Average MSE over all simulations
summary_res <- apply(res0, 2, mean, na.rm = TRUE)
print(summary_res)

# Average psMSE over all simulations
summary_res_pow <- apply(res_pow, 2, mean, na.rm = TRUE)
print(summary_res_pow)

# Average powMSE over all simulations
summary_res_ps <- apply(res_ps, 2, mean, na.rm = TRUE)
print(summary_res_ps)

# Make boxplots of mean-squared errors (MSE)
plot_MSE1 <- ggplot(data = melt(res0), aes(x = variable, y = value, color = variable)) + 
  geom_boxplot() +
  xlab("Method") +
  ylab("MSE") +
  scale_color_manual(values = c("#ca7dcc", "#1b98e0", "#353436", "#02e302"), name = "Method")

plot_MSE2 <- ggplot(data = melt(res0), aes(x = variable, y = value, color = variable)) +
  geom_boxplot() +
  xlab("Method") +
  ylab("MSE") +
  scale_color_manual(values = c("#ca7dcc", "#1b98e0", "#353436", "#02e302"), name = "Method") +
  coord_cartesian(ylim = c(0, 0.5))

# Make boxplots of propensity-overlap-weighted mean-squared errors (powMSE)
plot_powMSE1 <- ggplot(data = melt(res_pow), aes(x = variable, y = value, color = variable)) +
  geom_boxplot() +
  xlab("Method") +
  ylab("pow-MSE") +
  scale_color_manual(values = c("#ca7dcc", "#1b98e0", "#353436", "#02e302"), name = "Method")

plot_powMSE2 <- ggplot(data = melt(res_pow), aes(x = variable, y = value, color = variable)) +
  geom_boxplot() +
  xlab("Method") +
  ylab("pow-MSE") +
  scale_color_manual(values = c("#ca7dcc", "#1b98e0", "#353436", "#02e302"), name = "Method") +
  coord_cartesian(ylim = c(0, 0.5))

# Make boxplots of mean-squared errors in treated (psMSE)
plot_psMSE1 <- ggplot(data = melt(res_ps), aes(x = variable, y = value, color = variable)) +
  geom_boxplot() +
  xlab("Method") +
  scale_color_manual(values = c("#ca7dcc", "#1b98e0", "#353436", "#02e302", "red"), name = "Method") +
  ylab("MSE in treated")

plot_psMSE2 <- ggplot(data = melt(res_ps), aes(x = variable, y = value, color = variable)) +
  geom_boxplot() +
  xlab("Method") +
  ylab("MSE in treated") +
  scale_color_manual(values = c("#ca7dcc", "#1b98e0", "#353436", "#02e302", "red"), name = "Method") +
  coord_cartesian(ylim = c(0, 0.5))

figure_all <- ggarrange(plot_MSE1, plot_powMSE1, plot_psMSE1, plot_MSE2, plot_powMSE2, plot_psMSE2, 
                        labels = c("A", "B", "C", "D", "E", "F"), ncol = 3, nrow = 2)
figure_all
