
#################################################
# R-code to replicate the simulation set-up 2 from the paper
# "On a General Class of Orthogonal Learners for the Estimation of Heterogeneous Treatment Effects"
#################################################

library(ggpubr)
library(ggplot2)
library(reshape2)
library(SuperLearner)

# Set initial parameters (number of individuals in each of the data sets: train A, train B and test, and number of simulations)
n <- 500
nsim <- 2

p = 20 # Dimension of the covariates X
sigma = 0.5

# Select list of models to be included in the SuperLearner
SL.library = c("SL.glm", "SL.ranger", "SL.glmnet", "SL.xgboost")

# Simulation set-up 2 is based on the simulation set-up A from (Nie and Wager, 2021)
get.params = function() {
  X = matrix(runif(n * p, min = 0, max = 1), n, p)
  b = sin(pi * X[, 1] * X[, 2]) + 2 * (X[, 3] - 0.5) ^ 2 + X[, 4] + 0.5 * X[, 5] + X[, 6]
  eta = 0.1
  e = pmax(eta, pmin(sin(pi * X[, 1] * X[, 2] * X[, 3] * X[, 4]), 1 - eta))
  tau = (X[, 1] + X[, 2] + X[, 3]) / 2
  list(X = X, b = b, tau = tau, e = e)
}

make_matrix = function(x)
  stats::model.matrix( ~ . - 1, x)

# Data frame for the MSE of each method. "T", "IPW", "DR", "R" and "psDR" mean T-, IPW-, DR-, R- and propensity-score-weighted DR-Learner, respectively. "MSE"
# is mean-squared error, "psMSE" is mean-squared error in treated and "powMSE" is propensity-overlap-weighted mean-squared error.

res <- data.frame(matrix(nrow = nsim, ncol = 13))
colnames(res) <- c("T_MSE", "IPW_MSE", "DR_MSE", "R_MSE", "T_powMSE", "IPW_powMSE", "DR_powMSE", "R_powMSE", "T_psMSE", "IPW_psMSE", "DR_psMSE", "R_psMSE", "psDR_psMSE")

for (i in 1:nsim) {
  
  #####################################
  # Generate the training data sets A and B, and the test data set.
  # Y is the outcome, X are covariates and A is the exposure.
  #####################################
  
  # Generate the training data set A
  params_train_A = get.params()
  A_train_A = rbinom(n, 1, params_train_A$e)
  Y1_train_A = params_train_A$b + (1 - 0.5) * params_train_A$tau + sigma * rnorm(n)
  Y0_train_A = params_train_A$b + (0 - 0.5) * params_train_A$tau + sigma * rnorm(n)
  Y_train_A = A_train_A * Y1_train_A + (1 - A_train_A) * Y0_train_A
  
  # Generate the training data set B
  params_train_B = get.params()
  A_train_B = rbinom(n, 1, params_train_B$e)
  Y1_train_B = params_train_B$b + (1 - 0.5) * params_train_B$tau + sigma * rnorm(n)
  Y0_train_B = params_train_B$b + (0 - 0.5) * params_train_B$tau + sigma * rnorm(n)
  Y_train_B = A_train_B * Y1_train_B + (1 - A_train_B) * Y0_train_B
  
  # Generate the test data set
  params_test = get.params()
  A_test = rbinom(n, 1, params_test$e)
  Y1_test = params_test$b + (1 - 0.5) * params_test$tau + sigma * rnorm(n)
  Y0_test = params_test$b + (0 - 0.5) * params_test$tau + sigma * rnorm(n)
  Y_test = A_test * Y1_test + (1 - A_test) * Y0_test
  
  # Set of covariates in the training data sets A and B, and in the test set
  X_train_A = data.frame(params_train_A$X)
  X_train_B = data.frame(params_train_B$X)
  X_test = data.frame(params_test$X)
  
  # Combine the covariates and the exposure into a single data frame
  Z_train_A <- cbind(A_train_A, X_train_A)
  Z_train_B <- cbind(A_train_B, X_train_B)
  Z_test <- cbind(A_test, X_test)
  
  colnames(Z_train_A)[1] <- "a"
  colnames(Z_train_B)[1] <- "a"
  colnames(Z_test)[1] <- "a"
  
  data_train_A <- cbind(Y_train_A, Y1_train_A, Y0_train_A, A_train_A, X_train_A)
  data_train_B <- cbind(Y_train_B, Y1_train_B, Y0_train_B, A_train_B, X_train_B)
  data_test <- cbind(Y_test, Y1_test, Y0_test, A_test, X_test)
  
  colnames(data_train_A)[1:4] <- c("y", "y1", "y0", "a")
  colnames(data_train_B)[1:4] <- c("y", "y1", "y0", "a")
  colnames(data_test)[1:4] <- c("y", "y1", "y0", "a")
  
  # True propensity score in the test set (used for calculating "psMSE")
  data_test$ps <- params_test$e
  
  # True CATE
  data_train_A$CATE <- params_train_A$tau
  data_train_B$CATE <- params_train_B$tau
  data_test$CATE <- params_test$tau
  
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
  
  # Construct data sets were everyone/no one is treated
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
  
  # R-Learner (Propensity-overlap-weighted DR-Learner)
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
                                          X = X_train_AB[data_train_AB$a == 1,],
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
  data_test$pred_CATE_psDR[data_test$a == 1] <- as.vector(predict(model_pseudooutcome_psDR, newdata = X_test[data_test$a == 1,], onlySL = TRUE)$pred)
  
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
