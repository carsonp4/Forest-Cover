# Loading Packages
library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(rpart)
library(ranger)
library(stacks)
library(embed)
library(discrim)
library(naivebayes)
library(kknn)
library(kernlab)
library(themis)
library(keras)
library(bonsai)
library(lightgbm)
library(dbarts)

# Reading in Data
setwd("~/Desktop/Stat348/Forest-Cover/")
train <- vroom("train.csv")
test <- vroom("test.csv")
train$Cover_Type <- as.factor(train$Cover_Type)

my_recipe <- recipe(Cover_Type ~ ., data = train) %>%
  update_role(Id, new_role = "ID") %>%
  #step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  #step_other(all_factor_predictors(), threshold = .005) %>% # combines categorical values that occur <5% into an "other" value
  #step_dummy(all_nominal_predictors()) # dummy variable encoding
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(Cover_Type))  #target encoding


bake(prep(my_recipe), new_data = train)


# Random Forest -----------------------------------------------------------

rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

rf_tuning_grid <- grid_regular(mtry(c(1,ncol(train))), min_n(), levels=10)

folds <- vfold_cv(train, v = 3, repeats = 1)

tune_control <- control_grid(verbose = TRUE)

rf_results <- rf_wf %>% 
  tune_grid(resamples = folds,
            grid = rf_tuning_grid,
            metrics = metric_set(accuracy),
            control = tune_control)

rf_bestTune <- rf_results %>% 
  select_best("accuracy")

rf_final_wf <- rf_wf %>% 
  finalize_workflow(rf_bestTune) %>% 
  fit(data=train)

rf_preds <- predict(rf_final_wf,
                    new_data=test,
                    type="class")

rf_submit <- as.data.frame(cbind(test$id, as.character(rf_preds$.pred_class)))
colnames(rf_submit) <- c("id", "type")
write_csv(rf_submit, "rf_submit.csv")




# Logistic Regression -----------------------------------------------------

log_mod <- multinom_reg() %>% #Type of model
  set_engine("nnet")

log_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(log_mod) %>%
  fit(data = train) # Fit the workflow

log_preds <- predict(log_wf,
                     new_data=test,
                     type="class") # "class" or "prob" (see doc)

log_submit <- as.data.frame(cbind(as.integer(test$Id), as.character(log_preds$.pred_class)))
colnames(log_submit) <- c("Id", "Cover_Type")
write_csv(log_submit, "log_submit.csv")

# Naive Bayes -------------------------------------------------------------

nb_mod <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>% 
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(nb_mod)

nb_tuning_grid <- grid_regular(Laplace(),smoothness(), levels = 10)

folds <- vfold_cv(train, v = 5, repeats=1)

tune_control <- control_grid(verbose = TRUE)

nb_results <- nb_wf %>% 
  tune_grid(resamples = folds,
            grid = nb_tuning_grid,
            metrics = metric_set(accuracy),
            control = tune_control)

nb_bestTune <- nb_results %>% 
  select_best("accuracy")

nb_final_wf <- nb_wf %>% 
  finalize_workflow(nb_bestTune) %>% 
  fit(data=train)

nb_preds <- predict(nb_final_wf,
                    new_data=test,
                    type="class")

nb_submit <-  as.data.frame(cbind(as.integer(test$Id), as.character(nb_preds$.pred_class)))
colnames(nb_submit) <- c("Id", "Cover_Type")
write_csv(nb_submit, "nb_submit.csv")

# KNN ---------------------------------------------------------------------

knn_mod <- nearest_neighbor(neighbors = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")

knn_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(knn_mod)

knn_tuning_grid <- grid_regular(neighbors(),levels = 30)
folds <- vfold_cv(train, v = 5, repeats=1)
tune_control <- control_grid(verbose = TRUE)

knn_results <- knn_wf %>% 
  tune_grid(resamples = folds,
            grid = knn_tuning_grid,
            metrics = metric_set(accuracy),
            control = tune_control)

knn_bestTune <- knn_results %>% 
  select_best("accuracy")

knn_final_wf <- knn_wf %>% 
  finalize_workflow(knn_bestTune) %>% 
  fit(data=train)

knn_preds <- predict(knn_final_wf,
                     new_data=test,
                     type="class")

knn_submit <- as.data.frame(cbind(as.integer(test$Id), as.character(knn_preds$.pred_class)))
colnames(knn_submit) <- c("Id", "Cover_Type")
write_csv(knn_submit, "knn_submit.csv")

# SVM ---------------------------------------------------------------------

svm_mod <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% 
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(svm_mod)

svm_tuning_grid <- grid_regular(rbf_sigma(), cost(), levels = 30)

folds <- vfold_cv(train, v = 5, repeats=1)

tune_control <- control_grid(verbose = TRUE)

svm_results <- svm_wf %>% 
  tune_grid(resamples = folds,
            grid = svm_tuning_grid,
            metrics = metric_set(accuracy),
            control = tune_control)

svm_bestTune <- svm_results %>% 
  select_best("accuracy")

svm_final_wf <- svm_wf %>% 
  finalize_workflow(svm_bestTune) %>% 
  fit(data=train)

svm_preds <- predict(svm_final_wf,
                     new_data=test,
                     type="class")

svm_submit <- as.data.frame(cbind(as.integer(test$Id), as.character(svm_preds$.pred_class)))
colnames(svm_submit) <- c("Id", "Cover_Type")
write_csv(svm_submit, "svm_submit.csv")

# Boosted -----------------------------------------------------------------

boost_mod <- boost_tree(tree_depth = tune(),
                        trees = 1000,
                        learn_rate = tune(),
                        min_n = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

boost_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(boost_mod)

boost_tuning_grid <- grid_regular(tree_depth(), learn_rate(), min_n(), levels = 30)

folds <- vfold_cv(train, v = 5, repeats=1)

boost_results <- boost_wf %>% 
  tune_grid(resamples = folds,
            grid = boost_tuning_grid,
            metrics = metric_set(accuracy),
            control = tune::control_grid(verbose = TRUE))

boost_bestTune <- boost_results %>% 
  select_best("accuracy")

boost_final_wf <- boost_wf %>% 
  finalize_workflow(boost_bestTune) %>% 
  fit(data=train)

boost_preds <- predict(boost_final_wf,
                       new_data=test,
                       type="class")

boost_submit <- as.data.frame(cbind(as.integer(test$Id), as.character(boost_preds$.pred_class)))
colnames(boost_submit) <- c("Id", "Cover_Type")
write_csv(boost_submit, "boost_submit.csv")