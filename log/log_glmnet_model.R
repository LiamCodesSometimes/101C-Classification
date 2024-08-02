library(tidyverse)
library(tidymodels)
library(glmnet)
tidymodels_prefer()

train <- read_csv("../Data/train_class.csv")
test <- read_csv("../Data/test_class.csv")

train$winner <- as.factor(train$winner)
train <- train %>%
  select(-name)

set.seed(101)
train_folds <- vfold_cv(train, v = 10, strata = winner)

log_model <- logistic_reg() %>%
  set_engine("glmnet")

voting_recipe <- recipe(winner ~ ., data = train) %>%
  step_impute_bag(starts_with("income"), starts_with("gdp")) %>%
  step_mutate(male_prop = x0002e / x0001e,
              female_prop = x0003e / x0001e,
              black_prop = x0038e / x0001e,
              white_prop = x0037e / x0001e,
              native_prop = x0039e / x0001e,
              asian_prop = x0044e / x0001e,
              pacific_prop = x0052e / x0001e,
              other_prop = x0057e / x0001e,
              hispanic_prop = x0071e / x0001e,
              lessHS_prop = (c01_002e + c01_007e + c01_008e) / (c01_001e + c01_006e),
              gradHS_prop = (c01_003e +c01_009e) / (c01_001e + c01_006e),
              someAssoc_prop = (c01_004e +c01_010e) / (c01_001e + c01_006e),
              bachUp_prop = (c01_005e +c01_015e) / (c01_001e + c01_006e),
              under18_prop = x0019e / x0001e,
              over18_prop = x0025e / x0001e
  ) 


log_reg <- logistic_reg(mixture = tune(), penalty = tune()) %>%
  set_engine("glmnet")

grid <- grid_regular(mixture(), penalty(), levels = c(mixture = 4, penalty = 4))

log_reg_wf <- workflow() %>%
  add_model(log_reg) %>%
  add_recipe(voting_recipe)

log_reg_tuned <- tune_grid(
  log_reg_wf,
  resamples = train_folds,
  grid = grid,
  control = control_grid(save_pred = TRUE)
)

best_acc <- select_best(log_reg_tuned, metric = "accuracy")


log_reg_final <- finalize_model(log_reg, best_acc)

tuned_workflow <- workflow() %>%
  add_recipe(voting_recipe) %>%
  add_model(log_reg_final)

tuned_res <- tuned_workflow %>%
  fit_resamples(resamples = train_folds)

collect_metrics(tuned_res)


tuned_fit <- tuned_workflow %>%
  fit(data = train)

predictions <- tuned_fit %>%
  predict(new_data = test)

pred_table <- bind_cols(test %>% select(id), predictions) %>%
  rename(winner = .pred_class)