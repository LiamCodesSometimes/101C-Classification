library(tidyverse)
library(tidymodels)
library(xgboost)
library(vip)

train <- read_csv("../Data/train_class.csv")
test <- read_csv("../Data/test_class.csv")

train <- train %>%
  select(-'name') %>%
#  mutate(across('x2013_code', as.factor)) %>%
  mutate_if(is.character, factor)

test <- test %>%
#  mutate(across('x2013_code', as.factor)) %>%
  mutate_if(is.character, factor)

# Set engine
xgb_spec <- boost_tree(trees = 1000,
                       tree_depth = tune(), min_n = tune(), 
                       loss_reduction = tune(), sample_size = tune(), 
                       mtry = tune(),learn_rate = tune()) %>% 
  set_engine("xgboost") %>%
  set_mode("classification")

# Create Tuning grid
xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  learn_rate(),
  finalize(mtry(), train),
  size = 10
)


# Create workflow
xgb_wf <- workflow() %>%
  add_formula(winner~.) %>%
  add_model(xgb_spec)

# Add Recipe
xgb_recipe_wf <- workflow() %>%
  add_recipe(xgb_recipe) %>%
  add_model(xgb_spec)

# Cross Validation
set.seed(101)
vb_folds <- vfold_cv(train, strata = winner)

set.seed(202)

# Execute tuning grid
xgb_res <- tune_grid(
  xgb_wf, # Change this line to desired workflow
  resamples = vb_folds,
  grid = xgb_grid,
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

# Obtain optimal parameters
show_best(xgb_res, metric = "roc_auc")
best_auc <- select_best(xgb_res, metric = "roc_auc")
final_xgb <- finalize_workflow(xgb_wf, best_auc)

# Fit to full training set
final_res <- final_xgb %>%
  fit(data = train)

# Create predictions
predictions <- final_res %>%
  predict(new_data = test)

pred_table_2 <- bind_cols(test %>% select(id), predictions)

write_csv(pred_table_2, "xgb_predictions_tuned.csv")

