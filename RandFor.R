library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(doParallel)
library(themis)
library(lubridate)
library(recipes)

numcores <- parallel::detectCores() #How many cores do I have?
cl <- makePSOCKcluster(numcores-1)
registerDoParallel(cl)


train_data = vroom('train.csv')
test_data = vroom('test.csv')

train_data <- train_data %>% mutate(`Open Date` = mdy(`Open Date`))
test_data<- test_data %>% mutate(`Open Date` = mdy(`Open Date`))

train_data$revenue <- log(train_data$revenue)

recipe <- recipe(revenue ~ ., data = train_data) %>%
  step_date('Open Date', features = c("year", "month", "dow"), role = "predictor") %>%
  step_rm(Id, City, 'City Group', Type, 'Open Date') %>%  # Remove unnecessary columns
  step_dummy(all_factor_predictors(), one_hot = TRUE) %>%  # One-hot encode all factor variables
  step_range(all_numeric_predictors(), min = 0, max = 1)


# recipe <- recipe(revenue ~ ., data=train_data) %>%
#   step_mutate(`Open Date` = mdy(`Open Date`)) %>%
#   step_rm(`Open Date`)  %>%
#   step_other(all_nominal_predictors(), threshold = 0.05) %>%  # Combine rare levels
#   step_dummy(all_nominal_predictors())

prepped_rec <- prep(recipe)
baked <- bake(prepped_rec, new_data = train_data)

rand_for <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

## Workflow
rf_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(rand_for) 

## Grid of values to tune over (rand forest)
grid_of_tuning_params_randfor <- grid_regular(mtry(range = c(1,100)),
                                              min_n(),
                                              levels = 5)

## Split data for CV
folds <- vfold_cv(train_data, v = 5, repeats = 1)

## Run the CV
CV_results <- rf_wf %>%
  tune_grid(
    resamples = folds,
    grid = grid_of_tuning_params_randfor,
    metrics = metric_set(rmse, rsq, mae)
  )


## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "rmse")

## Finalize the Workflow & fit it
final_wf <- rf_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data)

#make predictions
preds_log <- predict(final_wf, new_data=test_data, type = 'numeric')
preds <- exp(preds_log)

## Format the Predictions for Submission to Kaggle
kaggle_submission <-  preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(Id, .pred) %>% #Just keep datetime and prediction variables
  rename(Prediction=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(Prediction=pmax(0, Prediction)) #pointwise max of (0, prediction)

sum(is.na(kaggle_submission))
## Write out file
vroom_write(x=kaggle_submission, file="./RandFor7.csv", delim=",")




stopCluster(cl)












