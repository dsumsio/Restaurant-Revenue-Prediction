library(vroom)
library(tidymodels)
library(tidyverse)

train_data <- vroom('train.csv')
test_data <- vroom('test.csv')

head(train_data)
summary(train_data)


recipe <- recipe(revenue~., data = train_data) %>%
  step_rm(Id, 'Open Date', City, 'City Group', Type)
  

prepped_rec <- prep(recipe)
baked <- bake(prepped_rec, new_data = train_data)

## knn model
knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_mode('regression') %>%
  set_engine('kknn')

# CV
## Grid of values to tune
tuning_grid <- grid_regular(neighbors(),
                            levels = 10)

## Split data for CV
folds <- vfold_cv(train_data, v = 5, repeats = 1)


knn_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(knn_model)

## Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid)

## Find best tuning params
bestTune <- CV_results %>%
  select_best(metric = 'rmse')

## Findlize the workflow & fit
final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

#make predictions
preds <- predict(final_wf, new_data=test_data, type = 'numeric')

## Format the Predictions for Submission to Kaggle
kaggle_submission <-  preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(Id, .pred) %>% #Just keep datetime and prediction variables
  rename(Prediction=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(Prediction=pmax(0, Prediction)) #pointwise max of (0, prediction)

sum(is.na(kaggle_submission))
## Write out file
vroom_write(x=kaggle_submission, file="./knn1.csv", delim=",")


