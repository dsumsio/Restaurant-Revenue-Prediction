library(tidyverse)
library(tidymodels)
library(vroom)
library(corrplot)
library(dplyr)

#  1,734,525

## Reading in the Data
traindata <- vroom("train.csv")
testdata <- vroom("test.csv")

## Define a recipe
recipe <- recipe(revenue~., data = traindata) %>%
  step_rm(Id, 'Open Date', City, 'City Group', Type) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)
  #step_mutate(Type = fct_collapse(Type, Other = c("MB"))) # %>%
  #step_mutate('City Group' = as.factor('City Group'))


## Define a Model
lin_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

## Combine into a Workflow and fit
workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(lin_model) %>%
  fit(data=traindata)

## Run all the steps on test data and change back to normal data
lin_preds <- predict(workflow, new_data = testdata)

## Format the Predictions for Submission to Kaggle
kaggle_submission <-  lin_preds %>%
  bind_cols(., testdata) %>% #Bind predictions with test data
  select(Id, .pred) %>% #Just keep datetime and prediction variables
  rename(Prediction=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(Prediction=pmax(0, Prediction)) #pointwise max of (0, prediction)
  
sum(is.na(kaggle_submission))
## Write out file
vroom_write(x=kaggle_submission, file="./LinearPreds2.csv", delim=",")






setdiff(names(traindata), names(testdata))
setdiff(names(testdata), names(traindata))

ncol(testdata)
ncol(traindata)




library(dplyr)

# Example tibble of results
# results <- tibble(Id = ..., prediction = ...)

# Select rows with NA in prediction
na_rows <- kaggle_submission %>%
  filter(is.na(Prediction))

# Filter rows in traindata based on Id
traindata_na <- traindata %>%
  filter(Id %in% na_rows$Id)

# Filter rows in testdata based on Id
testdata_na <- testdata %>%
  filter(Id %in% na_rows$Id)

# Display results
print(na_rows)
print(traindata_na)
print(testdata_na)

