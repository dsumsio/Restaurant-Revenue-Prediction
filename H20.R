library(h2o)
library(tidyverse)
library(vroom)

traindata <- vroom("train.csv")
testdata <- vroom("test.csv")

traindata <- traindata %>% select(-'Open Date', -City, -'City Group', -Type)
testdata <- testdata %>% select(-'Open Date', -City, -'City Group', -Type)

# Initialize H2O
h2o.init()

traindata_h2o <- as.h2o(traindata)
testdata_h2o <- as.h2o(testdata)

# Define target and features
target <- "revenue"
features <- setdiff(names(traindata_h2o), target)

# Train a model using H2O AutoML (works for regression tasks as well)
model <- h2o.automl(
  y = target,
  x = features,
  training_frame = traindata_h2o,
  max_models = 50,  # Increase the number of models trained
  max_runtime_secs = 600,  # Allow more training time (in seconds)
  seed = 1234  # For reproducibility
)

# View leaderboard of models
model@leaderboard

# Make predictions on test data
predictions <- h2o.predict(model, newdata = testdata_h2o)
preds <- as.data.frame(predictions)

kaggle_submission <-  preds %>%
  bind_cols(., testdata) %>% #Bind predictions with test data
  select(Id, predict) %>% #Just keep datetime and prediction variables
  rename(Prediction=predict) %>% #rename pred to count (for submission to Kaggle)
  mutate(Prediction=pmax(0, Prediction)) #pointwise max of (0, prediction)

sum(is.na(kaggle_submission))
## Write out file
vroom_write(x=kaggle_submission, file="./H20_2.csv", delim=",")
