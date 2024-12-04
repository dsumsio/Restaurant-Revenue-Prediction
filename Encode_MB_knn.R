library(caret)  # For kNN model
library(dplyr)
library(vroom)

train_data = vroom('train.csv')
test_data = vroom('test.csv')

# Step 1: Prepare the Training Data
# Convert 'Type' to a factor
train_data <- train_data %>%
  mutate(Type = as.factor(Type))

# Step 2: Identify Rows with 'MB' in Test Data
test_data_with_mb <- test_data %>%
  filter(Type == "MB")

test_data_without_mb <- test_data %>%
  filter(Type != "MB")

# Step 3: Prepare Data for kNN
# Combine training data and test_data_without_mb for consistent encoding
# Select only numeric columns
combined_data <- bind_rows(
  train_data %>% select_if(is.numeric) %>% bind_cols(select(train_data, Type)) %>% select(-revenue),  # Add Type from train_data
  test_data_without_mb %>% select_if(is.numeric) %>% bind_cols(select(test_data_without_mb, Type))  # Add Type from test_data_without_mb
)

# Ensure numeric variables are scaled for kNN
preprocess <- preProcess(combined_data, method = c("center", "scale"))
encoded_data <- predict(preprocess, combined_data)

# Step 4: Train kNN Model
# Use the training data portion for kNN
knn_model <- train(
  Type ~ ., 
  data = cbind(
    Type = train_data$Type, 
    encoded_data[1:nrow(train_data), ]
  ),
  method = "knn",
  tuneGrid = data.frame(k = 5)  # 5 nearest neighbors
)

# Step 5: Predict 'MB' Type in Test Data
# Predict using kNN model for rows with 'MB' in test data
mb_predictions <- predict(
  knn_model, 
  newdata = encoded_data[(nrow(train_data) + 1):nrow(encoded_data), ]
)
test_data_with_mb_encoded <- predict(preprocess, test_data_with_mb)
mb_predictions <- predict(knn_model, newdata = test_data_with_mb_encoded)
# Step 6: Replace 'MB' with Predicted Levels
test_data_with_mb <- test_data_with_mb %>%
  mutate(Type = mb_predictions)

# Combine Updated Test Data
final_test_data <- bind_rows(test_data_with_mb, test_data_without_mb)

final <- final_test_data %>%
  arrange(Id)

vroom_write(x=final, file="./test_knn.csv", delim=",")
