----------------------------#Loading Library----
#install.packages("ISLR")
#install.packages("FNN")

library(ISLR)
library(caret)
library(class)
library(FNN) 
library(dplyr)
library(ggplot2)
library(rBayesianOptimization)


----------------------------#Training Model----
#Load the necessary libraries

# Step 1: Select predictor columns, excluding "Married" (now the target variable)
Credit_subset <- Credit[, c("Income","Rating")]

# Convert categorical predictors to indicator variables
dummies <- dummyVars(~ ., data = Credit_subset)
Credit_dummies <- predict(dummies, newdata = Credit_subset)

# Step 2: Normalize the predictors
preProcValues <- preProcess(Credit_dummies, method = c("center", "scale"))   #("range") = Min-max scaler, ("center", "scale") = Standard scaler
Credit_normalized <- as.data.frame(predict(preProcValues, Credit_dummies))

# Add "Married" as the target variable to the normalized data
Credit_normalized$Married <- Credit$Married

# Step 3: Train-test split
set.seed(123)
trainIndex <- createDataPartition(Credit_normalized$Married, p = 0.7, list = FALSE)
trainData <- Credit_normalized[trainIndex, ]
testData <- Credit_normalized[-trainIndex, ]

# Step 4: Implement k-NN
# Prepare predictors and labels for k-NN
train_predictors <- trainData[, -which(names(trainData) == "Married")]
test_predictors <- testData[, -which(names(testData) == "Married")]
train_labels <- trainData$Married
test_labels <- testData$Married

----------------------------#Apply k-NN with k----
knn_pred <- knn(train = train_predictors, 
                test = test_predictors, 
                cl = train_labels, 
                k = 5,
                p = 2)  # 1:Manhattan, 2: Euclidan, inf:Chebyshev distance)

# Step 5: Evaluate the model accuracy
confusionMatrix(knn_pred, test_labels)

# Convert predictions and actual values to a data frame for visualization
results <- data.frame(Actual = test_labels, Predicted = knn_pred)

# Bar Plot of Predicted vs. Actual Counts
# Add counts for each category for better visualization
results_count <- results %>%
  count(Actual, Predicted)

ggplot(results_count, aes(x = Actual, y = n, fill = Predicted)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Bar Plot of Predicted vs Actual Married Categories",
       x = "Actual Married Category",
       y = "Count") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set2")


----------------------------#Hyperparameter Tuning K-NN-------------------------------
#Apply K-NN with grid-search----
# Define a custom trainControl for cross-validation
control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation

# Define the grid for `k` values
grid <- expand.grid(k = seq(1, 20, by = 2))  # Odd values of k from 1 to 19

# Train the k-NN model with grid search
set.seed(123)
knn_model <- train(Married ~ ., 
                   data = trainData, 
                   method = "knn", 
                   trControl = control, 
                   tuneGrid = grid)

# Print the best k value and results
print(knn_model)

# Make predictions using the optimized k-NN model
optimized_knn_pred <- predict(knn_model, newdata = testData)

# Evaluate the model
conf_matrix <- confusionMatrix(optimized_knn_pred, testData$Married)
print(conf_matrix)

# Step 6: Visualization of results
results <- data.frame(Actual = testData$Married, Predicted = optimized_knn_pred)

# Count predictions and actual values for visualization
results_count <- results %>%
  count(Actual, Predicted)

ggplot(results_count, aes(x = Actual, y = n, fill = Predicted)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Bar Plot of Predicted vs Actual Married Categories",
       x = "Actual Married Category",
       y = "Count") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set2")

#Apply K-NN with random-search----
# Define random search
control <- trainControl(method = "cv", number = 10, search = "random")  # 10-fold CV with random search

# Define a range for k values to sample randomly
set.seed(123)
knn_model_random <- train(Married ~ ., 
                          data = trainData, 
                          method = "knn", 
                          trControl = control, 
                          tuneLength = 10)  # Randomly samples 10 different k values

# Print the best k value from random search
best_k_random <- knn_model_random$bestTune$k
cat("The best k value from random search is:", best_k_random, "\n")

# Evaluate the optimized model on the test set
optimized_knn_pred_random <- predict(knn_model_random, newdata = testData)

# Evaluate the model
conf_matrix_random <- confusionMatrix(optimized_knn_pred_random, testData$Married)
print(conf_matrix_random)
#Apply K-NN with bayesian-search----

library(rBayesianOptimization)

#Define the objective function
simple_function <- function(k) {
  k <- round(k)  # Ensure k is an integer
  if (k < 1) return(list(Score = -Inf))  # Reject invalid k values
  
  # Objective: maximize negative quadratic function
  score <- -1 * (k - 10)^2
  return(list(Score = score))
}

# Run Bayesian Optimization
set.seed(123)  # For reproducibility
opt_results <- BayesianOptimization(
  FUN = simple_function,          # Objective function
  bounds = list(k = c(1, 20)),    # Search space for k
  init_points = 5,                # Number of initial random points
  n_iter = 10,                    # Number of iterations
  acq = "ucb",                    # Acquisition function: "ucb", "ei", or "poi"
  kappa = 2.576,                  # Exploration-exploitation tradeoff for UCB
  eps = 0.0                       # Exploration parameter
)

# Print the best parameter
print(opt_results$Best_Par)

# Extract the best parameter and round it to the nearest integer
best_k <- round(opt_results$Best_Par["k"])

knn_best_model <- knn(
  train = trainData[, -ncol(trainData)],
  test = testData[, -ncol(testData)],
  cl = trainData$Married,
  k = best_k
)

# Confusion matrix and accuracy
conf_matrix <- confusionMatrix(knn_best_model, testData$Married)
print(conf_matrix)