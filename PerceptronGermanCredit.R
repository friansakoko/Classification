# Load necessary libraries
library(caret)
library(ggplot2)
data("GermanCredit")
GermanCredit$Class <- factor(GermanCredit$Class, levels = c("Good", "Bad"))

# Data Preparation----
GermanCredit_subset <- GermanCredit[, c("Age", "Amount", "Class")]
GermanCredit_subset <- GermanCredit_subset[complete.cases(GermanCredit_subset), ]

# Normalize features
normalize <- function(x) (x - min(x)) / (max(x) - min(x))
GermanCredit_subset$Age <- normalize(GermanCredit_subset$Age)
GermanCredit_subset$Amount <- normalize(GermanCredit_subset$Amount)

# Split data
set.seed(123)
train_index <- createDataPartition(GermanCredit_subset$Class, p = 0.8, list = FALSE)
GermanCredit_Train <- GermanCredit_subset[train_index, ]
GermanCredit_Test <- GermanCredit_subset[-train_index, ]

GermanCredit_Train$Class <- ifelse(GermanCredit_Train$Class == "Good", 1, 0)
GermanCredit_Test$Class <- ifelse(GermanCredit_Test$Class == "Good", 1, 0)

# Train-Test----
X <- as.matrix(GermanCredit_Train[, c("Age", "Amount")])
y <- GermanCredit_Train$Class
X_test <- as.matrix(GermanCredit_Test[, c("Age", "Amount")])
y_test <- GermanCredit_Test$Class

# Perceptron Functions
act_func <- function(x) {
  ifelse(x >= 0, 1, 0)
}

train_perceptron <- function(X, y, lr = 0.1, epochs = 500) {
  weights <- runif(ncol(X))
  bias <- runif(1)
  
  for (epoch in 1:epochs) {
    for (i in 1:nrow(X)) {
      linear_output <- sum(X[i, ] * weights) + bias
      prediction <- act_func(linear_output)
      error <- y[i] - prediction
      weights <- weights + lr * error * X[i, ]
      bias <- bias + lr * error
    }
    if (epoch %% 50 == 0) {
      cat("Epoch:", epoch, "Weights:", weights, "Bias:", bias, "\n")
    }
  }
  list(weights = weights, bias = bias)
}

predict_perceptron <- function(model, X) {
  linear_output <- X %*% model$weights + model$bias
  act_func(linear_output)
}

# Test and Evaluate
set.seed(123)
model <- train_perceptron(X, y, lr = 0.1, epochs = 500)
perc_predictions <- predict_perceptron(model, X_test)

library(caret)
conf_matrix <- confusionMatrix(
  factor(perc_predictions, levels = c(0, 1)),
  factor(y_test, levels = c(0, 1))
)
print(conf_matrix)

# Visualize heatmap----
library(reshape2)

# Generate the confusion matrix (example using caret's confusionMatrix)
conf_matrix <- confusionMatrix(factor(perc_predictions), factor(GermanCredit_Test$Class))

# Convert confusion matrix to a data frame
conf_matrix_df <- as.data.frame(conf_matrix$table)

# Rename columns for clarity
colnames(conf_matrix_df) <- c("Actual", "Predicted", "Freq")

# Create a heatmap using ggplot2
ggplot(data = conf_matrix_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 5) + # Add text for frequency
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix Heatmap",
       x = "Predicted Class",
       y = "Actual Class",
       fill = "Frequency") +
  theme_minimal()


# Visualize the prediction test (Normalize)----

# Ensure predicted class numbers are converted to labels
GermanCredit_Test$Predicted <- ifelse(perc_predictions == 1, "Good", "Bad")
ggplot(GermanCredit_Test, aes(x = Age, y = Amount, color = as.factor(Predicted))) +
  geom_point(size = 3) +
  labs(title = "Perceptron Predictions (Normalized Scale)", color = "Predicted Class") +
  theme_minimal()

# Visualize the prediction test----

age_min <- min(GermanCredit[, "Age"], na.rm = TRUE)
age_max <- max(GermanCredit[, "Age"], na.rm = TRUE)
amount_min <- min(GermanCredit[, "Amount"], na.rm = TRUE)
amount_max <- max(GermanCredit[, "Amount"], na.rm = TRUE)

# Function to inverse normalize
inverse_normalize <- function(x, min_val, max_val) {
  x * (max_val - min_val) + min_val
}

# Apply inverse normalization to the test data
GermanCredit_Test$Age_Original <- inverse_normalize(GermanCredit_Test$Age, age_min, age_max)
GermanCredit_Test$Amount_Original <- inverse_normalize(GermanCredit_Test$Amount, amount_min, amount_max)

# Visualize predictions with original scales
ggplot(GermanCredit_Test, aes(x = Age_Original, y = Amount_Original, color = Predicted)) +
  geom_point(size = 3) +
  labs(
    title = "Perceptron Predictions (Original Scale)",
    x = "Age (Original Scale)",
    y = "Amount (Original Scale)",
    color = "Predicted Class"
  ) +
  theme_minimal()

# Hyperparameter: Grid-Search to find the best lr and epoch----
# Grid Search Function
grid_search_perceptron <- function(X_train, y_train, X_val, y_val, lr_values, epoch_values) {
  results <- data.frame(lr = numeric(), epochs = integer(), accuracy = numeric())
  
  for (lr in lr_values) {
    for (epochs in epoch_values) {
      # Train perceptron
      model <- train_perceptron(X_train, y_train, lr = lr, epochs = epochs)
      
      # Predict on validation set
      predictions <- predict_perceptron(model, X_val)
      
      # Evaluate performance
      accuracy <- mean(predictions == y_val)  # Accuracy metric
      
      # Store results
      results <- rbind(results, data.frame(lr = lr, epochs = epochs, accuracy = accuracy))
    }
  }
  
  # Return sorted results
  results[order(-results$accuracy), ]
}

# Define hyperparameter grid
lr_values <- c(0.01, 0.05, 0.1, 0.5, 1)  # Learning rate values to try
epoch_values <- c(100, 200, 500, 1000)   # Epoch values to try

# Split data into training and validation sets
set.seed(123)
val_index <- createDataPartition(GermanCredit_Train$Class, p = 0.2, list = FALSE)
X_train <- as.matrix(GermanCredit_Train[-val_index, c("Age", "Amount")])
y_train <- GermanCredit_Train$Class[-val_index]
X_val <- as.matrix(GermanCredit_Train[val_index, c("Age", "Amount")])
y_val <- GermanCredit_Train$Class[val_index]

# Perform grid search
grid_results <- grid_search_perceptron(X_train, 
                                       y_train, 
                                       X_val, 
                                       y_val, 
                                       lr_values, 
                                       epoch_values)

# Display best parameters
print(grid_results[1, ])  # Best combination

ggplot(grid_results, aes(x = lr, y = accuracy, color = as.factor(epochs))) +
  geom_point(size = 3) +
  labs(title = "Grid Search Results", color = "Epochs") +
  theme_minimal()

# Hyperparameter: Random-Search to find the best lr and epoch----
random_search_perceptron <- function(X_train, y_train, X_val, y_val, n_trials) {
  results <- data.frame(lr = numeric(), epochs = integer(), accuracy = numeric())
  
  for (trial in 1:n_trials) {
    # Randomly sample hyperparameters
    lr <- runif(1, min = 0.001, max = 1)  # Random learning rate
    epochs <- sample(100:1000, 1)         # Random number of epochs
    
    # Train perceptron
    model <- train_perceptron(X_train, y_train, lr = lr, epochs = epochs)
    
    # Predict on validation set
    predictions <- predict_perceptron(model, X_val)
    
    # Evaluate performance
    accuracy <- mean(predictions == y_val)  # Accuracy metric
    
    # Store results
    results <- rbind(results, data.frame(lr = lr, 
                                         epochs = epochs, 
                                         accuracy = accuracy))
  }
  
  # Return sorted results
  results[order(-results$accuracy), ]
}

# Split data into training and validation sets
set.seed(123)
val_index <- createDataPartition(GermanCredit_Train$Class, 
                                 p = 0.2, 
                                 list = FALSE)
X_train <- as.matrix(GermanCredit_Train[-val_index, c("Age", "Amount")])
y_train <- GermanCredit_Train$Class[-val_index]
X_val <- as.matrix(GermanCredit_Train[val_index, c("Age", "Amount")])
y_val <- GermanCredit_Train$Class[val_index]

# Perform random search
set.seed(123)
n_trials <- 20  # Number of random configurations to test
random_results <- random_search_perceptron(X_train, 
                                           y_train, 
                                           X_val, 
                                           y_val, 
                                           n_trials)

# Display best parameters
print(random_results[1, ])  # Best combination

ggplot(random_results, aes(x = lr, y = accuracy, color = as.factor(epochs))) +
  geom_point(size = 3) +
  labs(title = "Random Search Results", color = "Epochs") +
  theme_minimal()

