# Load and preprocess the data
library(caret)
data("GermanCredit")
View(GermanCredit)
GermanCredit$Class <- factor(GermanCredit$Class, levels = c("Good", "Bad"))

# Data Preparation----
# Select only numeric predictors and the target
GermanCredit_subset <- GermanCredit[, c("Age", "Amount","Class")]
GermanCredit_subset <- GermanCredit_subset[complete.cases(GermanCredit_subset), ]

# Normalize features
normalize <- function(x) (x - min(x)) / (max(x) - min(x))
GermanCredit_subset$Age <- normalize(GermanCredit_subset$Age)
GermanCredit_subset$Amount <- normalize(GermanCredit_subset$Amount)

# Split data into training and testing sets
set.seed(123)
train_index <- createDataPartition(GermanCredit_subset$Class, p = 0.8, list = FALSE)
GermanCredit_Train <- GermanCredit_subset[train_index, ]
GermanCredit_Test <- GermanCredit_subset[-train_index, ]

# Convert target variable to binary (Perceptron requires numeric output)
GermanCredit_Train$Class <- ifelse(GermanCredit_Train$Class == "Good", 1, 0)
GermanCredit_Test$Class <- ifelse(GermanCredit_Test$Class == "Good", 1, 0)

# Train-Test----
# Load necessary library
library(class)

# Extract features and target variable
train_features <- GermanCredit_Train[, c("Age", "Amount")]
train_labels <- GermanCredit_Train$Class
test_features <- GermanCredit_Test[, c("Age", "Amount")]
test_labels <- GermanCredit_Test$Class


# Train and predict using KNN
knn_predictions <- knn(
  train = train_features,
  test = test_features,
  cl = train_labels,
  k = 5,
  p = 2  # 1:Manhattan, 2: Euclidan, inf:Chebyshev distance)
)

# Evaluate performance
confusion_matrix <- table(Predicted = knn_predictions, Actual = test_labels)
print(confusion_matrix)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", accuracy, "\n")

# Visualize heatmap----
# Load necessary libraries
library(ggplot2)
library(reshape2)

# Generate the confusion matrix (example using caret's confusionMatrix)
conf_matrix <- confusionMatrix(factor(knn_predictions), factor(GermanCredit_Test$Class))

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
# Load ggplot2 library
library(ggplot2)

# Ensure predicted class numbers are converted to labels
GermanCredit_Test$Predicted <- ifelse(knn_predictions == 1, "Good", "Bad")

# Visualize predictions with normalized scales
ggplot(GermanCredit_Test, aes(x = Age, y = Amount, color = Predicted)) +
  geom_point(size = 3) +
  labs(
    title = "KNN Predictions (Normalized Scale)",
    x = "Age (Normalized)",
    y = "Amount (Normalized)",
    color = "Predicted Class"
  ) +
  theme_minimal()

# Visualize the prediction test----
# Save min and max values during normalization
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
    title = "KNN Predictions (Original Scale)",
    x = "Age (Original Scale)",
    y = "Amount (Original Scale)",
    color = "Predicted Class"
  ) +
  theme_minimal()




# Hyperparameter: Grid-Search to find the best k----
grid_search_knn <- function(train_features, train_labels, test_features, test_labels, k_values) {
  results <- data.frame(k = integer(), accuracy = numeric())
  
  for (k in k_values) {
    # Train and predict using KNN
    knn_predictions <- knn(
      train = train_features,
      test = test_features,
      cl = train_labels,
      k = k
    )
    # Evaluate accuracy
    accuracy <- mean(knn_predictions == test_labels)
    # Store results
    results <- rbind(results, data.frame(k = k, accuracy = accuracy))
  }
  # Return sorted results
  results[order(-results$accuracy), ]
}

# Define range of k values
k_values <- 1:30  # Test k values from 1 to 30

# Perform grid search
grid_results <- grid_search_knn(
  train_features = train_features,
  train_labels = train_labels,
  test_features = test_features,
  test_labels = test_labels,
  k_values = k_values
)
grid_results[order(grid_results$k), ]       #grid_results$k

# Best k
best_k <- grid_results[1, "k"]
cat("Best k:", best_k, "with accuracy:", grid_results[1, "accuracy"], "\n")

library(ggplot2)
ggplot(grid_results, aes(x = k, y = accuracy)) +
  geom_line() +
  geom_point() +
  labs(title = "Grid Search for Optimal k", x = "Number of Neighbors (k)", y = "Accuracy") +
  theme_minimal()


# Hyperparameter: Random-Search to find the best k----
# Random Search Function
random_search_knn <- function(train_features, train_labels, test_features, test_labels, k_range, n_trials) {
  results <- data.frame(k = integer(), accuracy = numeric())
  
  for (i in 1:n_trials) {
    # Randomly sample k
    k <- sample(k_range, 1)  # Sample one k from the range
    
    # Train and predict using KNN
    knn_predictions <- knn(
      train = train_features,
      test = test_features,
      cl = train_labels,
      k = k
    )
    
    # Evaluate accuracy
    accuracy <- mean(knn_predictions == test_labels)
    
    # Store results
    results <- rbind(results, data.frame(k = k, accuracy = accuracy))
  }
  
  # Return sorted results
  results[order(-results$accuracy), ]  # Sort by accuracy in descending order
}

# Parameters for Random Search
k_range <- 1:30  # Range of k-values to search
n_trials <- 15   # Number of random trials

# Perform random search
random_results <- random_search_knn(
  train_features = train_features,
  train_labels = train_labels,
  test_features = test_features,
  test_labels = test_labels,
  k_range = k_range,
  n_trials = n_trials
)

# Display results
print(random_results)

random_results[order(-random_results$accuracy), ]       #random_results$k

# Best k
best_k <- random_results[1, "k"]
cat("Best k:", best_k, "with accuracy:", random_results[1, "accuracy"], "\n")

library(ggplot2)
ggplot(random_results, aes(x = k, y = accuracy)) +
  geom_point(size = 3) +
  labs(title = "Random Search Results for KNN", 
       x = "k (Number of Neighbors)", 
       y = "Accuracy") +
  theme_minimal()

