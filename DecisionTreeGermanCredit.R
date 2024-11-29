library(rpart)
library(rpart.plot)
library(caret)

#help(package ="caret")
data(GermanCredit)
View(GermanCredit)

# Preprocess the data ----
# Convert the target variable (Class) to a factor if not already
GermanCredit$Class <- factor(GermanCredit$Class, levels = c("Good", "Bad"))
# Set a seed for reproducibility
set.seed(42)

# Split data into training and testing sets (80% train, 20% test) ----
train_index <- createDataPartition(GermanCredit$Class, p = 0.8, list = FALSE)
GermanCredit_Train <- GermanCredit[train_index, ]
GermanCredit_Test <- GermanCredit[-train_index, ]

#View(GermanCredit_Train)
#View(GermanCredit_Test)

# Set up train control with cross-validation
train_control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation

# Training the Decision Tree Prediction Model 1 ----
# Train a Decision Tree Prediction Model 
# All variables as predictors
PredictionModel_DecisionTree_Complete <- train(Class ~ ., data = GermanCredit_Train, method = "rpart", trControl = train_control)
PredictionModel_DecisionTree_Complete
# Visualize the Decision Tree Prediction Model (all variables as predictors)
rpart.plot(PredictionModel_DecisionTree_Complete$finalModel, type = 3, extra = 101, fallen.leaves = TRUE, main = "Decision Tree for German Credit Dataset")

# Training the Decision Tree Prediction Model 2----
# Train a Decision Tree Prediction Model 
# Three variables i.e. duration, amount, and installmentratepercentage as predictors)
PredictionModel_DecisionTree_3Variables <- train(Class ~ Duration + Amount + InstallmentRatePercentage, data = GermanCredit_Train, method = "rpart", trControl = train_control)
# Visualize the Decision Tree Prediction Model (three variables i.e. duration, amount, and installmentratepercentage as predictors)
rpart.plot(PredictionModel_DecisionTree_3Variables$finalModel, type = 3, extra = 101, fallen.leaves = TRUE, main = "Decision Tree for German Credit Dataset")

# Apply the prediction model on the test data (GermanCredit_Test) -----
PredictionClassComplete <- predict(PredictionModel_DecisionTree_Complete, newdata = GermanCredit_Test)
View(PredictionClassComplete)
PredictionClassComplete

# Evaluate the model's overall accuracy ----
PredictionAccuracyComplete <- sum(PredictionClassComplete == GermanCredit_Test$Class) / nrow(GermanCredit_Test)
cat("Model accuracy COMPLETE Decision Tree model on GermanCredit_Test data:", round(PredictionAccuracyComplete * 100, 2), "%\n")

# Evaluation ----
library(ggplot2)
# Generate confusion matrix
cm <- confusionMatrix(PredictionClassComplete, GermanCredit_Test$Class)
# Extract the confusion matrix table
cm_table <- as.table(cm$table)
# Convert to a data frame for ggplot
cm_df <- as.data.frame(cm_table)
colnames(cm_df) <- c("Prediction", "Actual", "Freq")
# Plot heatmap using ggplot
ggplot(data = cm_df, aes(x = Prediction, y = Actual, fill = Freq)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  theme_minimal() +
  labs(title = "Confusion Matrix Heatmap",
       x = "Predicted Class",
       y = "Actual Class",
       fill = "Frequency")


# Confusion matrix for a more detailed accuracy evaluation ----
# Load required libraries
library(ggplot2)
# Generate confusion matrix
cm <- confusionMatrix(PredictionClassComplete, GermanCredit_Test$Class)
# Extract the confusion matrix table
cm_table <- as.table(cm$table)
# Convert to a data frame for ggplot
cm_df <- as.data.frame(cm_table)
colnames(cm_df) <- c("Prediction", "Actual", "Freq")
# Plot heatmap using ggplot
ggplot(data = cm_df, aes(x = Prediction, y = Actual, fill = Freq)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  theme_minimal() +
  labs(title = "Confusion Matrix Heatmap",
       x = "Predicted Class",
       y = "Actual Class",
       fill = "Frequency")
