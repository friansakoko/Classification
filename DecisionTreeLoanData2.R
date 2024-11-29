# Load necessary libraries
library(caret)
library(rpart)
library(rpart.plot)
library(ggplot2)

# Load dataset (replace with the correct URL or dataset path)
url <- "https://raw.githubusercontent.com/friansakoko/Classification/refs/heads/main/loan_data.csv"
GermanCredit <- read.csv(url)

# View the first few rows of the data
View(GermanCredit)
str(GermanCredit)

# Preprocess the data ----
# Convert the target variable (loan_status) to a factor if not already
GermanCredit$loan_status <- factor(GermanCredit$loan_status, levels = c("1", "0"))

# Set a seed for reproducibility
set.seed(123)

# Split data into training and testing sets (80% train, 20% test) ----
train_index <- createDataPartition(GermanCredit$loan_status, p = 0.8, list = FALSE)
GermanCredit_Train <- GermanCredit[train_index, ]
GermanCredit_Test <- GermanCredit[-train_index, ]

# Set up train control with cross-validation
train_control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation





##### Model 1: Training the Decision Tree Prediction  ----
# Train a Decision Tree Prediction Model with all variables
PredictionModel_DecisionTree_Complete <- train(loan_status ~ ., 
                                               data = GermanCredit_Train, 
                                               method = "rpart", 
                                               trControl = train_control)
print(PredictionModel_DecisionTree_Complete)

# Visualize the Decision Tree Prediction Model (all variables as predictors)
rpart.plot(PredictionModel_DecisionTree_Complete$finalModel, type = 3, extra = 101, 
           fallen.leaves = TRUE, main = "Decision Tree for Loan Data")



##### Model 2: Training the Decision Tree Prediction ----
# Train a Decision Tree Prediction Model with selected variables
PredictionModel_DecisionTree_3Variables <- train(loan_status ~ loan_amnt + person_income + person_emp_exp, 
                                                 data = GermanCredit_Train, 
                                                 method = "rpart", 
                                                 trControl = train_control)
print(PredictionModel_DecisionTree_3Variables)

# Visualize the Decision Tree Prediction Model (selected variables)
rpart.plot(PredictionModel_DecisionTree_3Variables$finalModel, type = 3, extra = 101, 
           fallen.leaves = TRUE, main = "Decision Tree for Loan Dataset (Selected Variables)")





# Apply the prediction model on the test data -----
PredictionModel_DecisionTree_3Variables <- predict(PredictionModel_DecisionTree_3Variables, newdata = GermanCredit_Test)

# Ensure PredictionClassComplete is a factor with the same levels as loan_status
PredictionModel_DecisionTree_3Variables <- factor(PredictionModel_DecisionTree_3Variables, levels = levels(GermanCredit_Test$loan_status))

# Generate the confusion matrix
cm <- confusionMatrix(PredictionModel_DecisionTree_3Variables, GermanCredit_Test$loan_status)
print(cm)

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


# Map 1 to "Good" and 0 to "Bad" for better readability----
cm_df$Prediction <- factor(cm_df$Prediction, levels = c("1", "0"), labels = c("Good", "Bad"))
cm_df$Actual <- factor(cm_df$Actual, levels = c("1", "0"), labels = c("Good", "Bad"))

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
