##################################################
# ECON 418-518 Homework 3
# Karelly Gallegos
# The University of Arizona
# karellygallegos@arizona.edu 
# 8 December 2024
###################################################


#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# Load packages
pacman::p_load(data.table, caret, randomForest, glmnet)

# Set seed
set.seed(418518)

# Set the working directory
setwd("Desktop")

# Load the data and label it "data"
data <- read.csv("ECON_418-518_HW3_Data.csv")

#making sure I loaded it correctly
head(data)

#####################
# Problem 1
#####################


#################
# Question (i)
#################

# Drop the specified columns using subset function
#gave it a different name incase I needed the full data set again
dt <- subset(data, select = -c(fnlwgt, occupation, relationship, 
                                         capital.gain, capital.loss, 
                                         educational.num))

# Checking first few rows to make sure the columns were dropped correctly
head(dt)

#################
# Question (ii)
#################

##############
# Part (a)
##############

# Convert the 'income' column to a binary indicator (1 for '>50K', 0 otherwise)
dt$income <- ifelse(dt$income == ">50K", 1, 0)

# Check the first few rows to make sure I did it right
head(dt$income)

##############
# Part (b)
##############

# Convert the 'race' column to a binary indicator (1 for 'White', 0 otherwise)
dt$race <- ifelse(dt$race == "White", 1, 0)

# Check the first few rows to make sure I did it right
head(dt$race)

##############
# Part (c)
##############

# Convert the 'gender' column to a binary indicator (1 for 'Male', 0 otherwise)
dt$gender <- ifelse(dt$gender == "Male", 1, 0)

# Check the first few rows to make sure I did it right
head(dt$gender)

##############
# Part (d)
##############

# Convert the 'workclass' column to a binary indicator 
#(1 for 'Private', 0 otherwise)
dt$workclass <- ifelse(dt$workclass == "Private", 1, 0)

# Check the first few rows to make sure I did it right
head(dt$workclass)

##############
# Part (e)
##############

# Convert the 'native.country' column to a binary indicator 
#(1 for 'United-States', 0 otherwise)
dt$native.country <- ifelse(dt$native.country == 
                                        "United-States", 1, 0)

# Check the first few rows to make sure I did it right
head(dt$native.country)

##############
# Part (f)
##############

# Convert the 'marital.status' column to a binary indicator 
#(1 for 'Married-civ-spouse', 0 otherwise)
dt$marital.status <- ifelse(dt$marital.status == 
                                        "Married-civ-spouse", 1, 0)

# Check the first few rows to make sure I did it right
head(dt$marital.status)

##############
# Part (g)
##############

# Convert the 'education' column to a binary indicator 
#(1 for 'Bachelors' or 'Masters' or 'Doctorate, 0 otherwise)
dt$education <- ifelse(dt$education == "Bachelors" |
                                   dt$education == "Masters" |
                                   dt$education == "Doctorate", 1, 0)

# Check the first few rows to make sure I did it right
head(dt$education)

##############
# Part (h)
##############

# Create a new variable 'age sq' that is the square of the 'age' column
dt$age_sq <- dt$age^2

# Optionally, check the first few rows to confirm the new column is created
head(dt$age_sq)

##############
# Part (i)
##############

# Standardize age, age squared, and hours per week
dt$age <- scale(dt$age)
dt$age_sq <- scale(dt$age_sq)
dt$hours.per.week <- scale(dt$hours.per.week)

#################
# Question (iii)
#################

##############
# Part (a)
##############

# Proportion of individuals with income > 50K
#number pops up in "Values" in the Environment
income_above_50k <- mean(dt$income == 1, na.rm = TRUE)

##############
# Part (b)
##############

# Proportion of individuals in the private sector
in_private_sector <- mean(dt$workclass == 1, na.rm = TRUE)

##############
# Part (c)
##############

# Proportion of married individuals
is_married <- mean(dt$marital.status == 1, na.rm = TRUE)

##############
# Part (d)
##############

# Proportion of females
is_female <- mean(dt$gender == 0, na.rm = TRUE)

##############
# Part (e)
##############

#Total number of missing values (i.e., "?" values) in the dataset
missing_values <- sum(is.na(dt))

##############
# Part (f)
##############

#converting "income" to a factor
dt$income <- as.factor(dt$income)

#################
# Question (iv)
#################

##############
# Part (a)
##############

# Splitting data with a 70-30 training and testing split
train_size <- floor(0.7 * nrow(dt))

#Shuffle the data, not explicitly mentioned in directions, but feels right
shuffled_indices <- sample(nrow(dt))

##############
# Part (b)
##############

# Create the training data table 
train_data <- dt[1:train_size, ]

##############
# Part (c)
##############

# Create the testing data table
test_data <- dt[(train_size + 1):nrow(dt), ]

#################
# Question (v)
#################

##############
# Part (b)
##############

# set up a grid of lambda values between 10^5 and 10^(-2)
lambda_grid <- 10^seq(5, -2, length.out = 50)

# Train the Lasso model using 10-fold cross-validation
lasso_model <- train(income ~ ., 
                     data = train_data, 
                     method = "glmnet", 
                     tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid), 
                     trControl = trainControl(method = "cv", number = 10))

# Get the best value of lambda
best_lambda <- lasso_model$bestTune$lambda
cat("Best lambda:", best_lambda, "\n")

# View the cross-validation results
lasso_model$results

##############
# Part (c)
##############

#look at the highest accuracy for that lambda
max(lasso_model$results$Accuracy)

#look at the lambda from earlier
best_lambda

##############
# Part (d)
##############

#finding variables that have coefficient estimates that are nearly zero
lasso_coefficients <- coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda)
zero_coeff_vars <- rownames(lasso_coefficients)[abs(as.vector(lasso_coefficients)) < 1e-4]

##############
# Part (e)
##############

# Filter out non-zero variables from training data
non_zero_vars <- setdiff(colnames(train_data), zero_coeff_vars)

# Lasso Model with non-zero variables
lasso_refined <- train(
  income ~ ., 
  data = train_data, 
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid))

# Ridge Model with non-zero variables
ridge_refined <- train(income ~ ., 
                        data = train_data, 
                        method = "glmnet",
                        trControl = trainControl(method = "cv", number = 10),
                        tuneGrid = expand.grid(alpha = 0, lambda = lambda_grid))

# Compare Classification Accuracies
lasso_accuracy <- max(lasso_refined$results$Accuracy)
ridge_accuracy <- max(ridge_refined$results$Accuracy)

cat("Lasso Classification Accuracy:", lasso_accuracy, "\n")
cat("Ridge Classification Accuracy:", ridge_accuracy, "\n")

#################
# Question (vi)
#################

##############
# Part (b)
##############

# Define the grid 
tree_grid <- expand.grid(mtry = c(2, 5, 9)) 

# Train Random Forest models with 5-fold cross-validation
#First with 100 trees
randomf_model_100 <- train(income ~ ., 
                      data = train_data, 
                      method = "rf", trControl = trainControl
                      (method = "cv", number = 5, verboseIter = TRUE), 
                      tuneGrid = tree_grid, 
                      ntree = 100 )

#200 Trees
randomf_model_200 <- train(income ~ ., 
                          data = train_data, 
                          method = "rf", trControl = trainControl
                          (method = "cv", number = 5, verboseIter = TRUE), 
                          tuneGrid = tree_grid, 
                          ntree = 200 )

#300 Trees
randomf_model_300 <- train(income ~ ., 
                          data = train_data, 
                          method = "rf", trControl = trainControl
                          (method = "cv", number = 5, verboseIter = TRUE), 
                          tuneGrid = tree_grid, 
                          ntree = 300,)


##############
# Part (c)
##############

#Finding the model with the highest accuracy
randomf_model_100$results$Accuracy
randomf_model_200$results$Accuracy
randomf_model_300$results$Accuracy

##############
# Part (e)
##############

# Make predictions using the best random forest model 
predictions_rf <- predict(randomf_model_100, train_data)

# Generate the confusion matrix
cm_rf <- confusionMatrix(predictions_rf, train_data$income)

# Print confusion matrix
print(cm_rf)

# Check false negatives
cm_rf$table[2,1]

#Check for false positives
cm_rf$table[1,2]


#################
# Question (vii)
#################

# Make predictions using the best random forest model
predictions_rf_test <- predict(randomf_model_100, test_data)

# Evaluate the classification accuracy on the testing set
test_accuracy_rf <- mean(predictions_rf_test == test_data$income)
cat("Classification Accuracy on Testing Set:", test_accuracy_rf, "\n")
