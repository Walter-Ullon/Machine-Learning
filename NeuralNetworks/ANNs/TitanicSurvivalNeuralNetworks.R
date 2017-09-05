#Survival on the Titanic

# Load Data
titanic_train <- read.csv(url("https://storage.googleapis.com/kaggle-competitions-data/kaggle/3136/train.csv?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1501687577&Signature=gw0ie3ZX%2BY8qYQAFxw0roeKEaOwZVo3wgiiuZ%2BGM7Iu8Z32VtHFtYi%2BPYVuJL59PKukqUcQh9AeuaYVHFywWgP2vkWdWiZoEahnJXU1UUyhdMVciQLYtdEKjbIiwMPAwOXb17XTEJj49B%2FTIPE4joz3pTZ5zhw7c9uEWcPL1I4RSWQd8JVU5WQlAtAMluz7Nl72sQG5kl%2FBIwfTjczgQBENWtwQQJ4WPkcvOq1kFtt28B4%2FCq53LlNAHZS3VZmAmxeJIxGyo8ZhfRgPc8OblT6FB3KVMOC8va5OLALfo0WIBOZ%2F1u%2B26Ju8xbUiSMY1GfFw8KBGGtdaLYgPh2Bd%2BkA%3D%3D"), stringsAsFactors = FALSE)
titanic_test <- read.csv(url("https://storage.googleapis.com/kaggle-competitions-data/kaggle/3136/test.csv?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1501687636&Signature=k4LLD59yIvbFUoHsssUTTIO1MH1RRErWSDuafTZS%2B%2BZxx85PhiXJ6fXfBOS9zIaercPNAWKHft%2FeiQiFDglrAo3e9E1pOTjFkBjuqt7BwE9x9OkAi3TKf20%2Fvw%2B%2B%2BaNaJzT9Utyu0lFZi%2BfipicgOYsEwNldwu6GvXwI3YVdLCraQj8Ulu808%2FqAZ1D0B%2BKImrLzXGul5Q%2FYeHKL26JiMHbqag1%2FPW9ELOC1OXQPaeIO1cwDc8vGAOGF9mBV01t4HuuLjzDmrvedxCayoZrkVhwDENwIQbjc%2FvTy8hEf6Bcg41w7ZiIL5kKDF9saiuMtoMGheSvRKuLwAF5jVTdglQ%3D%3D"), stringsAsFactors = FALSE)

# Inspect data
str(titanic_test)
table(titanic_train$Sex)


# -------- Prepare data for cleaning. CLEAN it -------------
# Add Column to each data set. Column of TRUE if train set, FALSE is test set.
titanic_train$isTrainSet <- TRUE
titanic_test$isTrainSet <- FALSE

# check to make sure all "TrainSet" data points returnTRUE
tail(titanic_train$isTrainSet)

# Add "Survived" column to "test" dataset in order to match it with "train" set.
titanic_test$Survived <- NA
names(titanic_test)

# Bind both datasets into singe one (for cleaning purposes)
titanic_full <- rbind(titanic_train, titanic_test)

# Query "embarked" column to find any missing values. Replace missing values with the node.
table(titanic_full$Embarked) # Node = "S"
titanic_full[titanic_full$Embarked == "", "Embarked"] <- "S"

# Query Age column to see of there are any missing values.
table(is.na(titanic_full$Age))
boxplot(titanic_full$Age)
boxplot.stats(titanic_full$Age)

# Obtain max age from stats and use it to filter outliers.
maxAge <- boxplot.stats(titanic_full$Age)$stats[5]
age_filter <- titanic_full$Age < maxAge # creates rule
titanic_full[age_filter,] # applies rule to data

# Use regression to predict unknown "age" values.
age_eqn = "Age ~ Pclass + Sex + SibSp + Parch + Embarked" # i.e. predict age given Pclass, Sex,...
age_model <- lm(
  formula = age_eqn,
  data = titanic_full[age_filter,] # supply linear regression method with clean 'age' data (no outliers).
)

# Predict values
age_row <- titanic_full[is.na(titanic_full$Age), c("Pclass", "Sex", "Embarked", "SibSp", "Parch")]
# above: only query rows that have missing values in "age". Only want to see "Pclass", "Sex", ... Save.
age_predictions <- predict(age_model, newdata = age_row) # predict using "age_model", on age_row.

# Replace missing values with predicted values.
titanic_full[is.na(titanic_full$Age), "Age"] <- age_predictions

#age_median <- median(titanic_full$Age, na.rm = TRUE) # calculate median (remove missing values from calc.)
#titanic_full[is.na(titanic_full$Age), "Age"] <- age_median


# Query "Fare" column to see of there are any missing values. Replace with median.
table(is.na(titanic_full$Fare))
boxplot(titanic_full$Fare)
boxplot.stats(titanic_full$Fare) # returns median 1st Q, 3rd Q, etc.

# Obtain max fare from stats and use it to filter outliers.
maxFare <- boxplot.stats(titanic_full$Fare)$stats[5]
fare_filter <- titanic_full$Fare < maxFare # creates rule
titanic_full[fare_filter,] # applies rule to data

# Use regression to predict unknown "fare" values.
fare_eqn = "Fare ~ Pclass + Sex + SibSp + Parch + Embarked" # i.e. predict fare given Pclass, Sex,...
fare_model <- lm(
  formula = fare_eqn,
  data = titanic_full[fare_filter,] # supply linear regression method with clean fare data (no outliers).
)

# Predict values
fare_row <- titanic_full[is.na(titanic_full$Fare), c("Pclass", "Sex", "Embarked", "SibSp", "Parch")]
# above: only query rows that have missing values in "fare". Only want to see "Pclass", "Sex", ... Save.
fare_predictions <- predict(fare_model, newdata = fare_row) # predict using "fare_model", on fare_row.

# Replace missing values with predicted values.
titanic_full[is.na(titanic_full$Fare), "Fare"] <- fare_predictions

# Categorical casting
titanic_full$Pclass <- factor(titanic_full$Pclass, levels=c(3,2,1),ordered=TRUE)
titanic_full$Pclass <- as.factor(titanic_full$Pclass)
titanic_full$Sex <- as.factor(titanic_full$Sex)
titanic_full$Embarked <- as.factor(titanic_full$Embarked)

# Re-split data into train and test set
titanic_train <- titanic_full[titanic_full$isTrainSet==TRUE,]
titanic_test <- titanic_full[titanic_full$isTrainSet==FALSE,]

# Cast "Survived" column in training set as categorical.
titanic_train$Survived <- as.factor(titanic_train$Survived)

# ---------------------------------------------------------


# Remove unnecessary & unique variables
titanic_train <- titanic_train[c(-1, -4, -9, -11, -12, -13)]
titanic_test <- titanic_test[c(-1, -2, -4, -9, -11, -12, -13)]

titanic_train <- model.matrix(  ~ Survived + Pclass + Sex + Age + SibSp + Parch + Fare, data = titanic_train )
titanic_test <- model.matrix(  ~ Pclass + Sex + Age + SibSp + Parch + Fare, data = titanic_test )
titanic_train <- titanic_train[, -1]
titanic_test <- titanic_test[, -1]
head(titanic_train)
head(titanic_test)

# Train model using a multilayer "feedforward" neural network. Use a single "hidden" node.
titanic_model <- neuralnet(Survived1 ~ Pclass.L + Pclass.Q + Sexmale + Age + SibSp + Parch + Fare, data = titanic_train, hidden = 7)
plot(titanic_model)

# Submit model to test data.
model_results <- compute(titanic_model, titanic_test) # eliminate "survived" from test data.
predicted_survived <- round(model_results$net.result)


# Save as .CSV and prep data for submission to Kaggle.
titanic_pred <- as.data.frame(predicted_survived)
write.csv(titanic_pred, file = "titanicPredictionNeuralNet.csv")
