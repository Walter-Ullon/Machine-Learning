# Survival on the Titanic

# Load Data (links expire - must sign into kaggle and copy new links)
titanic_train <- read.csv(url("https://storage.googleapis.com/kaggle-competitions-data/kaggle/3136/train.csv?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1503149409&Signature=n7NVB%2Be2TMDsY9E9aygUxvAjyT6tTeP4nXfbZicZ8BDfXajpqi7937layk7%2Fi%2Bg%2Bq6dyDr%2BRbLuIN%2BtcK%2Br%2FrGHibYtrLHwfcfJAKoApD0HJCjbaJpaIs2AUEKU%2FL4qExeYVLcjFDHWkic1PBz62AfhRyT9RdTQ4VuyRkyvbDRhMWHMApvMUtam9F9fdvDrQwjvBzS7MFEYTBqT6%2BcKxcWtGMNO6t%2BrbwzzMyMNMPcdcbk0C%2BKEj7unRreO%2BQjSIyzLhxc47csT%2FuqfGz8qoxTyBfF3lpDuhfJUqVT1B56aUQZSViWTTSAGDu5KUEokkSuY0K8gCYa9e0abiThxjsg%3D%3D"), stringsAsFactors = FALSE)
#write.csv(titanic_train, file = "titanic_train.csv")
titanic_test <- read.csv(url("https://storage.googleapis.com/kaggle-competitions-data/kaggle/3136/test.csv?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1503149443&Signature=RCoWifzKTM5X7enJq9b0QccLq2ls8SlHRSYks7PgBXQ3Lrw3myskMGU3IsyhUbc6JnBxP%2FcDH5X6ohKOEujDMIpXxA5Pea5Kr6hSD%2FOvxEsbaKNIzTtVX3lWH0cBW0LpxtqqB%2FqA%2FcBJnwz39gOREhKFMXFD3vRVSlaPlmp6Ct4EEoGTUV1K7idWd7XEAUiNan9lIpHvqmQwOisKwRRBy4h69t9ef334mC47cUHDfz8RpfIK6htzpP0FKjynHa9ZE%2FT92hGGOWU0M6RfAJQvWNq02Vvh3KhUpPzmmAI5qk4%2B7dbr0qFY5ZIzzCTu6LWJga1zuDp1RgHzt3Cbz2sIMA%3D%3D"), stringsAsFactors = FALSE)
#write.csv(titanic_test, file = "titanic_test.csv")

#titanic_train <- read.csv("titanic_train.csv")
#titanic_test <- read.csv("titanic_test.csv")

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


#fare_median <- median(titanic_full$Fare, na.rm = TRUE) # calculate median (remove missing values from calc.)
#titanic_full[is.na(titanic_full$Fare), "Fare"] <- fare_median

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
titanic_train <- titanic_train[c(-1, -4, -9, -10, -11, -12, -13)]
titanic_test <- titanic_test[c(-1, -2, -4, -9, -10, -11, -12, -13)]

# Train model
titanic_model <- C5.0(titanic_train[-1], as.factor(titanic_train$Survived))
summary(titanic_model)

# Make prediction
titanic_pred <- predict(titanic_model, titanic_test)

# Obtain proportions
prop.table(table(titanic_test$Sex))
prop.table(table(titanic_pred))

# Save as .CSV and prep data for submission to Kaggle.
titanic_pred <- as.data.frame(titanic_pred)
write.csv(titanic_pred, file = "titanicPrediction.csv")

#--------------------------------------------------------------------
# Implement "boosting" algorithms to improve model. (See: AdaBoost).
#--------------------------------------------------------------------
# below: learn to predict "Survived" based on all of the categories found in data = titanic_train"
AdaB_model <- boosting(Survived ~ ., data = titanic_train, boos = TRUE)
predict_AdaB <- predict(AdaB_model, titanic_test)
str(predict_AdaB)

# Prep to save as CSV
colHeadings <- c("PassengerId", "Survived")
AdaB_predDataframe <- data.frame(c(892:1309), c(predict_AdaB$class))
names(AdaB_predDataframe) <- colHeadings
write.csv(AdaB_predDataframe, file = "AdaB_TitanicPred.csv", row.names = FALSE)




