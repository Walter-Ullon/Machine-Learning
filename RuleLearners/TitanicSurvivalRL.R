# Survival on the Titanic using Rule Learning algorithms.

# Load Data
titanic_train <- read.csv(url("https://storage.googleapis.com/kaggle-competitions-data/kaggle/3136/train.csv?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1501202527&Signature=kH%2B5q%2F3a9R9v4LVII%2FBnOYCfO0In8FiJMD7RNYEHtWWnY1HGYnR7KOJx0UWVwDaVpWx2Hw6Ds8%2FPSKzO3V8DjkjDByuBmNQunOCPWfwh2RH79cNBHGQDhHhoR6ttdV8gDPJq5EGwWvHT%2BTZbGGwKEDLrIChKpnV90ScbT9DwXELp1a%2BhciCPV4RKasuTIo37%2FvFuJY4MFCWbPj%2B9eFUfHxyfF3FEKEJ%2FN7gYznyb%2BMkUnr3d3Kuh%2BXFtGO3zwNXDx0LZI5FUtxkqeym1xKAtAUSDsIv1OW%2FIoL4pH9zHoysSU0JLltZOtXBY74zouw1xm23lfcA25AJULGQ47aitRg%3D%3D"), stringsAsFactors = FALSE)
titanic_test <- read.csv(url("https://storage.googleapis.com/kaggle-competitions-data/kaggle/3136/test.csv?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1501202573&Signature=hNq96PKu5916QwmTLsKox0afkVwsdRGWrsRlloxH5zwgXdJ2BCREDCOqX%2FsYraKszG7LkC%2FF1MFT%2FMJ1Ccb7vFmk6jlPCL9ic2L4S7PIBOBP29OsEhh8Wu0%2BXRbCjAwBvAzJVBEzSCDcIhemPLLTUXbh6hfXLOeKfFVbTwM6oqrUxGVQta6iJpnJmD96o2aOqsfdTtNNG8Hmxqhp15EDwWC7VcHOFxaZvwBrprWrbvxF82ILfDsjqPl5AnYzZhOCYu%2FF2VueKceTQ5eXTatKEM9%2B0D6K8XCIkswsehSJv8IrhYa%2BcoTiA%2FKsbcrnSLu7jdORwpHJBS6XaLDh3f8m5w%3D%3D"), stringsAsFactors = FALSE)

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


# Remove unnecessary & unique variables (Rule learners need strictly categorical predictors).
titanic_train <- titanic_train[c(-1, -4, -6, -9, -10, -11, -12, -13)]
titanic_test <- titanic_test[c(-1, -2, -4, -6, -9, -10, -11, -12, -13)]

# Train model
titanic_model <- JRip(Survived ~ ., data = titanic_train)
titanic_model
summary(titanic_model)


# Make prediction
titanic_pred <- predict(titanic_model, titanic_test)

# Obtain proportions
prop.table(table(titanic_test$Sex))
prop.table(table(titanic_pred))

# Save as .CSV and prep data for submission to Kaggle.
titanic_pred <- as.data.frame(titanic_pred)
write.csv(titanic_pred, file = "titanicPredictionRL.csv")

