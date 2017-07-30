# Application to "Optical Character Recognition" using SVMs.
# Data from UCI Machine Learning Data Repository.

# Install and load packages.
install.packages("kernlab")
library(kernlab)

# Load Data. Inspect.
# Note: SVMs require all features to be numeric. They should also be normalized.
letters <- read.csv(url("https://raw.githubusercontent.com/dataspelunking/MLwR/master/Machine%20Learning%20with%20R%20(2nd%20Ed.)/Chapter%2007/letterdata.csv"))
str(letters)

# Partition data into training and test sets (80-20)
letters_train <- letters[1:16000, ]
letters_test <- letters[16001:20000, ]


# Train classifier using a "vanilla" (linear kernel). Inspect.
letter_classifer <- ksvm(letter ~ ., data = letters_train, kernel = "vanilladot")
letter_classifer

# Evaluate model performance on test data.
letter_predictions <- predict(letter_classifer, letters_test, type = "response")

table(letter_predictions, letters_test$letter)
agreement <- letter_predictions == letters_test$letter
prop.table(table(agreement))


#--------------------- Improve Model ---------------------
# Employ Gaussian RBF Kernel.
letter_classifer_rbf <- ksvm(letter ~ ., data = letters_train, kernel = "rbfdot")

# Evaluate model performance on test data.
letter_predictions_rbf <- predict(letter_classifer_rbf, letters_test, type = "response")

agreement_rbf <- letter_predictions_rbf == letters_test$letter
prop.table(table(agreement_rbf))



