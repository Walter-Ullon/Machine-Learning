# Identification of rskiy bank loans using Decision Trees (germany -> DM = Deutsche Marks).

# Install and load packages
install.packages("C50")
library(C50)
library(gmodels)

# Import data from UCI Machine Learning Repository (http://archive.ics.uci.edu/ml)
# Inspect data
credit <- read.csv(url("https://raw.githubusercontent.com/dataspelunking/MLwR/master/Machine%20Learning%20with%20R%20(2nd%20Ed.)/Chapter%2005/credit.csv"))
str(credit)

# Inspect checking and savings balanace as possible indicators of default
table(credit$checking_balance)
table(credit$savings_balance)

# Obtain summary statistics on loan duration and loan amount
summary(credit$months_loan_duration)
summary(credit$amount)

# Obtain total number of defaults
table(credit$default)

# Obtain random sample to split between training and test data. Set seed.
# Returns a vector with 900 random values
set.seed(123)
train_sample <- sample(1000, 900)
str(train_sample)

# Split data into training and test sets.
# Check to see if both sets have ~ proportion of default loans (~30%, as in original set).
credit_train <- credit[train_sample, ]
credit_test <- credit[-train_sample, ]
prop.table(table(credit_train$default))
prop.table(table(credit_test$default))

# Train model.
# Exclude "default" class from training data (17th column) but supply it as target factor vector for classification (labels).
credit_model <- C5.0(credit_train[-17], credit_train$default)
credit_model # inspect...

# Inspect Tree's decisions and acquire stats on model performance on the training data.
summary(credit_model)

# Apply model to test data and predict...
# Compare prediction against true results.
credit_pred <- predict(credit_model, credit_test)
CrossTable(credit_test$default, credit_pred, prop.chisq = FALSE, prop.r = FALSE, dnn = c("actual default", "predicted default" ))

#-------------------------------------------------------------------------------------------
# NOTE: up to this point, the model performed rather poorly at identifying true defaults.
#       The proceeding code is intended to "fine-tune" our model to improve its performance.
#       See "Boosting" for more details.
#-------------------------------------------------------------------------------------------

# We employ the same classifying algorith (C5.0) but we "boost" it by adding "trials".
# The trials parameter indicates the additional number of separate decision trees to use. We set trials = 10.
credit_boost10 <- C5.0(credit_train[-17], credit_train$default, trials = 10)
credit_boost10

# Obtain new model's summary and peformance on the test data. Save to .txt file.
sink("credit_boost10.txt")
summary(credit_boost10)
sink()

# Test new model on test data
# Compare predicition against true results.
credit_boost10_pred <- predict(credit_boost10, credit_test)
CrossTable(credit_test$default, credit_boost10_pred, prop.chisq = FALSE, prop.r = FALSE, dnn = c("actual default", "predicted default" ))


#-------------------------------------------------------------------------------------------
# NOTE: the model is performing better, but still making costly "mistakes".
#       We will create a "cost" matrix to assign a penalty to each type of mistake. 
#-------------------------------------------------------------------------------------------

# Build matrix..
matrix_dimensions <- list(c("no", "yes"), c("no", "yes"))
names(matrix_dimensions) <- c("predicted", "actual")
matrix_dimensions

# Assign penalties for the various types of errors (4 values), in order:
    # predicted "no" -- actual "no"
    # predicted "yes" -- actual "no"
    # predicted "no" -- actual "yes"
    # predicted "yes" -- actual "yes"

# We assume false negatives (predicted no default, loanee defaulted) cost the bank 4 times as much as a missed opportunity.
error_cost <- matrix(c(0, 1, 4, 0), nrow = 2, dimnames = matrix_dimensions)
error_cost

# Apply cost matrix to C5.0 algo using "cost" parameter.
credit_cost <- C5.0(credit_train[-17], credit_train$default, costs = error_cost)
credit_cost_pred <- predict(credit_cost, credit_test)
CrossTable(credit_test$default, credit_cost_pred, prop.chisq = FALSE, prop.r = FALSE, dnn = c("actual default", "predicted default" ))

#-------------------------------------------------------------------------------------------
# NOTE: compared to the previous models, the version above produces more mistakes (37% vs. 18%).
#       However, the types of mistkes are very different (less false negatives -- 79% vs. 42% & 61%). 
#-------------------------------------------------------------------------------------------
