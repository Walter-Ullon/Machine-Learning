# Application of "Rule Learners" to the classification of Edible vs. Poisonous mushrooms.
# Data from UCI Machine Learning Repository.

# Install and load packages.
install.packages("RWeka")
library(RWeka)

# Import data, get snapshot.
shrooms <- read.csv(url("https://raw.githubusercontent.com/dataspelunking/MLwR/master/Machine%20Learning%20with%20R%20(2nd%20Ed.)/Chapter%2005/mushrooms.csv"))
str(shrooms)
table(shrooms$type)

# Remove "veil_type" variable since it is comprised of a singular value and won't be of any help.
shrooms$veil_type <- NULL

# Create rule by employing the "1R" algorithm.
shrooms_1RModel <- OneR(type ~ ., data = shrooms) # type ~ . -> "predict 'type' using the rest of the variables in the dataset"

# Print created rules. Obtain summary.
shrooms_1RModel
summary(shrooms_1RModel)

# Create rule by employing the "JRip" algorithm.
shrooms_JRipModel <- JRip(type ~ ., data = shrooms) # type ~ . -> "predict 'type' using the rest of the variables in the dataset"

# Print created rules. Obtain summary.
shrooms_JRipModel
summary(shrooms_JRipModel)












