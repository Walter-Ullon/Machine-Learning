# Mushroom classification of edible vs. poisonous. 
# Dataset downloaded from Kaggle.com

install.packages("C50")
library(C50)
library(gmodels)

# Import data from Kaggle (https://www.kaggle.com/uciml/mushroom-classification)
shrooms <- read.csv(url("https://storage.googleapis.com/kaggle-datasets/478/974/mushrooms.csv
                        ?GoogleAccessId=datasets@kaggle-161607.iam.gserviceaccount.com
                        &Expires=1501124301&Signature=jbRzz1BZ%2B0vuVCExQWpDZrinPfpx54HWrrWW
                        gSAlX64SLxugJ9c03TfmvxP7wSe0Miao5DutK5FiKDsdR4wMsjV16VObIoBnHYobAZHhE8j
                        TNFfn7%2FWjyQ3wnKsakiv0fK8zAwBuXw6r7C71DDmlPQi8CyQdtYdUUh4IQ893dnmfbF7t
                        FGlJd84UwTMQKciYHsGMz4L3vxDYFqygYwoKuKu7vB7btPaGRHbzIOXda96BYbyDOGumsIiwvgoh
                        OngO5Mpc1iO%2Bh2gepNImglISOgfdAzl7%2ByiOTCjCKp0T1jiYv2sOCsmBkJSMSjrMpxVJltxXRut
                        BHAgIr9lxN%2BPe3A%3D%3D"))
# Inspect data
str(shrooms)

# Distribution of edible vs poisonous shrooms
table(shrooms$class)

# Obtain random sample to split between training and test data. Set seed.
# Returns a vector with 900 random values
set.seed(7)
train_sample <- sample(8124, 7124)
str(train_sample)

# Split data into training and test sets.
# Check to see if both sets have ~ proportion of edible vs poisonous.
shrooms_train <- shrooms[train_sample, ]
shrooms_test <- shrooms[-train_sample, ]
prop.table(table(shrooms_train$class))
prop.table(table(shrooms_test$class))

# remove veil.type variable (17th col.) as it only contain one value "p" and tells us nothing.
shrooms_train <- shrooms_train[-17]
shrooms_test <- shrooms_test[-17]

# Train model.
# Exclude "class" class (edible, posonous) from training data (1st column) but supply it as target factor vector for classification (labels).
shrooms_model <- C5.0(shrooms_train[ ,-1], shrooms_train$class)
summary(shrooms_model)
pdf("shroomsTree.pdf", width=16,height=7)
plot(shrooms_model)
dev.off()

# Apply model to test data and predict...
# Compare prediction against true results.
shrooms_pred <- predict(shrooms_model, shrooms_test)
CrossTable(shrooms_test$class, shrooms_pred, prop.chisq = FALSE, prop.r = FALSE, dnn = c("actual", "predicted" ))


# MORAL OF THE STORY: always smell the mushrooms!!!

