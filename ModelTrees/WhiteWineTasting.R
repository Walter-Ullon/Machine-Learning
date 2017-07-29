# Application of Model & Regression Trees to estimation of wine quality .

# Install and load packages.
install.packages("rpart")
install.packages("RWeka")
library(RWeka)
library(rpart)
library(rpart.plot)

# Import data from UCI Machine Learning Data Repository
wine <- read.csv(url("https://raw.githubusercontent.com/dataspelunking/MLwR/master/Machine%20Learning%20with%20R%20(2nd%20Ed.)/Chapter%2006/whitewines.csv"))

# Explore data.
str(wine)
summary(wine)

# Examine "quality" distribution. Check for extremes.
pdf("QualityDistribution.pdf")
hist(wine$quality)
dev.off()

# Partition data into training and test sets (75/25 split).
wine_train <- wine[1:3750, ]
wine_test <- wine[3751:4898, ]

# Train "Regression Tree" model. Examine. Plot.
# Regression Trees use the average value of examples at leaf nodes to make numeric predictions.
m.rpart <- rpart(quality ~ ., data = wine_train)
m.rpart
summary(m.rpart)
pdf("TreePlot.pdf")
rpart.plot(m.rpart, digits = 3)
dev.off()


# Submit test data to model. Evaluate.
p.rpart <- predict(m.rpart, wine_test)
summary(p.rpart)
summary(wine_test$quality)

# Find correlation between predicted and actual values.
cor(p.rpart, wine_test$quality)

# Calculate the Absolute Mean Error (MAE).
MAE <- function(actual, predicted){
  mean(abs(actual - predicted))
}

MAE(p.rpart, wine_test$quality) # On average, the model is off 0.59 away from the actual (on a scale of 1 - 10)

# Obtain the MAE of the test set mean vs the predicted mean.
trainQualMean <- mean(wine_train$quality)
MAE(trainQualMean, wine_train$quality)

#-------------- Improve model Performance. Employ "Model Trees". -------------------
# Model Trees' leavs terminate in linear models. Each Model is then used to predict the value of 
# samples reaching this node.
# Use the M5' Algorithm. Examine.
m.m5p <- M5P(quality ~ ., data = wine_train)
m.m5p
summary(m.m5p)

# Apply model to test data. Evaluate. Find correlation between predicted and actual.
p.m5p <- predict(m.m5p, wine_test)
summary(p.m5p)
cor(p.m5p, wine_test$quality)

# Find MAE.
MAE(wine_test$quality, p.m5p)
