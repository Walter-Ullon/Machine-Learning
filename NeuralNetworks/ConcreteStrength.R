# Application of Neural Network Algorithms to Concrete Strength Modeling.
# Data collected from UCI Machine Learning Repository.

# Install and load packages.
install.packages("neuralnet")
library(neuralnet)

# Import data. Explore. Summarize.
concrete <- read.csv(url("https://raw.githubusercontent.com/dataspelunking/MLwR/master/Machine%20Learning%20with%20R%20(2nd%20Ed.)/Chapter%2007/concrete.csv"))
str(concrete)
summary(concrete)

# Normalize features in the data set to stop some variables from dominating others.
normalize <- function(x){
  return((x - min(x)) / (max(x) - min(x)))
}

concrete_norm <- as.data.frame(lapply(concrete, normalize))
summary(concrete_norm$strength)

# Split data into training and test sets (75-25).
concrete_train <- concrete_norm[1:773, ]
concrete_test <- concrete_norm[774:1030, ]

# Train model using a multilayer "feedforward" neural network. Use a single "hidden" node.
concrete_model <- neuralnet(strength ~ cement + slag + ash + water + superplastic + coarseagg + fineagg + age, data = concrete_train)
# Visualize model.
plot(concrete_model)

# Evaluate model performance. Apply model to test data.
model_results <- compute(concrete_model, concrete_test[1:8]) # eliminate "strength" from test data.
predicted_strength <- model_results$net.result

# Compute correlation between actual and predicted results.
cor(predicted_strength, concrete_test$strength)

#-------------- Improve Model -----------------
# We will add 5 "hidden" nodes.
concrete_model2 <- neuralnet(strength ~ cement + slag + ash + water + superplastic + coarseagg + fineagg + age, data = concrete_train, hidden = 5)
# Visualize model.
pdf("concreteNN2.pdf",width=6,height=4,paper='special')
plot(concrete_model2)
dev.off()

# Evaluate model performance. Apply model to test data.
model_results2 <- compute(concrete_model2, concrete_test[1:8]) # eliminate "strength" from test data.
predicted_strength2 <- model_results2$net.result

# Compute correlation between actual and predicted results.
cor(predicted_strength2, concrete_test$strength)


