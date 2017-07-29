# Application of Regression Models to Medical Insurance Data. Dependent variable: EXPENSES.
# Data simulated from demographic statistics from US Censues Bureau.

# Install and load packages.
install.packages("psych")
library(psych)

# Import data and convert nominal features to factors (stringAsFactors = TRUE).
insurance <- read.csv(url("https://raw.githubusercontent.com/dataspelunking/MLwR/master/Machine%20Learning%20with%20R%20(2nd%20Ed.)/Chapter%2006/insurance.csv"), stringsAsFactors = TRUE)
str(insurance)

# Check data for normality (linear models work better if data is normally distributed).
summary(insurance$expenses)
pdf('histogramTest.pdf')
hist(insurance$expenses)
dev.off()

# Explore relationships among features by usig a correlation matrix (use numeric fetures).
cor(insurance[c("age", "bmi", "children", "expenses")])

# Visualize correlations using a scatterplot matrix.
pdf('CorrelationMatrix.pdf')
pairs(insurance[c("age", "bmi", "children", "expenses")])
dev.off()

pdf('CorrelationMatrixPanels.pdf')
pairs.panels(insurance[c("age", "bmi", "children", "expenses")])
dev.off()

# Build regression model. Print coefficients.
ins_model <- lm(expenses ~ age + children + bmi + sex + smoker + region, data = insurance)
ins_model

# Evaluate model and obtain statistics.
summary(ins_model)

#--------------- Improve Model -------------------
# Add non-linear term (function of age^2).
# We assume the effects of age are disproportonally expensive for the oldest population.
insurance$age2 <- insurance$age^2

# Create binary indicator for BMI.
# We assume the effects of BMI on medical expenses is bigger if BMI > 30 (obese).
# Returns 1 if BMI >= 30, 0 otherwise.
insurance$bmi30 <- ifelse(insurance$bmi >= 30, 1, 0)

# Create "interaction effects" between variables. 
# We assume the effects of obesity and smoking coumpound to a bigger problem than their separate contributions.
# We pass it to the model as such: expenses ~ bmi30*smoker.








